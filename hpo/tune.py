"""
Optuna-based hyperparameter optimization for Neural Lyapunov.

Finds hyperparameters that minimize exclusion boundary size while maintaining
Lyapunov feasibility. Supports parallel workers via SQLite storage.

Usage:
    python tune.py --config ../configs/sta.yaml --n_trials 100
    python tune.py --config ../configs/sta.yaml --n_trials 50 --storage sqlite:///sta_hpo.db
"""
from __future__ import annotations
import os, sys, math, copy, argparse, time, glob, json
from typing import Dict, Any, Tuple, Optional, List, Callable
from datetime import datetime

import torch
import numpy as np

import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler

# Import the existing pipeline
try:
    from ..utils import load_yaml, ensure_dir, dump_yaml
    from ..train import train as train_fn
    from ..controllers import get_controller
    from ..models import SimpleLyapNet
    from ..gauges import OrientedEllipsoidGauge
    from ..validators import validate_global_outside, validate_global_outside_1d
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_yaml, ensure_dir, dump_yaml
    from train import train as train_fn
    from controllers import get_controller
    from models import SimpleLyapNet
    from gauges import OrientedEllipsoidGauge
    from validators import validate_global_outside, validate_global_outside_3d, validate_global_outside_1d


# =============================================================================
# Checkpoint & Evaluation Utilities
# =============================================================================

def _latest_subdir(root: str) -> str:
    subs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subs:
        raise RuntimeError(f"No subdirectories found under {root}")
    return max(subs, key=os.path.getmtime)


def _find_best_ckpt(run_dir: str) -> Optional[str]:
    for name in ["best_model_polished.pth", "best_model.pth", "last_model.pth"]:
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _load_ckpt_payload(ckpt_path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "polished" in ckpt:
        payload = ckpt["polished"]
        cfg = ckpt.get("cfg", {})
    elif "best" in ckpt:
        payload = ckpt["best"]
        cfg = ckpt.get("cfg", {})
    elif "V" in ckpt and "G" in ckpt:
        payload = {"V": ckpt["V"], "G": ckpt["G"], "iter": ckpt.get("iter", -1)}
        cfg = ckpt.get("cfg", {})
    else:
        raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}")
    return payload, cfg


def _build_models(payload: Dict[str, Any], cfg: Dict[str, Any], device: torch.device):
    ctrl_name = cfg["controller"]["name"]
    ctrl = get_controller(ctrl_name, **cfg["controller"]["params"])
    input_dim = ctrl.state_dim
    
    model_cfg = cfg.get("model", {})
    use_lifted = model_cfg.get("use_lifted_coords", False)
    lift_type = model_cfg.get("lift_type", "sta")
    
    # Extract only the parameters SimpleLyapNet accepts
    model_init_cfg = {
        "width": model_cfg.get("width", 128),
        "depth": model_cfg.get("depth", 3),
        "eps_quad": model_cfg.get("eps_quad", 1e-3),
        "alpha_bar": model_cfg.get("alpha_bar", 1e-3),
        "input_dim": input_dim,
    }
    
    gauge_cfg = cfg.get("gauge", {})
    gauge_init_cfg = {
        "initial_radius": gauge_cfg.get("initial_radius", 0.1),
        "input_dim": input_dim,
    }

    # Use LiftedLyapNet if configured
    if use_lifted:
        from models import LiftedLyapNet
        V = LiftedLyapNet(input_dim=input_dim, lift_type=lift_type, 
                          width=model_init_cfg["width"], depth=model_init_cfg["depth"],
                          eps_quad=model_init_cfg["eps_quad"], alpha_bar=model_init_cfg["alpha_bar"]).to(device)
    else:
        V = SimpleLyapNet(**model_init_cfg).to(device)
    
    G = OrientedEllipsoidGauge(**gauge_init_cfg).to(device)
    V.load_state_dict(payload["V"]); V.eval()
    G.load_state_dict(payload["G"]); G.eval()
    return V, G


@torch.no_grad()
def _estimate_area(G: torch.nn.Module, n_dir: int = 4096, device: Optional[torch.device] = None) -> Dict[str, float]:
    """Estimate boundary area/volume using radial sampling.

    For 2D: Returns area estimate using polar sampling.
    For 3D: Returns volume estimate using spherical sampling (Fibonacci lattice).
    """
    dev = device or torch.device("cpu")
    input_dim = G.input_dim

    if input_dim == 2:
        th = torch.linspace(0, 2 * math.pi, n_dir, device=dev)
        dirs = torch.stack([torch.cos(th), torch.sin(th)], 1)
        phi = G(dirs).clamp_min(1e-6)
        r = 1.0 / phi
        area = math.pi * torch.mean(r * r).item()
        return {
            "area": float(area),
            "mean_r": float(torch.mean(r).item()),
            "min_r": float(torch.min(r).item()),
            "max_r": float(torch.max(r).item()),
            "q95_r": float(torch.quantile(r, 0.95).item()),
        }
    elif input_dim == 3:
        # 3D: Fibonacci lattice for uniform sampling on sphere
        # Golden ratio spiral gives approximately uniform distribution
        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
        indices = torch.arange(n_dir, dtype=torch.float32, device=dev)

        # Fibonacci lattice: theta from golden ratio, phi from arccos
        theta = 2.0 * math.pi * indices / golden_ratio
        # Map index to z in [-1, 1] uniformly
        z = 1.0 - (2.0 * indices + 1.0) / n_dir
        z = z.clamp(-1.0, 1.0)  # Numerical safety
        xy_scale = torch.sqrt(1.0 - z * z)

        x = xy_scale * torch.cos(theta)
        y = xy_scale * torch.sin(theta)
        dirs = torch.stack([x, y, z], dim=1)  # (n_dir, 3)

        phi = G(dirs).clamp_min(1e-6)
        r = 1.0 / phi

        # Volume of ellipsoid-like region: (4/3)π * mean(r³) for sphere
        # For non-spherical, this is an approximation
        volume = (4.0 / 3.0) * math.pi * torch.mean(r ** 3).item()

        return {
            "area": float(volume),  # Use "area" key for consistency with objective
            "volume": float(volume),
            "mean_r": float(torch.mean(r).item()),
            "min_r": float(torch.min(r).item()),
            "max_r": float(torch.max(r).item()),
            "q95_r": float(torch.quantile(r, 0.95).item()),
        }
    elif input_dim == 1:
        # 1D: just the scalar radius (half-width of interval)
        r = float(torch.exp(G.log_radii).item())
        return {
            "area": 2 * r,  # Length of interval [-r, r]
            "length": 2 * r,
            "radius": r,
            "mean_r": r,
            "min_r": r,
            "max_r": r,
            "q95_r": r,
        }
    else:
        raise ValueError(f"Dimension {input_dim} not yet supported (only 1D, 2D and 3D)")


def _validate_model(V, G, cfg: Dict[str, Any], device: torch.device) -> Tuple[bool, Dict[str, float]]:
    """Run validation and return (feasible, metrics).

    Handles 1D (FOSMC), 2D (STA/CTA) and 3D (PID-SMC) configurations automatically.
    """
    ctrl_name = cfg["controller"]["name"]
    ctrl = get_controller(ctrl_name, **cfg["controller"]["params"])

    val_cfg = cfg.get("val", {})
    alpha_cfg = cfg["alpha"]
    box_cfg = cfg["box"]

    # Detect dimensionality from controller
    state_dim = ctrl.state_dim

    if state_dim == 1:
        # 1D validation (FOSMC)
        res = validate_global_outside_1d(
            V, G, ctrl,
            alpha_s=alpha_cfg.get("alpha_s", 0.03),
            train_box_s=box_cfg.get("train_s", 2.0),
            N=val_cfg.get("N", 1000),
            s_eps_val=val_cfg.get("s_eps_val", 1e-3),
            val_tol_max=val_cfg.get("tol_max", 0.0),
            val_tol_q=val_cfg.get("tol_q", 0.0),
            val_min_outside_frac=cfg.get("val_min_outside_frac", 0.02),
            val_min_outside_count=cfg.get("val_min_outside_count", 20),
            val_dtype=torch.float64,
        )
    elif state_dim == 3:
        # 3D validation (PID-SMC)
        res = validate_global_outside_3d(
            V, G, ctrl,
            alpha_x1=alpha_cfg.get("alpha_x1", 0.05),
            alpha_x2=alpha_cfg.get("alpha_x2", 0.05),
            alpha_x3=alpha_cfg.get("alpha_x3", 0.05),
            train_box_x1=box_cfg.get("train_x1", 4.0),
            train_box_x2=box_cfg.get("train_x2", 4.0),
            train_box_x3=box_cfg.get("train_x3", 4.0),
            N=val_cfg.get("N", 40),  # Smaller default for 3D (40³ = 64k points)
            s_eps_val=val_cfg.get("s_eps_val", 1e-3),
            val_tol_max=val_cfg.get("tol_max", 0.0),
            val_tol_q=val_cfg.get("tol_q", 0.0),
            val_min_outside_frac=cfg.get("val_min_outside_frac", 0.02),
            val_min_outside_count=cfg.get("val_min_outside_count", 200),
            val_dtype=torch.float64,
            axis_exclusion_eps=val_cfg.get("axis_exclusion_eps", 0.0),  # For lifted coords
        )
    else:
        # 2D validation (STA/CTA)
        res = validate_global_outside(
            V, G, ctrl,
            alpha_s=alpha_cfg.get("alpha_s", 0.03),
            alpha_v=alpha_cfg.get("alpha_v", 0.03),
            s_eps_val=val_cfg.get("s_eps_val", 1e-3),
            train_box_s=box_cfg.get("train_s", 2.0),
            train_box_v=box_cfg.get("train_v", 2.0),
            N=val_cfg.get("N", 256),
            val_tol_max=val_cfg.get("tol_max", 0.0),
            val_tol_q=val_cfg.get("tol_q", 0.0),
            val_min_outside_frac=cfg.get("val_min_outside_frac", 0.02),
            val_min_outside_count=cfg.get("val_min_outside_count", 200),
            val_dtype=torch.float64,
        )

    metrics = {
        "val_max": float(res.get("max", 1e10)),
        "val_p95": float(res.get("p95", 1e10)),
        "val_p99": float(res.get("p99", 1e10)),
        "val_frac": float(res.get("frac_pos", 1.0)),
    }
    return bool(res.get("is_ok", False)), metrics


def _evaluate_run(run_dir: str, device: torch.device) -> Tuple[bool, Dict[str, float], Dict[str, Any]]:
    """Evaluate a training run: load checkpoint, validate, compute metrics."""
    ckpt_path = _find_best_ckpt(run_dir)
    if ckpt_path is None:
        return False, {"area": float("inf")}, {}
    
    payload, cfg_ckpt = _load_ckpt_payload(ckpt_path, device)
    V, G = _build_models(payload, cfg_ckpt, device)
    
    feasible, val_metrics = _validate_model(V, G, cfg_ckpt, device)
    area_metrics = _estimate_area(G, n_dir=4096, device=device)
    
    return feasible, {**area_metrics, **val_metrics}, cfg_ckpt


# =============================================================================
# Parameter Suggestion (TUNED FOR MINIMAL BOUNDARY)
# =============================================================================

def _apply_trial_params_3d(cfg: Dict[str, Any], trial: optuna.trial.Trial) -> Dict[str, Any]:
    """
    Map Optuna suggestions -> cfg dict for 3D controllers (PID-SMC).

    REFINED based on Trial 16 (best valid):
    - model_width=128, depth=2 worked well
    - alpha_x1=0.09 (high), alpha_x2=0.02 (low), alpha_x3=0.06 (mid)
    - log_radii_min=-2.9 (close to -3.0 limit = smallest valid boundary)
    - optim_lr=0.0013
    """
    cfg = copy.deepcopy(cfg)

    # === Training Duration (12000 worked well) ===
    cfg["train"]["epochs"] = trial.suggest_categorical(
        "train_epochs", [10000, 12000]
    )

    # === Model Architecture (128, depth=2 worked well) ===
    cfg["model"]["width"] = trial.suggest_categorical(
        "model_width", [96, 128, 160]  # Centered around 128
    )
    cfg["model"]["depth"] = 2  # Fixed at 2 - consistently best
    cfg["model"]["eps_quad"] = trial.suggest_float("model_eps_quad", 5e-4, 5e-3, log=True)  # Narrowed around 0.0015
    cfg["model"]["alpha_bar"] = trial.suggest_float("model_alpha_bar", 5e-6, 1e-4, log=True)  # Narrowed around 1.8e-5

    # === Gauge Initial Radius (0.077 worked well) ===
    cfg["gauge"]["initial_radius"] = trial.suggest_float(
        "gauge_initial_radius", 0.06, 0.12, log=True  # Narrowed around 0.077
    )

    # === Optimizer (lr=0.0013 worked well) ===
    cfg["optim"]["lr"] = trial.suggest_float("optim_lr", 8e-4, 2e-3, log=True)  # Narrowed around 0.0013
    cfg["optim"]["weight_decay"] = trial.suggest_float("optim_wd", 3e-6, 3e-5, log=True)  # Narrowed around 7e-6
    cfg["train"]["lr"] = cfg["optim"]["lr"]

    # === Alpha values - ASYMMETRIC as discovered ===
    # Trial 16: alpha_x1=0.09 (high!), alpha_x2=0.02 (low), alpha_x3=0.06 (mid)
    alpha_x1 = trial.suggest_float("alpha_x1", 0.05, 0.12, log=True)  # Higher range
    alpha_x2 = trial.suggest_float("alpha_x2", 0.015, 0.04, log=True)  # Lower range
    alpha_x3 = trial.suggest_float("alpha_x3", 0.03, 0.10, log=True)  # Middle range
    cfg["alpha"]["alpha_x1"] = alpha_x1
    cfg["alpha"]["alpha_x2"] = alpha_x2
    cfg["alpha"]["alpha_x3"] = alpha_x3

    # === Filippov band (0.00017 worked well) ===
    cfg["train"]["s_eps_train"] = trial.suggest_float("s_eps_train", 1e-4, 5e-4, log=True)

    # === Sampling (10000/4096/1024 worked well) ===
    cfg["sample"]["m_out_global"] = trial.suggest_categorical(
        "m_out_global", [8192, 10000, 12000]
    )
    cfg["sample"]["m_ring"] = trial.suggest_categorical(
        "m_ring", [3072, 4096]  # Higher worked better
    )
    cfg["sample"]["m_size"] = trial.suggest_categorical(
        "m_size", [1024, 2048]
    )

    # === Ring Sampling (narrowed around Trial 16 values) ===
    cfg["ring"]["delta_rel"] = trial.suggest_float("ring_delta_rel", 0.30, 0.45)
    cfg["ring"]["delta_abs"] = trial.suggest_float("ring_delta_abs", 0.18, 0.28)
    cfg["ring"]["rel_frac"] = trial.suggest_float("ring_rel_frac", 0.45, 0.65)
    cfg["ring"]["dist_eps"] = trial.suggest_float("ring_dist_eps", 0.05, 0.10)
    cfg["ring"]["dist_gamma"] = trial.suggest_float("ring_dist_gamma", 0.8, 1.0)

    # === Shrink Gating (patience=6 worked well) ===
    cfg["size"]["feas_patience"] = trial.suggest_int("feas_patience", 5, 7)
    cfg["size"]["revoke_freeze"] = trial.suggest_int("revoke_freeze", 5, 7)
    cfg["size"]["ramp_iters"] = trial.suggest_int("ramp_iters", 1000, 1300)
    cfg["size"]["quantile"] = trial.suggest_float("size_quantile", 0.93, 0.97)

    # === Loss Weights (narrowed around Trial 16) ===
    cfg["weights"]["w_size_post"] = trial.suggest_float("w_size_post", 2.5, 4.0)
    cfg["weights"]["w_ring"] = trial.suggest_float("w_ring", 0.6, 0.9)
    cfg["weights"]["w_ang"] = trial.suggest_float("w_ang", 1e-6, 5e-6, log=True)
    cfg["weights"]["w_grad"] = trial.suggest_float("w_grad", 2e-4, 6e-4, log=True)
    cfg["weights"]["w_plain_mean"] = trial.suggest_float("w_plain_mean", 0.75, 0.90)
    cfg["weights"]["w_plain_tail"] = trial.suggest_float("w_plain_tail", 0.55, 0.75)
    cfg["weights"]["w_floor"] = trial.suggest_float("w_floor", 0.08, 0.13)

    # === Tail Aggregation ===
    cfg["tail"]["beta"] = trial.suggest_float("tail_beta", 25.0, 35.0)
    cfg["tail"]["topk_frac"] = trial.suggest_float("tail_topk_frac", 0.05, 0.09)

    # === Training Hyperparameters ===
    cfg["train"]["grad_clip"] = trial.suggest_float("grad_clip", 1.0, 1.5)
    cfg["train"]["dec_clip"] = trial.suggest_float("dec_clip", 28.0, 40.0)
    cfg["train"]["grad_target"] = trial.suggest_float("grad_target", 18.0, 28.0)

    # === Gauge Clamping ===
    # CRITICAL: min radius must be >= axis_exclusion_eps (0.05)
    # log(0.05) ≈ -3.0, Trial 16 had -2.9 (smallest valid = best)
    # Search near the lower bound to find smallest valid boundary
    axis_eps = cfg.get("val", {}).get("axis_exclusion_eps", 0.05)
    min_log_radius = math.log(axis_eps)  # -3.0 for eps=0.05
    log_radii_min = trial.suggest_float("log_radii_min", min_log_radius, -2.0)
    cfg["gauge_log_radii_clamp"] = (log_radii_min, 3.0)

    # === Multi-disturbance training (False worked in Trial 16) ===
    cfg["train_multi_disturbance"] = False  # Fixed based on Trial 16

    return cfg


def _apply_trial_params_1d(cfg: Dict[str, Any], trial: optuna.trial.Trial) -> Dict[str, Any]:
    """
    Map Optuna suggestions -> cfg dict for 1D controllers (FOSMC).

    GOAL: Minimize boundary size while achieving smooth V(s) resembling |s|.

    Key differences from 2D:
    - Only alpha_s (no alpha_v)
    - Extended log_radii_min to -6.0 (allows r ≈ 0.0025)
    - Deeper networks work better for 1D (6-8 layers)
    - Higher alpha_s enables tighter boundaries
    """
    cfg = copy.deepcopy(cfg)

    # === Training Duration (16000 for maximum shrinkage) ===
    cfg["train"]["epochs"] = trial.suggest_categorical(
        "train_epochs", [12000, 14000, 16000]
    )

    # === Model Architecture (deeper networks work for 1D) ===
    cfg["model"]["width"] = trial.suggest_categorical(
        "model_width", [48, 64, 80]
    )
    cfg["model"]["depth"] = trial.suggest_categorical(
        "model_depth", [4, 6, 8]
    )
    cfg["model"]["eps_quad"] = trial.suggest_float("model_eps_quad", 1e-4, 5e-3, log=True)
    cfg["model"]["alpha_bar"] = trial.suggest_float("model_alpha_bar", 1e-6, 5e-3, log=True)

    # === Gauge Initial Radius (start smaller for 1D) ===
    cfg["gauge"]["initial_radius"] = trial.suggest_float(
        "gauge_initial_radius", 0.03, 0.10, log=True
    )

    # === Optimizer ===
    cfg["optim"]["lr"] = trial.suggest_float("optim_lr", 1e-3, 3e-3, log=True)
    cfg["optim"]["weight_decay"] = trial.suggest_float("optim_wd", 1e-5, 1e-4, log=True)
    cfg["train"]["lr"] = cfg["optim"]["lr"]

    # === CRITICAL: Alpha (only alpha_s for 1D) ===
    # Higher alpha_s = tighter boundary (previous best was 0.09)
    alpha_s = trial.suggest_float("alpha_s", 0.03, 0.15, log=True)
    cfg["alpha"]["alpha_s"] = alpha_s

    # === Filippov band ===
    cfg["train"]["s_eps_train"] = trial.suggest_float("s_eps_train", 1e-3, 1e-2, log=True)

    # === Sampling ===
    cfg["sample"]["m_out_global"] = trial.suggest_categorical(
        "m_out_global", [2048, 4096, 6144]
    )
    cfg["sample"]["m_ring"] = trial.suggest_categorical(
        "m_ring", [512, 1024, 2048]
    )
    cfg["sample"]["m_size"] = trial.suggest_categorical(
        "m_size", [256, 512, 1024]
    )

    # === Ring Sampling ===
    cfg["ring"]["delta_rel"] = trial.suggest_float("ring_delta_rel", 0.04, 0.10)
    cfg["ring"]["delta_abs"] = trial.suggest_float("ring_delta_abs", 0.008, 0.02)
    cfg["ring"]["rel_frac"] = trial.suggest_float("ring_rel_frac", 0.4, 0.6)
    cfg["ring"]["dist_eps"] = trial.suggest_float("ring_dist_eps", 0.03, 0.08)
    cfg["ring"]["dist_gamma"] = trial.suggest_float("ring_dist_gamma", 0.8, 1.2)

    # === Shrink Gating ===
    cfg["size"]["feas_patience"] = trial.suggest_int("feas_patience", 3, 7)
    cfg["size"]["revoke_freeze"] = trial.suggest_int("revoke_freeze", 5, 12)
    cfg["size"]["ramp_iters"] = trial.suggest_int("ramp_iters", 800, 1200)
    cfg["size"]["quantile"] = trial.suggest_float("size_quantile", 0.92, 0.98)

    # === Loss Weights ===
    cfg["weights"]["w_size_post"] = trial.suggest_float("w_size_post", 2.5, 5.0)
    cfg["weights"]["w_ring"] = trial.suggest_float("w_ring", 0.4, 0.7)
    cfg["weights"]["w_ang"] = trial.suggest_float("w_ang", 1e-4, 1e-2, log=True)
    cfg["weights"]["w_grad"] = trial.suggest_float("w_grad", 1e-4, 1e-3, log=True)
    cfg["weights"]["w_plain_mean"] = trial.suggest_float("w_plain_mean", 1.0, 1.8)
    cfg["weights"]["w_plain_tail"] = trial.suggest_float("w_plain_tail", 0.5, 0.9)
    cfg["weights"]["w_floor"] = trial.suggest_float("w_floor", 0.05, 0.15)

    # === Tail Aggregation ===
    cfg["tail"]["beta"] = trial.suggest_float("tail_beta", 20.0, 35.0)
    cfg["tail"]["topk_frac"] = trial.suggest_float("tail_topk_frac", 0.03, 0.08)

    # === Training Hyperparameters ===
    cfg["train"]["grad_clip"] = trial.suggest_float("grad_clip", 1.5, 2.5)
    cfg["train"]["dec_clip"] = trial.suggest_float("dec_clip", 30.0, 50.0)
    cfg["train"]["grad_target"] = trial.suggest_float("grad_target", 20.0, 35.0)

    # === CRITICAL: Gauge Clamping ===
    # Extended to -6.0 to allow r ≈ 0.0025 (20x smaller than default!)
    log_radii_min = trial.suggest_float("log_radii_min", -6.0, -3.0)
    cfg["gauge_log_radii_clamp"] = (log_radii_min, 3.0)

    return cfg


def _apply_trial_params(cfg: Dict[str, Any], trial: optuna.trial.Trial) -> Dict[str, Any]:
    """
    Map Optuna suggestions -> cfg dict.

    FOCUS: Parameters that enable smaller boundaries while maintaining stability
    at the theoretical disturbance limit.

    V6 learnings from sta_limit_v5_refined (best trial #56, area=0.000056):
    - depth=2 consistently best (at MIN) -> fix at 2
    - width=80 worked well (near min) -> extend lower to 48
    - alpha_v near MIN (0.025) -> extend lower to 0.01
    - s_eps_train near MIN (1.56e-5) -> extend lower to 5e-6
    - topk_frac at MIN (0.07) -> extend lower to 0.03
    - w_size_post at MIN (2.01) -> extend lower to 1.0
    - train_multi_disturbance=True worked! (unexpected)
    - gauge_initial_radius=0.047 in lower half -> narrow to 0.03-0.08
    """
    cfg = copy.deepcopy(cfg)
    
    # === Training Duration (keep middle, 10-14k was sweet spot) ===
    cfg["train"]["epochs"] = trial.suggest_categorical(
        "train_epochs", [10000, 12000, 14000]
    )
    
    # === Model Architecture (EXTEND: smaller models) ===
    cfg["model"]["width"] = trial.suggest_categorical(
        "model_width", [48, 64, 80, 96]
    )
    cfg["model"]["depth"] = 2  # Fixed at 2 - consistently best
    cfg["model"]["eps_quad"] = trial.suggest_float("model_eps_quad", 1e-4, 5e-3, log=True)
    cfg["model"]["alpha_bar"] = trial.suggest_float("model_alpha_bar", 1e-6, 5e-3, log=True)
    
    # === Gauge Initial Radius (NARROWED around best=0.047) ===
    cfg["gauge"]["initial_radius"] = trial.suggest_float(
        "gauge_initial_radius", 0.03, 0.08, log=True  # Narrowed around best=0.047
    )
    
    # === Optimizer ===
    cfg["optim"]["lr"] = trial.suggest_float("optim_lr", 2e-4, 2e-3, log=True)
    cfg["optim"]["weight_decay"] = trial.suggest_float("optim_wd", 1e-6, 1e-4, log=True)
    cfg["train"]["lr"] = cfg["optim"]["lr"]
    
    # === CRITICAL: Alpha values (decay rates) ===
    # v5 best: alpha_s=0.028, alpha_v=0.025 (near MIN)
    alpha_s = trial.suggest_float("alpha_s", 1e-2, 4e-2, log=True)  # Narrowed around best=0.028
    alpha_v = trial.suggest_float("alpha_v", 1e-2, 4e-2, log=True)  # EXTENDED lower (was 0.025 at min)
    cfg["alpha"]["alpha_s"] = alpha_s
    cfg["alpha"]["alpha_v"] = alpha_v
    
    # === Filippov band (EXTENDED: allow even smaller) ===
    cfg["train"]["s_eps_train"] = trial.suggest_float("s_eps_train", 5e-6, 2e-4, log=True)  # EXTENDED lower (was 1.56e-5)
    
    # === Sampling ===
    cfg["sample"]["m_out_global"] = trial.suggest_categorical(
        "m_out_global", [4096, 6144, 8192]  # Best was 6144, focus there
    )
    cfg["sample"]["m_ring"] = trial.suggest_categorical(
        "m_ring", [1024, 2048, 3072]  # Best was 2048 (at limit), extend lower
    )
    cfg["sample"]["m_size"] = trial.suggest_categorical(
        "m_size", [512, 1024, 2048]
    )
    
    # === Ring Sampling ===
    cfg["ring"]["delta_rel"] = trial.suggest_float("ring_delta_rel", 0.25, 0.50)  # Narrowed around best=0.41
    cfg["ring"]["delta_abs"] = trial.suggest_float("ring_delta_abs", 0.2, 0.4)   # Narrowed around best=0.30
    cfg["ring"]["rel_frac"] = trial.suggest_float("ring_rel_frac", 0.25, 0.55)   # Narrowed around best=0.37
    cfg["ring"]["dist_eps"] = trial.suggest_float("ring_dist_eps", 0.08, 0.20)   # Extended upper, best=0.14
    cfg["ring"]["dist_gamma"] = trial.suggest_float("ring_dist_gamma", 0.8, 1.6)  # Narrowed around best=1.2
    
    # === CRITICAL: Shrink Gating ===
    cfg["size"]["feas_patience"] = trial.suggest_int("feas_patience", 2, 4)  # Narrowed around best=3
    cfg["size"]["revoke_freeze"] = trial.suggest_int("revoke_freeze", 2, 4)  # Narrowed around best=3
    cfg["size"]["ramp_iters"] = trial.suggest_int("ramp_iters", 500, 1000)   # Narrowed around best=761
    cfg["size"]["quantile"] = trial.suggest_float("size_quantile", 0.90, 0.98)  # Narrowed around best=0.95
    
    # === Loss Weights ===
    cfg["weights"]["w_size_post"] = trial.suggest_float("w_size_post", 1.0, 3.0)  # EXTENDED lower (was 2.01 at MIN)
    cfg["weights"]["w_ring"] = trial.suggest_float("w_ring", 0.4, 0.7)  # Narrowed around best=0.54
    cfg["weights"]["w_ang"] = trial.suggest_float("w_ang", 1e-6, 5e-5, log=True)  # EXTENDED lower, best=2.1e-5
    cfg["weights"]["w_grad"] = trial.suggest_float("w_grad", 3e-4, 1e-3, log=True)  # Narrowed around best=6.5e-4
    cfg["weights"]["w_plain_mean"] = trial.suggest_float("w_plain_mean", 0.7, 0.9)  # Narrowed around best=0.79
    cfg["weights"]["w_plain_tail"] = trial.suggest_float("w_plain_tail", 0.6, 0.8)  # Narrowed around best=0.72
    cfg["weights"]["w_floor"] = trial.suggest_float("w_floor", 0.05, 0.12)  # Narrowed around best=0.086
    
    # === Tail Aggregation ===
    cfg["tail"]["beta"] = trial.suggest_float("tail_beta", 40.0, 50.0)  # Narrowed (best=44.7)
    cfg["tail"]["topk_frac"] = trial.suggest_float("tail_topk_frac", 0.03, 0.10)  # EXTENDED lower (was 0.07 at MIN)
    
    # === Training Hyperparameters ===
    cfg["train"]["grad_clip"] = trial.suggest_float("grad_clip", 1.0, 2.0)  # Narrowed around best=1.52
    cfg["train"]["dec_clip"] = trial.suggest_float("dec_clip", 25.0, 50.0)  # Narrowed around best=36.8
    cfg["train"]["grad_target"] = trial.suggest_float("grad_target", 20.0, 35.0)  # Narrowed around best=25.3
    
    # === Gauge Clamping ===
    log_radii_min = trial.suggest_float("log_radii_min", -5.5, -3.5)  # Extended lower, best=-4.52
    cfg["gauge_log_radii_clamp"] = (log_radii_min, 3.0)
    
    # === Multi-disturbance training (v5 best was True!) ===
    cfg["train_multi_disturbance"] = trial.suggest_categorical(
        "train_multi_disturbance", [True, True, False]  # 2:1 bias toward True (worked in v5)
    )
    
    # === P99 re-enable threshold (NEW) ===
    cfg.setdefault("size", {})
    cfg["size"]["reenable_p99_eta"] = trial.suggest_float("reenable_p99_eta", 1e-4, 5e-3, log=True)
    
    return cfg


# =============================================================================
# Training Runner
# =============================================================================

def _run_training(cfg: Dict[str, Any], outputs_root: str) -> str:
    """Run one training and return the run directory."""
    if not os.path.isabs(outputs_root):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_root = os.path.join(base_dir, outputs_root)
    
    os.makedirs(outputs_root, exist_ok=True)
    cfg_local = copy.deepcopy(cfg)
    cfg_local.setdefault("io", {})
    cfg_local["io"]["outputs_root"] = outputs_root
    
    # Disable viz during HPO
    cfg_local.setdefault("viz", {})
    cfg_local["viz"]["enable"] = False
    
    try:
        train_fn(cfg_local)
    except Exception as e:
        print(f"[HPO] Training failed: {e}")
        raise
    
    return _latest_subdir(outputs_root)


# =============================================================================
# Objective Function
# =============================================================================

def _objective_factory(base_cfg: Dict[str, Any], args) -> Callable:
    """Create objective function for Optuna."""
    device = torch.device("cpu")  # Evaluation on CPU, training uses GPU

    # Detect dimensionality from controller
    ctrl_name = base_cfg["controller"]["name"]
    ctrl = get_controller(ctrl_name, **base_cfg["controller"]["params"])
    state_dim = ctrl.state_dim

    def objective(trial: optuna.trial.Trial) -> float:
        trial_start = time.time()

        # === SINGLE STAGE: Full training (no early pruning) ===
        # Use dimension-specific param function
        if state_dim == 1:
            cfg = _apply_trial_params_1d(base_cfg, trial)
        elif state_dim == 3:
            cfg = _apply_trial_params_3d(base_cfg, trial)
        else:
            cfg = _apply_trial_params(base_cfg, trial)
        cfg["train"]["epochs"] = args.stageB_epochs  # Use full epochs
        cfg.setdefault("polish", {})
        cfg["polish"]["enable"] = args.polish
        cfg["seed"] = int(base_cfg.get("seed", 1337)) + 1000 * trial.number
        
        out_root = os.path.join(args.outputs_root, f"trial{trial.number:04d}")
        
        try:
            run_dir = _run_training(cfg, out_root)
            feasible, metrics, cfg_final = _evaluate_run(run_dir, device)
        except Exception as e:
            print(f"[Trial {trial.number}] Training failed: {e}")
            return float("inf")  # Return bad score, don't prune
        
        area = metrics.get("area", float("inf"))
        val_max = metrics.get("val_max", float("inf"))
        
        # Final score: feasible trials get area, infeasible get penalized
        if feasible:
            final_score = area
        else:
            violation_penalty = max(0, val_max) * 10
            final_score = area + violation_penalty + 0.5
        
        elapsed = time.time() - trial_start
        print(f"\n[Trial {trial.number}] Completed in {elapsed/60:.1f}min")
        print(f"  Area: {area:.6f}, Feasible: {feasible}, Val_max: {val_max:.4f}")
        print(f"  Final score: {final_score:.6f}")
        
        # Store metrics for analysis
        trial.set_user_attr("area", area)
        trial.set_user_attr("feasible", feasible)
        trial.set_user_attr("val_max", val_max)
        trial.set_user_attr("elapsed_min", elapsed / 60)
        
        return final_score

    return objective


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    ap = argparse.ArgumentParser("HPO V2 for Minimal Boundary (Parallel Support)")
    
    # Core arguments
    ap.add_argument("--config", type=str, required=True, help="Base YAML config")
    ap.add_argument("--study", type=str, default="sta_minimal_boundary")
    ap.add_argument("--storage", type=str, default=None, 
                    help="Optuna storage URL, e.g., sqlite:///hpo.db or mysql://...")
    ap.add_argument("--outputs_root", type=str, default="outputs_hpo")
    
    # Trial counts
    ap.add_argument("--n_trials", type=int, default=100)
    
    # Training epochs (single stage now)
    ap.add_argument("--stageA_epochs", type=int, default=1500, 
                    help="(Deprecated, ignored)")
    ap.add_argument("--stageB_epochs", type=int, default=8000, 
                    help="Full training epochs per trial")
    ap.add_argument("--polish", action="store_true", help="Enable polish stage")
    
    # Parallel execution
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Number of parallel workers (use with shared storage)")
    ap.add_argument("--worker_id", type=int, default=0,
                    help="Worker ID for logging (informational)")
    
    args = ap.parse_args()
    
    # Load base config
    base_cfg = load_yaml(args.config)
    
    # Setup storage (required for parallel)
    if args.storage is None:
        # Default to SQLite for single worker
        os.makedirs(args.outputs_root, exist_ok=True)
        args.storage = f"sqlite:///{args.outputs_root}/{args.study}.db"
    
    print(f"\n{'='*60}")
    print(f"HPO V2 - Minimal Boundary Optimization")
    print(f"{'='*60}")
    print(f"Study: {args.study}")
    print(f"Storage: {args.storage}")
    print(f"Base config: {args.config}")
    print(f"Trials: {args.n_trials}")
    print(f"Stage A epochs: {args.stageA_epochs}")
    print(f"Stage B epochs: {args.stageB_epochs}")
    print(f"Polish: {args.polish}")
    print(f"Worker ID: {args.worker_id}")
    print(f"{'='*60}\n")
    
    # Setup pruner and sampler
    # DISABLED: MedianPruner was too aggressive for this hard problem
    # Most trials need full Stage A to converge, median comparison is premature
    from optuna.pruners import NopPruner
    pruner = NopPruner()  # No automatic pruning, only manual via val_max > 10
    
    sampler = TPESampler(
        multivariate=True,
        constant_liar=True,  # Important for parallel
        n_startup_trials=15,  # More exploration initially
    )
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        direction="minimize",  # Minimize area
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    
    # Run optimization
    objective = _objective_factory(base_cfg, args)
    
    study.optimize(
        objective, 
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=(args.n_jobs == 1),
        gc_after_trial=True,  # Help with memory
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("HPO Complete!")
    print(f"{'='*60}")
    
    total = len(study.trials)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    
    print(f"Total trials: {total}")
    print(f"  Completed: {completed}")
    print(f"  Pruned: {pruned}")
    print(f"  Failed: {failed}")
    
    if completed > 0:
        bt = study.best_trial
        print(f"\nBest Trial #{bt.number}:")
        print(f"  Area (score): {bt.value:.6f}")
        print(f"  Feasible: {bt.user_attrs.get('feasible_B', 'N/A')}")
        print(f"\nBest Parameters:")
        for k, v in sorted(bt.params.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")
        
        # Save best config
        best_cfg = copy.deepcopy(base_cfg)
        best_cfg = _apply_trial_params(best_cfg, bt)
        best_cfg_path = os.path.join(args.outputs_root, f"{args.study}_best_config.yaml")
        dump_yaml(best_cfg, best_cfg_path)
        print(f"\nBest config saved to: {best_cfg_path}")
    else:
        print("\nNo completed trials. Check logs for errors.")


if __name__ == "__main__":
    main()
