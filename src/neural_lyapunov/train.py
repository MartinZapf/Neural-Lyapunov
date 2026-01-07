from __future__ import annotations
import os
import argparse
import math
import copy
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

try:
    # Try relative imports first (when used as package)
    from .utils import set_seed, device, make_run_dir, dump_yaml, load_yaml, ensure_dir
    from .controllers import get_controller
    from .models import SimpleLyapNet, LiftedLyapNet
    from .gauges import FlexibleGauge
    from .sampling import (sample_ring_mixed, sample_global_outside, sample_ring_mixed_3d, sample_global_outside_3d,
                          sample_ring_mixed_1d, sample_global_outside_1d, boundary_radius)
    from .validators import validate_global_outside, validate_global_outside_1d
    from .losses import (outside_loss_strict, ring_loss_strict, size_objective, size_estimate,
                        angular_smoothness, floor_loss, grad_reg,
                        lipschitz_penalty_V, lipschitz_uniformity_penalty, curvature_penalty_V)
except ImportError:
    # Fall back to absolute imports (when run directly)
    from utils import set_seed, device, make_run_dir, dump_yaml, load_yaml, ensure_dir
    from controllers import get_controller
    from models import SimpleLyapNet, LiftedLyapNet
    from gauges import FlexibleGauge
    from sampling import (sample_ring_mixed, sample_global_outside, sample_ring_mixed_3d, sample_global_outside_3d,
                         sample_ring_mixed_1d, sample_global_outside_1d, boundary_radius)
    from validators import validate_global_outside, validate_global_outside_1d
    from losses import (outside_loss_strict, ring_loss_strict, size_objective, size_estimate,
                       angular_smoothness, floor_loss, grad_reg,
                       lipschitz_penalty_V, lipschitz_uniformity_penalty, curvature_penalty_V)


def validate_with_dimension(V, G, ctrl, cfg, val_cfg):
    """Helper function to validate based on controller dimension."""
    state_dim = ctrl.state_dim
    
    if state_dim == 1:
        return validate_global_outside_1d(
            V, G, ctrl,
            alpha_s=cfg["alpha"]["alpha_s"],
            train_box_s=cfg["box"]["train_s"],
            N=cfg["val"]["N"],
            **val_cfg
        )
    elif state_dim == 2:
        return validate_global_outside(
            V, G, ctrl,
            alpha_s=cfg["alpha"]["alpha_s"], alpha_v=cfg["alpha"]["alpha_v"],
            train_box_s=cfg["box"]["train_s"],
            train_box_v=cfg["box"]["train_v"], N=cfg["val"]["N"],
            **val_cfg
        )
    elif state_dim == 3:
        # Import 3D validation function
        try:
            from .validators import validate_global_outside_3d
        except ImportError:
            from validators import validate_global_outside_3d
            
        return validate_global_outside_3d(
            V, G, ctrl,
            alpha_x1=cfg["alpha"]["alpha_x1"], 
            alpha_x2=cfg["alpha"]["alpha_x2"], 
            alpha_x3=cfg["alpha"]["alpha_x3"],
            train_box_x1=cfg["box"]["train_x1"],
            train_box_x2=cfg["box"]["train_x2"], 
            train_box_x3=cfg["box"]["train_x3"], 
            N=cfg["val"].get("N", 40),  # Use smaller N for 3D (40³ = 64k points)
            **val_cfg
        )
    else:
        raise ValueError(f"Unsupported state dimension: {state_dim}")


def polish_with_fixed_boundary(V, G, ctrl, cfg, polish_cfg, device, ckpt_path):
    """
    Polish stage: Smooth V while keeping boundary φ FIXED.
    
    Design:
    - Stage 1 (main training): Found smallest feasible boundary → φ is validated
    - Stage 2 (this function): Optimize V for smoothness within that fixed boundary
    
    Strategy:
    - Freeze all G (gauge) parameters
    - Optimize V with Lipschitz penalties
    - Validate frequently (every K iterations)
    - Revert immediately if Lyapunov conditions violated
    - Stop if no improvement
    
    Args:
        V: Lyapunov network (will be optimized)
        G: Gauge network (will be FROZEN)
        ctrl: Controller
        cfg: Full config
        polish_cfg: Polish-specific config
        device: Torch device
        ckpt_path: Where to save polished model
        
    Returns:
        success (bool): True if polishing succeeded
    """
    from torch.nn.utils.clip_grad import clip_grad_norm_
    import copy
    
    state_dim = ctrl.state_dim
    
    # === Configuration ===
    max_epochs = polish_cfg.get("epochs", 500)
    lr = polish_cfg.get("lr", 1e-4)  # Absolute learning rate for polish
    validate_every = polish_cfg.get("validate_every", 10)
    patience = polish_cfg.get("patience", 50)
    
    # Dense validation configuration
    use_dense_validation = polish_cfg.get("use_dense_validation", False)
    val_N_dense = polish_cfg.get("val_N_dense", None)
    report_dense_stats = polish_cfg.get("report_dense_stats", True)
    
    # Determine validation grid size
    if use_dense_validation and val_N_dense is not None:
        val_N_polish = val_N_dense
        total_points = val_N_dense ** state_dim
        
        # Calculate grid spacing based on dimension
        if state_dim == 1:
            box_size = cfg["box"]["train_s"]
        elif state_dim == 2:
            box_size = cfg["box"]["train_s"]  # Use s dimension (same as v typically)
        elif state_dim == 3:
            box_size = cfg["box"]["train_x1"]  # Use x1 dimension
        else:
            box_size = 1.0  # Fallback
        
        grid_spacing = (2 * box_size) / (val_N_dense - 1)
        
        print(f"\n[POLISH] Dense validation ENABLED: {val_N_dense}^{state_dim} grid")
        print(f"[POLISH] Total validation points: {total_points:,}")
        print(f"[POLISH] Grid spacing: δ = {grid_spacing:.6f}")
        
        # Estimate validation time (rough)
        approx_time_per_val = total_points * 0.00001  # ~10μs per point
        print(f"[POLISH] Estimated time per validation: {approx_time_per_val:.1f}s")
    else:
        # Use same N as main training
        val_N_polish = cfg["val"]["N"]
        grid_spacing = None
        print(f"\n[POLISH] Using standard validation: {val_N_polish}^{state_dim} grid")
    
    # Lipschitz configuration
    lipschitz_weight = polish_cfg.get("lipschitz_weight", 1.0e-4)
    target_lipschitz = polish_cfg.get("target_lipschitz", 10.0)
    use_uniformity = polish_cfg.get("lipschitz_uniformity", True)
    
    # Curvature (second-order smoothness)
    curvature_weight = polish_cfg.get("curvature_weight", 1.0e-5)
    curvature_sigma = polish_cfg.get("curvature_sigma", 0.05)
    
    # Lyapunov enforcement weight (maintain feasibility)
    lyap_weight = polish_cfg.get("lyapunov_weight", 0.5)
    
    print("\n" + "="*70)
    print("[POLISH] Verification-Guided Smoothing with FIXED Boundary")
    if use_dense_validation:
        print(f"[POLISH] MODE: Dense Validation ({val_N_polish}^{state_dim} = {val_N_polish**state_dim:,} points)")
        if grid_spacing is not None:
            print(f"[POLISH] Grid spacing: δ = {grid_spacing:.6f}")
    else:
        print(f"[POLISH] MODE: Standard Validation ({val_N_polish}^{state_dim} points)")
    print("="*70)
    print(f"[POLISH] Config:")
    print(f"  - Epochs: {max_epochs}, LR: {lr:.1e}, Validate every: {validate_every}")
    print(f"  - Lipschitz: weight={lipschitz_weight:.1e}, target={target_lipschitz}")
    print(f"  - Curvature: weight={curvature_weight:.1e}, sigma={curvature_sigma}")
    print(f"  - Lyapunov enforcement: weight={lyap_weight}")
    print(f"  - Gauge parameters: FROZEN (boundary fixed)")
    print("="*70 + "\n")
    
    # === FREEZE GAUGE PARAMETERS ===
    for param in G.parameters():
        param.requires_grad = False
    
    # Verify gauge is frozen
    n_gauge_params = sum(p.numel() for p in G.parameters())
    n_gauge_trainable = sum(p.numel() for p in G.parameters() if p.requires_grad)
    n_V_params = sum(p.numel() for p in V.parameters())
    n_V_trainable = sum(p.numel() for p in V.parameters() if p.requires_grad)
    
    print(f"[POLISH] Parameter status:")
    print(f"  V parameters: {n_V_params} ({n_V_trainable} trainable)")
    print(f"  G parameters: {n_gauge_params} ({n_gauge_trainable} trainable)")
    assert n_gauge_trainable == 0, "ERROR: Gauge parameters not frozen!"
    print(f"[POLISH] ✓ Gauge parameters verified frozen\n")
    
    # === Create optimizer (V only) ===
    polish_opt = torch.optim.AdamW(
        V.parameters(),
        lr=lr,
        weight_decay=1e-6  # Light regularization
    )
    
    # === Save initial state ===
    initial_V_state = copy.deepcopy(V.state_dict())
    initial_G_state = copy.deepcopy(G.state_dict())  # Save for verification
    
    # === Checkpointing ===
    best_polish_state = None
    last_valid_V_state = copy.deepcopy(V.state_dict())
    last_valid_epoch = 0
    best_lipschitz = float('inf')
    
    # Validation config (use dense grid if enabled)
    val_cfg = {
        'val_min_outside_frac': cfg.get('val_min_outside_frac', 0.02),
        'val_min_outside_count': cfg.get('val_min_outside_count', 200),
        'val_tol_max': 0.0,
        'val_tol_q': 0.0,
        'val_dtype': getattr(torch, cfg.get('val_dtype', 'float64')),
        'axis_exclusion_eps': cfg["val"].get('axis_exclusion_eps', 0.0),  # For lifted coords
    }
    
    # Create modified config for dense validation if enabled
    val_cfg_polish = cfg.copy()
    val_cfg_polish["val"] = cfg["val"].copy()
    val_cfg_polish["val"]["N"] = val_N_polish
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    # === Polish Loop ===
    for epoch in range(1, max_epochs + 1):
        V.train()
        G.eval()  # Gauge stays in eval mode
        
        # Sample points (same as main training)
        if state_dim == 1:
            z_ring = sample_ring_mixed_1d(
                G, cfg["sample"]["m_ring"],
                cfg["ring"]["delta_rel"], cfg["ring"]["delta_abs"],
                cfg["ring"]["rel_frac"], device=device
            )
            z_glob = sample_global_outside_1d(
                G, cfg["sample"]["m_out_global"],
                cfg["box"]["train_s"], device=device
            )
        elif state_dim == 2:
            z_ring = sample_ring_mixed(
                G, cfg["sample"]["m_ring"],
                cfg["ring"]["delta_rel"], cfg["ring"]["delta_abs"],
                cfg["ring"]["rel_frac"], device=device
            )
            z_glob = sample_global_outside(
                G, cfg["sample"]["m_out_global"],
                cfg["box"]["train_s"], cfg["box"]["train_v"],
                device=device
            )
        elif state_dim == 3:
            z_ring = sample_ring_mixed_3d(
                G, cfg["sample"]["m_ring"],
                cfg["ring"]["delta_rel"], cfg["ring"]["delta_abs"],
                cfg["ring"]["rel_frac"], device=device
            )
            z_glob = sample_global_outside_3d(
                G, cfg["sample"]["m_out_global"],
                cfg["box"]["train_x1"], cfg["box"]["train_x2"],
                cfg["box"]["train_x3"], device=device
            )
        else:
            raise ValueError(f"Unsupported state dimension: {state_dim}")
        
        z_all = torch.cat([z_ring, z_glob], dim=0).requires_grad_(True)
        
        # === Compute Losses ===
        
        # Compute Lyapunov-related losses FIRST (they need z_all gradients)
        # 3. Lyapunov decrease (MAINTAIN feasibility)
        Vv = V(z_all)
        gV = torch.autograd.grad(Vv.sum(), z_all, create_graph=True)[0]
        
        # Choose between single worst-case or multi-disturbance training
        if cfg.get("train_multi_disturbance", False):
            dV = ctrl.worst_dV_multi(gV, z_all, include_nominal=True)
        else:
            dV = ctrl.worst_dV(gV, z_all)
        
        # Alpha regularization term (based on dimension)
        if state_dim == 1:
            alpha_all = cfg["alpha"]["alpha_s"] * torch.abs(z_all[:, 0])
        elif state_dim == 2:
            alpha_all = (cfg["alpha"]["alpha_s"] * torch.abs(z_all[:, 0]) +
                        cfg["alpha"]["alpha_v"] * torch.abs(z_all[:, 1]))
        elif state_dim == 3:
            alpha_all = (cfg["alpha"]["alpha_x1"] * torch.abs(z_all[:, 0]) +
                        cfg["alpha"]["alpha_x2"] * torch.abs(z_all[:, 1]) +
                        cfg["alpha"]["alpha_x3"] * torch.abs(z_all[:, 2]))
        else:
            raise ValueError(f"Unsupported state dimension: {state_dim}")
        
        # Lyapunov loss (enforce dV + α ≤ 0)
        dec = (dV + alpha_all).clamp(
            min=-cfg["train"]["dec_clip"],
            max=cfg["train"]["dec_clip"]
        )
        loss_lyap = F.softplus(dec.mean())
        
        # 4. Floor loss (keep V > 0 away from origin)
        loss_floor = floor_loss(Vv, z_all, cfg["train"]["c_lower"])
        
        # Now compute smoothness penalties on FRESH samples (avoid graph conflicts)
        # 1. Lipschitz smoothness (PRIMARY polish objective)
        if use_uniformity:
            loss_lip = lipschitz_uniformity_penalty(
                V, z_all.detach(), target_lipschitz=target_lipschitz
            )
        else:
            loss_lip = lipschitz_penalty_V(
                V, z_all.detach(), target_lipschitz=target_lipschitz
            )
        
        # 2. Curvature (second-order smoothness)
        loss_curv = curvature_penalty_V(
            V, z_all.detach(), sigma=curvature_sigma
        )
        
        # === Total Loss ===
        loss_total = (
            lipschitz_weight * loss_lip +
            curvature_weight * loss_curv +
            lyap_weight * loss_lyap +
            cfg["weights"]["w_floor"] * loss_floor
        )
        
        # === Optimize (V only) ===
        polish_opt.zero_grad(set_to_none=True)
        loss_total.backward()
        
        # Gentle gradient clipping
        clip_grad_norm_(V.parameters(), max_norm=1.0)
        
        polish_opt.step()
        
        # === Validation (Frequent) ===
        if epoch % validate_every == 0 or epoch == 1:
            V.eval()
            
            # Use dense validation config if enabled
            val = validate_with_dimension(V, G, ctrl, val_cfg_polish, val_cfg)
            
            # Report dense validation statistics periodically
            if use_dense_validation and report_dense_stats and (epoch % (validate_every * 5) == 0 or epoch == 1):
                print(f"\n[POLISH] Dense Validation Statistics (Epoch {epoch}):")
                print(f"  Grid: {val_N_polish}^{state_dim} = {val_N_polish**state_dim:,} points")
                if grid_spacing is not None:
                    print(f"  Spacing: δ = {grid_spacing:.6f}")
                print(f"  Max violation: {val.get('max', 0):.6e}")
                print(f"  P95 violation: {val.get('p95', 0):.6e}")
                print(f"  P99 violation: {val.get('p99', 0):.6e}")
                frac_pos = val.get('frac_pos', 0)
                if frac_pos > 0:
                    n_viol = int(frac_pos * val.get('total_points', 0))
                    print(f"  ⚠️  Violations: {n_viol:,} ({frac_pos*100:.2f}%)")
                else:
                    print(f"  ✓ Zero violations detected")
                print()
            
            if val.get('is_ok', False):
                # SUCCESS: Lyapunov conditions still satisfied
                consecutive_failures = 0
                
                # Update last valid checkpoint
                last_valid_V_state = copy.deepcopy(V.state_dict())
                last_valid_epoch = epoch
                
                # Track best (smoothest = lowest Lipschitz penalty)
                current_lipschitz = loss_lip.item()
                if current_lipschitz < best_lipschitz:
                    best_lipschitz = current_lipschitz
                    best_polish_state = {
                        "V": copy.deepcopy(V.state_dict()),
                        "G": copy.deepcopy(G.state_dict()),  # Include for completeness
                        "epoch": epoch,
                        "lipschitz_penalty": current_lipschitz,
                        "target_lipschitz": target_lipschitz,
                        "val": val
                    }
                
                print(f"[POLISH {epoch:03d}/{max_epochs}] "
                      f"lip={loss_lip.item():.3e} | "
                      f"curv={loss_curv.item():.3e} | "
                      f"lyap={loss_lyap.item():.3e} | "
                      f"val[max,p95]={val['max']:.3e},{val['p95']:.3e} | "
                      f"✓ VALID")
            
            else:
                # FAILURE: Lyapunov conditions violated
                consecutive_failures += 1
                
                if use_dense_validation:
                    frac_pos = val.get('frac_pos', 0)
                    n_viol = int(frac_pos * val.get('total_points', 0)) if frac_pos > 0 else 0
                    print(f"[POLISH {epoch:03d}/{max_epochs}] "
                          f"lip={loss_lip.item():.3e} | "
                          f"val[max,p95]={val.get('max', float('nan')):.3e},"
                          f"{val.get('p95', float('nan')):.3e} | "
                          f"✗ INVALID (fail #{consecutive_failures}/{max_consecutive_failures}) "
                          f"violations={n_viol:,} → reverting to epoch {last_valid_epoch}")
                else:
                    print(f"[POLISH {epoch:03d}/{max_epochs}] "
                          f"lip={loss_lip.item():.3e} | "
                          f"val[max,p95]={val.get('max', float('nan')):.3e},"
                          f"{val.get('p95', float('nan')):.3e} | "
                          f"✗ INVALID (fail #{consecutive_failures}/{max_consecutive_failures}) "
                          f"→ reverting to epoch {last_valid_epoch}")
                
                # REVERT to last valid state
                V.load_state_dict(last_valid_V_state)
                
                # Stop if too many consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    print(f"[POLISH] Stopping early: "
                          f"{consecutive_failures} consecutive validation failures")
                    break
        
        # === Patience check ===
        if epoch - last_valid_epoch > patience:
            print(f"[POLISH] Stopping early: "
                  f"No valid improvement for {patience} iterations")
            break
    
    # === Finalize ===
    
    # Run final dense validation report if enabled
    if use_dense_validation and best_polish_state is not None:
        V.load_state_dict(best_polish_state["V"])
        V.eval()
        final_val = validate_with_dimension(V, G, ctrl, val_cfg_polish, val_cfg)
        
        print("\n" + "="*70)
        print("[POLISH] Final Dense Validation Report")
        print("="*70)
        print(f"Dense validation grid: {val_N_polish}^{state_dim} = {val_N_polish**state_dim:,} points")
        if grid_spacing is not None:
            print(f"Grid spacing: δ = {grid_spacing:.6f}")
        
        print(f"\nFinal validation on dense grid:")
        print(f"  Max violation: {final_val.get('max', 0):.6e}")
        print(f"  P95 violation: {final_val.get('p95', 0):.6e}")
        print(f"  P99 violation: {final_val.get('p99', 0):.6e}")
        
        frac_pos = final_val.get('frac_pos', 0)
        if frac_pos > 0:
            n_viol = int(frac_pos * final_val.get('total_points', 0))
            print(f"  Violations: {n_viol:,} ({frac_pos*100:.2f}%)")
        else:
            print(f"  Violations: 0")
        
        if final_val.get('is_ok', False):
            print(f"\n✓ Polish succeeded with DENSE validation")
            print(f"  Zero violations on {val_N_polish**state_dim:,} point grid")
        else:
            print(f"\n✗ Polish has violations on dense grid")
        print("="*70 + "\n")
    
    # Verify gauge parameters unchanged
    final_G_state = G.state_dict()
    for key in initial_G_state.keys():
        assert torch.allclose(initial_G_state[key], final_G_state[key]), \
            f"ERROR: Gauge param {key} changed during polish!"
    print(f"[POLISH] ✓ Gauge parameters verified unchanged\n")
    
    # Unfreeze gauge parameters (restore trainability)
    for param in G.parameters():
        param.requires_grad = True
    
    if best_polish_state is not None:
        # Load best polished state
        V.load_state_dict(best_polish_state["V"])
        
        # Save checkpoint
        meta = {
            "controller": cfg.get("controller", {}).get("name", "unknown"),
            "controller_params": cfg.get("controller", {}).get("params", {}),
            "polish": {
                "epoch": best_polish_state["epoch"],
                "lipschitz_penalty": best_polish_state["lipschitz_penalty"],
                "target_lipschitz": target_lipschitz,
                "boundary_frozen": True
            }
        }
        
        torch.save({
            "polished": best_polish_state,
            "cfg": cfg,
            "meta": meta
        }, ckpt_path)
        
        print("\n" + "="*70)
        print(f"[POLISH] ✓ SUCCESS!")
        print(f"  - Best epoch: {best_polish_state['epoch']}")
        print(f"  - Lipschitz penalty: {best_polish_state['lipschitz_penalty']:.4e}")
        print(f"  - Validation: max={best_polish_state['val']['max']:.3e}, "
              f"p95={best_polish_state['val']['p95']:.3e}")
        print(f"  - Saved to: {ckpt_path}")
        print("="*70 + "\n")
        
        return True
    
    else:
        # Polish failed completely
        print("\n" + "="*70)
        print(f"[POLISH] ✗ FAILED: No valid polished states found")
        print(f"  - Reverting to pre-polish V")
        print("="*70 + "\n")
        
        V.load_state_dict(initial_V_state)
        return False


def build_argparser():
    p = argparse.ArgumentParser("slimode trainer")
    p.add_argument("--config", type=str, required=True, help="YAML config file (see configs/)")
    p.add_argument("--outputs_root", type=str, default="outputs", help="Root outputs folder")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p.add_argument("--no_plots", action="store_true")
    # Simplified arguments - strict-only now
    p.add_argument("--val_min_outside_frac", type=float, default=0.02, help="Min fraction of grid outside boundary")
    p.add_argument("--val_min_outside_count", type=int, default=200, help="Min absolute count outside boundary")
    p.add_argument("--deterministic", type=bool, default=True, help="Enable deterministic mode")
    return p


def train(cfg: dict):
    # Always strict mode - no warmup or margins
    deterministic = cfg.get("deterministic", True)
    set_seed(cfg["seed"], deterministic=deterministic)
    dev = device()

    # --- Controller with dimension detection
    ctrl_name = cfg["controller"]["name"]
    ctrl = get_controller(ctrl_name, **cfg["controller"]["params"])
    state_dim = ctrl.state_dim  # 2 or 3
    
    print(f"[INFO] Controller: {ctrl.name}, State dimension: {state_dim}")
    
    # Log disturbance handling configuration
    delta_bound = ctrl.get_delta_bound()
    if delta_bound > 0:
        print(f"[INFO] Disturbance handling enabled: δ₀ = {delta_bound} (Moreno & Osorio 2012)")
        print(f"[INFO] Disturbance channel: index {ctrl.disturbance_channel()}")
    else:
        print(f"[INFO] No disturbance (δ₀ = 0.0)")

    # --- Models with dimension support
    model_cfg = cfg["model"].copy()
    model_cfg["input_dim"] = state_dim
    
    # Check for lifted coordinates (Moreno-Osorio style transformation)
    use_lifted = model_cfg.pop("use_lifted_coords", False)
    lift_type = model_cfg.pop("lift_type", "sta")
    
    if use_lifted:
        if ctrl_name not in ["sta", "cta", "pid_smc", "pidsmc"]:  # STA, CTA and PID-SMC support lifting
            print(f"[WARN] Lifted coordinates only supported for STA/CTA/PID-SMC, disabling for {ctrl_name}")
            use_lifted = False
    
    if use_lifted:
        print(f"[INFO] Using LIFTED coordinates (Moreno-Osorio transformation) for {lift_type}")
        print(f"[INFO] → Enables theoretical disturbance bound: δ < k₂")
        model_cfg["lift_type"] = lift_type
        V = LiftedLyapNet(**model_cfg).to(dev)
    else:
        V = SimpleLyapNet(**model_cfg).to(dev)
    
    # Filter gauge config and add dimension
    gauge_cfg = cfg["gauge"]
    gauge_params = {"input_dim": int(state_dim)}  # type: ignore
    
    # Handle new oriented ellipsoid gauge parameters
    if "initial_radius" in gauge_cfg:
        gauge_params["initial_radius"] = float(gauge_cfg["initial_radius"])  # type: ignore
    else:
        # Default to 2.0 for backward compatibility
        gauge_params["initial_radius"] = 2.0  # type: ignore
    
    # Note: Legacy star-convex gauge parameters (n_planes, n_cones, etc.) are no longer used
    # The oriented ellipsoid gauge is much simpler: radii + rotation angles only
    G = FlexibleGauge(**gauge_params).to(dev)

    # --- Optimizer with separate param groups
    # V network: Allow weight decay for regularization
    # G (gauge): NO weight decay to prevent drift in angles/radii
    param_groups = [
        {
            'params': V.parameters(),
            'lr': cfg["optim"]["lr"],
            'weight_decay': cfg["optim"]["weight_decay"]  # Keep weight decay for V
        },
        {
            'params': G.parameters(),
            'lr': cfg["optim"]["lr"] * cfg["optim"].get("gauge_lr_mult", 0.5),  # Optionally slower LR for gauge
            'weight_decay': 0.0  # NO weight decay for gauge angles/radii
        }
    ]
    opt = torch.optim.AdamW(param_groups)

    # --- State tracking for shrink-verify loop
    val_tick = 0
    freeze_until = 0
    feasible_streak = 0
    
    # Track only the best model: smallest size among valid models
    best_state = None
    best_size = float("inf")

    # --- I/O
    out_dir = make_run_dir(cfg["io"]["outputs_root"], ctrl_name)
    ckpt_last = os.path.join(out_dir, "last_model.pth")
    cfg_path = os.path.join(out_dir, "config_used.yaml")
    dump_yaml(cfg, cfg_path)

    # --- Helper functions for gating logic
    def on_validate(result: dict) -> tuple[bool, bool, float]:
        """Update gating state based on validation result. Returns (feasible, shrink_enabled, last_p99)."""
        nonlocal val_tick, freeze_until, feasible_streak
        val_tick += 1
        
        if result.get('is_ok', False):
            feasible_streak += 1
        else:
            feasible_streak = 0
            freeze_until = val_tick + cfg.get("revoke_freeze", cfg["size"]["revoke_freeze"])
        
        feasible = (feasible_streak >= cfg["size"]["feas_patience"])
        
        # Add p99 safety margin before re-enabling shrink
        last_p99 = result.get('p99', 0.0)
        eta = cfg.get("size", {}).get("reenable_p99_eta", 1e-3)
        shrink_enabled = feasible and (val_tick >= freeze_until) and (last_p99 <= -eta)
        
        return feasible, shrink_enabled, last_p99

    # --- Loop
    for it in range(1, cfg["train"]["epochs"] + 1):
        
        # Enhanced sampling with outside guarantees
        sampling_cfg = {
            'min_outside_frac': cfg.get('min_outside_batch', 0.2),
            'expanded_box_scale': cfg.get('expanded_box_scale', 1.5),
            'far_outside_delta_min': cfg.get('far_outside_delta_min', 0.5),
            'far_outside_delta_max': cfg.get('far_outside_delta_max', 2.0)
        }
        
        # Dispatch sampling based on dimension
        if state_dim == 1:
            # 1D sampling
            z_ring = sample_ring_mixed_1d(
                G, cfg["sample"]["m_ring"], 
                cfg["ring"]["delta_rel"], cfg["ring"]["delta_abs"],
                cfg["ring"]["rel_frac"], device=dev
            )
            z_glob = sample_global_outside_1d(
                G, cfg["sample"]["m_out_global"],
                cfg["box"]["train_s"], device=dev, **sampling_cfg
            )
        elif state_dim == 2:
            z_ring = sample_ring_mixed(G, cfg["sample"]["m_ring"], cfg["ring"]["delta_rel"], cfg["ring"]["delta_abs"],
                                       cfg["ring"]["rel_frac"], device=dev)
            z_glob = sample_global_outside(G, cfg["sample"]["m_out_global"],
                                           cfg["box"]["train_s"], cfg["box"]["train_v"], device=dev, **sampling_cfg)
        elif state_dim == 3:
            z_ring = sample_ring_mixed_3d(G, cfg["sample"]["m_ring"], cfg["ring"]["delta_rel"], cfg["ring"]["delta_abs"],
                                          cfg["ring"]["rel_frac"], device=dev)
            z_glob = sample_global_outside_3d(G, cfg["sample"]["m_out_global"],
                                              cfg["box"]["train_x1"], cfg["box"]["train_x2"], cfg["box"]["train_x3"], 
                                              device=dev, **sampling_cfg)
        else:
            raise ValueError(f"Unsupported state dimension: {state_dim}")
            
        z_all = torch.cat([z_ring, z_glob], dim=0).requires_grad_(True)

        # Forward / grads
        Vv = V(z_all)
        gV = torch.autograd.grad(Vv.sum(), z_all, create_graph=True, retain_graph=True)[0]
        
        # Choose between single worst-case or multi-disturbance training
        if cfg.get("train_multi_disturbance", False):
            dV = ctrl.worst_dV_multi(gV, z_all, s_eps=cfg["train"]["s_eps_train"], include_nominal=True)
        else:
            dV = ctrl.worst_dV(gV, z_all, s_eps=cfg["train"]["s_eps_train"])

        # Alpha regularization: α(z) = weighted 1-norm (regularization parameters for training)
        if state_dim == 1:
            alpha_all = cfg["alpha"]["alpha_s"] * torch.abs(z_all[:, 0])
        elif state_dim == 2:
            alpha_all = cfg["alpha"]["alpha_s"] * torch.abs(z_all[:, 0]) + cfg["alpha"]["alpha_v"] * torch.abs(z_all[:, 1])
        elif state_dim == 3:
            alpha_all = (cfg["alpha"]["alpha_x1"] * torch.abs(z_all[:, 0]) + 
                        cfg["alpha"]["alpha_x2"] * torch.abs(z_all[:, 1]) +
                        cfg["alpha"]["alpha_x3"] * torch.abs(z_all[:, 2]))
        else:
            raise ValueError(f"Unsupported state dimension: {state_dim}")

        # Indices
        n_ring = z_ring.shape[0]
        idx_ring = torch.arange(0, n_ring, device=dev)
        idx_plain = torch.arange(0, z_all.shape[0], device=dev)

        # Loss: plain global outside (strict-only)
        loss_plain = outside_loss_strict(
            dV[idx_plain], alpha_all[idx_plain],
            dec_clip=cfg["train"]["dec_clip"],
            tail_mode=cfg["tail"]["mode"],
            tail_beta=cfg["tail"]["beta"],
            tail_topk=cfg["tail"]["topk_frac"],
            w_mean=cfg["weights"]["w_plain_mean"],
            w_tail=cfg["weights"]["w_plain_tail"]
        )
        
        # Loss: ring (strict-only) with optional gradient flow control
        detach_phi_ring = cfg.get("ring", {}).get("detach_phi", True)
        phi_all = G(z_all)
        if detach_phi_ring:
            phi_all = phi_all.detach()
        loss_ring = ring_loss_strict(
            dV[idx_ring], alpha_all[idx_ring], phi_all[idx_ring],
            dist_eps=cfg["ring"]["dist_eps"], dist_gamma=cfg["ring"]["dist_gamma"],
            dec_clip=cfg["train"]["dec_clip"]
        )

        # Size and angular smoothness objectives
        if state_dim == 1:
            # For 1D, sample uniform directions on [-1, +1] segment
            # This gives better coverage for asymmetric gauges
            n_dirs = cfg["sample"]["m_size"]
            dirs = 2.0 * torch.rand(n_dirs, 1, device=dev) - 1.0  # Uniform in [-1, 1]
            dirs = torch.sign(dirs)  # Convert to ±1 but keep randomness
            # Add a few specific directions for consistency
            if n_dirs >= 2:
                dirs[0, 0] = 1.0   # Ensure +1 is always included
                dirs[1, 0] = -1.0  # Ensure -1 is always included
        elif state_dim == 2:
            th = torch.rand(cfg["sample"]["m_size"], device=dev) * 2 * math.pi
            dirs = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
        elif state_dim == 3:
            # Sample directions uniformly on unit sphere in 3D
            dirs = torch.randn(cfg["sample"]["m_size"], 3, device=dev)
            dirs = torch.nn.functional.normalize(dirs, dim=1)
        else:
            raise ValueError(f"Unsupported state dimension: {state_dim}")
            
        phi_dirs = G(dirs)
        
        # Robust quantile-based size objective
        size_est = size_estimate(phi_dirs, dimension=state_dim)  # Dimension-aware logging
        loss_size = size_objective(phi_dirs, quantile=cfg.get("size", {}).get("quantile", 0.90))
        
        # Angular smoothness penalty
        loss_ang = angular_smoothness(phi_dirs)
        w_ang = cfg.get("weights", {}).get("w_ang", 1e-3)

        # Floors / regularizers
        loss_floor = floor_loss(Vv, z_all, cfg["train"]["c_lower"])
        loss_grad = grad_reg(gV, target=cfg["train"]["grad_target"], weight=cfg["weights"]["w_grad"])

        # Shrink–verify gating with proper cooldown
        feasible = (feasible_streak >= cfg["size"]["feas_patience"])
        shrink_enabled = feasible and (val_tick >= freeze_until)
        
        if shrink_enabled:
            prog = min(1.0, feasible_streak / cfg["size"]["ramp_iters"])
            w_size = cfg["weights"]["w_size_post"] * (0.3 + 0.7 * prog)
            w_plain = cfg["weights"]["w_plain_after"]
        else:
            w_size = cfg["weights"]["w_size_pre"]
            w_plain = cfg["weights"]["w_plain_before"]

        loss = (w_plain * loss_plain
                + cfg["weights"]["w_ring"] * loss_ring
                + w_size * loss_size
                + w_ang * loss_ang
                + cfg["weights"]["w_floor"] * loss_floor
                + loss_grad)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["train"]["grad_clip"] > 0:
            clip_grad_norm_(list(V.parameters()) + list(G.parameters()), cfg["train"]["grad_clip"])
        opt.step()

        with torch.no_grad():
            # Clamp log_radii to prevent extreme ellipsoid shapes
            # Default: radii between ~0.05 and ~20
            # HPO can tune this via gauge_log_radii_clamp config
            log_radii_clamp = cfg.get("gauge_log_radii_clamp", (-3.0, 3.0))
            G.log_radii.data.clamp_(min=log_radii_clamp[0], max=log_radii_clamp[1])
            # Angles are naturally bounded by trigonometric functions

        # Validate with enhanced validator
        if (it % cfg["val"]["every"] == 0) or it == 1:
            val_cfg = {
                'val_min_outside_frac': cfg.get('val_min_outside_frac', 0.02),
                'val_min_outside_count': cfg.get('val_min_outside_count', 200),
                'val_tol_max': cfg.get('val_tol_max', cfg["val"]["tol_max"]),
                'val_tol_q': cfg.get('val_tol_q', cfg["val"]["tol_q"]),
                'val_dtype': getattr(torch, cfg.get('val_dtype', 'float64')),
                'axis_exclusion_eps': cfg["val"].get('axis_exclusion_eps', 0.0),  # For lifted coords
            }
            
            # Validation based on dimension
            val = validate_with_dimension(V, G, ctrl, cfg, val_cfg)

            # Update gating based on validation
            feasible, shrink_enabled, last_p99 = on_validate(val)
            
            if val.get('is_ok', False):
                if feasible_streak == cfg["size"]["feas_patience"]:
                    print(f"[FEASIBLE] validator satisfied persistently at it={it}.")
            else:
                if feasible_streak == 0 and val_tick > 1:  # Just lost feasibility
                    reason = val.get('reason', 'validation_failed')
                    print(f"[REVOKE] validator failed at it={it}: {reason}. "
                          f"max={val.get('max', 'nan'):.3e}, p95={val.get('p95', 'nan'):.3e}. "
                          f"Freeze shrinking for {cfg['size']['revoke_freeze']} validations.")

            # Best snapshot tracking - dual checkpoints
            if val.get('is_ok', False):
                # Best model: smallest size among valid models
                if size_est.item() < best_size:
                    best_size = size_est.item()
                    best_state = {
                        "V": copy.deepcopy(V.state_dict()),
                        "G": copy.deepcopy(G.state_dict()),
                        "iter": it,
                        "size": best_size,
                        "val": val,
                        "val_score": val.get('max', float('inf')),  # Also track max violation for logging
                        "meta": {
                            "controller": ctrl_name,
                            "controller_params": cfg["controller"]["params"],
                            "alpha": cfg["alpha"],
                            "box": cfg["box"],
                            "train": cfg["train"],
                            "gauge": cfg["gauge"],
                            "model": cfg["model"],
                        }
                    }

            # Logging
            status = "OK" if val.get('is_ok', False) else "FAIL"
            print(f"[{it:05d}] loss={loss.item():.3e} | area={size_est.item():.3e} | "
                  f"val_max={val.get('max', float('nan')):.3e} | {status} | shrink={shrink_enabled}")

    # Save checkpoints
    torch.save({"V": V.state_dict(), "G": G.state_dict(), "cfg": cfg}, ckpt_last)
    
    # Save only the single best model (smallest size)
    if best_state is not None:
        ckpt_best = os.path.join(out_dir, "best_model.pth")
        torch.save({"best": best_state, "cfg": cfg}, ckpt_best)
        print(f"[SAVE] Best model: area={best_state['size']:.4e}, max_violation={best_state['val_score']:.4e} at iter {best_state['iter']}.")
        saved_any = True
    else:
        print("[WARN] No feasible snapshots saved.")
        saved_any = False

    # Polish stage if enabled - only polish the best model (smallest size)
    polish_cfg = cfg.get("polish", {})
    if polish_cfg.get("enable", True) and best_state is not None:
        print("\n[POLISH] Starting polish stage on best model (smallest boundary)...")
        
        # Load best model for polish
        V.load_state_dict(best_state["V"])
        G.load_state_dict(best_state["G"])
        
        ckpt_polished = os.path.join(out_dir, "best_model_polished.pth")
        polish_success = polish_with_fixed_boundary(V, G, ctrl, cfg, polish_cfg, dev, ckpt_polished)
        
        if polish_success:
            print(f"[POLISH] Success! Polished model saved to best_model_polished.pth")
        else:
            print(f"[POLISH] Failed - using original best_model.pth")

    # Automatic visualization generation if enabled
    if cfg.get("viz", {}).get("enable", True) and best_state is not None:
        print(f"\n[VIZ] Generating overview visualization...")
        
        # Determine best available model for visualization
        if polish_cfg.get("enable", True) and best_state is not None:
            # Check if polished model exists and use it, otherwise fall back to best model
            ckpt_polished = os.path.join(out_dir, "best_model_polished.pth")
            if os.path.exists(ckpt_polished):
                viz_model_path = ckpt_polished
                print(f"[VIZ] Using polished model: best_model_polished.pth")
            else:
                viz_model_path = os.path.join(out_dir, "best_model.pth")
                print(f"[VIZ] Using best model: best_model.pth")
        else:
            viz_model_path = os.path.join(out_dir, "best_model.pth")
            print(f"[VIZ] Using best model: best_model.pth")
        
        # Generate overview visualization using viz_overview.py module
        try:
            # Try relative import first (when used as package)
            try:
                from .viz.viz_overview import (load_checkpoint, build_models, 
                                             compute_grids, boundary_curve, radial_slices, make_figure,
                                             compute_grids_3d, boundary_surface_3d, cross_section_slices_3d, make_figure_3d,
                                             compute_grids_1d, make_figure_1d)
            except ImportError:
                # Fall back to absolute import (when run directly)
                from viz.viz_overview import (load_checkpoint, build_models, 
                                            compute_grids, boundary_curve, radial_slices, make_figure,
                                            compute_grids_3d, boundary_surface_3d, cross_section_slices_3d, make_figure_3d,
                                            compute_grids_1d, make_figure_1d)
            
            # Load the model
            V_state, G_state, viz_cfg, meta = load_checkpoint(viz_model_path, dev)
            V_viz, G_viz = build_models(V_state, G_state, viz_cfg, dev)
            
            # Set up visualization parameters
            grid_n = 220
            lim = 3.0
            overview_path = os.path.join(out_dir, "overview.png")
            
            # Detect dimension and choose appropriate visualization
            state_dim = getattr(ctrl, 'state_dim', 2)
            
            if state_dim == 2:
                # 2D visualization (STA, CTA)
                alpha_s = cfg.get("alpha", {}).get("alpha_s", 0.03)
                alpha_v = cfg.get("alpha", {}).get("alpha_v", 0.05) 
                s_eps = cfg.get("val", {}).get("s_eps_val", 1e-3)
                
                data = compute_grids(V_viz, G_viz, ctrl, dev, grid_n, lim, alpha_s, alpha_v, s_eps)
                xb, yb = boundary_curve(G_viz, dev, lim)
                
                # Radial slices
                angles_deg = [0, 45, 90, 135]
                slice_r = 220
                slices = radial_slices(V_viz, G_viz, angles_deg, slice_r, lim, dev)
                
                # Create 2D visualization
                make_figure(
                    data, xb, yb, slices, overview_path, lim,
                    zero_center=True, mask_inside=False, add_zero_contours=True,
                    show_train_box=True, train_box=cfg.get("box", {}).get("train_s", 1.6),
                    log_phi=False, enable_3d=True, surface_downsample=3, dpi=140
                )
                
            elif state_dim == 3:
                # 3D visualization (PID-SMC)
                alpha_x1 = cfg.get("alpha", {}).get("alpha_x1", 0.03)
                alpha_x2 = cfg.get("alpha", {}).get("alpha_x2", 0.05)
                alpha_x3 = cfg.get("alpha", {}).get("alpha_x3", 0.05)
                x1_eps = cfg.get("val", {}).get("s_eps_val", 1e-3)
                
                data = compute_grids_3d(V_viz, G_viz, ctrl, dev, grid_n, lim, alpha_x1, alpha_x2, alpha_x3, x1_eps)
                xb, yb, zb, mask = boundary_surface_3d(G_viz, dev, lim)
                
                # Cross-sectional slices
                angles_deg = [0, 45, 90, 135]
                slice_r = 220
                slices = cross_section_slices_3d(V_viz, G_viz, angles_deg, slice_r, lim, dev)
                
                # Create 3D visualization
                make_figure_3d(
                    data, xb, yb, zb, mask, slices, overview_path, lim,
                    surface_downsample=4, dpi=140
                )
                
            elif state_dim == 1:
                # 1D visualization (FOSMC)
                alpha_s = cfg.get("alpha", {}).get("alpha_s", 0.03)
                s_eps = cfg.get("val", {}).get("s_eps_val", 1e-3)
                
                data = compute_grids_1d(V_viz, G_viz, ctrl, dev, grid_n, lim, alpha_s, s_eps)
                make_figure_1d(data, overview_path, lim, dpi=140)
                
            else:
                print(f"[VIZ] Unsupported state dimension: {state_dim}. Skipping visualization.")
                return
            
            print(f"[VIZ] Overview saved to: {overview_path}")
            
        except ImportError as e:
            print(f"[VIZ] Could not import visualization modules: {e}")
        except Exception as e:
            print(f"[VIZ] Visualization generation failed: {e}")

    print(f"[DONE] outputs in: {out_dir}")


def main():
    ap = build_argparser()
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    # overrides
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.no_plots:
        cfg["viz"]["enable"] = False
    
    # Strict-only overrides
    cfg["val_min_outside_frac"] = args.val_min_outside_frac
    cfg["val_min_outside_count"] = args.val_min_outside_count
    cfg["deterministic"] = args.deterministic

    # ensure directories
    ensure_dir(cfg["io"]["outputs_root"])

    train(cfg)


if __name__ == "__main__":
    main()
