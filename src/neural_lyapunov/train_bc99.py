"""CLI driver for the BC99 trainer (no alpha, no gauge).

Reproduces the "Gauge = none" results reported in the paper for
controllers where the dense-grid decrease test passes at all nonzero grid
points in the training box.

Example:
    python -m neural_lyapunov.train_bc99 --config configs/fosmc.yaml --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from neural_lyapunov.bc_train import bc_train
from neural_lyapunov.controllers import get_controller
from neural_lyapunov.models import LiftedLyapNet, SimpleLyapNet
from neural_lyapunov.utils import load_yaml, set_seed


_BOX_KEYS_BY_DIM = {
    1: ("train_s",),
    2: ("train_s", "train_v"),
    3: ("train_s", "train_v", "train_a"),
}


def _box_from_cfg(cfg) -> tuple:
    """Resolve the training box (per-axis half-width tuple) from a config."""
    box_cfg = cfg.get("box", {})
    if isinstance(box_cfg, (list, tuple)):
        return tuple(float(b) for b in box_cfg)
    gauge_dim = int(cfg.get("gauge", {}).get("input_dim", 0)) or None
    keys = _BOX_KEYS_BY_DIM.get(gauge_dim)
    if keys and all(k in box_cfg for k in keys):
        return tuple(float(box_cfg[k]) for k in keys)
    # Fallback: take all numeric values in declaration order.
    vals = [float(v) for v in box_cfg.values() if isinstance(v, (int, float))]
    if vals:
        return tuple(vals)
    raise ValueError("Cannot infer training box from config.")


def build_V(cfg, dim, dtype, device):
    mc = dict(cfg.get("model", {}))
    mc.pop("alpha_bar", None)
    mc["alpha_bar"] = 0.0  # explicit: BC99 uses no alpha floor
    mc["input_dim"] = dim
    use_lifted = mc.pop("use_lifted_coords", False)
    lift_type = mc.pop("lift_type", "sta")
    if use_lifted:
        V = LiftedLyapNet(lift_type=lift_type, **mc)
    else:
        V = SimpleLyapNet(**mc)
    return V.to(device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lam_jac", type=float, default=1e-4)
    ap.add_argument("--lam_curv", type=float, default=1e-4)
    ap.add_argument("--out_dir", default="outputs_bc99")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    box = _box_from_cfg(cfg)
    dim = len(box)

    ctrl_cfg = cfg["controller"]
    ctrl = get_controller(ctrl_cfg["name"], **ctrl_cfg.get("params", {}))

    V = build_V(cfg, dim, dtype, device)
    n_params = sum(p.numel() for p in V.parameters())
    name = os.path.splitext(os.path.basename(args.config))[0]
    print(f"[bc99] config={name} seed={args.seed} dim={dim} params={n_params}")

    t0 = time.time()
    res = bc_train(
        V, ctrl, box=box, device=device,
        epochs=args.epochs, lr=args.lr,
        batch_per_step=args.batch,
        lam_jac=args.lam_jac, lam_curv=args.lam_curv,
        seed=args.seed, dtype=dtype,
    )
    elapsed = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{name}_seed{args.seed}"
    torch.save({"V": V.state_dict(), "cfg": cfg, "box": box},
               os.path.join(args.out_dir, f"{tag}.pth"))
    with open(os.path.join(args.out_dir, f"{tag}_history.json"), "w") as f:
        json.dump({"history": res["history"], "final_val_max": res["final_val_max"],
                   "best_val_max": res["best_val_max"], "elapsed_s": elapsed,
                   "args": vars(args)}, f, indent=2)
    final_val = res["final_val_max"]
    verdict = "Gauge = none" if final_val <= 0 else f"FAIL (val_max={final_val:+.4e} > 0)"
    print(f"[bc99] {tag}: final val_max = {final_val:+.4e}  ->  {verdict}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
