"""
BC99-style trainer (no alpha, no gauge).

Used to produce the "Gauge = none" results reported in the paper for
controllers where the dense-grid decrease test passes at all nonzero grid
points in the training box (FOSMC and STA in lifted coordinates).

Loss:
    L = mean(softplus(V_dot_worst))
      + (1/beta_lse) * (logsumexp(beta_lse * V_dot_worst) - log N)   # tail
      + lam_jac  * mean(|grad V|^2)
      + lam_curv * mean(|H V . v|^2)                                  # Hutchinson

where V_dot_worst is the controller's worst-case Filippov derivative
(ctrl.worst_dV), and v is a per-batch random vector for the stochastic
trace-of-Hessian-squared estimate.

Validation: max V_dot_worst on a uniform grid, masking only ||z|| < r_origin.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


def _worst_dV(V: nn.Module, ctrl, Z: Tensor, s_eps: float = 1e-3) -> Tensor:
    with torch.enable_grad():
        Z = Z.detach().requires_grad_(True)
        Vv = V(Z)
        gV = torch.autograd.grad(Vv.sum(), Z, create_graph=False, retain_graph=False)[0]
        return ctrl.worst_dV(gV, Z, s_eps=s_eps).detach()


def bc_train(V: nn.Module, ctrl, *, box: Tuple[float, ...], device: torch.device,
             epochs: int = 4000, lr: float = 1e-3,
             batch_per_step: int = 4096,
             beta_lse: float = 30.0,
             lam_jac: float = 1e-4,
             lam_curv: float = 1e-4,
             r_origin: float = 1e-6,
             s_eps: float = 1e-3,
             log_every: int = 200,
             val_every: int = 200,
             val_N: int = 401,
             dtype=torch.float64,
             seed: int = 0) -> dict:
    """Train V from scratch with no alpha and no gauge.

    Returns dict with history, final_val_max, best_val_max. Reloads V to the
    best validation state before returning.
    """
    V = V.to(device=device, dtype=dtype); V.train()
    opt = torch.optim.Adam(V.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    dim = len(box)
    box_t = torch.tensor(box, device=device, dtype=dtype)
    history = []
    best_val = float("inf")
    best_state = None

    def sample(B):
        z = (torch.rand(B, dim, generator=gen, device=device, dtype=dtype) * 2 - 1) * box_t
        n = torch.linalg.norm(z, dim=1)
        return z[n > r_origin]

    @torch.no_grad()
    def validate():
        if dim == 1:
            zs = torch.linspace(-box[0], box[0], val_N, device=device, dtype=dtype).reshape(-1, 1)
        elif dim == 2:
            s = torch.linspace(-box[0], box[0], val_N, device=device, dtype=dtype)
            v = torch.linspace(-box[1], box[1], val_N, device=device, dtype=dtype)
            S, Vg = torch.meshgrid(s, v, indexing="ij")
            zs = torch.stack([S.reshape(-1), Vg.reshape(-1)], 1)
        else:
            n_pts = val_N * val_N * 4
            zs = (torch.rand(n_pts, dim, generator=gen, device=device, dtype=dtype) * 2 - 1) * box_t
        n = torch.linalg.norm(zs, dim=1)
        zs = zs[n > r_origin]
        return float(_worst_dV(V, ctrl, zs).max().item())

    for it in range(1, epochs + 1):
        z = sample(batch_per_step)
        if z.shape[0] == 0:
            continue
        z = z.detach().requires_grad_(True)
        Vv = V(z)
        gV = torch.autograd.grad(Vv.sum(), z, create_graph=True)[0]
        dV = ctrl.worst_dV(gV, z, s_eps=s_eps)

        loss_mean = torch.nn.functional.softplus(dV).mean()
        loss_tail = (torch.logsumexp(beta_lse * dV, dim=0)
                     - torch.log(torch.tensor(float(dV.numel()), device=device))) / beta_lse
        loss_jac = (gV ** 2).sum(dim=1).mean()
        v_rand = torch.randn_like(z)
        Hv = torch.autograd.grad((gV * v_rand).sum(), z, create_graph=True)[0]
        loss_curv = (Hv ** 2).sum(dim=1).mean()

        loss = loss_mean + loss_tail + lam_jac * loss_jac + lam_curv * loss_curv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(V.parameters(), max_norm=10.0)
        opt.step()

        if it % log_every == 0:
            history.append(dict(iter=it, loss=float(loss.item()),
                                loss_mean=float(loss_mean.item()),
                                loss_tail=float(loss_tail.item()),
                                loss_jac=float(loss_jac.item()),
                                loss_curv=float(loss_curv.item())))
        if it % val_every == 0:
            vmax = validate()
            history[-1]["val_max"] = vmax
            if vmax < best_val:
                best_val = vmax
                best_state = {k: v.detach().clone() for k, v in V.state_dict().items()}
            print(f"  [bc-train] iter={it:5d}  loss={float(loss.item()):.4e}  "
                  f"val_max={vmax:+.4e}  best={best_val:+.4e}", flush=True)

    if best_state is not None:
        V.load_state_dict(best_state)
    final_val = validate()
    return dict(history=history, final_val_max=final_val, best_val_max=best_val)
