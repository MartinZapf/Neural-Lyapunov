from __future__ import annotations
from typing import Dict
import torch
try:
    from .gauges import FlexibleGauge
    from .controllers import BaseSMC
except ImportError:
    from gauges import FlexibleGauge
    from controllers import BaseSMC


def validate_global_outside(V, G: FlexibleGauge, ctrl: BaseSMC, *,
                            alpha_s: float, alpha_v: float,
                            train_box_s: float, train_box_v: float, N: int,
                            s_eps_val: float = 1e-3,
                            val_min_outside_frac: float = 0.02, val_min_outside_count: int = 200,
                            val_tol_max: float = 0.0, val_tol_q: float = 0.0,
                            val_dtype: torch.dtype = torch.float64,
                            axis_exclusion_eps: float = 0.0) -> Dict:
    """
    Rigorous grid-based validation with anti-vacuity checks.
    
    Validates the training condition: dV/dt + α(z) ≤ 0 on a uniform grid.
    Note: α(z) are regularization parameters for training, not theoretical constants.
    
    Checks:
      - Compute dec_plain = dV + alpha on uniform grid over train box (high precision)
      - Restrict to φ>1 and enforce minimum coverage requirements
      - Report max, p95, p99, fraction positive, and coverage metrics
      - Fail if insufficient outside coverage or any violations > tolerance
    """
    device = next(V.parameters()).device
    
    # Use high precision for validation
    s_vals = torch.linspace(-train_box_s, train_box_s, N, device=device, dtype=val_dtype)
    v_vals = torch.linspace(-train_box_v, train_box_v, N, device=device, dtype=val_dtype)
    Sg, Vg = torch.meshgrid(s_vals, v_vals, indexing="ij")
    Z = torch.stack([Sg.reshape(-1), Vg.reshape(-1)], dim=1).requires_grad_(True)

    # Cast models to validation precision temporarily
    original_dtype = next(V.parameters()).dtype
    if val_dtype != original_dtype:
        V = V.to(dtype=val_dtype)
        G = G.to(dtype=val_dtype)

    # Forward pass - need gradients for validation (no torch.no_grad() here)
    Z_val = Z.detach().requires_grad_(True)
    
    # Compute V and gradients
    Vv = V(Z_val)
    gV = torch.autograd.grad(Vv.sum(), Z_val, create_graph=False, retain_graph=False)[0]
    dV = ctrl.worst_dV(gV, Z_val, s_eps=s_eps_val)

    alpha = alpha_s * torch.abs(Z_val[:, 0]) + alpha_v * torch.abs(Z_val[:, 1])
    dec_plain = dV + alpha

    with torch.no_grad():
        phi = G(Z_val)
        outside_mask = phi > 1.0
        outside_count = int(outside_mask.sum().item())
        total_points = outside_mask.numel()
        outside_frac = outside_count / total_points
        
        # Restore original dtype
        if val_dtype != original_dtype:
            V = V.to(dtype=original_dtype)
            G = G.to(dtype=original_dtype)

        # Anti-vacuity checks: require minimum coverage
        if outside_count < val_min_outside_count or outside_frac < val_min_outside_frac:
            return {
                "is_ok": False,
                "reason": "insufficient_coverage", 
                "outside_count": outside_count,
                "outside_frac": outside_frac,
                "total_points": total_points,
                "min_required_count": val_min_outside_count,
                "min_required_frac": val_min_outside_frac,
                "max": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "frac_pos": float("nan")
            }
        
        # Compute statistics on outside region
        vals_outside = dec_plain[outside_mask]
        max_dec = vals_outside.max().item()
        p95_dec = torch.quantile(vals_outside, 0.95).item()
        p99_dec = torch.quantile(vals_outside, 0.99).item()
        frac_pos = (vals_outside > 0).float().mean().item()
        
        # Strict validation: check against tolerances
        is_ok = (max_dec <= val_tol_max) and (p95_dec <= val_tol_q)
        
        return {
            "is_ok": is_ok,
            "max": max_dec,
            "p95": p95_dec, 
            "p99": p99_dec,
            "frac_pos": frac_pos,
            "outside_count": outside_count,
            "outside_frac": outside_frac,
            "total_points": total_points,
            "coverage_ok": True  # we passed coverage checks to get here
        }


def validate_global_outside_3d(V, G: FlexibleGauge, ctrl: BaseSMC, *,
                              alpha_x1: float, alpha_x2: float, alpha_x3: float,
                              train_box_x1: float, train_box_x2: float, train_box_x3: float,
                              N: int = 40,  # Smaller default for 3D to manage memory (40³ = 64k points)
                              s_eps_val: float = 1e-3,
                              val_min_outside_frac: float = 0.02, val_min_outside_count: int = 200,
                              val_tol_max: float = 0.0, val_tol_q: float = 0.0,
                              val_dtype: torch.dtype = torch.float64,
                              axis_exclusion_eps: float = 0.0) -> Dict:
    """
    Rigorous 3D grid-based validation with anti-vacuity checks.

    Validates the training condition: dV/dt + α(z) ≤ 0 on a 3D uniform grid.
    Note: α(z) are regularization parameters for training, not theoretical constants.

    Checks:
      - Compute dec_plain = dV + alpha on uniform 3D grid over train box (high precision)
      - Restrict to phi>1 and enforce minimum coverage requirements
      - report max, p95, p99, fraction positive, and coverage metrics
      - fail if insufficient outside coverage or any violations > tolerance

    Memory usage: N³ points total, so N=40 gives 64k points, N=50 gives 125k points.
    Uses batching to avoid memory issues with large grids.

    Args:
        axis_exclusion_eps: If > 0, exclude points where |x1| < eps or |x2| < eps.
            This is needed for lifted coordinates (homogeneous transforms) where the
            Jacobian of the coordinate transform is singular at the coordinate planes.
            The physical dynamics are bounded, but gradient computation through the
            singular Jacobian causes numerical issues.
    """
    device = next(V.parameters()).device
    
    # Create 3D grid - be careful with memory usage
    x1_vals = torch.linspace(-train_box_x1, train_box_x1, N, device=device, dtype=val_dtype)
    x2_vals = torch.linspace(-train_box_x2, train_box_x2, N, device=device, dtype=val_dtype)
    x3_vals = torch.linspace(-train_box_x3, train_box_x3, N, device=device, dtype=val_dtype)
    
    X1g, X2g, X3g = torch.meshgrid(x1_vals, x2_vals, x3_vals, indexing="ij")
    Z = torch.stack([X1g.reshape(-1), X2g.reshape(-1), X3g.reshape(-1)], dim=1)
    
    # Cast models to validation precision temporarily
    original_dtype = next(V.parameters()).dtype
    if val_dtype != original_dtype:
        V = V.to(dtype=val_dtype)
        G = G.to(dtype=val_dtype)
    
    try:
        # Process in batches to avoid memory issues with large 3D grids
        batch_size = min(8192, Z.shape[0])  # Conservative batch size for 3D
        all_dec = []
        all_phi = []
        
        for i in range(0, Z.shape[0], batch_size):
            Z_batch = Z[i:i+batch_size].detach().requires_grad_(True)
            
            # Compute V and gradients for this batch
            Vv = V(Z_batch)
            gV = torch.autograd.grad(Vv.sum(), Z_batch, create_graph=False, retain_graph=False)[0]
            dV = ctrl.worst_dV(gV, Z_batch, s_eps=s_eps_val)
            
            # Alpha regularization for 3D
            alpha = (alpha_x1 * torch.abs(Z_batch[:, 0]) +
                    alpha_x2 * torch.abs(Z_batch[:, 1]) +
                    alpha_x3 * torch.abs(Z_batch[:, 2]))
            dec_plain = dV + alpha

            # Gauge values (no gradients needed)
            with torch.no_grad():
                phi = G(Z_batch)

            all_dec.append(dec_plain.detach())
            all_phi.append(phi)

        # Combine all batches
        dec_all = torch.cat(all_dec, dim=0)
        phi_all = torch.cat(all_phi, dim=0)

        # Axis exclusion for lifted coordinates (homogeneous transforms)
        # The Jacobian of the coordinate transform is singular at x1=0 and x2=0
        # which causes numerical issues even though the physical dynamics are bounded
        if axis_exclusion_eps > 0:
            axis_safe_mask = (torch.abs(Z[:, 0]) >= axis_exclusion_eps) & \
                             (torch.abs(Z[:, 1]) >= axis_exclusion_eps)
            excluded_count = int((~axis_safe_mask).sum().item())
        else:
            axis_safe_mask = torch.ones(Z.shape[0], dtype=torch.bool, device=Z.device)
            excluded_count = 0

        # Outside region analysis (combine with axis exclusion)
        outside_mask = (phi_all > 1.0) & axis_safe_mask
        outside_count = int(outside_mask.sum().item())
        total_points = int(axis_safe_mask.sum().item())  # Only count non-excluded points
        outside_frac = outside_count / max(total_points, 1)

        # Restore original dtype
        if val_dtype != original_dtype:
            V = V.to(dtype=original_dtype)
            G = G.to(dtype=original_dtype)
        
        # Anti-vacuity checks: require minimum coverage
        if outside_count < val_min_outside_count or outside_frac < val_min_outside_frac:
            return {
                "is_ok": False,
                "reason": "insufficient_coverage", 
                "outside_count": outside_count,
                "outside_frac": outside_frac,
                "total_points": total_points,
                "min_required_count": val_min_outside_count,
                "min_required_frac": val_min_outside_frac,
                "max": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "frac_pos": float("nan")
            }
        
        # Compute statistics on outside region
        vals_outside = dec_all[outside_mask]
        max_dec = vals_outside.max().item()
        p95_dec = torch.quantile(vals_outside, 0.95).item()
        p99_dec = torch.quantile(vals_outside, 0.99).item()
        frac_pos = (vals_outside > 0).float().mean().item()
        
        # Strict validation: check against tolerances
        is_ok = (max_dec <= val_tol_max) and (p95_dec <= val_tol_q)
        
        return {
            "is_ok": is_ok,
            "max": max_dec,
            "p95": p95_dec,
            "p99": p99_dec,
            "frac_pos": frac_pos,
            "outside_count": outside_count,
            "outside_frac": outside_frac,
            "total_points": total_points,
            "coverage_ok": True,  # we passed coverage checks to get here
            "excluded_count": excluded_count,  # Points excluded due to axis singularity
        }

    except Exception as e:
        # Restore dtype on error
        if val_dtype != original_dtype:
            V = V.to(dtype=original_dtype)
            G = G.to(dtype=original_dtype)
        
        print(f"[WARN] 3D validation failed: {e}")
        return {
            "is_ok": False,
            "reason": f"validation_error: {e}",
            "max": float('inf'),
            "p95": float('inf'),
            "p99": float('inf'),
            "frac_pos": 1.0,
            "outside_count": 0,
            "outside_frac": 0.0,
            "total_points": 0
        }


def validate_global_outside_1d(
    V, G, ctrl,
    alpha_s: float,
    train_box_s: float,
    N: int = 1000,
    s_eps_val: float = 1e-3,
    val_dtype=torch.float64,
    val_tol_max: float = 0.0,
    val_tol_q: float = 0.0,
    val_min_outside_frac: float = 0.02,
    val_min_outside_count: int = 20,
    axis_exclusion_eps: float = 0.0,
) -> dict:
    """
    Validate training condition on 1D grid.
    
    Checks: dV/dt + α(|s|) ≤ 0 for all s outside gauge boundary.
    Note: α is a regularization parameter for training, not a theoretical constant.
    
    Args:
        V: Lyapunov network
        G: Gauge function
        ctrl: 1D controller  
        alpha_s: Regularization weight for training
        train_box_s: Training box half-width
        N: Number of grid points
        val_dtype: Validation precision (float64 recommended)
        val_tol_max: Tolerance for maximum violation
        val_tol_q: Tolerance for quantile violation
        val_min_outside_frac: Minimum fraction outside required
        val_min_outside_count: Minimum absolute count outside required
        
    Returns:
        dict with keys: is_ok, max, p95, p99, frac_pos, outside_count, etc.
    """
    device = next(V.parameters()).device
    original_dtype = next(V.parameters()).dtype
    
    try:
        # Convert to validation precision
        if val_dtype != original_dtype:
            V = V.to(dtype=val_dtype)
            G = G.to(dtype=val_dtype)
        
        # Create 1D grid: s ∈ [-train_box_s, train_box_s]
        s_vals = torch.linspace(-train_box_s, train_box_s, N, 
                                device=device, dtype=val_dtype)
        Z = s_vals.unsqueeze(1)  # Shape (N, 1)
        
        # Compute V and gradient in batches (memory efficient)
        batch_size = 512
        all_dec = []
        all_phi = []
        
        for i in range(0, N, batch_size):
            Z_batch = Z[i:i+batch_size]
            
            # Compute V and gradient
            Z_batch.requires_grad_(True)
            Vv = V(Z_batch)
            gV = torch.autograd.grad(Vv.sum(), Z_batch, create_graph=False)[0]
            
            # Worst-case dV under Filippov dynamics
            dV = ctrl.worst_dV(gV, Z_batch, s_eps=s_eps_val)
            
            # Alpha regularization: α(|s|)
            alpha = alpha_s * torch.abs(Z_batch[:, 0])
            
            # Decrease condition
            dec_plain = dV + alpha
            
            # Gauge values (no gradients needed)
            with torch.no_grad():
                phi = G(Z_batch)
            
            all_dec.append(dec_plain.detach())
            all_phi.append(phi)
        
        # Combine all batches
        dec_all = torch.cat(all_dec, dim=0)
        phi_all = torch.cat(all_phi, dim=0)
        
        # Outside region analysis
        outside_mask = phi_all > 1.0
        outside_count = int(outside_mask.sum().item())
        total_points = outside_mask.numel()
        outside_frac = outside_count / total_points
        
        # Restore original dtype
        if val_dtype != original_dtype:
            V = V.to(dtype=original_dtype)
            G = G.to(dtype=original_dtype)
        
        # Anti-vacuity checks: require minimum outside coverage
        if outside_count < val_min_outside_count or outside_frac < val_min_outside_frac:
            return {
                "is_ok": False,
                "reason": "insufficient_coverage",
                "outside_count": outside_count,
                "outside_frac": outside_frac,
                "total_points": total_points,
                "min_required_count": val_min_outside_count,
                "min_required_frac": val_min_outside_frac,
                "max": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "frac_pos": float("nan")
            }
        
        # Compute statistics on outside region only
        vals_outside = dec_all[outside_mask]
        max_dec = vals_outside.max().item()
        p95_dec = torch.quantile(vals_outside, 0.95).item()
        p99_dec = torch.quantile(vals_outside, 0.99).item()
        frac_pos = (vals_outside > 0).float().mean().item()
        
        # Strict validation: check against tolerances
        is_ok = (max_dec <= val_tol_max) and (p95_dec <= val_tol_q)
        
        return {
            "is_ok": is_ok,
            "max": max_dec,
            "p95": p95_dec,
            "p99": p99_dec,
            "frac_pos": frac_pos,
            "outside_count": outside_count,
            "outside_frac": outside_frac,
            "total_points": total_points,
            "coverage_ok": True
        }
        
    except Exception as e:
        # Restore dtype on error
        if val_dtype != original_dtype:
            V = V.to(dtype=original_dtype)
            G = G.to(dtype=original_dtype)
        
        print(f"[WARN] 1D validation failed: {e}")
        return {
            "is_ok": False,
            "reason": f"validation_error: {e}",
            "max": float('inf'),
            "p95": float('inf'),
            "p99": float('inf'),
            "frac_pos": 1.0,
            "outside_count": 0,
            "outside_frac": 0.0,
            "total_points": 0
        }
