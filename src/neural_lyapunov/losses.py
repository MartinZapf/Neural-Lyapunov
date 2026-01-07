from __future__ import annotations
import torch
import torch.nn.functional as F


def outside_loss_strict(dV: torch.Tensor, alpha: torch.Tensor,
                        dec_clip: float,
                        tail_mode: str = "lse", tail_beta: float = 25.0, tail_topk: float = 0.05,
                        w_mean: float = 1.0, w_tail: float = 1.0) -> torch.Tensor:
    """Strict: margin==0 always. Aggregate once, then apply softplus once."""
    dec = (dV + alpha).clamp(min=-dec_clip, max=dec_clip)

    # mean component
    mean_term = F.softplus(dec.mean())

    # tail component (choose one aggregator)
    if tail_mode == "max":
        agg = dec.max()
    elif tail_mode == "topk":
        k = max(1, int(tail_topk * dec.numel()))
        agg, _ = torch.topk(dec, k, largest=True, sorted=False)
        agg = agg.mean()
    else:  # "lse"
        mval = dec.max()
        agg = mval + (1.0 / tail_beta) * torch.log(torch.exp(tail_beta * (dec - mval)).mean())

    tail_term = F.softplus(agg)
    return w_mean * mean_term + w_tail * tail_term


def ring_loss_strict(dV: torch.Tensor, alpha: torch.Tensor, phi_ring: torch.Tensor,
                     dist_eps: float, dist_gamma: float, dec_clip: float) -> torch.Tensor:
    """Ring loss for strict mode: no margins, always enforce dV + alpha*weight <= 0."""
    dist = torch.clamp(phi_ring - 1.0 + dist_eps, min=0.0)
    weight = dist.pow(dist_gamma)
    dec_ring = (dV + alpha * weight).clamp(min=-dec_clip, max=dec_clip)
    return F.softplus(dec_ring).mean()


def size_quantile_radius(phi_dirs: torch.Tensor, q: float = 0.90) -> torch.Tensor:
    """Minimize q-quantile of radius r=1/phi by averaging top-k radii."""
    r = 1.0 / phi_dirs.clamp_min(1e-6)
    m = r.numel()
    k = max(1, int(q * m))
    topk, _ = torch.topk(r, k, largest=True, sorted=False)
    return topk.mean()


def angular_smoothness(phi_dirs: torch.Tensor) -> torch.Tensor:
    """
    Dimension-agnostic angular smoothness penalty.
    
    For 2D: Uses periodic finite differences (original approach)
    For 3D: Uses variance-based smoothness (no natural ordering on sphere)
    
    The variance approach encourages similar gauge values across all directions,
    which promotes smooth boundaries without requiring spatial ordering.
    """
    # Use variance-based smoothness - works for any dimension
    # This encourages phi values to be similar across all sampled directions
    return phi_dirs.var()


def size_objective(phi_dirs: torch.Tensor, quantile: float = 0.90) -> torch.Tensor:
    """Size objective using quantile of radius for robustness."""
    return size_quantile_radius(phi_dirs, q=quantile)


def size_estimate(phi_dirs: torch.Tensor, dimension: int = 2) -> torch.Tensor:
    """
    Dimension-aware size estimate for logging.
    
    IMPORTANT: This must match the metric used in HPO (_estimate_area in optuna_tune.py)
    to avoid confusion. Both training logs and HPO should show the same number!
    
    For 1D: Length (2 × max radius)
    For 2D: Area (π × mean(r²))
    For 3D: Volume (4π/3 × mean(r³))
    """
    r = 1.0 / phi_dirs.clamp_min(1e-6)
    if dimension == 1:
        # 1D: Length = 2 × max_radius (full extent of interval)
        return 2.0 * r.max()
    elif dimension == 3:
        # 3D: Volume = (4π/3) × mean(r³)
        import math
        return (4.0 * math.pi / 3.0) * (r ** 3).mean()
    else:
        # 2D: Area = π × mean(r²)  [MATCHES HPO!]
        import math
        return math.pi * (r ** 2).mean()


def floor_loss(Vvals: torch.Tensor, z: torch.Tensor, c_lower: float) -> torch.Tensor:
    v_lower = c_lower * torch.sum(z * z, dim=1)
    return F.relu(v_lower - Vvals).mean()


def grad_reg(gV: torch.Tensor, target: float = 25.0, weight: float = 5e-4) -> torch.Tensor:
    gn = torch.linalg.norm(gV, dim=1)
    return weight * torch.mean(torch.clamp(gn - target, min=0.0) ** 2)


def lipschitz_penalty_V(
    V: torch.nn.Module,
    z: torch.Tensor,
    target_lipschitz: float = 10.0
) -> torch.Tensor:
    """
    Penalize when gradient norm of V exceeds target Lipschitz constant.

    Encourages smooth Lyapunov functions by enforcing ||∇V(z)|| ≤ L.
    Promotes smoother V for better interpolation and numerical stability.

    Args:
        V: Lyapunov network
        z: Batch of states, shape (B, state_dim)
        target_lipschitz: Target Lipschitz constant L

    Returns:
        Penalty term (scalar), 0 if all gradients within target
    """
    # Detach z to avoid double backward through controller dynamics
    z_copy = z.detach().clone().requires_grad_(True)
    
    # Compute V and its gradient
    V_vals = V(z_copy)
    grad_V = torch.autograd.grad(
        V_vals.sum(), z_copy,
        create_graph=True,
        retain_graph=True  # Keep graph alive for other losses
    )[0]
    
    # Compute gradient norm at each point
    grad_norm = torch.linalg.norm(grad_V, dim=1)
    
    # Soft penalty for exceeding target (squared hinge loss)
    violations = F.relu(grad_norm - target_lipschitz)
    penalty = (violations ** 2).mean()
    
    return penalty


def lipschitz_uniformity_penalty(
    V: torch.nn.Module,
    z: torch.Tensor,
    target_lipschitz: float = 10.0
) -> torch.Tensor:
    """
    Encourage uniform Lipschitz constant across all points.
    
    Penalizes both:
    - Gradients exceeding target
    - Large variance in gradient norms (encourages uniform smoothness)
    
    Args:
        V: Lyapunov network
        z: Batch of states, shape (B, state_dim)
        target_lipschitz: Target Lipschitz constant
        
    Returns:
        Combined penalty (scalar)
    """
    z_copy = z.detach().clone().requires_grad_(True)
    V_vals = V(z_copy)
    grad_V = torch.autograd.grad(
        V_vals.sum(), z_copy,
        create_graph=True,
        retain_graph=True  # Keep graph alive for other losses
    )[0]
    
    grad_norm = torch.linalg.norm(grad_V, dim=1)
    
    # Penalty 1: Exceeding target
    violations = F.relu(grad_norm - target_lipschitz)
    penalty_exceed = (violations ** 2).mean()
    
    # Penalty 2: Variance in gradient norms (encourage uniformity)
    penalty_variance = grad_norm.var()
    
    return penalty_exceed + 0.1 * penalty_variance


def curvature_penalty_V(
    V: torch.nn.Module,
    z: torch.Tensor,
    sigma: float = 0.05
) -> torch.Tensor:
    """
    Penalize high curvature: encourage similar gradients at nearby points.
    
    This is a second-order smoothness penalty (related to Hessian).
    Uses finite differences: penalize ||∇V(z) - ∇V(z + ε)||²
    
    Args:
        V: Lyapunov network
        z: Batch of states, shape (B, state_dim)
        sigma: Perturbation scale for finite differences
        
    Returns:
        Curvature penalty (scalar)
    """
    # Original points
    z_original = z.detach().clone().requires_grad_(True)
    V_original = V(z_original)
    grad_original = torch.autograd.grad(
        V_original.sum(), z_original,
        create_graph=True,
        retain_graph=True  # Keep graph alive for other losses
    )[0]
    
    # Perturbed points (random small perturbation)
    perturbation = torch.randn_like(z) * sigma
    z_perturbed = (z + perturbation).detach().clone().requires_grad_(True)
    V_perturbed = V(z_perturbed)
    grad_perturbed = torch.autograd.grad(
        V_perturbed.sum(), z_perturbed,
        create_graph=False  # Don't create graph for perturbed gradient
    )[0]
    
    # Penalize difference in gradients
    # grad_perturbed acts as a target (detached), grad_original has gradients
    grad_diff = grad_original - grad_perturbed.detach()
    penalty = (grad_diff ** 2).sum(dim=1).mean()
    
    return penalty
