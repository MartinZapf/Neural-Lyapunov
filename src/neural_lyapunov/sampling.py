from __future__ import annotations
import math
import torch
try:
    from .gauges import FlexibleGauge
except ImportError:
    from gauges import FlexibleGauge


@torch.no_grad()
def boundary_radius(gauge: FlexibleGauge, dirs: torch.Tensor) -> torch.Tensor:
    return 1.0 / gauge(dirs).clamp_min(1e-6)


@torch.no_grad()
def sample_ring_mixed(gauge: FlexibleGauge, m: int, delta_rel: float, delta_abs: float,
                      rel_frac: float, device) -> torch.Tensor:
    """
    Mix two collars:
      relative: r = r_b * (1 + delta_rel * U[0,1])
      absolute: r = r_b + delta_abs * U[0.7,1.3]
    """
    m_rel = int(m * rel_frac)
    m_abs = m - m_rel

    th = torch.rand(m, device=device) * 2 * math.pi
    dirs = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
    r_b = boundary_radius(gauge, dirs)

    out = []
    if m_rel > 0:
        u = torch.rand(m_rel, device=device)
        r = r_b[:m_rel] * (1.0 + delta_rel * u)
        out.append(dirs[:m_rel] * r.unsqueeze(1))
    if m_abs > 0:
        u = 0.7 + 0.6 * torch.rand(m_abs, device=device)
        r = r_b[m_rel:] + delta_abs * u
        out.append(dirs[m_rel:] * r.unsqueeze(1))
    return torch.cat(out, dim=0)


@torch.no_grad()
def sample_far_outside(gauge: FlexibleGauge, n: int, device,
                       delta_min: float = 0.5, delta_max: float = 2.0) -> torch.Tensor:
    """Sample points guaranteed to be far outside the boundary."""
    # Random directions
    th = torch.rand(n, device=device) * 2 * math.pi
    dirs = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
    
    # Get boundary radius
    r_b = boundary_radius(gauge, dirs)
    
    # Sample at radius r_b + delta where delta ~ U[delta_min, delta_max]
    delta = delta_min + (delta_max - delta_min) * torch.rand(n, device=device)
    r = r_b + delta
    
    return dirs * r.unsqueeze(1)


@torch.no_grad()
def sample_global_outside(gauge: FlexibleGauge, m: int, R_s: float, R_v: float, device,
                          max_tries: int = 10, min_outside_frac: float = 0.2,
                          expanded_box_scale: float = 1.5, 
                          far_outside_delta_min: float = 0.5,
                          far_outside_delta_max: float = 2.0) -> torch.Tensor:
    """
    Robust outside sampling with guaranteed minimum outside count.
    
    Strategy:
    1. Try rejection sampling in original box
    2. If insufficient, try expanded box  
    3. If still insufficient, supplement with far-outside samples
    """
    min_outside_batch = max(1, int(min_outside_frac * m))
    
    batch = []
    need = m
    tries = 0
    
    # Phase 1: Standard rejection sampling in original box
    while need > 0 and tries < max_tries:
        z = torch.empty(need * 3, 2, device=device)
        z[:, 0] = (2 * torch.rand(need * 3, device=device) - 1.0) * R_s
        z[:, 1] = (2 * torch.rand(need * 3, device=device) - 1.0) * R_v
        phi = gauge(z)
        keep = phi > 1.0
        if keep.any():
            chosen = z[keep]
            take = min(need, chosen.shape[0])
            batch.append(chosen[:take])
            need -= take
        tries += 1
    
    # Phase 2: If we don't have enough, try expanded box
    if len(batch) == 0 or torch.cat(batch, dim=0).size(0) < min_outside_batch:
        tries = 0
        while need > 0 and tries < max_tries:
            # Expanded box
            R_s_exp = R_s * expanded_box_scale
            R_v_exp = R_v * expanded_box_scale
            z = torch.empty(need * 3, 2, device=device)
            z[:, 0] = (2 * torch.rand(need * 3, device=device) - 1.0) * R_s_exp
            z[:, 1] = (2 * torch.rand(need * 3, device=device) - 1.0) * R_v_exp
            phi = gauge(z)
            keep = phi > 1.0
            if keep.any():
                chosen = z[keep]
                take = min(need, chosen.shape[0])
                batch.append(chosen[:take])
                need -= take
            tries += 1
    
    # Phase 3: Supplement with far-outside samples if still insufficient
    current_count = torch.cat(batch, dim=0).size(0) if batch else 0
    if current_count < min_outside_batch:
        supplement_needed = min_outside_batch - current_count
        if need > 0:
            # Take the minimum of what we still need and what's required for minimum
            supplement_count = min(need, supplement_needed) 
        else:
            supplement_count = supplement_needed
            
        far_samples = sample_far_outside(gauge, supplement_count, device,
                                       far_outside_delta_min, far_outside_delta_max)
        batch.append(far_samples)
        need -= supplement_count
    
    # Phase 4: If we still need more samples for the full batch size, use ring sampling
    if need > 0:
        filler = sample_ring_mixed(gauge, need, delta_rel=0.5, delta_abs=0.3, rel_frac=0.5, device=device)
        batch.append(filler)
    
    result = torch.cat(batch, dim=0)
    return result[:m]  # Trim to exact size requested


# ===================================
# 3D Sampling Functions  
# ===================================

@torch.no_grad()
def sample_ring_mixed_3d(gauge: FlexibleGauge, m: int, delta_rel: float, delta_abs: float,
                         rel_frac: float, device) -> torch.Tensor:
    """Sample ring around boundary in 3D using sphere sampling."""
    m_rel = int(m * rel_frac)
    m_abs = m - m_rel
    
    # Sample directions uniformly on unit sphere in 3D
    # Using normal distribution method
    dirs = torch.randn(m, 3, device=device)
    dirs = torch.nn.functional.normalize(dirs, dim=1)
    
    # Get boundary radii
    r_b = boundary_radius(gauge, dirs)
    
    out = []
    if m_rel > 0:
        u = torch.rand(m_rel, device=device)
        r = r_b[:m_rel] * (1.0 + delta_rel * u)
        out.append(dirs[:m_rel] * r.unsqueeze(1))
    if m_abs > 0:
        u = 0.7 + 0.6 * torch.rand(m_abs, device=device)
        r = r_b[m_rel:] + delta_abs * u
        out.append(dirs[m_rel:] * r.unsqueeze(1))
    
    return torch.cat(out, dim=0)


@torch.no_grad()
def sample_far_outside_3d(gauge: FlexibleGauge, n: int, device,
                          delta_min: float = 0.5, delta_max: float = 2.0) -> torch.Tensor:
    """Sample points guaranteed to be far outside the boundary in 3D."""
    # Random directions on 3D sphere
    dirs = torch.randn(n, 3, device=device)
    dirs = torch.nn.functional.normalize(dirs, dim=1)
    
    # Get boundary radius
    r_b = boundary_radius(gauge, dirs)
    
    # Sample at radius r_b + delta where delta ~ U[delta_min, delta_max]
    delta = delta_min + (delta_max - delta_min) * torch.rand(n, device=device)
    r = r_b + delta
    
    return dirs * r.unsqueeze(1)


@torch.no_grad()
def sample_global_outside_3d(gauge: FlexibleGauge, m: int, R_x1: float, R_x2: float, R_x3: float, device,
                             max_tries: int = 10, min_outside_frac: float = 0.02,
                             expanded_box_scale: float = 1.5,
                             far_outside_delta_min: float = 0.5,
                             far_outside_delta_max: float = 2.0) -> torch.Tensor:
    """
    Robust outside sampling with guaranteed minimum outside count in 3D.
    
    Strategy:
    1. Try rejection sampling in original box
    2. If insufficient, try expanded box  
    3. If still insufficient, supplement with far-outside samples
    """
    min_outside_batch = max(1, int(min_outside_frac * m))
    
    batch = []
    need = m
    tries = 0
    
    # Phase 1: Standard rejection sampling in original box
    while need > 0 and tries < max_tries:
        z = torch.empty(need * 4, 3, device=device)  # Use more candidates for 3D
        z[:, 0] = (2 * torch.rand(need * 4, device=device) - 1.0) * R_x1
        z[:, 1] = (2 * torch.rand(need * 4, device=device) - 1.0) * R_x2
        z[:, 2] = (2 * torch.rand(need * 4, device=device) - 1.0) * R_x3
        phi = gauge(z)
        keep = phi > 1.0
        if keep.any():
            chosen = z[keep]
            take = min(need, chosen.shape[0])
            batch.append(chosen[:take])
            need -= take
        tries += 1
    
    # Phase 2: If we don't have enough, try expanded box
    if len(batch) == 0 or torch.cat(batch, dim=0).size(0) < min_outside_batch:
        tries = 0
        while need > 0 and tries < max_tries:
            # Expanded box
            R_x1_exp = R_x1 * expanded_box_scale
            R_x2_exp = R_x2 * expanded_box_scale
            R_x3_exp = R_x3 * expanded_box_scale
            z = torch.empty(need * 4, 3, device=device)
            z[:, 0] = (2 * torch.rand(need * 4, device=device) - 1.0) * R_x1_exp
            z[:, 1] = (2 * torch.rand(need * 4, device=device) - 1.0) * R_x2_exp
            z[:, 2] = (2 * torch.rand(need * 4, device=device) - 1.0) * R_x3_exp
            phi = gauge(z)
            keep = phi > 1.0
            if keep.any():
                chosen = z[keep]
                take = min(need, chosen.shape[0])
                batch.append(chosen[:take])
                need -= take
            tries += 1
    
    # Phase 3: Supplement with far-outside samples if still insufficient
    current_count = torch.cat(batch, dim=0).size(0) if batch else 0
    if current_count < min_outside_batch:
        supplement_needed = min_outside_batch - current_count
        if need > 0:
            supplement_count = min(need, supplement_needed) 
        else:
            supplement_count = supplement_needed
            
        far_samples = sample_far_outside_3d(gauge, supplement_count, device,
                                           far_outside_delta_min, far_outside_delta_max)
        batch.append(far_samples)
        need -= supplement_count
    
    # Phase 4: If we still need more samples for the full batch size, use ring sampling
    if need > 0:
        filler = sample_ring_mixed_3d(gauge, need, delta_rel=0.5, delta_abs=0.3, rel_frac=0.5, device=device)
        batch.append(filler)
    
    result = torch.cat(batch, dim=0)
    return result[:m]  # Trim to exact size requested


# ========================
# 1D Sampling Functions
# ========================

@torch.no_grad()
def sample_ring_mixed_1d(gauge, m: int, delta_rel: float, delta_abs: float,
                         rel_frac: float, device) -> torch.Tensor:
    """
    Sample points in 1D ring around gauge boundary (VECTORIZED).
    
    For 1D, the 'ring' consists of two small intervals around
    the two boundary points (±r_boundary).
    
    Args:
        gauge: Gauge function φ
        m: Number of samples to generate
        delta_rel: Relative ring thickness
        delta_abs: Absolute ring thickness
        rel_frac: Fraction using relative vs absolute thickness
        device: Device to use
        
    Returns:
        Tensor of shape (m, 1) with samples in boundary ring
    """
    # Generate directions (+1 and -1) in equal proportions
    n_pos = m // 2
    n_neg = m - n_pos
    dirs = torch.cat([torch.ones(n_pos, 1, device=device),
                      -torch.ones(n_neg, 1, device=device)], dim=0)
    
    # Get boundary radius for each direction (vectorized)
    radii = boundary_radius(gauge, dirs).squeeze()  # Shape: (m,)
    
    # Split into relative and absolute samples
    n_rel = int(m * rel_frac)
    n_abs = m - n_rel
    
    # Vectorized relative ring sampling
    if n_rel > 0:
        delta_r = delta_rel * radii[:n_rel]  # Vectorized multiplication
        offsets = (torch.rand(n_rel, device=device) * 2 - 1) * delta_r  # Vectorized
        rel_samples = (radii[:n_rel] + offsets).unsqueeze(1) * dirs[:n_rel]  # Vectorized
    else:
        rel_samples = torch.empty(0, 1, device=device)
    
    # Vectorized absolute ring sampling  
    if n_abs > 0:
        offsets = (torch.rand(n_abs, device=device) * 2 - 1) * delta_abs  # Vectorized
        abs_samples = (radii[-n_abs:] + offsets).unsqueeze(1) * dirs[-n_abs:]  # Vectorized
    else:
        abs_samples = torch.empty(0, 1, device=device)
    
    # Combine samples
    result = torch.cat([rel_samples, abs_samples], dim=0)
    
    return result


@torch.no_grad()
def sample_global_outside_1d(gauge, m: int, train_box_s: float, device,
                             min_outside_frac: float = 0.2, expanded_box_scale: float = 1.5,
                             far_outside_delta_min: float = 0.5, far_outside_delta_max: float = 2.0,
                             max_tries: int = 50) -> torch.Tensor:
    """
    Efficient outside sampling for 1D using direct region sampling.
    
    In 1D, "outside" means |s| > r_boundary. Instead of rejection sampling
    (inefficient for 1D), we directly sample from the two outside regions.
    
    Args:
        gauge: Gauge function φ
        m: Number of samples
        train_box_s: Training box half-width
        device: Device
        (other params kept for compatibility but not used)
        
    Returns:
        Tensor of shape (m, 1) with samples outside gauge
    """
    # Get boundary radius (use zero direction since 1D is symmetric)
    zero_dir = torch.tensor([[1.0]], device=device)
    r_b = boundary_radius(gauge, zero_dir).item()
    
    # Split samples between left and right outside regions
    m_left = m // 2
    m_right = m - m_left
    
    batch = []
    
    # Sample left region: s ∈ [-train_box_s, -r_b]
    if train_box_s > r_b:
        s_left = -train_box_s + torch.rand(m_left, device=device) * (train_box_s - r_b)
        batch.append(s_left.unsqueeze(1))
    
    # Sample right region: s ∈ [r_b, train_box_s]
    if train_box_s > r_b:
        s_right = r_b + torch.rand(m_right, device=device) * (train_box_s - r_b)
        batch.append(s_right.unsqueeze(1))
    
    # If boundary is too large (r_b >= train_box_s), use far outside
    if not batch or len(torch.cat(batch, dim=0)) < m:
        needed = m - (len(torch.cat(batch, dim=0)) if batch else 0)
        far_samples = sample_far_outside_1d(gauge, needed, device,
                                           far_outside_delta_min, far_outside_delta_max)
        batch.append(far_samples)
    
    result = torch.cat(batch, dim=0) if batch else torch.empty(0, 1, device=device)
    return result[:m]


@torch.no_grad()
def sample_far_outside_1d(gauge, m: int, device, delta_min: float, delta_max: float) -> torch.Tensor:
    """
    Sample far outside gauge boundary in 1D.
    
    Samples at distance r_boundary + δ where δ ∈ [delta_min, delta_max].
    
    Args:
        gauge: Gauge function
        m: Number of samples
        device: Device
        delta_min: Minimum distance beyond boundary
        delta_max: Maximum distance beyond boundary
        
    Returns:
        Tensor of shape (m, 1)
    """
    # Sample directions (±1)
    dirs = torch.ones(m, 1, device=device)
    dirs[::2] *= -1  # Alternate signs
    
    # Get boundary radii
    radii = boundary_radius(gauge, dirs).squeeze()
    
    # Add random delta
    deltas = torch.rand(m, device=device) * (delta_max - delta_min) + delta_min
    far_radii = radii + deltas
    
    return (far_radii * dirs.squeeze()).unsqueeze(1)
