from __future__ import annotations
import torch
import torch.nn as nn


def sgnpow(x: torch.Tensor, p: float, eps: float = 1e-8) -> torch.Tensor:
    """Signed power: sign(x) * |x|^p with numerical stability."""
    return torch.sign(x) * (torch.abs(x) + eps).pow(p)


def lift_coords_sta(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Transform STA physical coordinates (s, v) to lifted/homogeneous coordinates (ξ₁, ξ₂).
    
    ξ₁ = |s|^(1/2) · sign(s)
    ξ₂ = v
    
    This is the Moreno-Osorio coordinate transformation that exposes the homogeneous
    structure of the Super-Twisting Algorithm.
    
    Key properties:
    - ξ₁² = |s|
    - sign(ξ₁) = sign(s)
    - Inverse: s = ξ₁ · |ξ₁| = sign(ξ₁) · ξ₁²
    
    Reference: Moreno & Osorio (2012) "Strict Lyapunov Functions for the Super-Twisting Algorithm"
    """
    s, v = z[..., 0], z[..., 1]
    xi1 = sgnpow(s, 0.5, eps)  # |s|^(1/2) * sign(s)
    xi2 = v
    return torch.stack([xi1, xi2], dim=-1)


def unlift_coords_sta(xi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Transform lifted coordinates (ξ₁, ξ₂) back to physical coordinates (s, v).

    s = ξ₁ · |ξ₁| = sign(ξ₁) · ξ₁²
    v = ξ₂
    """
    xi1, xi2 = xi[..., 0], xi[..., 1]
    s = sgnpow(xi1, 2.0, eps)  # ξ₁ * |ξ₁|
    v = xi2
    return torch.stack([s, v], dim=-1)


def lift_coords_pidsmc(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Transform PID-SMC physical coordinates (x1, x2, x3) to lifted coordinates (ξ₁, ξ₂, ξ₃).

    PID-SMC is homogeneous of degree -1 with dilation weights (3, 2, 1).
    The lifted coordinates normalize all weights to 1:

        ξ₁ = sign(x1) · |x1|^(1/3)
        ξ₂ = sign(x2) · |x2|^(1/2)
        ξ₃ = x3

    Key properties:
    - In lifted coords, Lyapunov function simplifies to polynomial form
    - ξ₁³ · sign(ξ₁) = x1, ξ₂² · sign(ξ₂) = x2
    - Inverse transformation recovers original coordinates

    Reference: Derived from homogeneity analysis of PID-SMC dynamics
    """
    x1, x2, x3 = z[..., 0], z[..., 1], z[..., 2]
    xi1 = sgnpow(x1, 1.0/3.0, eps)  # sign(x1) * |x1|^(1/3)
    xi2 = sgnpow(x2, 0.5, eps)       # sign(x2) * |x2|^(1/2)
    xi3 = x3
    return torch.stack([xi1, xi2, xi3], dim=-1)


def unlift_coords_pidsmc(xi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Transform lifted coordinates (ξ₁, ξ₂, ξ₃) back to physical coordinates (x1, x2, x3).

    x1 = sign(ξ₁) · |ξ₁|³
    x2 = sign(ξ₂) · |ξ₂|²
    x3 = ξ₃
    """
    xi1, xi2, xi3 = xi[..., 0], xi[..., 1], xi[..., 2]
    x1 = sgnpow(xi1, 3.0, eps)  # sign(ξ₁) * |ξ₁|³
    x2 = sgnpow(xi2, 2.0, eps)  # sign(ξ₂) * |ξ₂|²
    x3 = xi3
    return torch.stack([x1, x2, x3], dim=-1)


class SimpleLyapNet(nn.Module):
    """
    Lightweight MLP for V(z) with V(0)=0 enforced by feature centering.
    Now supports both 2D and 3D input.

    V(z) = ||h(z) - h(0)||^2 + eps_quad ||z||^2 + alpha_bar ||z||
    
    Note: h(0) is computed on-the-fly to ensure V(0)=0 always, even after parameter updates.
    """
    def __init__(self, width: int = 128, depth: int = 3, input_dim: int = 2,
                 eps_quad: float = 1e-3, alpha_bar: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        for i in range(depth):
            in_features = input_dim if i == 0 else width
            layers += [nn.Linear(in_features, width), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, 32)
        self.eps_quad = eps_quad
        self.alpha_bar = alpha_bar

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Input states of shape [..., input_dim]
            
        Returns:
            V: Lyapunov values of shape [...]
        """
        assert z.shape[-1] == self.input_dim, \
            f"Expected input_dim={self.input_dim}, got {z.shape[-1]}"
        
        h = self.head(self.backbone(z))
        
        # Compute center from current weights on-the-fly
        z0 = torch.zeros(1, z.size(-1), device=z.device, dtype=z.dtype)
        h0 = self.head(self.backbone(z0))
        
        V_res = torch.sum((h - h0) ** 2, dim=1)
        return V_res + self.eps_quad * torch.sum(z * z, dim=1) + self.alpha_bar * torch.linalg.norm(z, dim=1)


class LiftedLyapNet(nn.Module):
    """
    Lyapunov network with coordinate transformation for homogeneous systems.

    Supported systems:
    - STA/CTA (2D): Moreno-Osorio transform (ξ₁, ξ₂) = (|s|^(1/2)·sign(s), v)
    - PID-SMC (3D): Homogeneous transform (ξ₁, ξ₂, ξ₃) = (|x1|^(1/3)·sign(x1), |x2|^(1/2)·sign(x2), x3)

    This enables learning Lyapunov functions with the correct structure for
    homogeneous systems. In lifted coordinates, Lyapunov functions often
    simplify to polynomial form.

    Key benefits:
    1. Level sets have natural elliptical shape in lifted space
    2. Disturbance rejection ratio stays bounded
    3. Aligns with analytical Lyapunov function structure
    4. Simplifies symbolic regression (polynomial operators sufficient)

    The gradient ∇V w.r.t. physical coordinates is computed automatically via
    autograd through the coordinate transformation.

    V(z) = V_base(lift(z))
    """

    def __init__(self, width: int = 128, depth: int = 3, input_dim: int = 2,
                 eps_quad: float = 1e-3, alpha_bar: float = 1e-3,
                 lift_type: str = "sta"):
        """
        Args:
            width: Hidden layer width
            depth: Number of hidden layers
            input_dim: Input dimension (2 for STA/CTA, 3 for PID-SMC)
            eps_quad: Quadratic regularization coefficient (applied in lifted space)
            alpha_bar: Linear regularization coefficient (applied in lifted space)
            lift_type: Type of coordinate lifting ("sta", "cta", "pid_smc", "pidsmc")
        """
        super().__init__()
        self.input_dim = input_dim
        self.lift_type = lift_type
        self.eps_quad = eps_quad
        self.alpha_bar = alpha_bar

        # Validate lift type and dimension
        if lift_type not in ("sta", "cta", "pid_smc", "pidsmc"):
            raise ValueError(f"Unsupported lift_type: {lift_type}. Supported: 'sta', 'cta', 'pid_smc'")
        if lift_type in ("sta", "cta") and input_dim != 2:
            raise ValueError(f"LiftedLyapNet with lift_type='{lift_type}' requires input_dim=2, got {input_dim}")
        if lift_type in ("pid_smc", "pidsmc") and input_dim != 3:
            raise ValueError(f"LiftedLyapNet with lift_type='pid_smc' requires input_dim=3, got {input_dim}")
        
        # Build backbone MLP (operates in lifted space)
        layers = []
        for i in range(depth):
            in_features = input_dim if i == 0 else width
            layers += [nn.Linear(in_features, width), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, 32)
    
    def _lift(self, z: torch.Tensor) -> torch.Tensor:
        """Apply coordinate transformation."""
        if self.lift_type in ("sta", "cta"):
            return lift_coords_sta(z)
        elif self.lift_type in ("pid_smc", "pidsmc"):
            return lift_coords_pidsmc(z)
        else:
            return z  # Identity for unsupported types
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute V(z) by first transforming to lifted coordinates.
        
        Args:
            z: Physical coordinates of shape [..., input_dim]
            
        Returns:
            V: Lyapunov values of shape [...]
        """
        assert z.shape[-1] == self.input_dim, \
            f"Expected input_dim={self.input_dim}, got {z.shape[-1]}"
        
        # Transform to lifted coordinates
        xi = self._lift(z)
        
        # Forward through network (in lifted space)
        h = self.head(self.backbone(xi))
        
        # Compute center from current weights on-the-fly
        xi0 = torch.zeros(1, xi.size(-1), device=z.device, dtype=z.dtype)
        h0 = self.head(self.backbone(xi0))
        
        # V(0) = 0 is enforced: when z=0, xi=0, so h-h0=0
        V_res = torch.sum((h - h0) ** 2, dim=1)
        
        # Quadratic and linear terms in LIFTED space (key for homogeneity!)
        # This gives V ~ ||ξ||² structure which is correct for STA
        return V_res + self.eps_quad * torch.sum(xi * xi, dim=1) + self.alpha_bar * torch.linalg.norm(xi, dim=1)

