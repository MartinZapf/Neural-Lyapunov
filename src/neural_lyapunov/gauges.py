from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OrientedEllipsoidGauge(nn.Module):
    """
    Oriented ellipsoid gauge φ(z) with learnable radii and rotation.
    
    For 2D: 2 radii + 1 rotation angle (3 parameters)
    For 3D: 3 radii + 3 rotation angles (6 parameters)
    
    Provides 96.2-99.4% parameter reduction compared to prior star-convex gauges
    (160-200 parameters) while maintaining excellent performance and natural
    convexity properties of ellipsoids.
    """
    def __init__(self, input_dim: int = 2, initial_radius: float = 2.0):
        super().__init__()
        self.input_dim = input_dim
        
        # Learnable radii (log scale for positivity)
        if input_dim == 1:
            # 1D: single radius (just a scaling factor)
            self.log_radii = nn.Parameter(torch.log(torch.tensor([initial_radius])))
            # No rotation in 1D
            self.angles = None
        elif input_dim == 2:
            self.log_radii = nn.Parameter(torch.log(torch.tensor([initial_radius, initial_radius])))
            # Single rotation angle for 2D
            self.angles = nn.Parameter(torch.zeros(1))
        elif input_dim == 3:
            self.log_radii = nn.Parameter(torch.log(torch.tensor([initial_radius, initial_radius, initial_radius])))
            # Three Euler angles for 3D rotation
            self.angles = nn.Parameter(torch.zeros(3))
        else:
            raise ValueError(f"Unsupported input_dim: {input_dim}. Only 1D, 2D and 3D supported.")
    
    def _get_rotation_matrix(self) -> torch.Tensor:
        """Generate rotation matrix from angles."""
        if self.input_dim == 1:
            # 1D: identity (no rotation) - create on same device as parameters
            return torch.tensor([[1.0]], device=self.log_radii.device, dtype=self.log_radii.dtype)
        elif self.input_dim == 2:
            # 2D rotation matrix
            assert self.angles is not None, "Angles should not be None for 2D"
            angle = self.angles[0]
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            return torch.stack([
                torch.stack([cos_a, -sin_a]),
                torch.stack([sin_a, cos_a])
            ])
        else:
            # 3D rotation matrix using ZYX Euler angles
            assert self.angles is not None, "Angles should not be None for 3D"
            alpha, beta, gamma = self.angles
            
            # Individual rotation matrices
            cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
            cos_b, sin_b = torch.cos(beta), torch.sin(beta)
            cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)
            
            # Combined rotation matrix (ZYX order)
            R = torch.stack([
                torch.stack([cos_a*cos_b, cos_a*sin_b*sin_g - sin_a*cos_g, cos_a*sin_b*cos_g + sin_a*sin_g]),
                torch.stack([sin_a*cos_b, sin_a*sin_b*sin_g + cos_a*cos_g, sin_a*sin_b*cos_g - cos_a*sin_g]),
                torch.stack([-sin_b, cos_b*sin_g, cos_b*cos_g])
            ])
            return R

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Input states of shape [..., input_dim]
            
        Returns:
            phi: Gauge values of shape [...]
        """
        assert z.shape[-1] == self.input_dim, \
            f"Expected input_dim={self.input_dim}, got {z.shape[-1]}"
        
        # Get radii (ensure positivity)
        radii = torch.exp(self.log_radii).clamp_min(1e-6)
        
        # Get rotation matrix
        R = self._get_rotation_matrix()
        
        # Apply inverse rotation to transform to axis-aligned ellipsoid space
        # Rotation: Applies R^T z in column-vector notation.
        # Implementation: z @ R (row-vector convention where z is shape [..., D])
        # These are equivalent: row-vector z @ R = (R^T @ z^T)^T for column-vector z
        # Since R is orthogonal, R^T = R^{-1}
        z_rot = torch.matmul(z, R)  # Broadcasting handles batch dimensions
        
        # Compute ellipsoid distance in rotated space
        # phi(z) = sqrt(sum((z_rot / radii)^2))
        normalized = z_rot / radii
        phi = torch.sqrt(torch.sum(normalized * normalized, dim=-1).clamp_min(1e-6))
        
        return phi
    
    def get_parameter_count(self) -> int:
        """Return number of learnable parameters."""
        if self.input_dim == 1:
            return 1  # 1 radius
        elif self.input_dim == 2:
            return 3  # 2 radii + 1 angle
        else:
            return 6  # 3 radii + 3 angles


# Legacy alias for backward compatibility
FlexibleGauge = OrientedEllipsoidGauge
