"""Tests for gauge functions."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_lyapunov.gauges import OrientedEllipsoidGauge


class TestOrientedEllipsoidGauge:
    """Test OrientedEllipsoidGauge properties."""

    def test_homogeneity_1d(self):
        """phi(lambda * z) = lambda * phi(z) for lambda > 0."""
        gauge = OrientedEllipsoidGauge(input_dim=1, initial_radius=1.0)
        z = torch.randn(10, 1)
        lam = 2.0
        phi_z = gauge(z)
        phi_lam_z = gauge(lam * z)
        assert torch.allclose(phi_lam_z, lam * phi_z, rtol=1e-5)

    def test_homogeneity_2d(self):
        """phi(lambda * z) = lambda * phi(z) for lambda > 0."""
        gauge = OrientedEllipsoidGauge(input_dim=2, initial_radius=1.0)
        z = torch.randn(10, 2)
        lam = 2.0
        phi_z = gauge(z)
        phi_lam_z = gauge(lam * z)
        assert torch.allclose(phi_lam_z, lam * phi_z, rtol=1e-5)

    def test_homogeneity_3d(self):
        """phi(lambda * z) = lambda * phi(z) for lambda > 0."""
        gauge = OrientedEllipsoidGauge(input_dim=3, initial_radius=1.0)
        z = torch.randn(10, 3)
        lam = 2.0
        phi_z = gauge(z)
        phi_lam_z = gauge(lam * z)
        assert torch.allclose(phi_lam_z, lam * phi_z, rtol=1e-5)

    def test_positive_definite(self):
        """phi(z) > 0 for z != 0."""
        gauge = OrientedEllipsoidGauge(input_dim=2, initial_radius=1.0)
        z = torch.randn(10, 2)
        z = z / z.norm(dim=1, keepdim=True)  # Unit vectors
        phi = gauge(z)
        assert (phi > 0).all()

    def test_zero_at_origin(self):
        """phi(0) is small (limited by numerical stability clamp)."""
        gauge = OrientedEllipsoidGauge(input_dim=2, initial_radius=1.0)
        z0 = torch.zeros(1, 2)
        phi0 = gauge(z0)
        # sqrt(clamp_min(0, 1e-6)) = 1e-3, so allow slightly larger
        assert torch.abs(phi0).item() < 2e-3

    def test_parameter_count_1d(self):
        """1D gauge has 1 parameter."""
        gauge = OrientedEllipsoidGauge(input_dim=1)
        assert gauge.get_parameter_count() == 1

    def test_parameter_count_2d(self):
        """2D gauge has 3 parameters (2 radii + 1 angle)."""
        gauge = OrientedEllipsoidGauge(input_dim=2)
        assert gauge.get_parameter_count() == 3

    def test_parameter_count_3d(self):
        """3D gauge has 6 parameters (3 radii + 3 angles)."""
        gauge = OrientedEllipsoidGauge(input_dim=3)
        assert gauge.get_parameter_count() == 6

    def test_unsupported_dimension(self):
        """Unsupported dimensions should raise error."""
        with pytest.raises(ValueError):
            OrientedEllipsoidGauge(input_dim=4)
