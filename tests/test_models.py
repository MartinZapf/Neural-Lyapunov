"""Tests for Lyapunov network models."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_lyapunov.models import SimpleLyapNet, LiftedLyapNet, lift_coords_sta, lift_coords_pidsmc


class TestSimpleLyapNet:
    """Test SimpleLyapNet properties."""

    def test_v_zero_at_origin_2d(self):
        """V(0) = 0 must hold."""
        net = SimpleLyapNet(width=32, depth=2, input_dim=2)
        z0 = torch.zeros(1, 2)
        V0 = net(z0)
        assert torch.abs(V0).item() < 1e-6

    def test_v_zero_at_origin_3d(self):
        """V(0) = 0 must hold for 3D input."""
        net = SimpleLyapNet(width=32, depth=2, input_dim=3)
        z0 = torch.zeros(1, 3)
        V0 = net(z0)
        assert torch.abs(V0).item() < 1e-6

    def test_positive_away_from_origin(self):
        """V(z) > 0 for z != 0 (due to eps_quad term)."""
        net = SimpleLyapNet(width=32, depth=2, input_dim=2, eps_quad=0.01)
        z = torch.randn(10, 2)
        z = z / z.norm(dim=1, keepdim=True)  # Unit vectors
        V = net(z)
        assert (V > 0).all()

    def test_gradient_computation(self):
        """Gradients should be computable."""
        net = SimpleLyapNet(width=32, depth=2, input_dim=2)
        z = torch.randn(10, 2, requires_grad=True)
        V = net(z)
        gV = torch.autograd.grad(V.sum(), z)[0]
        assert gV.shape == (10, 2)
        assert not torch.isnan(gV).any()


class TestLiftedLyapNet:
    """Test LiftedLyapNet properties."""

    def test_v_zero_at_origin_sta(self):
        """V(0) = 0 must hold for STA lift."""
        net = LiftedLyapNet(width=32, depth=2, input_dim=2, lift_type="sta")
        z0 = torch.zeros(1, 2)
        V0 = net(z0)
        assert torch.abs(V0).item() < 1e-6

    def test_v_zero_at_origin_pidsmc(self):
        """V(0) = 0 must hold for PID-SMC lift."""
        net = LiftedLyapNet(width=32, depth=2, input_dim=3, lift_type="pid_smc")
        z0 = torch.zeros(1, 3)
        V0 = net(z0)
        assert torch.abs(V0).item() < 1e-6

    def test_invalid_lift_type(self):
        """Invalid lift type should raise error."""
        with pytest.raises(ValueError):
            LiftedLyapNet(width=32, depth=2, input_dim=2, lift_type="invalid")

    def test_dimension_mismatch(self):
        """Mismatched dimensions should raise error."""
        with pytest.raises(ValueError):
            LiftedLyapNet(width=32, depth=2, input_dim=3, lift_type="sta")


class TestCoordinateTransforms:
    """Test coordinate lifting functions."""

    def test_sta_lift_origin(self):
        """Origin maps to origin."""
        z = torch.zeros(1, 2)
        xi = lift_coords_sta(z)
        assert torch.allclose(xi, torch.zeros(1, 2), atol=1e-6)

    def test_sta_lift_shape(self):
        """Output shape matches input."""
        z = torch.randn(10, 2)
        xi = lift_coords_sta(z)
        assert xi.shape == z.shape

    def test_pidsmc_lift_origin(self):
        """Origin maps to origin."""
        z = torch.zeros(1, 3)
        xi = lift_coords_pidsmc(z)
        assert torch.allclose(xi, torch.zeros(1, 3), atol=1e-6)

    def test_pidsmc_lift_shape(self):
        """Output shape matches input."""
        z = torch.randn(10, 3)
        xi = lift_coords_pidsmc(z)
        assert xi.shape == z.shape
