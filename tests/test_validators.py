"""Tests for validation functions."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_lyapunov.models import SimpleLyapNet
from neural_lyapunov.gauges import OrientedEllipsoidGauge
from neural_lyapunov.controllers import FOSMC, STA
from neural_lyapunov.validators import (
    validate_global_outside_1d,
    validate_global_outside,
)


class TestValidation1D:
    """Test 1D validation function."""

    def test_returns_dict(self):
        """Validation should return a dict with expected keys."""
        V = SimpleLyapNet(width=32, depth=2, input_dim=1)
        G = OrientedEllipsoidGauge(input_dim=1, initial_radius=0.1)
        ctrl = FOSMC()

        result = validate_global_outside_1d(
            V, G, ctrl,
            alpha_s=0.01,
            train_box_s=2.0,
            N=100,
            val_min_outside_count=10,
            val_min_outside_frac=0.01,
        )

        assert isinstance(result, dict)
        assert "is_ok" in result
        assert "max" in result
        assert "outside_count" in result

    def test_coverage_check(self):
        """Large gauge should fail coverage check."""
        V = SimpleLyapNet(width=32, depth=2, input_dim=1)
        G = OrientedEllipsoidGauge(input_dim=1, initial_radius=10.0)  # Very large
        ctrl = FOSMC()

        result = validate_global_outside_1d(
            V, G, ctrl,
            alpha_s=0.01,
            train_box_s=2.0,
            N=100,
            val_min_outside_count=50,
            val_min_outside_frac=0.5,
        )

        # Should fail due to insufficient coverage
        assert result["is_ok"] == False
        assert result.get("reason") == "insufficient_coverage"


class TestValidation2D:
    """Test 2D validation function."""

    def test_returns_dict(self):
        """Validation should return a dict with expected keys."""
        V = SimpleLyapNet(width=32, depth=2, input_dim=2)
        G = OrientedEllipsoidGauge(input_dim=2, initial_radius=0.1)
        ctrl = STA()

        result = validate_global_outside(
            V, G, ctrl,
            alpha_s=0.01,
            alpha_v=0.01,
            train_box_s=2.0,
            train_box_v=2.0,
            N=50,
            val_min_outside_count=10,
            val_min_outside_frac=0.01,
        )

        assert isinstance(result, dict)
        assert "is_ok" in result
        assert "max" in result
        assert "outside_count" in result
        assert "outside_frac" in result

    def test_coverage_check(self):
        """Large gauge should fail coverage check."""
        V = SimpleLyapNet(width=32, depth=2, input_dim=2)
        G = OrientedEllipsoidGauge(input_dim=2, initial_radius=10.0)
        ctrl = STA()

        result = validate_global_outside(
            V, G, ctrl,
            alpha_s=0.01,
            alpha_v=0.01,
            train_box_s=2.0,
            train_box_v=2.0,
            N=50,
            val_min_outside_count=100,
            val_min_outside_frac=0.5,
        )

        assert result["is_ok"] == False
        assert result.get("reason") == "insufficient_coverage"
