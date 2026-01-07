"""Tests for controller implementations."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_lyapunov.controllers import STA, CTA, FOSMC, PIDSMC, get_controller


class TestControllerInstantiation:
    """Test that all controllers can be created."""

    def test_sta_creation(self):
        ctrl = STA(k1=1.5, k2=1.0)
        assert ctrl.name == "sta"
        assert ctrl.state_dim == 2

    def test_cta_creation(self):
        ctrl = CTA(k1=1.0, k2=1.0, k3=0.3)
        assert ctrl.name == "cta"
        assert ctrl.state_dim == 2

    def test_fosmc_creation(self):
        ctrl = FOSMC(k=1.0)
        assert ctrl.name == "fosmc"
        assert ctrl.state_dim == 1

    def test_pidsmc_creation(self):
        ctrl = PIDSMC(k1=2.7, k2=5.345, k3=1.1)
        assert ctrl.name == "pid_smc"
        assert ctrl.state_dim == 3


class TestControllerModes:
    """Test Filippov mode computations."""

    def test_sta_modes_shape(self):
        ctrl = STA()
        z = torch.randn(10, 2)
        f_plus, f_minus = ctrl.modes(z)
        assert f_plus.shape == (10, 2)
        assert f_minus.shape == (10, 2)

    def test_fosmc_modes_shape(self):
        ctrl = FOSMC()
        z = torch.randn(10, 1)
        f_plus, f_minus = ctrl.modes(z)
        assert f_plus.shape == (10, 1)
        assert f_minus.shape == (10, 1)

    def test_pidsmc_modes_shape(self):
        ctrl = PIDSMC()
        z = torch.randn(10, 3)
        f_plus, f_minus = ctrl.modes(z)
        assert f_plus.shape == (10, 3)
        assert f_minus.shape == (10, 3)


class TestWorstDV:
    """Test worst-case dV/dt computation."""

    def test_worst_dv_sta(self):
        ctrl = STA()
        z = torch.randn(10, 2)
        gradV = torch.randn(10, 2)
        dV = ctrl.worst_dV(gradV, z)
        assert dV.shape == (10,)

    def test_worst_dv_fosmc(self):
        ctrl = FOSMC()
        z = torch.randn(10, 1)
        gradV = torch.randn(10, 1)
        dV = ctrl.worst_dV(gradV, z)
        assert dV.shape == (10,)


class TestFactory:
    """Test get_controller factory."""

    def test_factory_sta(self):
        ctrl = get_controller("sta", k1=1.5, k2=1.0)
        assert isinstance(ctrl, STA)

    def test_factory_cta(self):
        ctrl = get_controller("cta")
        assert isinstance(ctrl, CTA)

    def test_factory_fosmc(self):
        ctrl = get_controller("fosmc")
        assert isinstance(ctrl, FOSMC)

    def test_factory_pidsmc(self):
        ctrl = get_controller("pid_smc")
        assert isinstance(ctrl, PIDSMC)
