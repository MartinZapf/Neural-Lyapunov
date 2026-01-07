from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Any
import torch

Tensor = torch.Tensor


class BaseSMC:
    """
    Base interface for 'sliding-mode-style' controllers on 2D or 3D state z.

    Every controller must implement:
      - modes(z): -> (f_plus, f_minus)   (B,state_dim) each
        where '+' means the mode corresponding to sign(z[0])=+1 and '-' to sign(z[0])=-1.
      - name: a short string identifier (e.g., 'sta', 'cta', 'smc1', 'pid_smc').
      - state_dim: dimension of the state space (2 or 3).
      - params: controller parameters (dataclass with delta_bound attribute)

    We also provide a common Filippov 'worst-case' directional derivative helper.
    """

    name: str = "base"
    state_dim: int = 2  # Default to 2D for backward compatibility
    params: Any  # Controller parameters (subclass-specific dataclass)

    def modes(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (f_plus, f_minus) for Filippov dynamics.
        
        Args:
            z: States of shape [..., state_dim]
            
        Returns:
            f_plus: Dynamics in plus mode, shape [..., state_dim]
            f_minus: Dynamics in minus mode, shape [..., state_dim]
        """
        raise NotImplementedError

    def disturbance_channel(self) -> int:
        """
        Return which state index disturbance enters.

        DEFAULT behavior based on matched disturbance theory (Moreno & Osorio 2012):
        - For 1D systems (FOSMC): disturbance enters ṡ (index 0)
        - For 2D systems (STA, CTA): disturbance enters v̇ (index 1)
        - For 3D systems: disturbance enters last state (index 2)

        NOTE: Subclasses may override this if their plant dynamics differ.
        For example, PIDSMC overrides to return 1 because its plant has
        disturbance entering ẋ₂, not ẋ₃.

        Returns:
            int: State index where disturbance enters
        """
        if self.state_dim == 1:
            return 0  # ṡ = f_s(s) + Δ
        elif self.state_dim == 2:
            return 1  # v̇ = f_v(s,v) + Δ
        elif self.state_dim == 3:
            return 2  # ẋ₃ = f₃(x) + Δ
        else:
            raise ValueError(f"Unsupported dimension: {self.state_dim}")
    
    def get_delta_bound(self) -> float:
        """Get disturbance bound from params. Returns 0.0 if not set."""
        return getattr(self.params, 'delta_bound', 0.0)

    def _compute_dV_at_disturbance(
        self, gradV: Tensor, z: Tensor, delta: float, s_eps: float = 1e-3
    ) -> Tensor:
        """
        Compute dV/dt at a specific disturbance level.
        
        Args:
            gradV: Gradient of V wrt state, shape (B, state_dim)
            z: State, shape (B, state_dim)
            delta: Disturbance magnitude (worst-case sign is chosen)
            s_eps: Filippov band width
            
        Returns:
            dV/dt at this disturbance level, shape (B,)
        """
        s = z[:, 0]
        dist_channel = self.disturbance_channel()
        
        f_plus, f_minus = self.modes(z)
        
        # Worst-case disturbance contribution at this delta level
        disturbance_contribution = torch.abs(gradV[:, dist_channel]) * delta
        
        near = torch.abs(s) <= s_eps
        far = ~near
        
        dV = torch.empty_like(s)
        
        if far.any():
            sig = torch.sign(s[far]).clamp(-1, 1).unsqueeze(1)
            f_far = torch.where(sig > 0, f_plus[far], f_minus[far])
            dV[far] = torch.sum(gradV[far] * f_far, dim=1) + disturbance_contribution[far]
        
        if near.any():
            dVp = torch.sum(gradV[near] * f_plus[near], dim=1)
            dVm = torch.sum(gradV[near] * f_minus[near], dim=1)
            dV[near] = torch.maximum(dVp, dVm) + disturbance_contribution[near]
        
        return dV

    def worst_dV(self, gradV: Tensor, z: Tensor, s_eps: float = 1e-3) -> Tensor:
        """
        Compute worst-case dV/dt under:
        1. Filippov set-valued dynamics (modes +/-)
        2. Bounded matched disturbance |Δ| ≤ delta_bound
        
        Theoretical foundation (Moreno & Osorio 2012):
        - Disturbance enters control channel (matched disturbance)
        - Worst-case: dV/dt = (∇V · f) + |∇V[channel]| · δ₀
        - Absolute value ensures we pick the sign that maximizes dV/dt
        
        The Lyapunov condition is: dV/dt + α(z) ≤ 0
        where α(z) is a tunable weighted 1-norm (e.g., α_s|s| + α_v|v|).
        
        Note: α(z) are hyperparameters that aid training by preventing trivial
        solutions. They do NOT provide exponential convergence (α→0 as z→0).
        
        Args:
            gradV: Gradient of V wrt state, shape (B, state_dim)
            z: State, shape (B, state_dim)
            s_eps: Filippov band width
            
        Returns:
            Worst-case dV/dt, shape (B,)
        """
        delta_bound = self.get_delta_bound()
        return self._compute_dV_at_disturbance(gradV, z, delta_bound, s_eps)

    def worst_dV_multi(
        self, gradV: Tensor, z: Tensor, s_eps: float = 1e-3, 
        include_nominal: bool = True
    ) -> Tensor:
        """
        Compute worst-case dV/dt over multiple disturbance levels.
        
        This ensures the Lyapunov function is robust across the FULL disturbance
        range [0, delta_bound], not just at the extreme. Since disturbance enters
        linearly, checking endpoints (δ=0 and δ=δ_max) is sufficient by convexity.
        
        Rationale:
        - A Lyapunov function trained only for δ_max might have suboptimal
          behavior at δ=0 (nominal system)
        - Training on both endpoints ensures good margins everywhere
        - This is especially useful near the theoretical stability limit
        
        Args:
            gradV: Gradient of V wrt state, shape (B, state_dim)
            z: State, shape (B, state_dim)
            s_eps: Filippov band width
            include_nominal: If True, also check δ=0 (nominal system)
            
        Returns:
            Maximum dV/dt over all disturbance levels, shape (B,)
        """
        delta_bound = self.get_delta_bound()
        
        # Always compute at maximum disturbance
        dV_max = self._compute_dV_at_disturbance(gradV, z, delta_bound, s_eps)
        
        if include_nominal and delta_bound > 0:
            # Also compute at nominal (δ=0)
            dV_nominal = self._compute_dV_at_disturbance(gradV, z, 0.0, s_eps)
            # Take pointwise maximum
            return torch.maximum(dV_max, dV_nominal)
        
        return dV_max


# -----------------------
# Super-Twisting Algorithm
# -----------------------
@dataclass
class STAParams:
    k1: float = 1.2
    k2: float = 1.0
    delta_bound: float = 0.0  # Maximum disturbance magnitude


class STA(BaseSMC):
    """Super-Twisting dynamics on (s, v)."""
    name = "sta"
    state_dim = 2  # 2D system

    def __init__(self, k1: float = 1.2, k2: float = 1.0, delta_bound: float = 0.0):
        self.params = STAParams(k1=k1, k2=k2, delta_bound=delta_bound)

    def modes(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        s, v = z[:, 0], z[:, 1]
        rt = torch.sqrt(torch.abs(s) + 1e-8)

        # + mode corresponds to sign(s)=+1
        sdot_p = -self.params.k1 * rt + v
        vdot_p = -self.params.k2 * torch.ones_like(s)

        # - mode corresponds to sign(s)=-1
        sdot_m = +self.params.k1 * rt + v
        vdot_m = +self.params.k2 * torch.ones_like(s)

        f_plus = torch.stack([sdot_p, vdot_p], dim=1)
        f_minus = torch.stack([sdot_m, vdot_m], dim=1)
        return f_plus, f_minus


# -------------------------
# Continuous Twisting (CTA)
# -------------------------
@dataclass
class CTAParams:
    k1: float = 1.0
    k2: float = 1.0
    k3: float = 0.3  # small continuous stabilizer on v-dot
    delta_bound: float = 0.0  # Maximum disturbance magnitude


class CTA(BaseSMC):
    """
    A commonly used 'continuous twisting' variant in the same (s,v) coords:
        sdot = -k1 * sqrt(|s|) * sign(s) + v
        vdot = -k2 * sign(s) - k3 * s
    This adds a continuous linear term (-k3 s) to smooth vdot.
    """
    name = "cta"
    state_dim = 2  # 2D system

    def __init__(self, k1: float = 1.0, k2: float = 1.0, k3: float = 0.3, delta_bound: float = 0.0):
        self.params = CTAParams(k1=k1, k2=k2, k3=k3, delta_bound=delta_bound)

    def modes(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        s, v = z[:, 0], z[:, 1]
        rt = torch.sqrt(torch.abs(s) + 1e-8)
        # + mode
        sdot_p = -self.params.k1 * rt + v
        vdot_p = -self.params.k2 * torch.ones_like(s) - self.params.k3 * s
        # - mode
        sdot_m = +self.params.k1 * rt + v
        vdot_m = +self.params.k2 * torch.ones_like(s) - self.params.k3 * s

        f_plus = torch.stack([sdot_p, vdot_p], dim=1)
        f_minus = torch.stack([sdot_m, vdot_m], dim=1)
        return f_plus, f_minus


# ----------------------------------
# 1st-Order SMC (true 1D system)
# ----------------------------------
@dataclass
class FOSMCParams:
    k: float = 1.0  # Single gain for 1D system
    delta_bound: float = 0.0  # Maximum disturbance magnitude


class FOSMC(BaseSMC):
    """
    True First-Order Sliding Mode Control - 1D system.
    
    System: ṡ = -k·sign(s)
    
    State space: (s)  - single sliding variable
    Discontinuity: At s = 0
    """
    name = "fosmc"
    state_dim = 1  # TRUE 1D SYSTEM
    
    def __init__(self, k: float = 1.0, delta_bound: float = 0.0):
        self.params = FOSMCParams(k=k, delta_bound=delta_bound)
    
    def modes(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Return Filippov modes for 1D FOSMC.
        
        Args:
            z: States [s] of shape [..., 1]
            
        Returns:
            f_plus: ṡ for s > 0 (mode with sign(s)=+1), shape [..., 1]
            f_minus: ṡ for s < 0 (mode with sign(s)=-1), shape [..., 1]
        """
        # Plus mode: ṡ = -k (drives s toward zero from above)
        f_plus = -self.params.k * torch.ones_like(z)
        
        # Minus mode: ṡ = +k (drives s toward zero from below)  
        f_minus = +self.params.k * torch.ones_like(z)
        
        return f_plus, f_minus


# ----------------------------------
# PID-like SMC (3-state system)
# ----------------------------------
@dataclass
class PIDSMCParams:
    k1: float = 7.89   # From CDC'25 paper: k1 = 2.7 * L^(2/3) with L=5
    k2: float = 11.95  # From CDC'25 paper: k2 = 5.345 * L^(1/2) with L=5
    k3: float = 5.5    # From CDC'25 paper: k3 = 1.1 * L with L=5
    delta_bound: float = 0.0  # Maximum disturbance magnitude


class PIDSMC(BaseSMC):
    """
    Full 3-state PID-like Sliding Mode Controller.
    
    System:
        ẋ₁ = x₂
        ẋ₂ = -k₁⌊x₁⌉^(1/3) - k₂⌊x₂⌉^(1/2) + x₃
        ẋ₃ = -k₃⌊x₁⌉^0
    
    Where ⌊x⌉^α = |x|^α · sign(x) and ⌊x⌉^0 = sign(x).
    
    State space: (x₁, x₂, x₃)
    Discontinuity: At x₁ = 0
    """
    name = "pid_smc"
    state_dim = 3  # THIS IS A 3D SYSTEM
    
    def __init__(self, k1: float = 7.89, k2: float = 11.95, k3: float = 5.5, delta_bound: float = 0.0):
        self.params = PIDSMCParams(k1=k1, k2=k2, k3=k3, delta_bound=delta_bound)
    
    def modes(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Return Filippov modes for PID-SMC.
        
        Args:
            z: States [x₁, x₂, x₃] of shape [..., 3]
            
        Returns:
            f_plus: Dynamics for x₁ > 0, shape [..., 3]
            f_minus: Dynamics for x₁ < 0, shape [..., 3]
        """
        x1, x2, x3 = z[:, 0], z[:, 1], z[:, 2]
        
        # Compute fractional powers carefully
        abs_x1 = torch.abs(x1) + 1e-8
        abs_x2 = torch.abs(x2) + 1e-8
        
        x1_third = torch.pow(abs_x1, 1.0/3.0)  # |x₁|^(1/3)
        x2_half = torch.sqrt(abs_x2)           # |x₂|^(1/2)
        
        sign_x2 = torch.sign(x2).clamp(-1, 1)
        
        # Common dynamics (same for both modes)
        dx1_dt = x2  # ẋ₁ = x₂ always
        
        # Plus mode (x₁ > 0 → sign(x₁) = +1)
        dx2_dt_plus = -self.params.k1 * x1_third - self.params.k2 * x2_half * sign_x2 + x3
        dx3_dt_plus = -self.params.k3 * torch.ones_like(x1)
        
        # Minus mode (x₁ < 0 → sign(x₁) = -1)
        dx2_dt_minus = +self.params.k1 * x1_third - self.params.k2 * x2_half * sign_x2 + x3
        dx3_dt_minus = +self.params.k3 * torch.ones_like(x1)
        
        # Stack into vectors
        f_plus = torch.stack([dx1_dt, dx2_dt_plus, dx3_dt_plus], dim=1)
        f_minus = torch.stack([dx1_dt, dx2_dt_minus, dx3_dt_minus], dim=1)

        return f_plus, f_minus

    def disturbance_channel(self) -> int:
        """
        Return disturbance channel for PID-SMC.

        PID-SMC plant dynamics (from user's specification):
            ẋ₁ = x₂
            ẋ₂ = u + φ    ← matched disturbance φ enters HERE (index 1)
            φ̇ = Δ(t)

        where u = -k₁⌊x₁⌉^(1/3) - k₂⌊x₂⌉^(1/2) + x₃ is the control input.

        The matched disturbance φ enters the ẋ₂ equation, which corresponds
        to state index 1 (zero-indexed), NOT index 2.

        Returns:
            int: 1 (disturbance enters ẋ₂ channel)
        """
        return 1  # Disturbance enters ẋ₂, not ẋ₃!


# -----------------------
# Factory
# -----------------------
def get_controller(name: str, **kwargs) -> BaseSMC:
    n = name.lower().strip()
    if n == "sta":
        return STA(**kwargs)
    if n == "cta":
        return CTA(**kwargs)
    if n in ("smc1", "fosmc"):
        return FOSMC(**kwargs)
    if n in ("pid_smc", "pidsmc"):
        return PIDSMC(**kwargs)
    raise ValueError(f"Unknown controller '{name}'. Valid: sta, cta, smc1, pid_smc")
