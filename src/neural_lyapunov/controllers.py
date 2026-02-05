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
    k1: float = 2.7    # Gain for |x₁|^(1/3)·sign(x₁)
    k2: float = 5.345  # Gain for |x₂|^(1/2)·sign(x₂)
    k3: float = 1.1    # Gain for sign(x₁) in twisting term
    k4: float = 1.1    # Gain for sign(x₂) in twisting term
    delta_bound: float = 0.0  # Maximum disturbance derivative magnitude


class CTA(BaseSMC):
    """
    Continuous Twisting Algorithm (Torres-González et al., 2017).
    
    True CTA is a 3-state system with Twisting structure in the integral:
        ẋ₁ = x₂
        ẋ₂ = -k₁⌊x₁⌉^(1/3) - k₂⌊x₂⌉^(1/2) + x₃
        ẋ₃ = -k₃⌊x₁⌉^0 - k₄⌊x₂⌉^0 + Δ̇(t)
    
    Where ⌊x⌉^α = |x|^α · sign(x) and ⌊x⌉^0 = sign(x).
    
    State space: (x₁, x₂, x₃)
    Discontinuities: At x₁ = 0 AND x₂ = 0 (two switching surfaces)
    
    This requires 4 Filippov modes for all sign combinations.
    """
    name = "cta"
    state_dim = 3  # THIS IS A 3D SYSTEM

    def __init__(self, k1: float = 2.7, k2: float = 5.345, k3: float = 1.1, k4: float = 1.1, delta_bound: float = 0.0):
        self.params = CTAParams(k1=k1, k2=k2, k3=k3, k4=k4, delta_bound=delta_bound)

    def modes(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Return Filippov modes for CTA along the x₁=0 switching surface.
        
        Note: CTA has discontinuities at BOTH x₁=0 and x₂=0. This method
        returns modes varying with sign(x₁). The x₂ discontinuity is handled
        via the overridden worst_dV method.
        
        Args:
            z: States [x₁, x₂, x₃] of shape [..., 3]
            
        Returns:
            f_plus: Dynamics for x₁ > 0, shape [..., 3]
            f_minus: Dynamics for x₁ < 0, shape [..., 3]
        """
        x1, x2, x3 = z[:, 0], z[:, 1], z[:, 2]
        
        # Compute fractional powers
        abs_x1 = torch.abs(x1) + 1e-8
        abs_x2 = torch.abs(x2) + 1e-8
        
        x1_third = torch.pow(abs_x1, 1.0/3.0)  # |x₁|^(1/3)
        x2_half = torch.sqrt(abs_x2)           # |x₂|^(1/2)
        
        sign_x2 = torch.sign(x2).clamp(-1, 1)
        
        # Common dynamics
        dx1_dt = x2  # ẋ₁ = x₂ always
        
        # Plus mode (x₁ > 0 → sign(x₁) = +1)
        # ẋ₂ = -k₁|x₁|^(1/3) - k₂|x₂|^(1/2)·sign(x₂) + x₃
        dx2_dt_plus = -self.params.k1 * x1_third - self.params.k2 * x2_half * sign_x2 + x3
        # ẋ₃ = -k₃ - k₄·sign(x₂)  (using sign(x₂) value)
        dx3_dt_plus = -self.params.k3 * torch.ones_like(x1) - self.params.k4 * sign_x2
        
        # Minus mode (x₁ < 0 → sign(x₁) = -1)
        dx2_dt_minus = +self.params.k1 * x1_third - self.params.k2 * x2_half * sign_x2 + x3
        dx3_dt_minus = +self.params.k3 * torch.ones_like(x1) - self.params.k4 * sign_x2
        
        f_plus = torch.stack([dx1_dt, dx2_dt_plus, dx3_dt_plus], dim=1)
        f_minus = torch.stack([dx1_dt, dx2_dt_minus, dx3_dt_minus], dim=1)

        return f_plus, f_minus
    
    def modes_all(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return all 4 Filippov modes for CTA.
        
        CTA has discontinuities at x₁=0 and x₂=0, requiring 4 modes:
            (sign(x₁)=+1, sign(x₂)=+1) -> f_pp
            (sign(x₁)=+1, sign(x₂)=-1) -> f_pm
            (sign(x₁)=-1, sign(x₂)=+1) -> f_mp
            (sign(x₁)=-1, sign(x₂)=-1) -> f_mm
        """
        x1, x2, x3 = z[:, 0], z[:, 1], z[:, 2]
        
        abs_x1 = torch.abs(x1) + 1e-8
        abs_x2 = torch.abs(x2) + 1e-8
        
        x1_third = torch.pow(abs_x1, 1.0/3.0)
        x2_half = torch.sqrt(abs_x2)
        
        dx1_dt = x2
        
        # Mode (++): sign(x₁)=+1, sign(x₂)=+1
        dx2_pp = -self.params.k1 * x1_third - self.params.k2 * x2_half + x3
        dx3_pp = -self.params.k3 - self.params.k4 * torch.ones_like(x1)
        
        # Mode (+-): sign(x₁)=+1, sign(x₂)=-1
        dx2_pm = -self.params.k1 * x1_third + self.params.k2 * x2_half + x3
        dx3_pm = -self.params.k3 + self.params.k4 * torch.ones_like(x1)
        
        # Mode (-+): sign(x₁)=-1, sign(x₂)=+1
        dx2_mp = +self.params.k1 * x1_third - self.params.k2 * x2_half + x3
        dx3_mp = +self.params.k3 - self.params.k4 * torch.ones_like(x1)
        
        # Mode (--): sign(x₁)=-1, sign(x₂)=-1
        dx2_mm = +self.params.k1 * x1_third + self.params.k2 * x2_half + x3
        dx3_mm = +self.params.k3 + self.params.k4 * torch.ones_like(x1)
        
        f_pp = torch.stack([dx1_dt, dx2_pp, dx3_pp], dim=1)
        f_pm = torch.stack([dx1_dt, dx2_pm, dx3_pm], dim=1)
        f_mp = torch.stack([dx1_dt, dx2_mp, dx3_mp], dim=1)
        f_mm = torch.stack([dx1_dt, dx2_mm, dx3_mm], dim=1)
        
        return f_pp, f_pm, f_mp, f_mm
    
    def disturbance_channel(self) -> int:
        """Return disturbance channel - disturbance derivative enters ẋ₃."""
        return 2
    
    def _compute_dV_at_disturbance(
        self, gradV: Tensor, z: Tensor, delta: float, s_eps: float = 1e-3
    ) -> Tensor:
        """
        Compute dV/dt for CTA considering BOTH discontinuity surfaces.
        
        CTA has discontinuities at x₁=0 and x₂=0, so we need to consider
        the Filippov set-valued dynamics at both surfaces.
        """
        x1, x2 = z[:, 0], z[:, 1]
        dist_channel = self.disturbance_channel()
        
        # Get all 4 modes
        f_pp, f_pm, f_mp, f_mm = self.modes_all(z)
        
        # Worst-case disturbance contribution
        disturbance_contrib = torch.abs(gradV[:, dist_channel]) * delta
        
        # Determine proximity to each switching surface
        near_x1 = torch.abs(x1) <= s_eps
        near_x2 = torch.abs(x2) <= s_eps
        
        # 4 regions based on proximity to switching surfaces
        far_both = ~near_x1 & ~near_x2         # Far from both surfaces
        near_x1_only = near_x1 & ~near_x2      # Near x₁=0 only
        near_x2_only = ~near_x1 & near_x2      # Near x₂=0 only
        near_both = near_x1 & near_x2          # Near both surfaces
        
        dV = torch.empty_like(x1)
        
        # Region 1: Far from both surfaces - use actual signs
        if far_both.any():
            sig1 = torch.sign(x1[far_both]).clamp(-1, 1)
            sig2 = torch.sign(x2[far_both]).clamp(-1, 1)
            
            # Select the correct mode based on signs
            f_far = torch.where(
                (sig1 > 0).unsqueeze(1),
                torch.where((sig2 > 0).unsqueeze(1), f_pp[far_both], f_pm[far_both]),
                torch.where((sig2 > 0).unsqueeze(1), f_mp[far_both], f_mm[far_both])
            )
            dV[far_both] = torch.sum(gradV[far_both] * f_far, dim=1) + disturbance_contrib[far_both]
        
        # Region 2: Near x₁=0 only - max over sign(x₁)
        if near_x1_only.any():
            sig2 = torch.sign(x2[near_x1_only]).clamp(-1, 1)
            # Select based on sign(x₂), max over sign(x₁)
            f_p = torch.where((sig2 > 0).unsqueeze(1), f_pp[near_x1_only], f_pm[near_x1_only])
            f_m = torch.where((sig2 > 0).unsqueeze(1), f_mp[near_x1_only], f_mm[near_x1_only])
            dVp = torch.sum(gradV[near_x1_only] * f_p, dim=1)
            dVm = torch.sum(gradV[near_x1_only] * f_m, dim=1)
            dV[near_x1_only] = torch.maximum(dVp, dVm) + disturbance_contrib[near_x1_only]
        
        # Region 3: Near x₂=0 only - max over sign(x₂)
        if near_x2_only.any():
            sig1 = torch.sign(x1[near_x2_only]).clamp(-1, 1)
            # Select based on sign(x₁), max over sign(x₂)
            f_p = torch.where((sig1 > 0).unsqueeze(1), f_pp[near_x2_only], f_mp[near_x2_only])
            f_m = torch.where((sig1 > 0).unsqueeze(1), f_pm[near_x2_only], f_mm[near_x2_only])
            dVp = torch.sum(gradV[near_x2_only] * f_p, dim=1)
            dVm = torch.sum(gradV[near_x2_only] * f_m, dim=1)
            dV[near_x2_only] = torch.maximum(dVp, dVm) + disturbance_contrib[near_x2_only]
        
        # Region 4: Near both surfaces - max over all 4 modes
        if near_both.any():
            dVpp = torch.sum(gradV[near_both] * f_pp[near_both], dim=1)
            dVpm = torch.sum(gradV[near_both] * f_pm[near_both], dim=1)
            dVmp = torch.sum(gradV[near_both] * f_mp[near_both], dim=1)
            dVmm = torch.sum(gradV[near_both] * f_mm[near_both], dim=1)
            dV[near_both] = torch.maximum(
                torch.maximum(dVpp, dVpm),
                torch.maximum(dVmp, dVmm)
            ) + disturbance_contrib[near_both]
        
        return dV


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
