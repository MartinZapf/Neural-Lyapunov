from __future__ import annotations
import argparse, os, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Flexible imports to handle both relative and absolute imports
try:
    from ..controllers import get_controller
    from ..models import SimpleLyapNet, LiftedLyapNet
    from ..gauges import FlexibleGauge
except ImportError:
    # Fall back to absolute imports when relative imports fail
    import sys
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from controllers import get_controller
    from models import SimpleLyapNet, LiftedLyapNet
    from gauges import FlexibleGauge


def load_checkpoint(path: str, device: torch.device):
    """
    Supports checkpoint formats:
      - {"best": {...}, "cfg": cfg} (best model format)
      - {"polished": {...}, "cfg": cfg} (polish stage format)  
      - {"V":..., "G":..., "cfg": cfg} (direct format)
    Returns (V_state, G_state, cfg, meta) where meta contains controller info.
    """
    ckpt = torch.load(path, map_location=device)
    if "best" in ckpt:
        payload = ckpt["best"]
        cfg = ckpt.get("cfg", {})
        V_state = payload["V"]; G_state = payload["G"]
        meta = payload.get("meta", {})
    elif "polished" in ckpt:
        payload = ckpt["polished"]
        cfg = ckpt.get("cfg", {})
        V_state = payload["V"]; G_state = payload["G"]
        meta = payload.get("meta", {})
        # If meta is empty, construct it from config like the direct format case
        if not meta:
            meta = {"controller": cfg.get("controller", {}).get("name", "sta"),
                    "controller_params": cfg.get("controller", {}).get("params", {})}
        print(f"[VIZ] Loading polished checkpoint from iteration {payload.get('iter', 'unknown')}")
    elif "V" in ckpt and "G" in ckpt:
        cfg = ckpt.get("cfg", {})
        V_state = ckpt["V"]; G_state = ckpt["G"]
        meta = {"controller": cfg.get("controller", {}).get("name", "sta"),
                "controller_params": cfg.get("controller", {}).get("params", {})}
    else:
        raise ValueError("Unrecognized checkpoint format. Expected 'best', 'polished', or direct 'V'/'G' keys.")
    return V_state, G_state, cfg, meta


def build_models(V_state, G_state, cfg, device):
    model_cfg = cfg.get("model", {"width": 128, "depth": 3, "eps_quad": 1e-3, "alpha_bar": 1e-3})
    gauge_cfg = cfg.get("gauge", {"dim": 2, "n_planes": 48, "n_cones": 24, "eps_ball": 5e-3})
    
    # Detect dimension from controller configuration
    ctrl_name = cfg.get("controller", {}).get("name", "sta")
    if ctrl_name == "pid_smc":
        input_dim = 3
    elif ctrl_name == "fosmc":
        input_dim = 1
    else:
        input_dim = 2
    
    # Add input_dim to model configs
    model_cfg = model_cfg.copy()
    model_cfg["input_dim"] = input_dim
    
    # Handle gauge params for new oriented ellipsoid gauge
    gauge_params = {"input_dim": input_dim}  # type: ignore
    if "initial_radius" in gauge_cfg:
        gauge_params["initial_radius"] = float(gauge_cfg["initial_radius"])  # type: ignore
    else:
        # Default to 2.0 for backward compatibility
        gauge_params["initial_radius"] = 2.0  # type: ignore
    
    # Note: Legacy star-convex gauge parameters (n_planes, n_cones, etc.) are no longer used

    # Detect if lifted coordinates are being used
    use_lifted = model_cfg.get("use_lifted_coords", False)
    lift_type = model_cfg.get("lift_type", "sta")
    
    # Remove use_lifted_coords from model_cfg as it's not a constructor parameter
    model_cfg_clean = {k: v for k, v in model_cfg.items() if k != "use_lifted_coords"}
    
    if use_lifted:
        # Create LiftedLyapNet (lift_type is a valid parameter)
        V = LiftedLyapNet(**model_cfg_clean).to(device)
    else:
        # Create SimpleLyapNet (doesn't use lift_type)
        model_cfg_simple = {k: v for k, v in model_cfg_clean.items() if k != "lift_type"}
        V = SimpleLyapNet(**model_cfg_simple).to(device)
    G = FlexibleGauge(**gauge_params).to(device)
    V.load_state_dict(V_state); V.eval()
    G.load_state_dict(G_state); G.eval()
    return V, G


def compute_grids(V, G, ctrl, device, grid_n, lim, alpha_s, alpha_v, s_eps):
    xs = torch.linspace(-lim, lim, grid_n, device=device)
    ys = torch.linspace(-lim, lim, grid_n, device=device)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    ZZ = torch.stack([XX.reshape(-1), YY.reshape(-1)], 1).requires_grad_(True)

    # Phi (no grad), V with grad for dV
    with torch.no_grad():
        phi = G(ZZ)
    Vv = V(ZZ)
    gV = torch.autograd.grad(Vv.sum(), ZZ, create_graph=False, retain_graph=False)[0]
    dV = ctrl.worst_dV(gV, ZZ, s_eps=s_eps)
    alpha = alpha_s * torch.abs(ZZ[:, 0]) + alpha_v * torch.abs(ZZ[:, 1])
    dec = dV + alpha

    shp = (grid_n, grid_n)
    data = {
        "V":   Vv.detach().cpu().reshape(shp),
        "dV":  dV.detach().cpu().reshape(shp),
        "dec": dec.detach().cpu().reshape(shp),
        "phi": phi.detach().cpu().reshape(shp),
        "grad": torch.linalg.norm(gV, dim=1).detach().cpu().reshape(shp),
        "X":   XX.cpu(),
        "Y":   YY.cpu(),
    }
    return data


@torch.no_grad()
def boundary_curve(G, device, lim, num=1440):
    th = torch.linspace(0, 2 * math.pi, num, device=device)
    dirs = torch.stack([torch.cos(th), torch.sin(th)], 1)
    r = 1.0 / G(dirs).clamp_min(1e-6)
    xb = (dirs[:, 0] * r).cpu().numpy()
    yb = (dirs[:, 1] * r).cpu().numpy()
    mask = (np.abs(xb) <= lim * 1.05) & (np.abs(yb) <= lim * 1.05)
    return xb[mask], yb[mask]


@torch.no_grad()
def radial_slices(V, G, angles_deg, slice_r, lim, device):
    angles = [math.radians(a) for a in angles_deg]
    r = torch.linspace(0, lim, slice_r, device=device)
    slices = []
    for ang, adeg in zip(angles, angles_deg):
        d = torch.tensor([math.cos(ang), math.sin(ang)], device=device)
        pts = r.unsqueeze(1) * d.unsqueeze(0)
        Vvals = V(pts).cpu().numpy()
        rb = float((1.0 / G(d.unsqueeze(0)).clamp_min(1e-6)).item())
        slices.append({"angle": adeg, "r": r.cpu().numpy(), "V": Vvals, "rb": rb})
    return slices


def make_figure(
    data, xb, yb, slices, out_path, lim, *,
    zero_center=False, mask_inside=False, add_zero_contours=False,
    show_train_box=False, train_box=1.6, log_phi=False,
    enable_3d=True, surface_downsample=3, dpi=140
):
    """
    2D visualization with 4 focused plots:
    1. V contours (Lyapunov function)
    2. dV/dt worst-case
    3. Radial V slices
    4. V surface (3D)
    """
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2)

    # Prepare masked arrays for outside-only display if requested
    phi_np = data["phi"].numpy()
    if mask_inside:
        mask = phi_np <= 1.0
        dV = data["dV"].numpy().copy();  dV[mask]  = np.nan
    else:
        dV = data["dV"].numpy()

    # Symmetric color limits about 0 if requested
    def symm_limits(a: np.ndarray):
        vmax = np.nanmax(np.abs(a))
        return (-vmax, vmax) if np.isfinite(vmax) and vmax > 0 else (None, None)

    vlim_dV  = symm_limits(dV)  if zero_center else (None, None)

    # Panel 1: V contours + boundary
    ax1 = fig.add_subplot(gs[0, 0])
    cs = ax1.contour(data["X"], data["Y"], data["V"], levels=25, cmap="viridis")
    ax1.clabel(cs, inline=True, fontsize=6)
    ax1.plot(xb, yb, "r-", lw=1.5, label="Boundary φ=1")
    ax1.set_title("V Contours", fontsize=13, fontweight="bold")
    ax1.set_xlabel("s", fontsize=10)
    ax1.set_ylabel("v", fontsize=10)
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_aspect("equal")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, linestyle=":")

    # Panel 2: dV heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    vmin_dV = vlim_dV[0] if vlim_dV[0] is not None else np.nanmin(dV)
    vmax_dV = vlim_dV[1] if vlim_dV[1] is not None else np.nanmax(dV)
    im2 = ax2.imshow(
        dV.T, extent=[-lim, lim, -lim, lim], origin="lower",  # type: ignore
        cmap="coolwarm", vmin=vmin_dV, vmax=vmax_dV
    )
    ax2.plot(xb, yb, "k-", lw=1.2)
    ax2.set_title("dV/dt worst-case" + (" (outside)" if mask_inside else ""), fontsize=13, fontweight="bold")
    ax2.set_xlabel("s", fontsize=10)
    ax2.set_ylabel("v", fontsize=10)
    fig.colorbar(im2, ax=ax2, shrink=0.85, label="dV/dt")

    # Optional 0-level contours
    if add_zero_contours:
        Xn = data["X"].numpy(); Yn = data["Y"].numpy()
        try: ax2.contour(Xn, Yn, dV.T,  levels=[0.0], colors="lime", linewidths=1.2)
        except Exception: pass

    # Optional train box overlay
    if show_train_box:
        b = float(train_box)
        ax2.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], "yellow", linestyle="--", lw=1.5, label="Train box")
        ax2.legend(fontsize=8)

    # Panel 3: radial slices of V
    ax3 = fig.add_subplot(gs[1, 0])
    for sl in slices:
        ax3.plot(sl["r"], sl["V"], label=f"θ={sl['angle']}°", linewidth=1.5)
        if sl["rb"] < lim * 1.05:
            ax3.axvline(sl["rb"], color=ax3.lines[-1].get_color(), linestyle="--", linewidth=1.2, alpha=0.6)
    ax3.set_xlabel("Radius", fontsize=10)
    ax3.set_ylabel("V(r,θ)", fontsize=10)
    ax3.set_title("Radial V Slices", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(alpha=0.3, linestyle=":")

    # Panel 4: 3D surface
    if enable_3d:
        ax4 = fig.add_subplot(gs[1, 1], projection="3d")
        ds = max(1, surface_downsample)
        Xd = data["X"][::ds, ::ds]
        Yd = data["Y"][::ds, ::ds]
        Vd = data["V"][::ds, ::ds]
        ax4.plot_surface(Xd, Yd, Vd, cmap="viridis", linewidth=0, antialiased=True,  # type: ignore
                         rcount=Xd.shape[0], ccount=Xd.shape[1], alpha=0.9)
        ax4.plot(xb, yb, np.full_like(xb, Vd.min()), "r-", linewidth=1.5)
        ax4.set_title("V Surface", fontsize=13, fontweight="bold")
        ax4.set_xlabel("s", fontsize=9)
        ax4.set_ylabel("v", fontsize=9)
        ax4.set_zlabel("V", fontsize=9)  # type: ignore
        ax4.set_xlim(-lim, lim); ax4.set_ylim(-lim, lim)
        ax4.view_init(elev=25, azim=-60)
    else:
        # Fallback to gradient magnitude if 3D disabled
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(data["grad"].T, extent=[-lim, lim, -lim, lim], origin="lower", cmap="plasma")  # type: ignore
        ax4.plot(xb, yb, "w-", lw=0.8)
        ax4.set_title("|∇V| (3D disabled)", fontsize=13, fontweight="bold")
        fig.colorbar(im4, ax=ax4, shrink=0.85, label="|∇V|")

    fig.suptitle("2D Neural Lyapunov Overview", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # type: ignore
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved 2D overview to {out_path}")


def build_argparser():
    p = argparse.ArgumentParser("Overview viz")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--grid_n", type=int, default=220)
    p.add_argument("--lim", type=float, default=3.0)
    p.add_argument("--angles", type=str, default="0,45,90,135")
    p.add_argument("--slice_r", type=int, default=220)
    p.add_argument("--surface_downsample", type=int, default=3)
    p.add_argument("--no_3d", action="store_true")
    p.add_argument("--log_phi", action="store_true")
    p.add_argument("--mask_inside", action="store_true", help="Hide phi<=1 region in dV/dec panels")
    p.add_argument("--zero_center", action="store_true", help="Force symmetric colormap around 0 for dV/dec")
    p.add_argument("--add_zero_contours", action="store_true", help="Overlay 0-level contour lines on dV/dec")
    p.add_argument("--show_train_box", action="store_true", help="Draw dashed training box on dV/dec panels")
    p.add_argument("--train_box", type=float, default=1.6, help="Half-width of square training box")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dpi", type=int, default=140)
    # Optional overrides
    p.add_argument("--alpha_s", type=float, default=None)
    p.add_argument("--alpha_v", type=float, default=None)
    p.add_argument("--s_eps", type=float, default=None)
    return p


# ================ 1D VISUALIZATION FUNCTIONS ================

def compute_grids_1d(V, G, ctrl, device, grid_n, lim, alpha_s, s_eps):
    """
    Compute 1D arrays for visualization of Lyapunov function.
    
    Args:
        V: Lyapunov network
        G: Gauge function
        ctrl: 1D controller
        device: Torch device
        grid_n: Number of grid points
        lim: Spatial limit (plots [-lim, lim])
        alpha_s: Alpha regularization weight
        s_eps: Filippov smoothing parameter
        
    Returns:
        dict with keys: s, V, dV, dec, phi, grad (all 1D numpy arrays)
    """
    s_vals = torch.linspace(-lim, lim, grid_n, device=device)
    Z = s_vals.unsqueeze(1)  # Shape (grid_n, 1)
    
    # Compute V values
    with torch.no_grad():
        Vvals = V(Z).squeeze()
        phi_vals = G(Z).squeeze()
    
    # Compute gradient and worst-case dV
    Z.requires_grad_(True)
    Vv = V(Z)
    gV = torch.autograd.grad(Vv.sum(), Z, create_graph=False)[0]
    dV = ctrl.worst_dV(gV, Z, s_eps=s_eps)
    
    # Alpha regularization and decrease
    alpha = alpha_s * torch.abs(Z[:, 0])
    dec = dV + alpha
    
    # Gradient magnitude
    grad_mag = torch.abs(gV[:, 0])
    
    return {
        "s": s_vals.cpu().numpy(),
        "V": Vvals.detach().cpu().numpy(),
        "dV": dV.detach().cpu().numpy(),
        "dec": dec.detach().cpu().numpy(),
        "phi": phi_vals.detach().cpu().numpy(),
        "grad": grad_mag.detach().cpu().numpy()
    }


def make_figure_1d(data, out_path, lim, **kwargs):
    """
    1D visualization with 2 focused plots:
    1. Lyapunov function V(s)
    2. Derivative dV/dt
    """
    dpi = kwargs.get('dpi', 140)
    
    s = data["s"]
    V = data["V"]
    dV = data["dV"]
    phi = data["phi"]
    
    # Find boundary points where φ ≈ 1
    boundary_mask = np.abs(phi - 1.0) < 0.05
    s_boundaries = []
    if boundary_mask.any():
        # Get all boundary crossings
        boundary_indices = np.where(boundary_mask)[0]
        if len(boundary_indices) > 0:
            # Cluster nearby indices
            clusters = [[boundary_indices[0]]]
            for idx in boundary_indices[1:]:
                if idx - clusters[-1][-1] <= 3:
                    clusters[-1].append(idx)
                else:
                    clusters.append([idx])
            # Take middle of each cluster
            for cluster in clusters:
                mid_idx = cluster[len(cluster) // 2]
                s_boundaries.append(s[mid_idx])
    
    # Create figure with 1 row × 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Lyapunov function V(s)
    ax1.plot(s, V, 'b-', linewidth=2.5, label='V(s)')
    ax1.axhline(0, color='k', linestyle='-', linewidth=1.5, alpha=0.5)
    ax1.fill_between(s, 0, V, where=(phi > 1.0), alpha=0.15, color='blue', label='Outside φ>1')
    
    # Mark boundaries
    for sb in s_boundaries:
        ax1.axvline(sb, color='r', linestyle='--', linewidth=2, alpha=0.7)
    
    # Highlight the stable region (inside boundary)
    if len(s_boundaries) >= 2:
        ax1.axvspan(s_boundaries[0], s_boundaries[1], alpha=0.1, color='green', label='Stable region φ≤1')
    
    ax1.set_title("Lyapunov Function V(s)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("s (sliding surface)", fontsize=11)
    ax1.set_ylabel("V", fontsize=11)
    ax1.set_xlim(-lim, lim)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=10, loc='upper center')
    
    # Plot 2: Derivative dV/dt
    ax2.plot(s, dV, 'r-', linewidth=2.5, label='dV/dt')
    ax2.axhline(0, color='k', linestyle='-', linewidth=1.5, alpha=0.5, label='Zero line')
    ax2.fill_between(s, dV, 0, where=(dV < 0) & (phi > 1.0), alpha=0.25, color='green', label='Decreasing (dV<0)')
    ax2.fill_between(s, dV, 0, where=(dV > 0) & (phi > 1.0), alpha=0.25, color='red', label='Increasing (dV>0)')
    
    # Mark boundaries
    for sb in s_boundaries:
        ax2.axvline(sb, color='r', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_title("Lyapunov Derivative dV/dt", fontsize=14, fontweight='bold')
    ax2.set_xlabel("s (sliding surface)", fontsize=11)
    ax2.set_ylabel("dV/dt", fontsize=11)
    ax2.set_xlim(-lim, lim)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=10, loc='upper right')
    
    plt.suptitle("1D Neural Lyapunov Overview - First-Order SMC", 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] 1D visualization saved: {out_path}")


def main():
    args = build_argparser().parse_args()
    dev = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    V_state, G_state, cfg, meta = load_checkpoint(args.model_path, dev)
    V, G = build_models(V_state, G_state, cfg, dev)

    # Controller (for worst-case dV)
    ctrl_name = meta.get("controller", cfg.get("controller", {}).get("name", "sta"))
    ctrl = get_controller(ctrl_name, **meta.get("controller_params", cfg.get("controller", {}).get("params", {})))

    # Parameters from cfg with optional overrides
    alpha_s = args.alpha_s if args.alpha_s is not None else cfg.get("alpha", {}).get("alpha_s", 0.03)
    alpha_v = args.alpha_v if args.alpha_v is not None else cfg.get("alpha", {}).get("alpha_v", 0.05)
    s_eps   = args.s_eps   if args.s_eps   is not None else cfg.get("val",   {}).get("s_eps_val", 1e-3)

    # Check dimensionality and call appropriate visualization
    input_dim = G.input_dim
    if input_dim == 3:
        # 3D visualization for PID-SMC
        alpha_x1 = cfg.get("alpha", {}).get("alpha_x1", alpha_s)
        alpha_x2 = cfg.get("alpha", {}).get("alpha_x2", alpha_v) 
        alpha_x3 = cfg.get("alpha", {}).get("alpha_x3", alpha_v)
        x1_eps = s_eps
        
        data = compute_grids_3d(V, G, ctrl, dev, args.grid_n, args.lim, alpha_x1, alpha_x2, alpha_x3, x1_eps)
        xb, yb, zb, mask = boundary_surface_3d(G, dev, args.lim)
        
        angles_deg = [int(a.strip()) for a in args.angles.split(",") if a.strip()]
        slices = cross_section_slices_3d(V, G, angles_deg, args.slice_r, args.lim, dev)
        
        # Default output path: alongside the model
        out_path = args.out or os.path.join(os.path.dirname(args.model_path), "overview.png")
        
        make_figure_3d(
            data, xb, yb, zb, mask, slices, out_path, args.lim,
            dpi=args.dpi
        )
        print(f"3D visualization saved: {out_path}")
        
    elif input_dim == 2:
        # 2D visualization for STA/CTA
        data = compute_grids(V, G, ctrl, dev, args.grid_n, args.lim, alpha_s, alpha_v, s_eps)
        xb, yb = boundary_curve(G, dev, args.lim)

        angles_deg = [int(a.strip()) for a in args.angles.split(",") if a.strip()]
        slices = radial_slices(V, G, angles_deg, args.slice_r, args.lim, dev)

        # Default output path: alongside the model
        out_path = args.out or os.path.join(os.path.dirname(args.model_path), "overview.png")

        make_figure(
            data, xb, yb, slices, out_path, args.lim,
            zero_center=args.zero_center, mask_inside=args.mask_inside,
            add_zero_contours=args.add_zero_contours,
            show_train_box=args.show_train_box, train_box=args.train_box,
            log_phi=args.log_phi, enable_3d=(not args.no_3d),
            surface_downsample=args.surface_downsample, dpi=args.dpi
        )
        
    elif input_dim == 1:
        # 1D visualization for FOSMC
        data = compute_grids_1d(V, G, ctrl, dev, args.grid_n, args.lim, alpha_s, s_eps)
        
        # Default output path: alongside the model
        out_path = args.out or os.path.join(os.path.dirname(args.model_path), "overview.png")
        
        make_figure_1d(
            data, out_path, args.lim,
            dpi=args.dpi
        )
        
    else:
        raise ValueError(f"Unsupported dimensionality: {input_dim}. Expected 1, 2, or 3.")


# ================ 3D VISUALIZATION FUNCTIONS ================

def compute_grids_3d(V, G, ctrl, device, grid_n, lim, alpha_x1, alpha_x2, alpha_x3, x1_eps):
    """
    Compute 3D grids for visualization. For PID-SMC with state (x1, x2, x3).
    Creates cross-sectional slices through the 3D state space.
    """
    # Create 3D grid
    xs = torch.linspace(-lim, lim, grid_n, device=device)
    ys = torch.linspace(-lim, lim, grid_n, device=device) 
    zs = torch.linspace(-lim, lim, grid_n, device=device)
    
    # Generate slice data at z=0 (x3=0 plane)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    ZZ_slice = torch.zeros_like(XX)
    points_2d = torch.stack([XX.reshape(-1), YY.reshape(-1), ZZ_slice.reshape(-1)], 1).requires_grad_(True)

    # Compute values for x3=0 slice
    with torch.no_grad():
        phi_slice = G(points_2d)
    V_slice = V(points_2d)
    gV_slice = torch.autograd.grad(V_slice.sum(), points_2d, create_graph=False, retain_graph=False)[0]
    dV_slice = ctrl.worst_dV(gV_slice, points_2d, s_eps=x1_eps)
    alpha_slice = alpha_x1 * torch.abs(points_2d[:, 0]) + alpha_x2 * torch.abs(points_2d[:, 1]) + alpha_x3 * torch.abs(points_2d[:, 2])
    dec_slice = dV_slice + alpha_slice

    shp = (grid_n, grid_n)
    data = {
        "V": V_slice.detach().cpu().reshape(shp),
        "dV": dV_slice.detach().cpu().reshape(shp),
        "dec": dec_slice.detach().cpu().reshape(shp),
        "phi": phi_slice.detach().cpu().reshape(shp),
        "grad": torch.linalg.norm(gV_slice, dim=1).detach().cpu().reshape(shp),
        "X": XX.cpu(),
        "Y": YY.cpu(),
        "slice_plane": "x3=0"
    }
    return data


@torch.no_grad()
def boundary_surface_3d(G, device, lim, num_theta=36, num_phi=18):
    """
    Compute 3D boundary surface for visualization.
    Returns points on the boundary surface where G(x) = 1.
    """
    # Spherical coordinates
    theta = torch.linspace(0, 2 * math.pi, num_theta, device=device)
    phi = torch.linspace(0, math.pi, num_phi, device=device)
    TH, PH = torch.meshgrid(theta, phi, indexing="ij")
    
    # Convert to Cartesian unit directions
    dirs_x = torch.sin(PH) * torch.cos(TH)
    dirs_y = torch.sin(PH) * torch.sin(TH) 
    dirs_z = torch.cos(PH)
    dirs = torch.stack([dirs_x.reshape(-1), dirs_y.reshape(-1), dirs_z.reshape(-1)], 1)
    
    # Compute boundary radius
    r = 1.0 / G(dirs).clamp_min(1e-6)
    
    # Get boundary points
    xb = (dirs[:, 0] * r).cpu().numpy().reshape(num_theta, num_phi)
    yb = (dirs[:, 1] * r).cpu().numpy().reshape(num_theta, num_phi)
    zb = (dirs[:, 2] * r).cpu().numpy().reshape(num_theta, num_phi)
    
    # Use adaptive masking based on actual boundary extent
    max_extent = max(np.abs(xb).max(), np.abs(yb).max(), np.abs(zb).max())
    adaptive_lim = max(lim, max_extent * 1.2)  # At least 20% larger than boundary
    mask = (np.abs(xb) <= adaptive_lim) & (np.abs(yb) <= adaptive_lim) & (np.abs(zb) <= adaptive_lim)
    
    return xb, yb, zb, mask


@torch.no_grad()
def cross_section_slices_3d(V, G, angles_deg, slice_r, lim, device):
    """
    Generate cross-sectional slices through 3D space for visualization.
    """
    angles = [math.radians(a) for a in angles_deg]
    r = torch.linspace(0, lim, slice_r, device=device)
    slices = []
    
    for ang, adeg in zip(angles, angles_deg):
        # Create slice in x1-x2 plane (x3=0)
        d = torch.tensor([math.cos(ang), math.sin(ang), 0.0], device=device)
        pts = torch.zeros((slice_r, 3), device=device)
        pts[:, :3] = r.unsqueeze(1) * d.unsqueeze(0)
        
        Vvals = V(pts).cpu().numpy()
        rb = float((1.0 / G(d.unsqueeze(0)).clamp_min(1e-6)).item())
        slices.append({"angle": adeg, "r": r.cpu().numpy(), "V": Vvals, "rb": rb})
    
    return slices


def make_figure_3d(data, xb, yb, zb, mask, slices, out_path, lim, **kwargs):
    """
    3D visualization with 4 focused plots:
    1. V cross-section (x3=0 plane)
    2. dV/dt cross-section (x3=0 plane)
    3. 3D boundary surface
    4. Radial V slices
    
    These provide the best insight into 3D Lyapunov analysis:
    - Cross-sections show behavior in a key plane
    - Boundary surface shows full 3D stability region
    - Radial slices show V structure along different directions
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(f"3D Neural Lyapunov Overview ({data['slice_plane']} slice)", 
                 fontsize=16, fontweight='bold')
    
    # 2x2 layout for 4 focused plots
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3, projection='3d')
    ax4 = plt.subplot(2, 2, 4)
    
    # Plot 1: V cross-section at x3=0
    im1 = ax1.imshow(data["V"].T, extent=(-lim, lim, -lim, lim), origin="lower", cmap="viridis")
    ax1.set_title(f"V (Lyapunov) - {data['slice_plane']}", fontsize=13, fontweight='bold')
    ax1.set_xlabel("x₁", fontsize=10)
    ax1.set_ylabel("x₂", fontsize=10)
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.85)
    cb1.set_label("V", fontsize=9)
    
    # Add boundary curve to slice
    slice_boundary_x = None
    slice_boundary_y = None
    if len(slices) > 0:
        slice_boundary_r = [s["rb"] for s in slices]
        slice_boundary_angles = [math.radians(s["angle"]) for s in slices]
        slice_boundary_x = [r * math.cos(a) for r, a in zip(slice_boundary_r, slice_boundary_angles)]
        slice_boundary_y = [r * math.sin(a) for r, a in zip(slice_boundary_r, slice_boundary_angles)]
        ax1.plot(slice_boundary_x, slice_boundary_y, 'r--', linewidth=1.5, alpha=0.8, label='Boundary φ=1')
        ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, linestyle=':')
    
    # Plot 2: dV/dt cross-section at x3=0
    im2 = ax2.imshow(data["dV"].T, extent=(-lim, lim, -lim, lim), origin="lower", cmap="coolwarm")
    ax2.set_title(f"dV/dt (worst-case) - {data['slice_plane']}", fontsize=13, fontweight='bold')
    ax2.set_xlabel("x₁", fontsize=10)
    ax2.set_ylabel("x₂", fontsize=10)
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.85)
    cb2.set_label("dV/dt", fontsize=9)
    
    # Add boundary curve
    if slice_boundary_x is not None and slice_boundary_y is not None:
        ax2.plot(slice_boundary_x, slice_boundary_y, 'k-', linewidth=1.2, alpha=0.7)
    ax2.grid(alpha=0.3, linestyle=':')
    
    # Plot 3: 3D boundary surface
    if xb is not None and yb is not None and zb is not None:
        xb_masked = np.where(mask, xb, np.nan)
        yb_masked = np.where(mask, yb, np.nan)
        zb_masked = np.where(mask, zb, np.nan)
        ax3.plot_surface(xb_masked, yb_masked, zb_masked, alpha=0.4, color='red', edgecolor='darkred', linewidth=0.2)  # type: ignore
    ax3.set_title("3D Boundary Surface (φ=1)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("x₁", fontsize=9)
    ax3.set_ylabel("x₂", fontsize=9)
    ax3.set_zlabel("x₃", fontsize=9)  # type: ignore
    ax3.view_init(elev=20, azim=-60)  # type: ignore
    
    # Plot 4: Radial slices
    if slices:
        for s in slices:
            ax4.plot(s["r"], s["V"], label=f"θ={s['angle']}°", linewidth=1.5)
            if s["rb"] < lim * 1.05:
                ax4.axvline(s["rb"], color=ax4.lines[-1].get_color(), linestyle="--", linewidth=1.2, alpha=0.6)
    ax4.set_title("Radial V Slices", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Radius", fontsize=10)
    ax4.set_ylabel("V(r,θ)", fontsize=10)
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=kwargs.get('dpi', 140), bbox_inches='tight')
    plt.close()
    print(f"[OK] 3D visualization saved: {out_path}")


if __name__ == "__main__":
    main()
