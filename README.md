# Neural Lyapunov Functions for Sliding Mode Controllers

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

Code for the paper *Neural Lyapunov Functions for Sliding Mode Based Feedback
Loops*. The framework jointly trains a neural Lyapunov function `V(z)` and an
oriented ellipsoidal **gauge** `phi(z)` for closed-loop sliding-mode dynamics.
The Lyapunov decrease test is enforced (and validated on a dense `float64`
grid) outside the gauge interior `{phi <= 1}`; the gauge marks the part of the
training domain where the dense-grid decrease test is not enforced. A
shrink-and-verify loop reduces the gauge while keeping the grid validation
passing. If the dense-grid decrease test passes at all nonzero grid points in
the training box, no positive-volume gauge is reported (`Gauge = none`).

**Paper:** *to appear* — citation block at the bottom of this README.

## Installation

```bash
git clone https://github.com/MartinZapf/Neural-Lyapunov.git
cd Neural-Lyapunov
pip install -e .

# For hyperparameter optimization
pip install -e ".[hpo]"
```

## Quick Start

Train a Lyapunov function and gauge for the Super-Twisting Algorithm:

```bash
python -m neural_lyapunov.train --config configs/sta.yaml
```

Visualize the resulting `(V, phi)` pair:

```bash
python viz/overview.py --model_path outputs/<run_dir>/best_model.pth
```

The same trainer runs every reported controller — only the YAML config differs.

## Reported results

The five YAML files in `configs/` are the final configurations used to
produce the results reported in the paper.

| Controller   | Dim | Config                | Gauge                      | Validation grid | Train time |
|--------------|-----|-----------------------|----------------------------|-----------------|------------|
| FOSMC        | 1   | `fosmc.yaml`          | none                       | 1000            | 8 min      |
| STA          | 2   | `sta.yaml`            | area `A = 0.003`           | 301²            | 12 min     |
| STA (lifted) | 2   | `sta_lifted.yaml`     | none                       | 301²            | 18 min     |
| CTA          | 3   | `cta.yaml`            | volume `V = 3.0e-5`        | 50³             | 5 min      |
| PID-SMC      | 3   | `pidsmc.yaml`         | volume `V = 5.2e-4`        | 50³             | 25 min     |

**What `none` means.** The gauge module is still trained, but the reported
result is accepted as `none` when the dense-grid decrease test passes on every
nonzero grid point in the training box. The equilibrium itself is excluded
from the derivative validation, since the `eps_quad·‖z‖²` term in the
Lyapunov ansatz is nonsmooth at the origin under the `α(z)` regularizer used
for training.

**STA standard vs lifted.** The same closed-loop STA dynamics admit two
parameterizations of the Lyapunov network. In standard coordinates `(s, v)`
the framework returns a small but nonzero gauge near the switching line. In
lifted coordinates `(ξ₁, ξ₂) = (|s|^(1/2) sign(s), v)` the framework returns
no gauge at all. The method is not coordinate invariant; this pair of
configurations illustrates that observation. See the paper, §IV.C.

## How it works

The trainer learns `V` and `phi` jointly such that:

1. `V(z) > 0` for `z ≠ 0` (positive definite by construction).
2. `dV/dt < 0` outside the gauge `{z : phi(z) > 1}`, evaluated as a worst case
   over the Filippov differential inclusion (see below).
3. The gauge volume is reduced as far as the dense-grid validator allows
   while the strict decrease test continues to pass.

**Filippov vertex-max.** On the switching surface the right-hand side is
replaced by the convex hull of finitely many limiting branches
`f^(1), …, f^(K)`. For a `C¹` Lyapunov candidate `V`, the supremum of
`∇V·v` over the Filippov set equals the maximum over the branches:

```
sup_{v ∈ F[f](z)} ∇V(z)·v = max_k ∇V(z)·f^(k)(z)
```

(Linearity of the inner product; the maximum of a linear functional over a
convex hull of finitely many points is attained at a vertex.) The trainer
implements this directly via `vdot_worst_filippov` / `controller.worst_dV`,
so the per-point Filippov treatment runs identically inside and outside the
gauge. A small numerical band `|s| <= 1e-3` around the switching surface uses
the same `max` as a conservative over-approximation of the single-valued
branch, to avoid missing the surface on a finite training grid; this band is
not a smoothing of the dynamics.

**Matched scalar disturbances.** For controllers whose disturbance enters as
a matched additive scalar term `δ ∈ [-δ₀, +δ₀]`, the worst-case Lyapunov
derivative is affine in `δ`, so checking the two endpoints
`δ ∈ {-δ₀, +δ₀}` is equivalent to checking the entire interval. The same
argument extends to box-bounded vector disturbances by checking the box
vertices. The validator uses this directly.

## Training your own controller

### 1. Implement the controller

Add to `src/neural_lyapunov/controllers.py`:

```python
class MyController(BaseSMC):
    def __init__(self, k1: float = 1.0, **kwargs):
        super().__init__()
        self.name = "my_controller"
        self.state_dim = 2
        self.k1 = k1

    def modes(self, z: torch.Tensor):
        """Return (f_plus, f_minus): branches selected by sign(s)."""
        ...

    def worst_dV(self, gradV, z, s_eps=1e-3):
        f_plus, f_minus = self.modes(z)
        return torch.maximum((gradV * f_plus).sum(1), (gradV * f_minus).sum(1))
```

Register it in `get_controller()` in the same file.

### 2. Create a config file

Start from the config most similar to your controller:

- 1D systems: `configs/fosmc.yaml`
- 2D systems: `configs/sta.yaml` (raw) or `configs/sta_lifted.yaml` (lifted)
- 3D systems: `configs/cta.yaml` or `configs/pidsmc.yaml`

Key parameters to adjust:

```yaml
controller:
  name: my_controller
  params:
    k1: 1.0

model:
  width: 96
  depth: 2

gauge:
  input_dim: 2
  initial_radius: 0.1   # initial gauge size; shrunk during training

alpha:
  alpha_s: 0.02
  alpha_v: 0.02

box:
  train_s: 4
  train_v: 4
```

### 3. Train

```bash
python -m neural_lyapunov.train --config configs/my_controller.yaml
```

Training output reports `val_max` (worst grid violation; must be `≤ 0` for a
passing validation), the current gauge size, and `OK/FAIL` validation status.

### 4. Tune (optional)

`hpo/tune.py` contains the Optuna search machinery used to select training
hyperparameters for the reported configurations.

```bash
python hpo/tune.py --config configs/my_controller.yaml --n_trials 100
```

The full HPO sweeps used in the paper (≈ 39 000 STA training runs for the
parameter-space scan in §IV.C) are not archived in this release due to size;
the included YAMLs are the final selected configurations.

## Configuration reference

### Core parameters

| Parameter             | Description                                  | Typical range  |
|-----------------------|----------------------------------------------|----------------|
| `model.width`         | Hidden layer width                           | 64–256         |
| `model.depth`         | Number of hidden layers                      | 2–3            |
| `model.eps_quad`      | Quadratic regularization in `V`              | 1e-4 to 1e-3   |
| `gauge.initial_radius`| Starting gauge size                          | 0.01–0.5       |
| `train.epochs`        | Training iterations                          | 5000–15000     |
| `train.lr`            | Learning rate                                | 1e-4 to 2e-3   |

### Sampling

| Parameter             | Description                                   |
|-----------------------|-----------------------------------------------|
| `sample.m_ring`       | Points sampled near the gauge boundary        |
| `sample.m_out_global` | Points sampled globally outside the gauge     |
| `sample.m_size`       | Points used for the size objective            |

### Validation

| Parameter   | Description                                          |
|-------------|------------------------------------------------------|
| `val.N`     | Grid resolution per dimension                        |
| `val.every` | Validation frequency (epochs)                        |
| `val_dtype` | Validation precision (`float64` recommended)         |

## Output structure

Each training run creates:

```
outputs/YYYY-MM-DD_HH-MM-SS_<controller>/
├── config_used.yaml         # Full config with defaults filled in
├── best_model.pth           # Best (V, phi) pair seen during shrink-and-verify
├── best_model_polished.pth  # Optional polish stage output (if enabled)
└── overview.png             # Visualization (if enabled)
```

After training, visualize with `python viz/overview.py --model_path
outputs/<run_dir>/best_model.pth`. Pretrained checkpoints are not part of
this release; reproduce by training from the included configs.

## Citation

To appear:

```bibtex
@article{zapf2026neural,
  title={Neural Lyapunov Functions for Sliding Mode Based Feedback Loops},
  author={Zapf, Martin},
  year={2026},
  note={To appear}
}
```

## License

MIT License — see [LICENSE](LICENSE).
