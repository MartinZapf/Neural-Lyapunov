# Neural Lyapunov Functions for Sliding Mode Controllers

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

Learn neural Lyapunov functions that verify stability of sliding mode controllers with certified exclusion regions.

**Paper:** Neural Lyapunov Functions for Sliding Mode Based Feedback Loops (arXiv link coming soon)

## Installation

```bash
git clone https://github.com/MartinZapf/Neural-Lyapunov.git
cd Neural-Lyapunov
pip install -e .

# For hyperparameter optimization
pip install -e ".[hpo]"
```

## Quick Start

Train a Lyapunov function for the Super-Twisting Algorithm:

```bash
python -m neural_lyapunov.train --config configs/sta.yaml
```

Visualize results:

```bash
python viz/overview.py --model_path outputs/<run_dir>/best_model.pth
```

## Supported Controllers

| Controller | Dim | Config | Exclusion | Description |
|------------|-----|--------|-----------|-------------|
| FOSMC | 1D | `fosmc.yaml` | r = 0.005 | First-order sliding mode |
| STA | 2D | `sta.yaml` | A = 0.003 | Super-Twisting Algorithm |
| CTA | 2D | `cta.yaml` | A = 0.004 | Continuous Twisting Algorithm |
| PID-SMC | 3D | `pidsmc.yaml` | V = 5.2e-3 | PID-like sliding mode |

## How It Works

The method learns a neural Lyapunov function V(z) and exclusion boundary phi(z) such that:

1. **V(z) > 0** for z != 0 (positive definite)
2. **dV/dt < 0** outside the exclusion region {z : phi(z) <= 1}
3. The exclusion region is minimized during training

The training uses a **shrink-verify** loop:
- Start with a feasible (possibly large) exclusion region
- Gradually shrink the boundary while maintaining the decrease condition
- Validate on a dense grid with float64 precision
- Polish the final solution

For discontinuous dynamics (sliding mode), we use **Filippov differential inclusions**: dV/dt is computed as the worst case over the set-valued right-hand side.

## Training Your Own Controller

### Step 1: Implement the Controller

Add your controller to `src/neural_lyapunov/controllers.py`:

```python
class MyController(BaseSMC):
    def __init__(self, k1: float = 1.0, **kwargs):
        super().__init__()
        self.name = "my_controller"
        self.state_dim = 2  # Dimension of state space
        self.k1 = k1

    def modes(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return f+ and f- for Filippov inclusion."""
        # Implement your controller dynamics here
        # f+ = dynamics when sign(s) = +1
        # f- = dynamics when sign(s) = -1
        pass

    def worst_dV(self, gradV: torch.Tensor, z: torch.Tensor,
                 s_eps: float = 1e-3) -> torch.Tensor:
        """Compute worst-case dV/dt over Filippov set."""
        f_plus, f_minus = self.modes(z)
        dV_plus = torch.sum(gradV * f_plus, dim=1)
        dV_minus = torch.sum(gradV * f_minus, dim=1)
        return torch.maximum(dV_plus, dV_minus)
```

Register it in `get_controller()`:
```python
def get_controller(name: str, **kwargs) -> BaseSMC:
    controllers = {
        "sta": STA,
        "cta": CTA,
        "fosmc": FOSMC,
        "pid_smc": PIDSMC,
        "my_controller": MyController,  # Add here
    }
    return controllers[name](**kwargs)
```

### Step 2: Create a Config File

Start from the config most similar to your controller:
- **1D systems:** Start from `configs/fosmc.yaml`
- **2D systems:** Start from `configs/sta.yaml` or `configs/cta.yaml`
- **3D systems:** Start from `configs/pidsmc.yaml`

Key parameters to adjust:

```yaml
controller:
  name: my_controller
  params:
    k1: 1.0  # Your controller gains

model:
  width: 96        # Network width (64-256)
  depth: 2         # Network depth (2-3)

gauge:
  input_dim: 2     # Must match controller state_dim
  initial_radius: 0.1  # Starting exclusion size

alpha:
  alpha_s: 0.02    # Regularization weight per state dimension
  alpha_v: 0.02

box:
  train_s: 4       # Training domain half-width per dimension
  train_v: 4
```

### Step 3: Train

```bash
python -m neural_lyapunov.train --config configs/my_controller.yaml
```

Monitor the output for:
- `val_max`: Maximum violation (should be <= 0 for valid Lyapunov function)
- `area`: Current exclusion region size
- `OK/FAIL`: Validation status

### Step 4: Tune (Optional)

Use Optuna for hyperparameter optimization:

```bash
# Single worker
python hpo/tune.py --config configs/my_controller.yaml --n_trials 100

# Parallel workers (recommended)
cd hpo
./launch_parallel.sh ../configs/my_controller.yaml 50 4  # 50 trials, 4 workers
```

Monitor with Optuna dashboard:
```bash
optuna-dashboard sqlite:///my_controller_hpo.db
```

## Configuration Reference

### Core Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `model.width` | Hidden layer width | 64-256 |
| `model.depth` | Number of hidden layers | 2-3 |
| `model.eps_quad` | Quadratic regularization in V | 1e-4 to 1e-3 |
| `gauge.initial_radius` | Starting exclusion size | 0.01-0.5 |
| `train.epochs` | Training iterations | 5000-15000 |
| `train.lr` | Learning rate | 1e-4 to 2e-3 |

### Sampling Parameters

| Parameter | Description |
|-----------|-------------|
| `sample.m_ring` | Points in ring region near boundary |
| `sample.m_out_global` | Points sampled globally outside |
| `sample.m_size` | Points for size estimation |

### Validation Parameters

| Parameter | Description |
|-----------|-------------|
| `val.N` | Grid resolution (N x N for 2D) |
| `val.every` | Validation frequency (epochs) |
| `val_dtype` | Precision (`float64` recommended) |

## Output Structure

Each training run creates:

```
outputs/YYYY-MM-DD_HH-MM-SS_<controller>/
├── config_used.yaml      # Full config with defaults
├── best_model.pth        # Best model checkpoint
├── best_model_polished.pth  # Polished model (if enabled)
└── overview.png          # Visualization (if enabled)
```

## Citation

If you use this code, please cite:

```bibtex
@article{zapf2026neural,
  title={Neural Lyapunov Functions for Sliding Mode Based Feedback Loops},
  author={Zapf, Martin},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
