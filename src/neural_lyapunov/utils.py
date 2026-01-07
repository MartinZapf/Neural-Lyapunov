from __future__ import annotations
import os
import time
import yaml
import torch
import random
import numpy as np
from typing import Dict, Any


def set_seed(seed: int = 1337, deterministic: bool = True):
    import warnings
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Set environment variable before calling deterministic algorithms
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # Suppress the specific CUDA deterministic warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Deterministic behavior was enabled.*")
            torch.use_deterministic_algorithms(True, warn_only=True)
            
        if torch.cuda.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: Dict[str, Any], path: str):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def make_run_dir(base: str, controller_name: str) -> str:
    """Create timestamped run directory for outputs.

    Args:
        base: Base output directory (absolute or relative to cwd)
        controller_name: Name of controller for directory suffix

    Returns:
        Path to created run directory
    """
    run = f"{timestamp()}_{controller_name}"
    out = os.path.join(base, run)
    ensure_dir(out)
    return out
