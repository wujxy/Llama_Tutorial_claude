"""
Utility module for setting random seeds for reproducibility.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: Optional[bool] = False) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
