import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global random seeds for reproducible experiments.

    This function seeds:
        - Python's built-in random module
        - NumPy
        - PyTorch (if installed)

    Args:
        seed (int): Seed value to use.
        deterministic (bool): If True, enables extra deterministic flags
            in PyTorch backend where available. This can make training
            slightly slower but improves reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Some libraries read this environment variable for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch is optional for some workflows; skip if not installed.
        pass


def get_seed_from_config(cfg: dict, default: Optional[int] = None) -> int:
    """
    Safely extract seed from configuration dictionary.

    Args:
        cfg (dict): Master configuration dictionary.
        default (Optional[int]): Fallback seed if not defined in cfg.

    Returns:
        int: Seed value to use.
    """
    if cfg is None:
        return default if default is not None else 42

    return int(cfg.get("seed", default if default is not None else 42))

