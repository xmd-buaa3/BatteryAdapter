"""Deterministic seeding shared by adapter training and final test evaluation.

BatteryMFormer pins its RNG surface in ``set_seed`` (``reference/BatteryMFormer/
run_main.py`` and ``evaluate_model.py``): ``random``, ``numpy``, ``torch`` and the
CUDA generators are reseeded from the fold seed, and cuDNN is forced onto
deterministic kernels. Adapter runs must start from the same state so that a
fold's seed means the same thing in both pipelines.

The cuDNN flags are set unconditionally: they only take effect on CUDA, and
pinning them here removes the dependency on when ``torch.cuda.is_available()``
first becomes true.
"""
import random

import numpy as np
import torch


def set_deterministic_seed(seed):
    """Reseed every generator BatteryMFormer reseeds and pin deterministic cuDNN."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
