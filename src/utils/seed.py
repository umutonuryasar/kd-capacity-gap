import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all RNG sources for reproducibility.

    Sets seeds for Python random, NumPy, PyTorch CPU/CUDA, and pins cuDNN
    to deterministic algorithms (with benchmark disabled).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
