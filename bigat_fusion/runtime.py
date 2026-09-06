"""Random state and accelerator selection."""

import random

import numpy as np
import torch


def seed_everything(seed):
    """Initialize Python, NumPy, and PyTorch random states."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested):
    """Resolve an explicit or automatic PyTorch compute device."""
    if requested == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "xpu" and not (
        hasattr(torch, "xpu") and torch.xpu.is_available()
    ):
        raise RuntimeError(
            "Intel GPU (XPU) was requested but is unavailable. Install the XPU build "
            "of PyTorch and update the Intel GPU driver."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)
