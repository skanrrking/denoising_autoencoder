"""Pick the fastest available hardware and DataLoader worker count."""

import sys

import torch


def get_device() -> torch.device:
    """Step 1: Try CUDA (NVIDIA GPU), then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_num_workers() -> int:
    """Step 2: Windows cannot safely spawn many DataLoader workers, so use 0."""
    if sys.platform == "win32":
        return 0
    return 4


def use_amp(device: torch.device) -> bool:
    """Step 3: Mixed precision only runs on CUDA; disable on MPS/CPU."""
    return device.type == "cuda"
