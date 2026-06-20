import sys

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_num_workers() -> int:
    if sys.platform == "win32":
        return 0
    return 4


def use_amp(device: torch.device) -> bool:
    return device.type == "cuda"
