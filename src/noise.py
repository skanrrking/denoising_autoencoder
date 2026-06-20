"""On-device Gaussian noise injection (never in the Dataset class)."""

import torch


def add_gaussian_noise(x: torch.Tensor, noise_factor: float) -> torch.Tensor:
    """
    Step 1: Sample random noise on the same device as x (no CPU->GPU copy).
    Step 2: Scale noise and add it to the clean image.
    Step 3: Clamp to [0, 1] so pixel values stay in valid image range.
    """
    noise = torch.randn_like(x) * noise_factor
    return torch.clamp(x + noise, 0.0, 1.0)
