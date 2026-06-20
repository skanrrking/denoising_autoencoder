import torch


def add_gaussian_noise(x: torch.Tensor, noise_factor: float) -> torch.Tensor:
    noise = torch.randn_like(x) * noise_factor
    return torch.clamp(x + noise, 0.0, 1.0)
