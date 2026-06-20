import sys

import torch
import torch.nn as nn

from src.model import DenoisingAutoencoder
from src.noise import add_gaussian_noise


def run_preflight_tests(
    in_channels: int = 1,
    image_size: int = 28,
    noise_factor: float = 0.3,
) -> None:
    device = torch.device("cpu")
    model = DenoisingAutoencoder(in_channels=in_channels).to(device)
    model.eval()

    dummy = torch.randn(8, in_channels, image_size, image_size, device=device)
    with torch.no_grad():
        output = model(dummy)

    assert output.shape == dummy.shape, (
        f"Shape mismatch: expected {dummy.shape}, got {output.shape}"
    )

    assert output.min().item() >= 0.0, "Output minimum is below 0.0"
    assert output.max().item() <= 1.0, "Output maximum is above 1.0"

    model.train()
    torch.manual_seed(0)
    clean = torch.rand(2, in_channels, image_size, image_size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    sanity_noise = min(noise_factor, 0.2)

    final_loss = float("inf")
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        noisy = add_gaussian_noise(clean, sanity_noise)
        recon = model(noisy)
        loss = criterion(recon, clean)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    if final_loss >= 0.05:
        print(
            f"Preflight sanity test failed: loss={final_loss:.6f} (expected < 0.05)",
            file=sys.stderr,
        )
        sys.exit(1)
