"""Pre-flight checks run before any real data is loaded."""

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
    """
    Abort early if the model architecture or training loop is broken.

    Test 1 - Shape integrity: output must match input shape.
    Test 2 - Boundary enforcement: Sigmoid must keep outputs in [0, 1].
    Test 3 - Local convergence: model must overfit 2 tiny images in 50 steps.
    """
    device = torch.device("cpu")
    model = DenoisingAutoencoder(in_channels=in_channels).to(device)
    model.eval()

    # Test 1: shape integrity.
    dummy = torch.randn(8, in_channels, image_size, image_size, device=device)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == dummy.shape, (
        f"Shape mismatch: expected {dummy.shape}, got {output.shape}"
    )

    # Test 2: boundary enforcement.
    assert output.min().item() >= 0.0, "Output minimum is below 0.0"
    assert output.max().item() <= 1.0, "Output maximum is above 1.0"

    # Test 3: local convergence sanity check on 2 mock images.
    model.train()
    torch.manual_seed(0)  # Fixed seed so this test is reproducible.
    clean = torch.rand(2, in_channels, image_size, image_size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    sanity_noise = min(noise_factor, 0.2)  # Slightly easier noise for quick overfit.

    final_loss = float("inf")
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        noisy = add_gaussian_noise(clean, sanity_noise)
        recon = model(noisy)
        loss = criterion(recon, clean)  # Target is clean, never noisy.
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    if final_loss >= 0.05:
        print(
            f"Preflight sanity test failed: loss={final_loss:.6f} (expected < 0.05)",
            file=sys.stderr,
        )
        sys.exit(1)
