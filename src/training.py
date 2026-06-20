"""One epoch of training and one pass of validation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.noise import add_gaussian_noise


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    noise_factor: float,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
    log_every_n_batches: int,
) -> float:
    """Train for one epoch and return average MSE against clean targets."""
    model.train()
    running_loss = 0.0
    sample_count = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (clean, _) in enumerate(progress, start=1):
        # Step 1: Move clean batch to GPU/CPU asynchronously.
        clean = clean.to(device, non_blocking=True)

        # Step 2: Add noise on the same device right before the forward pass.
        noisy = add_gaussian_noise(clean, noise_factor)

        # Step 3: Clear old gradients without zeroing memory in place.
        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and scaler is not None:
            # CUDA path: mixed precision forward + scaled backward pass.
            with torch.amp.autocast("cuda"):
                recon = model(noisy)
                loss = criterion(recon, clean)  # Compare to clean, never noisy.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # MPS/CPU path: standard full-precision training.
            recon = model(noisy)
            loss = criterion(recon, clean)
            loss.backward()
            optimizer.step()

        # Step 4: Track scalar loss only (.item()) to avoid VRAM leaks.
        batch_size = clean.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

        if batch_idx % log_every_n_batches == 0:
            progress.set_postfix(loss=f"{loss.item():.6f}")

    return running_loss / sample_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    noise_factor: float,
    amp_enabled: bool,
) -> float:
    """Measure average reconstruction error on the test set."""
    model.eval()
    running_loss = 0.0
    sample_count = 0

    for clean, _ in loader:
        clean = clean.to(device, non_blocking=True)
        noisy = add_gaussian_noise(clean, noise_factor)

        if amp_enabled:
            with torch.amp.autocast("cuda"):
                recon = model(noisy)
                loss = criterion(recon, clean)
        else:
            recon = model(noisy)
            loss = criterion(recon, clean)

        running_loss += loss.item() * clean.size(0)
        sample_count += clean.size(0)

    return running_loss / sample_count
