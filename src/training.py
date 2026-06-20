import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.device import use_amp
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
    model.train()
    running_loss = 0.0
    sample_count = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (clean, _) in enumerate(progress, start=1):
        clean = clean.to(device, non_blocking=True)
        noisy = add_gaussian_noise(clean, noise_factor)

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and scaler is not None:
            with torch.amp.autocast("cuda"):
                recon = model(noisy)
                loss = criterion(recon, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon = model(noisy)
            loss = criterion(recon, clean)
            loss.backward()
            optimizer.step()

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

        batch_size = clean.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / sample_count
