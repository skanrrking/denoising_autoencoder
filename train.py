"""
Main training script.

Flow:
  1. Parse CLI overrides
  2. Run preflight tests
  3. Load clean dataset
  4. Train denoising autoencoder
  5. Save best checkpoint by validation loss
"""

import argparse
import random

import numpy as np
import torch

from config import Config
from src.dataset import get_dataloaders
from src.device import get_device, use_amp
from src.model import DenoisingAutoencoder
from src.preflight import run_preflight_tests
from src.training import evaluate, train_epoch


def set_seed(seed: int) -> None:
    """Make randomness reproducible across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a high-efficiency denoising autoencoder.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--noise-factor", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    # Step 1: Load defaults, then apply any command-line overrides.
    args = parse_args()
    config = Config()

    if args.dataset:
        config.dataset = args.dataset
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.noise_factor:
        config.noise_factor = args.noise_factor

    set_seed(config.seed)
    device = get_device()
    amp_enabled = use_amp(device)
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    # Step 2: Validate model architecture before touching real data.
    print("Running preflight tests...")
    run_preflight_tests(
        in_channels=config.in_channels,
        image_size=config.image_size,
        noise_factor=config.noise_factor,
    )
    print(f"Preflight tests passed. Device: {device} | AMP: {amp_enabled}")

    # Step 3: Build dataloaders and model.
    config.ensure_dirs()
    train_loader, test_loader = get_dataloaders(
        dataset_name=config.dataset,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
    )

    model = DenoisingAutoencoder(in_channels=config.in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    checkpoint_path = config.checkpoint_dir / "best_model.pt"

    # Step 4: Train epoch by epoch.
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            noise_factor=config.noise_factor,
            amp_enabled=amp_enabled,
            scaler=scaler,
            log_every_n_batches=config.log_every_n_batches,
        )
        val_loss = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            noise_factor=config.noise_factor,
            amp_enabled=amp_enabled,
        )
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        # Step 5: Keep only the best checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(config),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
