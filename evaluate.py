import argparse
from pathlib import Path

import torch
import torch.nn as nn

from config import Config
from src.dataset import get_dataloaders
from src.device import get_device, use_amp
from src.model import DenoisingAutoencoder
from src.noise import add_gaussian_noise
from src.visualize import save_reconstruction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DAE and save reconstruction grid.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation.png"),
    )
    parser.add_argument("--num-samples", type=int, default=10)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = Config()
    device = get_device()
    amp_enabled = use_amp(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = checkpoint.get("config", {})
    config.dataset = saved.get("dataset", config.dataset)
    config.noise_factor = saved.get("noise_factor", config.noise_factor)
    config.batch_size = saved.get("batch_size", config.batch_size)

    _, test_loader = get_dataloaders(
        dataset_name=config.dataset,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
    )

    model = DenoisingAutoencoder(in_channels=config.in_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    clean, _ = next(iter(test_loader))
    clean = clean.to(device, non_blocking=True)
    noisy = add_gaussian_noise(clean, config.noise_factor)

    if amp_enabled:
        with torch.amp.autocast("cuda"):
            denoised = model(noisy)
    else:
        denoised = model(noisy)

    mse = nn.MSELoss()(denoised, clean).item()
    print(f"Test MSE: {mse:.6f}")

    save_reconstruction_grid(
        noisy[: args.num_samples],
        denoised[: args.num_samples],
        clean[: args.num_samples],
        args.output,
        num_samples=args.num_samples,
    )
    print(f"Saved grid to {args.output}")
    print("Open it with: open outputs/evaluation.png")


if __name__ == "__main__":
    main()
