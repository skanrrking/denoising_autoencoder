"""Default hyperparameters and paths for training/evaluation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Which dataset to download and train on.
    dataset: str = "mnist"
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")

    # Training settings.
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    noise_factor: float = 0.3  # Gaussian noise strength added during training.
    log_every_n_batches: int = 50  # Throttle batch-level logging to avoid I/O lag.
    seed: int = 42

    def ensure_dirs(self) -> None:
        """Create folders for downloaded data and saved model weights."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def in_channels(self) -> int:
        # MNIST is grayscale (1 channel); CIFAR-10 is RGB (3 channels).
        return 1 if self.dataset == "mnist" else 3

    @property
    def image_size(self) -> int:
        # MNIST images are 28x28; CIFAR-10 images are 32x32.
        return 28 if self.dataset == "mnist" else 32
