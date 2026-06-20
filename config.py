from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    dataset: str = "mnist"
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")

    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    noise_factor: float = 0.3
    log_every_n_batches: int = 50
    seed: int = 42

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def in_channels(self) -> int:
        return 1 if self.dataset == "mnist" else 3

    @property
    def image_size(self) -> int:
        return 28 if self.dataset == "mnist" else 32
