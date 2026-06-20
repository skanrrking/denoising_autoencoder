"""Load clean image datasets and build high-throughput DataLoaders."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.device import get_num_workers


def get_dataloaders(
    dataset_name: str,
    data_dir: Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """
    Step 1: Convert PIL images to tensors in [0, 1].
    Step 2: Download/load a clean dataset (no noise added here).
    Step 3: Wrap datasets in DataLoaders tuned for fast GPU feeding.
    """
    transform = transforms.ToTensor()
    num_workers = get_num_workers()

    # Shared DataLoader settings for throughput.
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "pin_memory": True,  # Page-locked memory for faster host->device copies.
        "drop_last": True,  # Drop uneven final batch to keep vectorized ops steady.
    }
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = True  # Avoid respawning workers each epoch.

    # Step 2: Pick dataset. Labels are ignored because this is not classification.
    if dataset_name == "mnist":
        train_set = datasets.MNIST(
            root=str(data_dir), train=True, download=True, transform=transform
        )
        test_set = datasets.MNIST(
            root=str(data_dir), train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        train_set = datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=transform
        )
        test_set = datasets.CIFAR10(
            root=str(data_dir), train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'mnist' or 'cifar10'.")

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
