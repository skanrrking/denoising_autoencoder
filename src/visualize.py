"""Save a visual grid comparing noisy, denoised, and clean images."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_reconstruction_grid(
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    clean: torch.Tensor,
    output_path: Path,
    num_samples: int = 10,
) -> None:
    """
    Build a 3-row image grid:
      row 0 = noisy input fed to the model
      row 1 = model reconstruction
      row 2 = clean ground-truth target
    """
    num_samples = min(num_samples, noisy.size(0))
    is_color = noisy.size(1) == 3
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))

    if num_samples == 1:
        axes = axes.reshape(3, 1)

    for i in range(num_samples):
        for row, images in enumerate([noisy, denoised, clean]):
            img = images[i].detach().cpu()
            if is_color:
                img = img.permute(1, 2, 0).numpy()
                axes[row, i].imshow(img.clip(0, 1))
            else:
                axes[row, i].imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, i].axis("off")

    for row, label in enumerate(["Noisy", "Denoised", "Clean"]):
        axes[row, 0].set_ylabel(label, fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
