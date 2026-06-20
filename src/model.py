"""Convolutional denoising autoencoder architecture."""

import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """
    Noisy image -> encoder (compress) -> decoder (reconstruct) -> denoised image.

    The model never sees clean images during the forward pass; it only receives
    noisy inputs and learns to output something close to the original clean target.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Encoder: shrink spatial size with strided convolutions (no MaxPool).
        # 28x28 -> 14x14 -> 7x7 for MNIST.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder: grow spatial size back with transposed convolutions.
        # 7x7 -> 14x14 -> 28x28 for MNIST.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),  # Force outputs into [0, 1] to match image pixel range.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run noisy input through encoder then decoder."""
        return self.decoder(self.encoder(x))
