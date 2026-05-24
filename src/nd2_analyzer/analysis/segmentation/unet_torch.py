"""
unet_torch.py

PyTorch U-Net with a Keras-compatible interface.

Usage (drop-in replacement for the TF unet_segmentation):

    from .unet_torch import unet_segmentation

    model = unet_segmentation(
        input_size=(512, 512, 1),
        pretrained_weights="./checkpoints/my_model.pt",
    )
    output = model.predict(images_nhwc)   # numpy (N, H, W, 1), float32 in [0,1]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _SamePad2x2(nn.Module):
    """Replicates Keras Conv2D kernel-size-2 'same' padding (pad right + bottom)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 1, 0, 1))


def _he(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def _conv3(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(_he(nn.Conv2d(in_ch, out_ch, 3, padding=1)), nn.ReLU(inplace=True))


def _conv2_same(in_ch: int, out_ch: int) -> nn.Sequential:
    """2×2 conv with 'same' padding — matches the Keras UpSampling conv."""
    return nn.Sequential(_SamePad2x2(), _he(nn.Conv2d(in_ch, out_ch, 2, padding=0)), nn.ReLU(inplace=True))


class _ContractingBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = _conv3(in_ch, out_ch)
        self.conv2 = _conv3(out_ch, out_ch)
        self.drop: Optional[nn.Module] = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.drop(x) if self.drop else x


class _ExpandingBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = _conv2_same(in_ch, out_ch)
        self.conv2 = _conv3(out_ch + skip_ch, out_ch)
        self.conv3 = _conv3(out_ch, out_ch)
        self.drop: Optional[nn.Module] = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv1(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.drop(x) if self.drop else x


# ---------------------------------------------------------------------------
# U-Net core
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """
    Generic U-Net, architecture-equivalent to the Keras version.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale).
    output_classes : int
        Number of output channels.
    dropout : float
        Dropout rate in contracting/expanding blocks (0 = disabled).
    levels : int
        Total depth of the network (including the base level-0 block).
    """

    def __init__(
        self,
        in_channels: int = 1,
        output_classes: int = 1,
        dropout: float = 0.0,
        levels: int = 5,
    ) -> None:
        super().__init__()
        base = 64

        # Level-0 input block (no pooling)
        self.level0_conv1 = _conv3(in_channels, base)
        self.level0_conv2 = _conv3(base, base)

        # Contracting path
        self.contracting = nn.ModuleList()
        fi = base
        for _ in range(1, levels):
            fo = fi * 2
            self.contracting.append(_ContractingBlock(fi, fo, dropout))
            fi = fo

        # Expanding path
        self.expanding = nn.ModuleList()
        for _ in range(levels - 1):
            self.expanding.append(_ExpandingBlock(fi, fi // 2, fi // 2, dropout))
            fi = fi // 2

        # Final 1×1 output conv (sigmoid applied in forward)
        self.final_conv = _he(nn.Conv2d(fi, output_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.level0_conv1(x)
        x = self.level0_conv2(x)

        skips = [x]
        for block in self.contracting:
            x = block(x)
            skips.append(x)

        x = skips.pop()
        for block in self.expanding:
            x = block(x, skips.pop())

        return torch.sigmoid(self.final_conv(x))


# ---------------------------------------------------------------------------
# Keras-compatible wrapper
# ---------------------------------------------------------------------------


class UNetSegmentation:
    """
    Thin wrapper around :class:`UNet` that mirrors the Keras model interface
    used throughout the project:

    * ``model.predict(images_nhwc)`` — accepts numpy ``(N, H, W, C)`` float32,
      returns numpy ``(N, H, W, 1)`` float32.
    * Loads weights from a ``.pt`` state-dict file saved by ``convert.py``.
    * Automatically uses CUDA if available, falls back to CPU.

    Parameters
    ----------
    input_size : (H, W, C)
        Input tensor shape (excluding batch).  Only C is used to configure the
        network; H/W are handled dynamically by the convolutional layers.
    pretrained_weights : str or Path, optional
        Path to a ``.pt`` file produced by ``convert.py``.
    levels : int
        Number of U-Net levels (must match the value used during training).
    dropout : float
        Dropout rate (0 = disabled; keep at 0 for inference).
    device : str or torch.device, optional
        Override the auto-detected device (e.g. ``"cpu"``, ``"cuda:1"``).
    batch_size : int
        Number of images processed per forward pass in :meth:`predict`.
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (256, 32, 1),
        pretrained_weights: Optional[str] = None,
        levels: int = 5,
        dropout: float = 0.0,
        device: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        in_channels = input_size[2]

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = UNet(
            in_channels=in_channels,
            output_classes=1,
            dropout=dropout,
            levels=levels,
        ).to(self.device)

        if pretrained_weights is not None:
            self.load_weights(pretrained_weights)

        self._model.eval()
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_weights(self, path: str) -> None:
        """Load a PyTorch state-dict ``.pt`` file."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of images.

        Parameters
        ----------
        images : np.ndarray
            Shape ``(N, H, W, C)`` or ``(H, W, C)``, dtype float32.
            Values should be in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Predictions with shape ``(N, H, W, 1)`` (or ``(H, W, 1)`` if a
            single image was passed), dtype float32 in ``[0, 1]``.
        """
        squeeze = images.ndim == 3
        if squeeze:
            images = images[np.newaxis]  # add batch dim

        images = images.astype(np.float32)
        n = images.shape[0]
        results: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, n, self._batch_size):
                chunk = images[start : start + self._batch_size]
                # NHWC → NCHW
                tensor = torch.from_numpy(chunk.transpose(0, 3, 1, 2)).to(self.device)
                out = self._model(tensor)  # (N, 1, H, W)
                # NCHW → NHWC
                results.append(out.cpu().numpy().transpose(0, 2, 3, 1))

        output = np.concatenate(results, axis=0)
        return output[0] if squeeze else output

    # Keep a summary method for convenience
    def summary(self) -> None:
        """Print a brief parameter count summary."""
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print(f"UNetSegmentation — total params: {total:,}  trainable: {trainable:,}")
        print(f"Device: {self.device}")


# ---------------------------------------------------------------------------
# Factory function — exact same signature as the Keras unet_segmentation()
# ---------------------------------------------------------------------------


def unet_segmentation(
    pretrained_weights: Optional[str] = None,
    input_size: Tuple[int, int, int] = (256, 32, 1),
    levels: int = 5,
    dropout: float = 0.0,
    device: Optional[str] = None,
    batch_size: int = 8,
) -> UNetSegmentation:
    """
    Build (and optionally load) a PyTorch U-Net for segmentation.

    Drop-in replacement for the Keras ``unet_segmentation()``.

    Parameters
    ----------
    pretrained_weights : str, optional
        Path to a ``.pt`` state-dict saved by ``convert.py``.
    input_size : (H, W, C)
        Input shape, excluding batch.
    levels : int
        U-Net depth.
    dropout : float
        Dropout rate (0 = off).
    device : str, optional
        PyTorch device string.  Auto-detected when *None*.
    batch_size : int
        Chunk size for :meth:`UNetSegmentation.predict`.

    Returns
    -------
    UNetSegmentation
        Ready-to-use model with a ``.predict()`` interface.
    """
    return UNetSegmentation(
        input_size=input_size,
        pretrained_weights=pretrained_weights,
        levels=levels,
        dropout=dropout,
        device=device,
        batch_size=batch_size,
    )

