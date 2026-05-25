"""
convert.py

Convert a Keras `.h5` U-Net weights file into a PyTorch `.pt` state-dict
compatible with :class:`unet_torch.UNet`.

Usage
-----
    python -m nd2_analyzer.analysis.segmentation.convert \\
        /path/to/weights.h5 [--out /path/to/weights.pt] [--levels 5]

Reads weights directly with ``h5py`` — TensorFlow / Keras are NOT required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch

from .unet_torch import UNet


def _keras_to_torch_kernel(arr: np.ndarray) -> torch.Tensor:
    # Keras Conv2D kernel: (kH, kW, in_ch, out_ch)
    # PyTorch Conv2d weight: (out_ch, in_ch, kH, kW)
    return torch.from_numpy(arr.transpose(3, 2, 0, 1).copy())


def _layer_map(levels: int) -> Dict[str, str]:
    """Map Keras layer name → PyTorch parameter prefix (without `.weight`/`.bias`)."""
    mapping: Dict[str, str] = {
        "Level0_Conv2D_1": "level0_conv1.0",
        "Level0_Conv2D_2": "level0_conv2.0",
        "true_output": "final_conv",
    }

    # Contracting: Keras Level{1..levels-1} → PyTorch contracting[{level-1}]
    for level in range(1, levels):
        idx = level - 1
        mapping[f"Level{level}_Contracting_Conv2D_1"] = f"contracting.{idx}.conv1.0"
        mapping[f"Level{level}_Contracting_Conv2D_2"] = f"contracting.{idx}.conv2.0"

    # Expanding: Keras Level{levels-2..0} → PyTorch expanding[0..levels-2]
    # i.e. PyTorch index = (levels - 2) - keras_level
    for level in range(levels - 1):
        idx = (levels - 2) - level
        # Conv2D_1 is the 2x2 conv after upsample — Sequential[Pad, Conv, ReLU], conv at index 1
        mapping[f"Level{level}_Expanding_Conv2D_1"] = f"expanding.{idx}.conv1.1"
        mapping[f"Level{level}_Expanding_Conv2D_2"] = f"expanding.{idx}.conv2.0"
        mapping[f"Level{level}_Expanding_Conv2D_3"] = f"expanding.{idx}.conv3.0"

    return mapping


def convert(h5_path: Path, pt_path: Path, levels: int = 5, in_channels: int = 1) -> None:
    mapping = _layer_map(levels)
    model = UNet(in_channels=in_channels, output_classes=1, dropout=0.0, levels=levels)
    state_dict = model.state_dict()

    copied: set[str] = set()
    with h5py.File(h5_path, "r") as f:
        for keras_name, torch_prefix in mapping.items():
            if keras_name not in f:
                raise KeyError(f"Layer '{keras_name}' missing from {h5_path}")
            # Keras nests one more group with the same name.
            grp = f[keras_name][keras_name]
            kernel = np.asarray(grp["kernel:0"])
            bias = np.asarray(grp["bias:0"])

            weight_key = f"{torch_prefix}.weight"
            bias_key = f"{torch_prefix}.bias"
            if weight_key not in state_dict or bias_key not in state_dict:
                raise KeyError(
                    f"Target params '{weight_key}'/'{bias_key}' not in PyTorch model "
                    f"(check `levels` matches the Keras model)."
                )

            target_weight = state_dict[weight_key]
            converted_weight = _keras_to_torch_kernel(kernel)
            if converted_weight.shape != target_weight.shape:
                raise ValueError(
                    f"Shape mismatch for {keras_name} → {weight_key}: "
                    f"got {tuple(converted_weight.shape)}, expected {tuple(target_weight.shape)}"
                )
            state_dict[weight_key] = converted_weight
            state_dict[bias_key] = torch.from_numpy(bias.copy())
            copied.add(weight_key)
            copied.add(bias_key)

    expected = set(state_dict.keys())
    missing = expected - copied
    if missing:
        raise RuntimeError(f"Did not copy weights for {len(missing)} params: {sorted(missing)[:5]}…")

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, pt_path)
    print(f"Wrote {pt_path}  ({sum(v.numel() for v in state_dict.values()):,} params)")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Keras .h5 U-Net weights → PyTorch .pt state-dict")
    p.add_argument("h5", type=Path, help="Input Keras .h5 weights file")
    p.add_argument("--out", type=Path, default=None, help="Output .pt path (default: alongside .h5)")
    p.add_argument("--levels", type=int, default=5, help="U-Net depth (must match training)")
    p.add_argument("--in-channels", type=int, default=1, help="Input channels (1 for grayscale)")
    args = p.parse_args()

    out = args.out or args.h5.with_suffix(".pt")
    convert(args.h5, out, levels=args.levels, in_channels=args.in_channels)


if __name__ == "__main__":
    main()
