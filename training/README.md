# Training code (not part of the installed package)

This folder contains the original **TensorFlow/Keras** code used to define and train
the custom U-Net segmentation model described in the manuscript. It is kept here for
reproducibility and is **not** part of the installed `partaker` package: Partaker is an
analysis/inference tool and does not train models at runtime, so the shipped package
depends only on PyTorch and does **not** require TensorFlow.

## Files

- `unet_keras.py` — the TensorFlow/Keras U-Net architecture, the custom pixel-wise
  weighted binary cross-entropy loss, and the model/compile setup used for training.

## Workflow

The model was trained offline in TensorFlow/Keras, producing Keras `.h5` weights. Those
weights were then converted to a PyTorch state-dict (`.pt`) for inference in the released
package:

```
TensorFlow/Keras training (training/unet_keras.py)
        │  produces
        ▼
   Keras weights (.h5)
        │  converted by src/partaker/analysis/segmentation/convert.py
        ▼
  PyTorch weights (.pt)   ← loaded at inference by unet_torch.py
```

Both the `.h5` and `.pt` weights, and the training/benchmark datasets, are archived on
Zenodo: https://doi.org/10.5281/zenodo.20577330

## Requirements

Running this training code requires TensorFlow/Keras, which are **not** installed by the
`partaker` package. Install them separately only if you wish to retrain the model. The
`partaker` application itself, including U-Net inference, runs entirely on PyTorch.
