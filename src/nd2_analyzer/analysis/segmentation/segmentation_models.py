"""
segmentation_models.py

Singleton that owns all segmentation backends.

Contract
--------
Every public method returns ``list[np.ndarray]`` where each array is
``uint16``, shape ``(H, W)``, and pixel value equals the cell label
(0 = background, 1..N = individual cells).  Colour mapping is the
responsibility of SegmentationService — this module never touches it.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops

try:
    from cellpose_omni import models as cellpose_models

    _CELLPOSE_AVAILABLE = True
except ImportError:
    _CELLPOSE_AVAILABLE = False

from nd2_analyzer.utils.sliding_prediction import sliding_window_predict
from .unet_torch import unet_segmentation


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Normalise to [0, 1] then apply a light Gaussian denoise.

    Parameters
    ----------
    image : np.ndarray
        Raw input frame (any numeric dtype).

    Returns
    -------
    np.ndarray
        Float32 array in [0, 1].
    """
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        normalised = (image - img_min) / (img_max - img_min)
    else:
        normalised = np.zeros_like(image, dtype=np.float32)

    denoised = gaussian_filter(normalised.astype(np.float32), sigma=1)
    return denoised


# ---------------------------------------------------------------------------
# SegmentationModels singleton
# ---------------------------------------------------------------------------


class SegmentationModels:
    """
    Lazy-loading singleton for all segmentation backends.

    All ``segment_images`` calls return ``list[np.ndarray]`` of ``uint16``
    labeled arrays — pixel value is the cell ID, 0 is background.
    """

    CELLPOSE_BACT_PHASE = "bact_phase_cp3"
    CELLPOSE_BACT_FLUOR = "bact_fluor_cp3"
    OMNIPOSE_BACT_PHASE = "omnipose_bact_phase"
    OMNIPOSE_BACT_FLUOR = "omnipose_bact_fluo"
    UNET = "unet"
    CELLPOSE = "cellpose_deepbacs"

    available_models: list[str] = [
        OMNIPOSE_BACT_PHASE,
        OMNIPOSE_BACT_FLUOR,
        CELLPOSE_BACT_PHASE,
        CELLPOSE_BACT_FLUOR,
        UNET,
        CELLPOSE,
    ]

    UNET_TILE_SIZE: int = 512
    UNET_OVERLAP: int = 64

    _instance: Optional["SegmentationModels"] = None

    def __new__(cls, *args, **kwargs) -> "SegmentationModels":
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._models: dict = {}
        return cls._instance

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def segment_images(
        self,
        images: list[np.ndarray],
        mode: str,
        progress: Optional[Callable] = None,
        preprocess: bool = True,
    ) -> list[np.ndarray]:
        """
        Segment a list of images with the requested model.

        Parameters
        ----------
        images :
            List of 2-D (grayscale) NumPy arrays at their native resolution.
        mode :
            One of the ``SegmentationModels.*`` class constants.
        progress :
            Optional callable or PyQt signal for progress updates.
        preprocess :
            Apply normalise + Gaussian denoise pre-processing when ``True``.

        Returns
        -------
        list[np.ndarray]
            ``uint16`` labeled arrays, shape ``(H, W)``, at the original
            input resolution.  Pixel value = cell ID, 0 = background.
        """
        print(f"[SegmentationModels] Using '{mode}' model")

        if preprocess:
            images = [preprocess_image(img) for img in images]

        if mode == self.UNET:
            results = self._segment_unet(images)

        elif mode in (
            self.CELLPOSE,
            self.CELLPOSE_BACT_PHASE,
            self.CELLPOSE_BACT_FLUOR,
        ):
            model_type_map = {
                self.CELLPOSE: "deepbacs_cp3",
                self.CELLPOSE_BACT_PHASE: "bact_phase_cp3",
                self.CELLPOSE_BACT_FLUOR: "bact_fluor_cp3",
            }
            cellpose_inst = self._get_cellpose_model(mode, model_type_map[mode])
            results = self._segment_cellpose(images, progress, cellpose_inst)

        elif mode in (self.OMNIPOSE_BACT_PHASE, self.OMNIPOSE_BACT_FLUOR):
            model_type_map = {
                self.OMNIPOSE_BACT_PHASE: "bact_phase_omni",
                self.OMNIPOSE_BACT_FLUOR: "bact_fluor_omni",
            }
            omni_model = self._get_omnipose_model(mode, model_type_map[mode])
            results = self._segment_omnipose(images, progress, omni_model)

        else:
            raise ValueError(f"Unknown segmentation mode: '{mode}'")

        self._log_label_count(results)
        return results

    # ------------------------------------------------------------------
    # U-Net — sliding-window at native resolution
    # ------------------------------------------------------------------

    def _segment_unet(self, images: list[np.ndarray]) -> list[np.ndarray]:
        model = self._get_unet_model()
        results: list[np.ndarray] = []

        for img in images:
            img_f = img.squeeze().astype(np.float32)
            if img_f.max() > 1.0:
                img_f = img_f / 255.0

            prob_map = sliding_window_predict(
                model,
                img_f,
                tile_size=self.UNET_TILE_SIZE,
                overlap=self.UNET_OVERLAP,
            )

            binary = (prob_map > 0.5).astype(np.uint8)
            labeled = label(binary).astype(np.uint16)
            print(
                f"[UNet] labeled: dtype={labeled.dtype}, shape={labeled.shape}, n_cells={labeled.max()}"
            )
            results.append(labeled)

        return results

    def _get_unet_model(self):
        if self.UNET not in self._models:
            weights_path = os.environ.get("PARTAKER_UNET_WEIGHTS")
            if not weights_path:
                raise EnvironmentError(
                    "Environment variable 'PARTAKER_UNET_WEIGHTS' is not set. "
                    "Point it to a .pt state-dict produced by convert.py."
                )
            self._models[self.UNET] = unet_segmentation(
                input_size=(self.UNET_TILE_SIZE, self.UNET_TILE_SIZE, 1),
                pretrained_weights=weights_path,
            )
        return self._models[self.UNET]

    # ------------------------------------------------------------------
    # CellPose / DeepBacs
    # ------------------------------------------------------------------

    def _get_cellpose_model(self, key: str, model_type: str):
        if key not in self._models:
            if not _CELLPOSE_AVAILABLE:
                raise ImportError("cellpose_omni is not installed.")
            use_gpu = os.environ.get("PARTAKER_GPU") == "1"
            self._models[key] = cellpose_models.CellposeModel(
                gpu=use_gpu, model_type=model_type
            )
        return self._models[key]

    def _segment_cellpose(
        self,
        images: list[np.ndarray],
        progress: Optional[Callable],
        cellpose_inst,
    ) -> list[np.ndarray]:
        images = [img.squeeze() if img.ndim > 2 else img for img in images]

        try:
            # eval() already returns integer-labeled masks (cell ID per pixel)
            masks, _, _ = cellpose_inst.eval(images, diameter=None, channels=[0, 0])
        except Exception as exc:
            print(f"[SegmentationModels] CellPose error: {exc}")
            return [np.zeros(img.shape[:2], dtype=np.uint16) for img in images]

        self._emit_progress(progress, len(images))

        results = [m.astype(np.uint16) for m in masks]
        print(
            f"[CellPose] labeled: dtype={results[0].dtype}, shape={results[0].shape}, n_cells={results[0].max()}"
        )
        return results

    # ------------------------------------------------------------------
    # OmniPose
    # ------------------------------------------------------------------

    def _get_omnipose_model(self, key: str, model_type: str):
        if key not in self._models:
            if not _CELLPOSE_AVAILABLE:
                raise ImportError("cellpose_omni is not installed.")
            use_gpu = os.environ.get("PARTAKER_GPU") == "1"
            self._models[key] = cellpose_models.CellposeModel(
                gpu=use_gpu, model_type=model_type
            )
        return self._models[key]

    def _segment_omnipose(
        self,
        images: list[np.ndarray],
        progress: Optional[Callable],
        model,
    ) -> list[np.ndarray]:
        masks, _, _ = model.eval(
            images,
            channels=[0, 0],
            rescale=None,
            mask_threshold=-2,
            flow_threshold=0,
            transparency=True,
            omni=True,
            cluster=True,
            resample=True,
            verbose=False,
            tile=False,
            niter=None,
            augment=False,
        )
        self._emit_progress(progress, len(masks))

        results = [m.astype(np.uint16) for m in masks]
        print(
            f"[OmniPose] labeled: dtype={results[0].dtype}, shape={results[0].shape}, n_cells={results[0].max()}"
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_progress(progress, n: int) -> None:
        if progress is None:
            return
        if callable(progress):
            progress(n)
        else:
            progress.emit(n)

    @staticmethod
    def _log_label_count(masks: list[np.ndarray]) -> None:
        if masks:
            n = int(masks[0].max())
            print(f"[SegmentationModels] First frame: {n} cell(s) labeled")

    # ------------------------------------------------------------------
    # Post-processing utilities
    # ------------------------------------------------------------------

    def remove_artifacts_from_mask(
        self, mask: np.ndarray, min_area_ratio: float = 0.2
    ) -> np.ndarray:
        """
        Remove labeled regions whose area is below ``min_area_ratio`` of the
        mean region area.  Input and output are ``uint16`` labeled arrays.
        """
        regions = regionprops(mask.astype(np.int32))
        if not regions:
            return mask

        threshold = np.mean([r.area for r in regions]) * min_area_ratio
        clean = np.zeros_like(mask)
        for r in regions:
            if r.area >= threshold:
                clean[mask == r.label] = r.label
        return clean

    def apply_morphological_erosion(
        self, masks: list[np.ndarray], kernel_size: int = 3
    ) -> list[np.ndarray]:
        """
        Erode each labeled mask with an elliptical structuring element.
        Labels are preserved: erosion is performed per-label on a binary
        scratch image, then the surviving pixels are written back.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        eroded_masks = []
        for mask in masks:
            result = np.zeros_like(mask)
            for cell_id in np.unique(mask):
                if cell_id == 0:
                    continue
                cell_binary = (mask == cell_id).astype(np.uint8)
                cell_eroded = cv2.erode(cell_binary, kernel, iterations=1)
                result[cell_eroded > 0] = cell_id
            eroded_masks.append(result)
        return eroded_masks
