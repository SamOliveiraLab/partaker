"""
segmentation_service.py

Orchestrates segmentation requests: retrieves labeled frames from the cache,
applies post-processing (artifact removal, ROI masking), drives display
rendering, and publishes results on the pubsub bus.

Segmentation contract
---------------------
All labeled arrays are uint16, shape (H, W), pixel value = cell ID (0 = bg).
"""

from __future__ import annotations

import logging

import cv2
import matplotlib.colors as mcolors
import numpy as np
from pubsub import pub
from scipy.ndimage import binary_dilation
from skimage.measure import label, regionprops

from partaker.utils.image_functions import normalize_image, convert_image

logger = logging.getLogger(__name__)


class SegmentationService:
    """Orchestrates segmentation, post-processing, and display rendering."""

    def __init__(self, cache, models, data_getter) -> None:
        """
        Args:
            cache:       SegmentationCache instance.
            models:      SegmentationModels instance.
            data_getter: Callable (t, p, c) -> np.ndarray returning raw images.
        """
        self.cache = cache
        self.models = models
        self.get_raw_image = data_getter
        self.roi_masks: dict = {}
        self.crop_coordinates: list | None = None

        pub.subscribe(self.handle_image_request, "segmented_image_request")
        pub.subscribe(self.on_roi_selected, "roi_selected")
        pub.subscribe(self.on_reset_roi, "roi_reset")
        pub.subscribe(self.on_crop_selected, "crop_selected")
        pub.subscribe(self.on_crop_reset, "crop_reset")

    # ------------------------------------------------------------------
    # PubSub event handlers
    # ------------------------------------------------------------------

    def on_roi_selected(self, mask: np.ndarray, p) -> None:
        self.roi_masks[p] = mask

    def on_reset_roi(self, p) -> None:
        self.roi_masks.pop(p, None)

    def on_crop_selected(self, coords: list) -> None:
        self.crop_coordinates = coords

    def on_crop_reset(self) -> None:
        self.crop_coordinates = None

    # ------------------------------------------------------------------
    # Request handler
    # ------------------------------------------------------------------

    def handle_image_request(
        self,
        time,
        position,
        channel,
        mode: str,
        model: str | None = None,
        overlay_channel: int | None = None,
    ) -> None:
        """Handle a display request that requires segmentation."""
        if mode == "normal":
            return

        if not model:
            raise ValueError(
                "A segmentation model must be specified for non-normal modes."
            )

        labeled = self._get_labeled_frame(time, position, channel, model)

        # Spatial cropping — applied once here so all downstream arrays match.
        roi_mask = None
        if self.crop_coordinates is not None:
            x, y, w, h = self.crop_coordinates
            labeled = labeled[y : y + h, x : x + w]
            if position in self.roi_masks:
                roi_mask = self.roi_masks[position][y : y + h, x : x + w]
        elif position in self.roi_masks:
            roi_mask = self.roi_masks[position]

        if roi_mask is not None:
            labeled = self._apply_roi_mask(labeled, roi_mask)

        active_channel = overlay_channel if overlay_channel is not None else channel
        processed = self._post_process(
            raw_image=self.get_raw_image(time, position, active_channel),
            labeled=labeled,
            mode=mode,
            channel=active_channel,
        )

        pub.sendMessage(
            "image_ready",
            image=processed,
            time=time,
            position=position,
            channel=channel,
            mode=mode,
        )

    def _get_labeled_frame(self, time, position, channel, model: str) -> np.ndarray:
        """
        Return the uint16 labeled frame for (time, position).

        Artifact removal is applied once on first access and written back so
        subsequent reads return the cleaned result immediately.
        ``frame_segmented`` is published exactly once per (time, position, model).
        """
        key = (time, position)

        if not self.cache.is_computed(model, key):
            # First access: cache runs segmentation internally on __getitem__.
            raw = self.cache.with_model(model)[key]  # triggers _compute_and_store
            labeled = self.remove_artifacts(raw)
            self.cache.with_model(model)[key] = labeled  # overwrite with clean version
        else:
            labeled = self.cache.with_model(model)[key]

        _frame = labeled
        # Check if ROI exists, if yes compile it for metrics
        if position in self.roi_masks:
            roi_mask = self.roi_masks[position]
            if roi_mask is not None:
                _frame = self._apply_roi_mask(labeled, roi_mask)

        pub.sendMessage(
            "frame_segmented",
            labeled_frame=_frame,
            time=time,
            position=position,
            channel=channel,
            model=model,
        )

        return labeled

    # ------------------------------------------------------------------
    # Artifact removal
    # ------------------------------------------------------------------

    def remove_artifacts(
        self, labeled: np.ndarray, min_area_ratio: float = 0.2
    ) -> np.ndarray:
        """
        Remove labeled regions whose area is below *min_area_ratio* of the
        mean region area.

        Args:
            labeled:         uint16 label image from SegmentationModels.
            min_area_ratio:  Fraction of mean area below which a region is
                             considered an artifact (default 0.2).

        Returns:
            uint16 label image with artifact regions zeroed out.
        """
        regions = regionprops(labeled.astype(np.int32))
        if not regions:
            return labeled

        mean_area = np.mean([r.area for r in regions])
        threshold = mean_area * min_area_ratio

        result = np.zeros_like(labeled, dtype=np.uint16)
        for r in regions:
            if r.area >= threshold:
                result[labeled == r.label] = r.label

        return result

    # ------------------------------------------------------------------
    # ROI masking
    # ------------------------------------------------------------------

    def _apply_roi_mask(self, labeled: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Zero out cells that fall outside *roi_mask* or touch its boundary.

        Both arrays must already be cropped to the same spatial window.

        Args:
            labeled:  uint16 label image, shape (H, W).
            roi_mask: Binary mask, shape (H, W).

        Returns:
            uint16 label image with out-of-ROI cells zeroed.
        """
        # Defensive re-label if a binary mask somehow arrives here.
        if len(np.unique(labeled)) <= 2:
            labeled = label(labeled > 0).astype(np.uint16)

        boundary = binary_dilation(roi_mask) & ~roi_mask
        boundary_labels = set(np.unique(labeled[boundary])) - {0}

        result = np.zeros_like(labeled, dtype=np.uint16)
        for cell_id in np.unique(labeled):
            if cell_id == 0:
                continue
            if cell_id not in boundary_labels and np.any(
                (labeled == cell_id) & roi_mask
            ):
                result[labeled == cell_id] = cell_id

        return result

    # ------------------------------------------------------------------
    # Colourmap
    # ------------------------------------------------------------------

    def _build_label_color_map(self, labeled: np.ndarray) -> dict[int, tuple]:
        """
        Return {cell_id: (R, G, B)} for all non-zero labels.

        Uses a fixed seed so colours are stable across frames.
        """
        unique_labels = [int(lid) for lid in np.unique(labeled) if lid != 0]
        n = len(unique_labels)
        if n == 0:
            return {}

        rng = np.random.default_rng(42)
        hues = rng.permutation(n) / n

        return {
            lid: tuple(
                (
                    mcolors.hsv_to_rgb(
                        [
                            float(hues[i]),
                            0.8 + 0.2 * rng.random(),
                            0.8 + 0.2 * rng.random(),
                        ]
                    )
                    * 255
                )
                .astype(np.uint8)
                .tolist()
            )
            for i, lid in enumerate(unique_labels)
        }

    def _apply_colormap(self, labeled: np.ndarray) -> np.ndarray:
        """Return a uint8 RGB image where every pixel is coloured by its cell ID."""
        if len(np.unique(labeled)) <= 2:
            labeled = label(labeled > 0).astype(np.uint16)

        color_map = self._build_label_color_map(labeled)
        colored = np.zeros((*labeled.shape, 3), dtype=np.uint8)
        for lid, rgb in color_map.items():
            colored[labeled == lid] = rgb

        return colored

    # ------------------------------------------------------------------
    # Overlay rendering
    # ------------------------------------------------------------------

    def _create_overlay(
        self,
        raw_image: np.ndarray,
        labeled: np.ndarray,
        channel: int,
        contour_thickness: int = 2,
    ) -> np.ndarray:
        """
        Tint *raw_image* by channel colour and draw per-cell contours on top.

        Args:
            raw_image:         Single-channel 2-D image (any numeric dtype).
            labeled:           uint16 label image (0 = background).
            channel:           Channel index for tinting (0 = grey, 1 = red/mCherry, 2 = yellow/YFP).
            contour_thickness: Contour line width in pixels.

        Returns:
            uint8 RGB image.
        """
        if raw_image.ndim != 2:
            raise ValueError("Overlay requires a single-channel (2-D) raw image.")

        raw_8bit = convert_image(normalize_image(raw_image), np.uint8)

        rgb_tinted = np.zeros((*raw_8bit.shape, 3), dtype=np.uint8)
        if channel == 0:
            rgb_tinted[:, :] = raw_8bit[:, :, np.newaxis]
        elif channel == 1:  # mCherry — red
            rgb_tinted[:, :, 0] = raw_8bit
            rgb_tinted[:, :, 1] = (raw_8bit * 0.1).astype(np.uint8)
            rgb_tinted[:, :, 2] = (raw_8bit * 0.1).astype(np.uint8)
        elif channel == 2:  # YFP — yellow
            rgb_tinted[:, :, 0] = (raw_8bit * 0.8).astype(np.uint8)
            rgb_tinted[:, :, 1] = (raw_8bit * 0.7).astype(np.uint8)
            rgb_tinted[:, :, 2] = (raw_8bit * 0.1).astype(np.uint8)

        if len(np.unique(labeled)) <= 2:
            labeled = label(labeled > 0).astype(np.uint16)

        color_map = self._build_label_color_map(labeled)
        overlay = rgb_tinted.copy()
        for lid, rgb in color_map.items():
            mask = (labeled == lid).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, rgb, thickness=contour_thickness)

        return overlay

    # ------------------------------------------------------------------
    # Post-processing dispatcher
    # ------------------------------------------------------------------

    def _post_process(
        self,
        raw_image: np.ndarray,
        labeled: np.ndarray,
        mode: str,
        channel: int,
    ) -> np.ndarray:
        """
        Returns:
            "segmented": uint16 label array (passthrough)
            "overlay":   uint8 RGB with per-cell contours on tinted background
            "labeled":   uint8 RGB colourmap
        """
        if mode == "segmented":
            return labeled
        if mode == "overlay":
            return self._create_overlay(raw_image, labeled, channel)
        if mode == "labeled":
            return self._apply_colormap(labeled)
        raise ValueError(f"Unknown display mode: {mode!r}")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def update_parameters(self, overlay_color=None, colormap=None) -> None:
        """Kept for API compatibility."""
        pass
