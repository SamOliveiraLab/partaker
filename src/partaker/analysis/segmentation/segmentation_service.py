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
from skimage.segmentation import find_boundaries

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

        # Initialize default parameters
        self.overlay_color = (0, 255, 0)  # Green for outlines
        self.label_colormap = "viridis"

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

        key = (time, position)
        frame_was_computed = self.cache.is_computed(model, key)
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

        if mode == "segmented" and not frame_was_computed:
            pub.sendMessage(
                "frame_segmented",
                labeled_frame=labeled,
                time=time,
                position=position,
                channel=channel,
                model=model,
            )

    def _get_labeled_frame(self, time, position, channel, model: str) -> np.ndarray:
        """Return the uint16 labeled frame for (time, position)."""
        model_cache = self.cache.with_model(model)
        cache_key = (time, position)

        # This retrieves an existing segmentation or generates and caches a new one.
        segmented = np.asarray(model_cache[cache_key])

        return segmented

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
        """Apply distinct colors using HSV"""
        # Check if segmentation is already labeled (OmniPose/Cellpose)
        # or binary (UNET)
        max_value = labeled.max()
        unique_values = len(np.unique(labeled))

        # If max value > 255 or many unique values, it's already labeled
        # Binary masks typically have only 0 and 255 (or 0 and 1)
        if max_value > 255 or unique_values > 100:
            labels = labeled
            n_labels = labels.max()
        else:
            labels = label(labeled)
            n_labels = labels.max()

        if n_labels == 0:
            return np.zeros((*labels.shape, 3), dtype=np.uint8)

        # Generate random hues with fixed high saturation and value for vivid colors
        np.random.seed(42)  # Optional: for reproducibility
        hues = np.random.permutation(n_labels) / n_labels  # Evenly distributed hues

        # Create color lookup table
        lut = np.zeros((n_labels + 1, 3))
        lut[0] = [0, 0, 0]  # Background is black

        for i in range(1, n_labels + 1):
            # HSV: random hue, high saturation (0.8-1.0), high value (0.8-1.0)
            h = hues[i - 1]
            s = 0.8 + 0.2 * np.random.rand()  # Saturation 80-100%
            v = 0.8 + 0.2 * np.random.rand()  # Brightness 80-100%
            lut[i] = mcolors.hsv_to_rgb([h, s, v])

        # Map labels to colors
        colored = lut[labels]
        colored = (colored * 255).astype(np.uint8)

        # Add cell ID text labels on each cell
        regions = regionprops(labels)

        for region in regions:
            cell_id = region.label
            centroid_y, centroid_x = region.centroid

            # Convert to integer coordinates
            x, y = int(centroid_x), int(centroid_y)

            # Add white text with black outline for visibility
            text = str(cell_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1

            # Get text size to center it better
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Adjust position to center text
            text_x = x - text_width // 2
            text_y = y + text_height // 2

            # Draw black outline (thicker)
            cv2.putText(
                colored,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 1,
                cv2.LINE_AA,
            )
            # Draw white text on top
            cv2.putText(
                colored,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        print(f"  ✅ Added {len(regions)} cell ID labels")

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
        """Create overlay of segmentation outlines on raw image"""
        # Convert raw image to RGB if needed
        if len(raw_image.shape) == 2:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)

        # Find segmentation boundaries
        boundaries = find_boundaries(labeled, mode="inner")

        # Apply overlay
        overlay = raw_image.copy()
        overlay[boundaries] = self.overlay_color
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
        """Update visualization parameters"""
        if overlay_color:
            self.overlay_color = overlay_color
        if colormap:
            self.label_colormap = colormap