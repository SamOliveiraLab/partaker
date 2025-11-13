from typing import Optional

import cv2
import numpy as np
from pubsub import pub
from skimage.measure import label
from skimage.segmentation import find_boundaries
from skimage.exposure import rescale_intensity

from nd2_analyzer.utils.image_functions import normalize_image, convert_image


class SegmentationService:
    """Service handling image segmentation and cache management"""

    def __init__(self, cache, models, data_getter):
        """
        Args:
            cache: SegmentationCache instance
            models: SegmentationModels instance
            data_getter: Function to retrieve raw images (t, p, c) -> np.ndarray
        """
        self.cache = cache
        self.models = models
        self.get_raw_image = data_getter
        self.roi_mask: Optional[np.ndarray] = None
        self.crop_coordinates = None

        pub.subscribe(self.handle_image_request, "segmented_image_request")
        pub.subscribe(self.on_roi_selected, "roi_selected")
        pub.subscribe(self.on_reset_roi, "roi_reset")
        pub.subscribe(self.on_crop_selected, "crop_selected")
        pub.subscribe(self.on_crop_reset, "crop_reset")

        # Initialize default parameters
        self.overlay_color = (0, 255, 0)  # Green for outlines
        self.label_colormap = "viridis"

    def on_roi_selected(self, mask: np.ndarray) -> None:
        """
        Saves the ROI mask to the service
        """
        self.roi_mask = mask

    def on_reset_roi(self) -> None:
        self.roi_mask = None

    def on_crop_selected(self, coords: list):
        self.crop_coordinates = coords

    def on_crop_reset(self):
        self.crop_coordinates = None

    def handle_image_request(self, time, position, channel, mode, model=None, overlay_channel=None):
        """Handle image requests requiring segmentation"""
        if mode == "normal":
            return

        if not model:
            raise ValueError("Segmentation model must be specified for non-normal modes")

        cache_key = (time, position, channel, model)
        segmented = self.cache.with_model(model)[cache_key]

        if self.crop_coordinates is not None:
            x, y, width, height = self.crop_coordinates
            segmented = segmented[y: y + height, x: x + width]

        if self.roi_mask is not None:
            segmented = self._apply_roi_mask(segmented)

        # Post-process based on mode
        processed_image = self._post_process(
            raw_image=self.get_raw_image(time, position, overlay_channel if overlay_channel is not None else channel),
            segmented=segmented,
            mode=mode,
            channel=overlay_channel if overlay_channel is not None else channel
        )

        pub.sendMessage(
            "image_ready",
            image=processed_image,
            time=time,
            position=position,
            channel=channel,
            mode=mode,
        )

    def _create_overlay(self, raw_image, segmented, channel, contour_thickness=2,
                            hue_shifts_deg=None, sat_scale=1.0, val_scale=1.0):
        """
        Create an HSV-based colorization overlay for a given channel, then
        draw contours from segmentation colors.

        - raw_image: single-channel 8-bit image
        - segmented: segmentation labels
        - channel: integer index of channel
        - hue_shifts_deg: dict channel -> degrees to shift hue (0-360). If None, use a default map.
        - keeps final output as uint8 RGB
        """

        if raw_image.ndim != 2:
            raise AssertionError("Overlay path expects a single-channel raw image")

        raw_image = normalize_image(raw_image)
        raw_8bit = convert_image(raw_image, np.uint8)

        # Converting to color based on the channel
        rgb_tinted = np.zeros(raw_8bit.shape + (3,), dtype=np.uint8)
        if channel == 0:
            rgb_tinted[:, :, 0] = raw_8bit
            rgb_tinted[:, :, 1] = raw_8bit
            rgb_tinted[:, :, 2] = raw_8bit
        if channel == 1: # mCherry
            rgb_tinted[:, :, 0] = raw_8bit
            rgb_tinted[:, :, 1] = (raw_8bit * 0.1).astype(np.uint8)
            rgb_tinted[:, :, 2] = (raw_8bit * 0.1).astype(np.uint8)
        if channel == 2: # YFP
            rgb_tinted[:, :, 0] = (raw_8bit * 0.8).astype(np.uint8)
            rgb_tinted[:, :, 1] = (raw_8bit * 0.7).astype(np.uint8)
            rgb_tinted[:, :, 2] = (raw_8bit * 0.1).astype(np.uint8)

        if segmented.max() <= 1:
            from skimage.measure import label
            labeled = label(segmented > 0)
        else:
            labeled = segmented

        colored_palette = self._apply_colormap(segmented)
        if colored_palette.dtype != np.uint8:
            colored_palette = colored_palette.astype(np.uint8)

        overlay = rgb_tinted.copy()

        unique_labels = np.unique(labeled)
        for lid in unique_labels:
            if lid == 0:
                continue
            mask = (labeled == lid).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            coords = np.where(labeled == lid)
            if coords[0].size > 0:
                color = tuple(map(int, colored_palette[coords[0][0], coords[1][0], :]))
            else:
                color = (255, 255, 255)

            cv2.drawContours(overlay, contours, -1, color, thickness=contour_thickness)

        return overlay

    def _apply_colormap(self, segmented):
        """Apply distinct colors using HSV (shared logic for labeled mode)"""
        labels = label(segmented) if segmented.max() <= 1 else segmented
        n_labels = labels.max()

        # Generate random hues with fixed high saturation and value
        np.random.seed(42)
        hues = np.random.permutation(n_labels) / n_labels

        # Create color lookup table
        lut = np.zeros((n_labels + 1, 3))
        lut[0] = [0, 0, 0]  # Background is black

        for i in range(1, n_labels + 1):
            h = hues[i - 1]
            s = 0.8 + 0.2 * np.random.rand()
            v = 0.8 + 0.2 * np.random.rand()

            import matplotlib.colors as mcolors
            lut[i] = mcolors.hsv_to_rgb([h, s, v])

        # Map labels to colors
        colored = lut[labels]
        colored = (colored * 255).astype(np.uint8)

        return colored

    def _apply_roi_mask(self, segmented: np.ndarray) -> np.ndarray:
        """
        Apply binary mask to segmentation results.
        Discard segmentations outside the mask and those touching the mask boundary.

        Parameters:
            segmented (np.ndarray): Segmented image with labeled regions

        Returns:
            np.ndarray: Masked segmentation
        """

        is_labeled = True if len(np.unique(segmented)) > 2 else False

        # Convert binary segmentation to labeled regions if needed
        if not is_labeled:
            from skimage.measure import label

            labeled_frame = label(segmented)
        else:
            labeled_frame = segmented

        # Find regions that overlap with the mask boundary
        from scipy.ndimage import binary_dilation

        # Crop roi_mask
        crop_roi_mask = self.roi_mask
        if self.crop_coordinates is not None:
            x, y, width, height = self.crop_coordinates
            crop_roi_mask = crop_roi_mask[y: y + height, x: x + width]

        mask_boundary = binary_dilation(crop_roi_mask) & ~crop_roi_mask

        # Get labels of regions touching the boundary
        boundary_labels = set(np.unique(labeled_frame * mask_boundary))
        if 0 in boundary_labels:
            boundary_labels.remove(0)  # Remove background label

        # Create a new segmentation with only regions inside mask and not touching boundary
        result = np.zeros_like(segmented)
        for label_id in np.unique(labeled_frame):
            if label_id > 0:  # Skip background
                if label_id not in boundary_labels and np.any(
                    (labeled_frame == label_id) & crop_roi_mask
                ):
                    result[labeled_frame == label_id] = (
                        255 if np.max(segmented) <= 255 else label_id
                    )

        return result

    def _post_process(self, raw_image, segmented, mode, channel):
        """Apply final transformations based on display mode

        Returns:
            np.ndarray: uint8 for overlay/labeled modes, preserves dtype for segmented
        """
        if mode == "segmented":
            return segmented  # Preserve original dtype

        if mode == "overlay":
            # Always returns uint8
            return self._create_overlay(raw_image, segmented, channel)

        if mode == "labeled":
            # Always returns uint8
            return self._apply_colormap(segmented)

        raise ValueError(f"Unknown display mode: {mode}")

    def _validate_dtype(self, array: np.ndarray, expected_dtype: np.dtype,
                        array_name: str = "array") -> None:
        """Validate array dtype matches expected"""
        if array.dtype != expected_dtype:
            raise TypeError(
                f"{array_name} has dtype {array.dtype}, expected {expected_dtype}"
            )

    def _ensure_dtype(self, array: np.ndarray, target_dtype: np.dtype,
                      scale: bool = True) -> np.ndarray:
        """Convert array to target dtype with optional scaling"""
        if array.dtype == target_dtype:
            return array

        if scale and np.issubdtype(array.dtype, np.integer) and np.issubdtype(target_dtype, np.integer):
            # Scale between integer types
            src_info = np.iinfo(array.dtype)
            dst_info = np.iinfo(target_dtype)
            scaled = (array.astype(np.float64) - src_info.min) / (src_info.max - src_info.min)
            return (scaled * (dst_info.max - dst_info.min) + dst_info.min).astype(target_dtype)
        else:
            return array.astype(target_dtype)

    def update_parameters(self, overlay_color=None, colormap=None):
        """Update visualization parameters"""
        if overlay_color:
            self.overlay_color = overlay_color
        if colormap:
            self.label_colormap = colormap
