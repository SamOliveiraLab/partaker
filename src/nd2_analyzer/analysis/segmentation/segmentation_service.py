from typing import Optional

import cv2
import numpy as np
from pubsub import pub
from skimage.measure import label
from skimage.segmentation import find_boundaries


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

        pub.subscribe(self.handle_image_request, "segmented_image_request")
        pub.subscribe(self.on_roi_selected, "roi_selected")
        pub.subscribe(self.on_reset_roi, "roi_reset")

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

    def handle_image_request(self, time, position, channel, mode, model=None):
        """Handle image requests requiring segmentation"""
        if mode == "normal":
            return  # Let ImageData handle normal requests

        if not model:
            raise ValueError(
                "Segmentation model must be specified for non-normal modes"
            )

        # Request cached, which will segment if not present
        cache_key = (time, position, channel, model)
        segmented = self.cache.with_model(model)[cache_key]

        if self.roi_mask is not None:
            segmented = self._apply_roi_mask(segmented)

        # Post-process based on mode
        processed_image = self._post_process(
            raw_image=self.get_raw_image(time, position, channel),
            segmented=segmented,
            mode=mode,
        )

        pub.sendMessage(
            "image_ready",
            image=processed_image,
            time=time,
            position=position,
            channel=channel,
            mode=mode,
        )

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

        mask_boundary = binary_dilation(self.roi_mask) & ~self.roi_mask

        # Get labels of regions touching the boundary
        boundary_labels = set(np.unique(labeled_frame * mask_boundary))
        if 0 in boundary_labels:
            boundary_labels.remove(0)  # Remove background label

        # Create a new segmentation with only regions inside mask and not touching boundary
        result = np.zeros_like(segmented)
        for label_id in np.unique(labeled_frame):
            if label_id > 0:  # Skip background
                if label_id not in boundary_labels and np.any(
                    (labeled_frame == label_id) & self.roi_mask
                ):
                    result[labeled_frame == label_id] = (
                        255 if np.max(segmented) <= 255 else label_id
                    )

        return result

    def _post_process(self, raw_image, segmented, mode):
        """Apply final transformations based on display mode"""
        if mode == "segmented":
            return segmented

        if mode == "overlay":
            return self._create_overlay(raw_image, segmented)

        if mode == "labeled":
            return self._apply_colormap(segmented)

        raise ValueError(f"Unknown display mode: {mode}")

    # TODO: move this logic to view area, as this is just visual
    def _create_overlay(self, raw_image, segmented):
        """Create overlay of segmentation outlines on raw image"""
        # Convert raw image to RGB if needed
        if len(raw_image.shape) == 2:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)

        # Find segmentation boundaries
        boundaries = find_boundaries(segmented, mode="inner")

        # Apply overlay
        overlay = raw_image.copy()
        overlay[boundaries] = self.overlay_color
        return overlay

    def _apply_colormap(self, segmented):
        """Apply distinct colors using HSV"""
        import numpy as np
        import matplotlib.colors as mcolors
        from skimage.measure import label

        labels = label(segmented)
        n_labels = labels.max()

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

        return colored

    def update_parameters(self, overlay_color=None, colormap=None):
        """Update visualization parameters"""
        if overlay_color:
            self.overlay_color = overlay_color
        if colormap:
            self.label_colormap = colormap
