import numpy as np
from .segmentation_models import SegmentationModels


def slices_to_indices(slice_indices):

    shape = tuple(s.stop - s.start for s in slice_indices)

    for index in np.ndindex(shape):
        actual_index = tuple(
            slice_indices[i].start +
            index[i] for i in range(
                len(slice_indices)))
        yield actual_index


class SegmentationCache:
    def __init__(self, nd2_data):
        self.nd2_data = nd2_data
        self.shape = nd2_data.shape
        self.mmap_arrays_idx = {}
        self.model_name = None
        self.seg = SegmentationModels()

    def with_model(self, model_name):
        self.model_name = model_name
        if model_name not in self.mmap_arrays_idx:
            self.mmap_arrays_idx[model_name] = (
                np.zeros(self.shape, dtype=np.uint8), set())
        return self

    def set_binary_mask(self, binary_mask):
        """
        Set a binary mask to crop segmentation results.
        
        Parameters:
            binary_mask (np.ndarray): Binary mask where True/1 indicates regions to keep
        """
        if binary_mask.shape != self.shape[-2:]:  # Check if mask matches image dimensions
            raise ValueError(f"Binary mask shape {binary_mask.shape} does not match image dimensions {self.shape[-2:]}")
        self.binary_mask = binary_mask
        return self

    def apply_binary_mask(self, segmented_frame):
        """
        Apply binary mask to segmentation results.
        Discard segmentations outside the mask and those touching the mask boundary.
        
        Parameters:
            segmented_frame (np.ndarray): Segmented image with labeled regions
        
        Returns:
            np.ndarray: Masked segmentation
        """

        is_labeled = True if len(np.unique(segmented_frame)) > 2 else False

        # Convert binary segmentation to labeled regions if needed
        if not is_labeled:
            from skimage.measure import label
            labeled_frame = label(segmented_frame)
        else:
            labeled_frame = segmented_frame
        
        # Find regions that overlap with the mask boundary
        from scipy.ndimage import binary_dilation
        mask_boundary = binary_dilation(self.binary_mask) & ~self.binary_mask
        
        # Get labels of regions touching the boundary
        boundary_labels = set(np.unique(labeled_frame * mask_boundary))
        if 0 in boundary_labels:
            boundary_labels.remove(0)  # Remove background label
        
        # Create a new segmentation with only regions inside mask and not touching boundary
        result = np.zeros_like(segmented_frame)
        for label_id in np.unique(labeled_frame):
            if label_id > 0:  # Skip background
                if label_id not in boundary_labels and np.any((labeled_frame == label_id) & self.binary_mask):
                    result[labeled_frame == label_id] = 255 if np.max(segmented_frame) <= 255 else label_id
        
        return result

    def remove_artifacts(self, segmented_frame):
        """
        Remove artifacts based on cell area.
        Keep only regions with area >= 20% of the most common cell area.
        
        Parameters:
            segmented_frame (np.ndarray): Segmented image
        
        Returns:
            np.ndarray: Cleaned segmentation with artifacts removed
        """
        from skimage.measure import label, regionprops
        from scipy import stats
        
        # Convert binary segmentation to labeled regions if needed
        if np.max(segmented_frame) <= 1:
            labeled_frame = label(segmented_frame)
        else:
            labeled_frame = segmented_frame.copy()
        
        # Calculate areas of all regions
        regions = regionprops(labeled_frame)
        if not regions:
            return segmented_frame  # No regions found
        
        areas = [region.area for region in regions]
        
        # Find the most common cell area (mode)
        if len(areas) > 1:
            mode_area = stats.mode(areas, keepdims=True)[0][0]
        else:
            mode_area = areas[0]  # If only one region, use its area
        
        # Set area threshold as 20% of the mode area
        area_threshold = mode_area * 0.2
        
        # Create a new segmentation with only regions above the threshold
        result = np.zeros_like(segmented_frame)
        for region in regions:
            if region.area >= area_threshold:
                if np.max(segmented_frame) <= 255:
                    result[labeled_frame == region.label] = 255
                else:
                    result[labeled_frame == region.label] = region.label
        
        return result


    def __getitem__(self, slice_indices):
        if self.model_name is None:
            raise ValueError(
                "Model name must be set using with_model() before accessing data.")

        if isinstance(
                slice_indices,
                tuple) and all(
                isinstance(
                i,
                int) for i in slice_indices):
            slice_indices = tuple(slice(i, i + 1) for i in slice_indices)

        mmap_array, indices = self.mmap_arrays_idx[self.model_name]

        for actual_index in slices_to_indices(slice_indices):
            if actual_index not in indices:
                frame = self.nd2_data[actual_index].compute()
                indices.add(actual_index)
                segmented_frame = self.seg.segment_images(
                    np.array([frame]), mode=self.model_name)[0]
            
                # Apply binary mask if available
                if hasattr(self, 'binary_mask') and self.binary_mask is not None:
                    segmented_frame = self.apply_binary_mask(segmented_frame)
                    
                # Artifact removal
                segmented_frame = self.remove_artifacts(segmented_frame)
            
                mmap_array[actual_index][:, :] = np.copy(segmented_frame)
        return mmap_array[slice_indices][0, 0, 0]
