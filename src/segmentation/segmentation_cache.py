from itertools import product
import numpy as np
from .segmentation_models import SegmentationModels

class SegmentationCache:
    def __init__(self, nd2_data):
        self.nd2_data = nd2_data
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

    def __getitem__(self, key):
        if self.model_name is None:
            raise ValueError("Model name must be set using with_model() before accessing data.")

        # Get the memory-mapped array and indices for current model
        mmap_array, indices = self.mmap_arrays_idx[self.model_name]
        
        # Convert various index types to standardized form
        key = self._normalize_index(key)
        
        # Calculate actual shape and indices
        requested_shape = self._get_requested_shape(key)
        all_indices = self._expand_indices(key)
        
        # Process unprocessed frames
        for idx in all_indices:
            if idx not in indices:
                self._process_frame(mmap_array, indices, idx)
        
        # Return data with proper dimensions
        return np.squeeze(mmap_array[key])

    def _normalize_index(self, key):
        """Convert various index types to tuple of slice objects"""
        if not isinstance(key, tuple):
            key = (key,)
        
        normalized = []
        for k in key:
            if isinstance(k, int):
                # Convert single integers to slices to maintain dimensions
                normalized.append(slice(k, k+1))
            elif isinstance(k, Ellipsis.__class__):  # Handle ellipsis
                remaining_dims = self.ndim - len(key) + 1
                normalized.extend([slice(None)] * remaining_dims)
            else:
                normalized.append(k)
        
        # Fill missing dimensions with full slices
        while len(normalized) < self.ndim:
            normalized.append(slice(None))
            
        return tuple(normalized[:self.ndim])

    def _get_requested_shape(self, key):
        """Calculate the shape of the requested array portion"""
        shape = []
        for k, dim_size in zip(key, self.shape):
            if isinstance(k, slice):
                shape.append(len(range(*k.indices(dim_size))))
            elif isinstance(k, int):
                shape.append(1)
            else:
                raise IndexError(f"Unsupported index type: {type(k)}")
        return tuple(shape)

    def _expand_indices(self, key):
        """Generate all actual indices from slices"""
        indices = []
        for dim_slice, dim_size in zip(key, self.shape):
            if isinstance(dim_slice, slice):
                start, stop, step = dim_slice.indices(dim_size)
                indices.append(range(start, stop, step))
            elif isinstance(dim_slice, int):
                indices.append([dim_slice])
            else:
                raise IndexError(f"Unsupported index type: {type(dim_slice)}")
        
        return product(*indices)

    def _process_frame(self, mmap_array, indices, idx):
        """Process and cache a single frame"""
        # Convert negative indices to positive
        idx = tuple(i % s for i, s in zip(idx, self.shape))
        
        try:
            frame = self.nd2_data[idx].compute()
            segmented_frame = self.seg.segment_images([frame], mode=self.model_name)[0]
            
            if hasattr(self, 'binary_mask') and self.binary_mask is not None:
                segmented_frame = self.apply_binary_mask(segmented_frame)
                
            segmented_frame = self.remove_artifacts(segmented_frame)
            mmap_array[idx] = segmented_frame
            indices.add(idx)
        except Exception as e:
            raise IndexError(f"Failed to process frame {idx}") from e

    @property
    def shape(self):
        return self.nd2_data.shape

    @property
    def ndim(self):
        return self.nd2_data.ndim - 2 # Since the last 2 will be image X and Y

    @property
    def dtype(self):
        return self.mmap_array.dtype

    def __array__(self):
        return self.mmap_array[:]
