import numpy as np
from .segmentation_models import SegmentationModels

def slices_to_indices(slice_indices):

    shape = tuple(s.stop - s.start for s in slice_indices)

    for index in np.ndindex(shape):
        actual_index = tuple(slice_indices[i].start + index[i] for i in range(len(slice_indices)))
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
            self.mmap_arrays_idx[model_name] = (np.zeros(self.shape, dtype=np.uint8), set())
        return self

    def __getitem__(self, slice_indices):
        if self.model_name is None:
            raise ValueError("Model name must be set using with_model() before accessing data.")
        
        if isinstance(slice_indices, tuple) and all(isinstance(i, int) for i in slice_indices):
            slice_indices = tuple(slice(i, i + 1) for i in slice_indices)

        mmap_array, indices = self.mmap_arrays_idx[self.model_name]
        
        for actual_index in slices_to_indices(slice_indices):
            if actual_index not in indices:
                frame = self.nd2_data[actual_index].compute()
                indices.add(actual_index)
                segmented_frame = self.seg.segment_images(np.array([frame]), mode=self.model_name)[0]
                mmap_array[actual_index][:, :] = np.copy(segmented_frame)
        return mmap_array[slice_indices][0, 0, 0]
