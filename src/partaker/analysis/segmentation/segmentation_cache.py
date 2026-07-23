import json
import os
from itertools import product

import cv2
import h5py
import numpy as np
from pubsub import pub

from .segmentation_models import SegmentationModels

# from .losses import pixelwise_weighted_binary_crossentropy_seg


class SegmentationCache:
    def __init__(self, image_data, phc_channel=0):
        self.image_data = image_data
        self.phc_channel = phc_channel
        self.mmap_arrays_idx = {}
        self.model_name = None

        pub.subscribe(self.invalidate_all, "invalidate_segmentation_cache")

    def with_model(self, model_name):
        self.model_name = model_name
        if model_name not in self.mmap_arrays_idx:
            self.mmap_arrays_idx[model_name] = (
                np.zeros(self.shape, dtype=np.uint16),
                set(),
            )
        return self

    def save(self, file_path):
        """Save the cache state to an HDF5 file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with h5py.File(file_path, "w") as f:
            # Store basic attributes
            f.attrs["model_name"] = self.model_name if self.model_name else ""
            f.attrs["phc_channel"] = self.phc_channel

            # Create a group for mmap arrays
            mmap_group = f.create_group("mmap_arrays")

            # Store each model's data
            for model_name, (array, index_set) in self.mmap_arrays_idx.items():
                model_group = mmap_group.create_group(model_name)
                # Store the array
                model_group.create_dataset("array", data=array, compression="gzip")
                # Store the set as a JSON string
                index_list = [list(idx) for idx in index_set]
                model_group.attrs["index_set"] = json.dumps(index_list)

            # Store image_data reference or metadata
            # Note: We don't store the actual image_data, just metadata about it
            if self.image_data is not None:
                image_group = f.create_group("image_metadata")
                # Store file path if available
                if hasattr(self.image_data, "filename"):
                    image_group.attrs["filename"] = str(self.image_data.filename)
                # Store shape if available
                if hasattr(self.image_data, "shape"):
                    image_shape = self.image_data.shape
                    image_group.attrs["shape"] = json.dumps(image_shape)

    @classmethod
    def load(cls, file_path, image_data=None):
        """Load cache state from an HDF5 file"""
        with h5py.File(file_path, "r") as f:
            phc_channel = int(f.attrs.get("phc_channel", 0))
            cache = cls(image_data, phc_channel=phc_channel)

            # Load basic attributes
            cache.model_name = f.attrs.get("model_name", "") or None

            if (
                "mmap_arrays" in f
            ):  # reload each segmentation cache with the associated cache index
                mmap_group = f["mmap_arrays"]
                for model_name in mmap_group:
                    model_group = mmap_group[model_name]
                    array = model_group["array"][:]

                    # Reconstruct the indices set
                    index_set = set()
                    if "index_set" in model_group.attrs:
                        index_list_str = model_group.attrs["index_set"]
                        try:
                            # Load the indices from JSON string
                            index_list = json.loads(index_list_str)

                            # Convert each index to a proper tuple
                            for idx in index_list:
                                # Ensure each index is a tuple of integers
                                if isinstance(idx, list):
                                    index_set.add(tuple(int(i) for i in idx[:2]))
                                else:
                                    print(f"WARNING: Unexpected index format: {idx}")

                            print(
                                f"Successfully loaded {len(index_set)} indices for model {model_name}"
                            )
                        except Exception as e:
                            print(
                                f"Error parsing indices for model {model_name}: {str(e)}"
                            )
                            # Continue with empty set

                    # Store the array and indices
                    cache.mmap_arrays_idx[model_name] = (array, index_set)

        return cache

    def __getitem__(self, key):
        if self.model_name is None:
            raise ValueError(
                "Model name must be set using with_model() before accessing data."
            )

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

    def __setitem__(self, key, value):
        if self.model_name is None:
            raise ValueError(
                "Model name must be set using with_model() before accessing data."
            )

        mmap_array, indices = self.mmap_arrays_idx[self.model_name]
        key = self._normalize_index(key)
        all_indices = list(self._expand_indices(key))

        if len(all_indices) != 1:
            raise ValueError("SegmentationCache assignment requires one frame.")

        idx = all_indices[0]
        mmap_array[idx] = np.asarray(value, dtype=np.uint16)
        indices.add(idx)

    def is_computed(self, model_name, key):
        if model_name not in self.mmap_arrays_idx:
            return False

        _, indices = self.mmap_arrays_idx[model_name]
        key = self._normalize_index(key)
        return all(idx in indices for idx in self._expand_indices(key))

    def invalidate_all(self):
        self.mmap_arrays_idx = {}
        self.model_name = None

    def _normalize_index(self, key):
        """Convert various index types to tuple of slice objects"""
        if not isinstance(key, tuple):
            key = (key,)

        key = key[: self.ndim]
        normalized = []
        for k in key:
            if isinstance(k, int):
                # Convert single integers to slices to maintain dimensions
                normalized.append(slice(k, k + 1))
            elif isinstance(k, Ellipsis.__class__):  # Handle ellipsis
                remaining_dims = self.ndim - len(key) + 1
                normalized.extend([slice(None)] * remaining_dims)
            else:
                normalized.append(k)

        # Fill missing dimensions with full slices
        while len(normalized) < self.ndim:
            normalized.append(slice(None))

        return tuple(normalized[: self.ndim])

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
            t, p = idx

            from partaker.data.image_data import ImageData

            frame = ImageData.get_instance().get(t, p, self.phc_channel)
            if hasattr(frame, "compute"):
                frame = frame.compute()

            if frame.dtype == bool:
                frame = frame.astype(np.uint8) * 255
            elif frame.dtype != np.uint8 and frame.dtype != np.uint16:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                )

            # TODO: migrate to segmentation service
            # Normalize frame first
            if self.model_name in [SegmentationModels().UNET]:
                frame = cv2.normalize(frame, None, 0, 65535, cv2.NORM_MINMAX).astype(
                    np.uint16
                )

            segmented_frame = SegmentationModels().segment_images(
                [frame], mode=self.model_name
            )[0]

            mmap_array[idx] = segmented_frame
            indices.add(idx)
        except Exception as e:
            raise IndexError(f"Failed to process frame {idx}") from e

    @property
    def shape(self):
        full_shape = self.image_data.shape
        return (*full_shape[:2], *full_shape[-2:])

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        if self.model_name is None:
            return np.dtype(np.uint16)
        mmap_array, _ = self.mmap_arrays_idx[self.model_name]
        return mmap_array.dtype

    def __array__(self):
        if self.model_name is None:
            raise ValueError(
                "Model name must be set using with_model() before accessing data."
            )
        mmap_array, _ = self.mmap_arrays_idx[self.model_name]
        return mmap_array[:]