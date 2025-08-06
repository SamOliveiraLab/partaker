import json
import os
from pathlib import Path
from typing import Union, Sequence

import dask.array as da
import nd2
from pubsub import pub

from nd2_analyzer.analysis.segmentation.segmentation_cache import SegmentationCache
from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
from nd2_analyzer.analysis.segmentation.segmentation_service import SegmentationService

"""
Can hold either an ND2 file or a series of images
"""

class ImageData:
    def __init__(self, data, path, is_nd2=True):
        self.data = data
        self.nd2_filename = path
        self.processed_images = []
        self.is_nd2 = is_nd2

        # Initialize segmentation components
        self.segmentation_cache = SegmentationCache(data)
        self.segmentation_service = SegmentationService(
            cache=self.segmentation_cache,
            models=SegmentationModels(),
            data_getter=self._get_raw_image
        )

        pub.subscribe(self._access, "raw_image_request")
        pub.sendMessage("image_data_loaded", image_data=self)

    def _get_raw_image(self, t, p, c):
        """Helper method to retrieve raw images"""

        if len(self.data.shape) == 5:  # - has channel
            raw_image = self.data[t, p, c]
        elif len(self.data.shape) == 4:  # - no channel
            raw_image = self.data[t, p]
        else:
            print(f"Unusual data format: {len(self.data.shape)} dimensions")
            if len(self.data.shape) >= 3:
                raw_image = self.data[t, p]
            else:
                raw_image = self.data[t]

        # Compute if it's a dask array
        if hasattr(raw_image, 'compute'):
            raw_image = raw_image.compute()

        return raw_image

    def _access(self, time, position, channel):

        image = self._get_raw_image(time, position, channel)
        pub.sendMessage("image_ready",
                        image=image,
                        time=time,
                        position=position,
                        channel=channel,
                        mode='normal')

    @classmethod
    def load_nd2(cls, file_paths: Union[str, Sequence[str]]):
        """
        Load one or more ND2 files, verify that channel count, image height and width match,
        crop the P-dimension (second axis) to the smallest found, concatenate along time axis,
        and print the final shape.
        """

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        arrays = []
        p_dims = []
        channels = height = width = None

        for path in file_paths:
            arr = nd2.imread(path, dask=True)  # Lazy load dask
            shape = arr.shape
            if len(shape) < 5:
                raise ValueError(
                    f"File {path} has shape {shape}; expected (T, P, C, Y, X)"
                )

            T, P, C, Y, X = shape

            if channels is None:
                # Set C, Y, X on the first file
                channels, height, width = C, Y, X
            else:
                # Check if files are "castable"
                if C != channels:
                    raise ValueError(f"{path}: channels {C} != {channels}")
                if Y != height:
                    raise ValueError(f"{path}: height {Y} != {height}")
                if X != width:
                    raise ValueError(f"{path}: width {X} != {width}")

            p_dims.append(P)
            arrays.append(arr)

        # Crop all files to the smallest P
        # TODO: check if this is valid
        min_p = min(p_dims)
        cropped = [arr[:, :min_p, :, :, :] for arr in arrays]

        full_data = da.concatenate(cropped, axis=0)

        print(f"Loaded {len(file_paths)} file(s). "
              f"Cropped P to {min_p}. Final array shape: {full_data.shape}")

        return cls(data=full_data, path=file_paths, is_nd2=True)

    def save(self, filename: str):
        """Saves state to file
        Doesn't save nd2 since it is already stored in a file
        """
        base_dir = Path(filename)
        os.makedirs(base_dir, exist_ok=True)

        # Save segmentation cache if it exists
        if self.segmentation_cache is not None:
            cache_path = base_dir / "segmentation_cache.h5"
            self.segmentation_cache.save(str(cache_path))

        # Save other container data
        container_data = {
            'nd2_filename': self.nd2_filename,
            'is_nd2': self.is_nd2
        }

        # Save container metadata
        with open(base_dir / "image_data.json", 'w') as f:
            json.dump(container_data, f)

    @classmethod
    def load(cls, filename):
        """Load imagedata from path"""
        base_dir = Path(filename)

        # Load imagedata metadata
        meta_path = base_dir / "image_data.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta_json = json.load(f)
                nd2_filename = meta_json.get('nd2_filename')
                is_nd2 = meta_json.get('is_nd2', True)

                files_ok = (isinstance(nd2_filename, str) and os.path.exists(nd2_filename)) or (
                            isinstance(nd2_filename, list) and all(os.path.exists(_fname) for _fname in nd2_filename))

                if files_ok:
                    image_data = cls.load_nd2(nd2_filename)

                    # Load segmentation cache if file exists
                    cache_path = base_dir / "segmentation_cache.h5"
                    if cache_path.exists():
                        image_data.segmentation_cache = SegmentationCache.load(
                            str(cache_path), image_data.data)
                        image_data.segmentation_service = SegmentationService(
                            cache=image_data.segmentation_cache,
                            models=SegmentationModels(),
                            data_getter=image_data._get_raw_image
                        )

                    return image_data
                else:
                    raise FileNotFoundError(
                        f"ND2 file not found: {nd2_filename}")
        else:
            raise FileNotFoundError(f"Metadata file not found in {filename}")
