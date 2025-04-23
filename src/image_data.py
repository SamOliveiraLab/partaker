from pathlib import Path
import json
import os

import nd2

from segmentation.segmentation_cache import SegmentationCache
from segmentation.segmentation_service import SegmentationService
from segmentation.segmentation_models import SegmentationModels

from pubsub import pub

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
        return self.data[t, p, c].compute()

    def _access(self, time, position, channel):
        
        image = self._get_raw_image(time, position, channel)
        pub.sendMessage("image_ready",
                       image=image,
                       time=time,
                       position=position,
                       channel=channel,
                       mode='normal')

    @classmethod
    def load_nd2(cls, file_path: str):
        with nd2.ND2File (file_path) as nd2_file:
            image_data = ImageData(data=nd2.imread(
                file_path, dask=True), path=file_path, is_nd2=True)

        return image_data

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
        image_data = cls()
        base_dir = Path(filename)

        # Load imagedata and nd2
        meta_path = base_dir / "image_data.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                _json = json.load(f)
                image_data.nd2_filename = _json.get('nd2_filename')
                image_data.is_nd2 = _json.get('is_nd2')
                # Load other attributes as needed

        # Load nd2
        image_data._load_nd2(image_data.nd2_filename)

        # Load segmentation cache if file exists
        cache_path = base_dir / "segmentation_cache.h5"
        if cache_path.exists():
            image_data.segmentation_cache = SegmentationCache.load(
                str(cache_path), image_data.data)

        return image_data
