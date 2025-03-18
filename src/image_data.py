from pathlib import Path
import json
import os

import nd2

from segmentation.segmentation_cache import SegmentationCache

"""
Can hold either an ND2 file or a series of images
"""


class ImageData:
    def __init__(self, data=None, path=None, is_nd2=False):
        self.data = data
        self.nd2_filename = path
        self.processed_images = []
        self.is_nd2 = is_nd2
        if data is not None:
            self.segmentation_cache = SegmentationCache(data)
        else:
            self.segmentation_cache = None

    @classmethod
    def load_nd2(cls, filename: str):
        image_data = cls()
        image_data.nd2_filename = filename
        image_data._load_nd2(filename)

        return image_data

    def _load_nd2(self, filename: str):
        with nd2.ND2File(filename) as nd2_file:
            self.data = nd2.imread(filename, dask=True)
        if self.segmentation_cache is not None:
            self.segmentation_cache.clear()

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
