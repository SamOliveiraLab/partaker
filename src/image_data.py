from segmentation.segmentation_cache import SegmentationCache

"""
Can hold either an ND2 file or a series of images
"""
class ImageData:
    def __init__(self, data, is_nd2=False):
        self.data = data
        self.processed_images = []
        self.is_nd2 = is_nd2
        self.segmentation_cache = {}
        self.seg_cache = SegmentationCache(data)
