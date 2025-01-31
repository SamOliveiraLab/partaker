from pathlib import Path
import matplotlib.pyplot as plt
import math
import numpy as np
import imageio.v2 as imageio
import os
import cv2

from cachier import cachier
import datetime

from cellpose import models, io

from .unet import unet_segmentation

from cellSAM import segment_cellular_image, get_model

# # Example usage
# segmentation_models = SegmentationModels()
# # Accessing the CELLPOSE model
# cellpose_model = segmentation_models.get_model(SegmentationModels.CELLPOSE)
# # Accessing the UNET model
# unet_model = segmentation_models.get_model(SegmentationModels.UNET)
# # Segmenting images using the CELLPOSE model
# segmented_images = segmentation_models.segment_images(images, mode=SegmentationModels.CELLPOSE, progress=progress)
# # Segmenting images using the UNET model
# segmented_images = segmentation_models.segment_images(images, mode=SegmentationModels.UNET, progress=progress)

# # @cachier(stale_after=datetime.timedelta(days=3))
# def segment_images(directory='/Users/hiram/Workspace/OliveiraLab/PartakerV3/src/aligned_data/XY8_Long_PHC', weights='/Users/hiram/Workspace/OliveiraLab/PartakerV3/src/checkpoints/delta_2_20_02_24_600eps.index'):

#     target_size_seg = (512, 512)
#     model = unet_segmentation(input_size = target_size_seg + (1,))
#     test_images = Path(directory)
#     imgs = list(map(lambda x : cv2.imread(str(x), cv2.IMREAD_GRAYSCALE), sorted([img for img in test_images.iterdir()], key=lambda x : int(x.stem))))

#     def my_resize(img):
#         a = img[55:960, 150:810]
#         a = cv2.resize(a, (512, 512))
#         a = np.expand_dims(a, axis=-1)
#         return a

#     pred_imgs = model.predict(np.array(list(map(my_resize, imgs))))
#     return pred_imgs

# TODO: Implement caching for segmentation models

"""
Container class for multiple segmentation models
"""
class SegmentationModels:
    CELLPOSE = 'cellpose'
    UNET = 'unet'
    CELLSAM = 'cellsam'

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SegmentationModels, cls).__new__(cls, *args, **kwargs)
            cls._instance.models = {}
        return cls._instance
    
    """
    Segments a single image using cellsam
    """
    def segment_cellsam(self, images):
        _img = np.expand_dims(images[0], axis=2)
        _cellSAM_masks, _, _ = segment_cellular_image(_img, device='cpu')
        return [_cellSAM_masks] # Doing this to ensure API compatibility with other segmentation methods

    def segment_unet(self, images):
        model = self.models[SegmentationModels.UNET]
        pred_imgs = model.predict(np.array([np.expand_dims(cv2.resize(img, (512, 512)), axis=-1) for img in images]))

        print(pred_imgs.shape)
        return pred_imgs[:, :, :, 0]
    
    def segment_cellpose(self, images, progress):
        cellpose_inst = self.models[SegmentationModels.CELLPOSE]

        # Ensure images are in the correct format
        images = [img.squeeze() if img.ndim > 2 else img for img in images]

        # Run segmentation
        try:
            masks, _, _ = cellpose_inst.eval(images, diameter=None, channels=[0, 0])
            masks = np.array(masks)  # Ensure masks are a NumPy array
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return None

        # Create binary black-and-white masks
        try:
            bw_images = np.zeros_like(masks, dtype=np.uint8)
            bw_images[masks > 0] = 255  # Convert labeled masks to binary
        except Exception as e:
            print(f"Error converting masks to binary: {e}")
            return None

        # Update progress if a callback is provided
        if progress:
            if callable(progress):  # If it's a function
                progress(len(images))
            else:  # Assume it's a PyQt signal
                progress.emit(len(images))

        return bw_images        

    def segment_images(self, images, mode, progress=None):
        print(f"Segmenting images using {mode} model")

        if mode == SegmentationModels.CELLPOSE:
            if SegmentationModels.CELLPOSE not in self.models:
                if "PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1":
                    self.models[self.CELLPOSE] = models.CellposeModel(gpu=True, model_type='deepbacs_cp3')
                else:
                    self.models[self.CELLPOSE] = models.CellposeModel(gpu=False, model_type='deepbacs_cp3')
            
            return self.segment_cellpose(images, progress)
        
        elif mode == SegmentationModels.UNET:
            if SegmentationModels.UNET not in self.models:
                target_size_seg = (512, 512)
                self.models[SegmentationModels.UNET] = unet_segmentation(input_size=target_size_seg + (1,))
            
            return self.segment_unet(images)

        elif mode == SegmentationModels.CELLSAM:
            return self.segment_cellsam(images)

        else:
            raise ValueError(f"Invalid segmentation mode: {mode}")
