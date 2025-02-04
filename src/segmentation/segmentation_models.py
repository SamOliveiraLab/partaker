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
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.restoration import richardson_lucy


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
    CELLPOSE_FT_0 = 'cellpose_finetuned'

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

    """
    Segments using U-Net
    - Patches image
    - Removes artifacts
    - Indentifies singles components
    """
    def segment_unet(self, images):
        model = self.models[SegmentationModels.UNET]
        patches = []
        patch_indices = []

        # Divide each image into 512x512 patches
        for img_idx, img in enumerate(images):
            img = np.array(img)
            height, width = img.shape[:2]

            for i in range(0, height, 512):
                for j in range(0, width, 512):
                    patch = img[i:i+512, j:j+512]

                    # If the patch is smaller than 512x512, pad it with zeros
                    if patch.shape[0] < 512 or patch.shape[1] < 512:
                        padded_patch = np.zeros((512, 512), dtype=patch.dtype)
                        padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                        patch = padded_patch

                    patches.append(np.expand_dims(patch, axis=-1))
                    patch_indices.append((img_idx, i, j))

        # Segment all patches at once
        patches = np.array(patches)
        segmented_patches = model.predict(patches)

        # Create an empty array to store the segmented results
        segmented_images = [np.zeros_like(img, dtype=np.uint8) for img in images]

        # Combine the segmented patches back into the original images
        for idx, (img_idx, i, j) in enumerate(patch_indices):
            segmented_patch = segmented_patches[idx, :, :, 0]

            # Remove padding if necessary
            if i + 512 > segmented_images[img_idx].shape[0]:
                segmented_patch = segmented_patch[:segmented_images[img_idx].shape[0] - i, :]
            if j + 512 > segmented_images[img_idx].shape[1]:
                segmented_patch = segmented_patch[:, :segmented_images[img_idx].shape[1] - j]

            segmented_images[img_idx][i:i+512, j:j+512] = segmented_patch

        return segmented_images
    # def segment_unet(self, images):
    #     model = self.models[SegmentationModels.UNET]
    #     pred_imgs = model.predict(np.array([np.expand_dims(cv2.resize(np.array(img), (512, 512)), axis=-1) for img in images]))

    #     print(pred_imgs.shape)
    #     return pred_imgs[:, :, :, 0]
    
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
        
        
        
        
    def segment_images(self, images, mode, model_type=None, progress=None, preprocess=True):
        
            # Preprocess images if the flag is enabled
            if preprocess:
                images = [preprocess_image(img) for img in images]

            if mode == SegmentationModels.CELLPOSE:
                if SegmentationModels.CELLPOSE not in self.models:
                    if "PARTAKER_GPU" in os.environ and os.environ["PARTAKER_GPU"] == "1":
                        self.models[self.CELLPOSE] = models.CellposeModel(gpu=True, model_type=model_type)
                    else:
                        self.models[self.CELLPOSE] = models.CellposeModel(gpu=False, model_type=model_type)
                
                # Ensure the selected model type is applied dynamically
                self.models[self.CELLPOSE].model_type = model_type
                
                return self.segment_cellpose(images, progress)
            
            elif mode == SegmentationModels.UNET:
                if SegmentationModels.UNET not in self.models:
                    target_size_seg = (512, 512)
                    self.models[SegmentationModels.UNET] = unet_segmentation(input_size=target_size_seg + (1,))
                
                return self.segment_unet(images)

            else:
                raise ValueError(f"Invalid segmentation mode: {mode}")        
            



def preprocess_image(image):
        """
        Preprocess an image by applying Gaussian blur, CLAHE, and Richardson-Lucy deblurring.

        Parameters:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        normalized_frame = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        denoised_frame = gaussian_filter(normalized_frame, sigma=1)

        # Apply CLAHE to improve contrast
        clahe = exposure.equalize_adapthist(denoised_frame, clip_limit=0.03)

        # Step 3: Deblur the image
        psf = np.ones((5, 5)) / 25  # Example PSF
        deblurred_frame = richardson_lucy(denoised_frame, psf, num_iter=30)

        return deblurred_frame