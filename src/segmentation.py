from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import imageio.v2 as imageio
import os

from cachier import cachier
import datetime

from cellpose import models, io

# def preprocess_image(image):
#     # Apply contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_image = clahe.apply(image)

#     # Apply Gaussian blur for denoising
#     blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

#     return blurred_image

# def segment_this_image(image):
#     # Preprocess image before segmentation
#     preprocessed_image = preprocess_image(image)

#     # Use Cellpose for segmentation
#     cellpose_inst = CellposeModelSingleton().model
#     masks, flows, styles = cellpose_inst.eval(preprocessed_image, diameter=None, channels=[0, 0])

#     # Create binary mask
#     bw_image = np.zeros_like(masks, dtype=np.uint8)
#     bw_image[masks > 0] = 255
    
#     # Debug: Show binary mask
#     # cv2.imshow("Binary Mask", bw_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     return bw_image

# def segment_all_images(images, progress=None):
#     # Get the Cellpose model instance
#     cellpose_inst = CellposeModelSingleton().model

#     # Ensure images are in the correct format
#     images = [img.squeeze() if img.ndim > 2 else img for img in images]

#     # Run segmentation
#     try:
#         masks, _, _ = cellpose_inst.eval(images, diameter=None, channels=[0, 0])
#         masks = np.array(masks)  # Ensure masks are a NumPy array
#     except Exception as e:
#         print(f"Error during segmentation: {e}")
#         return None

#     # Create binary black-and-white masks
#     try:
#         bw_images = np.zeros_like(masks, dtype=np.uint8)
#         bw_images[masks > 0] = 255  # Convert labeled masks to binary
#     except Exception as e:
#         print(f"Error converting masks to binary: {e}")
#         return None

#     # Update progress if a callback is provided
#     if progress:
#         if callable(progress):  # If it's a function
#             progress(len(images))
#         else:  # Assume it's a PyQt signal
#             progress.emit(len(images))

#     return bw_images

# """
# Segments one image and returns it, in a single channel
# """
# def _segment_this_image(image):
#     image = np.array(image)

#     target_size_seg = (512, 512)
#     model = unet_segmentation(input_size = target_size_seg + (1,))

#     def my_resize(img):
#         print(img.shape)
#         # a = img[55:960][150:810]
#         a = img
#         a = cv2.resize(a, (512, 512))
#         a = np.expand_dims(a, axis=-1)
#         return a

#     pred_imgs = model.predict(np.array([my_resize(image)]))

#     print(pred_imgs.shape)
#     return pred_imgs[0, :, :, 0]
