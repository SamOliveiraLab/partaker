import numpy as np
import cv2
from scipy.ndimage import label
from .rpu import RPUParams

FLUO_EPSILON = 0.001

"""
Isolates the fluorescence against each connected component (cell) in the binary image
It gets the mean fluorescence value for each connected component

binary_images: array of binary images
fluorescence_images: array of fluorescence images

Returns:
sc_fluo: list of lists of fluorescence values for each connected component
timestamps: timestamp of each valid fluorescence image
"""
def analyze_fluorescence_singlecell(binary_images, fluorescence_images, rpu: RPUParams = None):
    results = []
    timestamps = []

    for i, (binary_image, fluorescence_image) in enumerate(zip(binary_images, fluorescence_images)):
        result = []
        labeled_array, num_features = label(binary_image)
        for component in range(1, num_features + 1):
            mask = labeled_array == component
            fluorescence_avg = fluorescence_image[mask].flatten().mean()
            if fluorescence_avg <= FLUO_EPSILON:
                continue

            result.append(rpu.compute(fluorescence_avg) if rpu else fluorescence_avg)
        
        if len(result) == 0:
            continue
        timestamps.append(i)
        results.append(result)

    return results, timestamps

def analyze_fluorescence_total(fluorescence_images, rpu: RPUParams = None):
    results = []
    timestamps = []

    for i, fluorescence_image in enumerate(fluorescence_images):
        fluorescence_avg = fluorescence_image.flatten().mean()
        
        if fluorescence_avg <= FLUO_EPSILON:
            continue
        
        results.append(rpu.compute(fluorescence_avg) if rpu else fluorescence_avg)
        timestamps.append(i)

    return results, timestamps
