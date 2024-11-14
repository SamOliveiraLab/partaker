import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_cell_morphologies(binary_image: np.array) -> pd.DataFrame:
    """
    Extracts cell morphologies from a binarized image.

    Parameters:
    binary_image (numpy.ndarray): Binarized image where cells are white (255) and background is black (0).

    Returns:
    list: A list of dictionaries containing morphology properties for each cell.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    morphologies = []
    
    for contour in contours:
        try:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h
            
            # Calculate extent (ratio of contour area to bounding box area)
            rect_area = w * h
            extent = float(area) / rect_area
            
            # Calculate convex hull and solidity (ratio of contour area to convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            
            # Calculate equivalent diameter (diameter of the circle with the same area as the contour)
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            
            # Calculate orientation (angle at which the object is directed)
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            
            # Store morphology properties in a dictionary
            morphology = {
                'area': area,
                'perimeter': perimeter,
                'bounding_box': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity,
                'equivalent_diameter': equivalent_diameter,
                'orientation': angle
            }
            
            morphologies.append(morphology)
        except Exception as e:
            print("Found an error: ", e)
            continue

    # Convert list of dictionaries to a pandas DataFrame
    morphologies_df = pd.DataFrame(morphologies)
    
    return morphologies_df

def extract_cell_morphologies_time(segmented_imgs: np.array, **kwargs) -> pd.DataFrame:
    """
    Extracts cell morphologies from a binarized image.

    Parameters:
    segmented_imgs (numpy.ndarray): Binarized image where cells are white (255) and background is black (0).

    Returns:
    list: A list of dictionaries containing morphology properties for each cell.
    """
    metrics = []
    
    for binary_image in tqdm(segmented_imgs):
        
        _metrics = extract_cell_morphologies(binary_image)
        metrics.append(_metrics.mean(axis=0))
        
    return pd.concat(metrics, ignore_index=True)
