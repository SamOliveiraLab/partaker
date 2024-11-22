import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_cell_morphologies(binary_image: np.array) -> pd.DataFrame:
    """
    Extracts cell morphologies from a binarized image with robust error handling.

    Parameters:
    binary_image (numpy.ndarray): Binarized image where cells are white (255) and background is black (0).

    Returns:
    pd.DataFrame: DataFrame containing morphology properties for each cell.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    morphologies = []

    for contour in contours:
        try:
            # Filter out tiny contours (noise)
            area = cv2.contourArea(contour)
            if area < 5:  # Ignore contours with area < 5 pixels
                continue
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Avoid division by zero in aspect ratio and extent calculations
            if h == 0 or w == 0:
                continue
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h
            
            # Calculate extent (ratio of contour area to bounding box area)
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # Calculate convex hull and solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Calculate equivalent diameter
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            
            # Calculate orientation
            if len(contour) >= 5:  # fitEllipse requires at least 5 points
                (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            else:
                angle = np.nan  # Use NaN for undefined orientation
            
            # Store morphology properties
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
            print(f"Error processing contour: {e}")
            continue

    # Convert list of dictionaries to a pandas DataFrame
    if morphologies:
        morphologies_df = pd.DataFrame(morphologies)
    else:
        # Return an empty DataFrame with predefined columns if no valid contours
        morphologies_df = pd.DataFrame(columns=[
            'area', 'perimeter', 'bounding_box', 'aspect_ratio',
            'extent', 'solidity', 'equivalent_diameter', 'orientation'
        ])
    
    return morphologies_df



def extract_cell_morphologies_time(segmented_imgs: np.array, **kwargs) -> pd.DataFrame:
    """
    Extracts cell morphologies from a series of segmented images.

    Parameters:
    segmented_imgs (numpy.ndarray): 3D array of segmented binary images.

    Returns:
    pd.DataFrame: A DataFrame with averaged metrics for each time frame.
    """
    metrics = []
    for binary_image in tqdm(segmented_imgs):
        try:
            _metrics = extract_cell_morphologies(binary_image)
            if not _metrics.empty:
                metrics.append(_metrics.mean(axis=0))
            else:
                print("No valid contours found in a frame.")
        except Exception as e:
            print(f"Error processing frame: {e}")

    if metrics:
        result = pd.DataFrame(metrics)
        print("Aggregated results:\n", result)
        return result
    else:
        print("No valid metrics to aggregate.")
        return pd.DataFrame()
    