import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_individual_cells(image, segmented_image):
    """
    Extracts individual cells from the original image based on the segmented mask.

    Parameters:
    -----------
    image : np.ndarray
        The original grayscale or raw image.
    segmented_image : np.ndarray
        The binary segmented image where each cell is labeled uniquely.

    Returns:
    --------
    List of tuples where each tuple contains:
        - Cropped cell image (np.ndarray)
        - Bounding box (x, y, w, h)
    """
    # Ensure the images are the same size
    assert image.shape == segmented_image.shape, "Image and segmented image must have the same dimensions."

    # Find connected components in the segmented mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        segmented_image, connectivity=8)

    # Extract individual cells
    extracted_cells = []
    for label in range(1, num_labels):  # Skip the background (label=0)
        # Extract bounding box for the label
        x, y, w, h, area = stats[label]

        # Skip small regions (noise)
        if area < 50:
            continue

        # Crop the corresponding region from the original image
        cropped_cell = image[y:y + h, x:x + w]
        extracted_cells.append((cropped_cell, (x, y, w, h)))

    return extracted_cells


def classify_morphology(metrics):
    """
    Classify cell morphology based on its metrics.

    Parameters:
    - metrics: dict, a dictionary containing cell metrics (area, aspect_ratio, circularity, perimeter, solidity, orientation).

    Returns:
    - str, the morphology class (e.g., 'Small', 'Round', 'Normal', 'Elongated', 'Deformed').
    """
    area = metrics.get("area", 0)
    aspect_ratio = metrics.get("aspect_ratio", 0)
    circularity = metrics.get("circularity", 0)
    perimeter = metrics.get("perimeter", 0)
    solidity = metrics.get("solidity", 1)
    orientation = metrics.get("orientation", 0)

    # Small Cells
    if area < 1500 and perimeter < 200 and aspect_ratio < 3:
        return "Small"

    # Round Cells
    elif 1500 <= area < 3000 and 0.85 <= solidity <= 0.95 and circularity > 0.6 and 3 <= aspect_ratio < 6:
        return "Small"

    # Normal Cells
    elif 0.7 <= circularity <= 0.9 and 1.2 <= aspect_ratio < 3 and 0.9 <= solidity <= 1.0:
        return "Normal"

    # Elongated Cells
    elif area >= 3000 and aspect_ratio >= 6 and circularity < 0.3:
        return "Elongated"

    # Deformed Cells
    else:
        return "Normal"


def extract_cells_and_metrics(image, segmented_image):
    """
    Extract individual cells, their bounding boxes, and metrics from a segmented image.

    Parameters:
    - image: np.ndarray, the original grayscale image.
    - segmented_image: np.ndarray, the binary segmented image.

    Returns:
    - cell_mapping: dict, a dictionary with cell IDs as keys and a dictionary of metrics and bounding boxes as values.
    """
    from skimage.measure import regionprops, label
    from skimage.color import rgb2gray
    from skimage.transform import resize

    # Debugging: print input shapes
    print(f"Original image shape: {image.shape}")
    print(f"Segmented image shape: {segmented_image.shape}")

    # Ensure the intensity image is single-channel (convert if multi-channel)
    if image.ndim == 3 and image.shape[-1] in [3, 4]:  # RGB or RGBA
        print("Converting multi-channel image to grayscale.")
        image = rgb2gray(image)

    # Check and handle shape mismatches between the intensity image and the
    # segmented image
    if image.shape != segmented_image.shape:
        print(
            f"Resizing intensity image from {image.shape} to {segmented_image.shape}")
        image = resize(
            image,
            segmented_image.shape,
            preserve_range=True,
            anti_aliasing=True)

    # Label connected regions in the segmented image
    labeled_image = label(segmented_image)

    # Debugging: print labeled image shape
    print(f"Labeled image shape: {labeled_image.shape}")

    # Extract properties for each labeled region
    cell_mapping = {}
    for region in regionprops(labeled_image, intensity_image=image):
        if region.area < 50:  # Filter out small regions (noise)
            continue

        # Calculate bounding box and metrics
        x1, y1, x2, y2 = region.bbox  # Bounding box coordinates
        metrics = {
            "area": region.area,
            "perimeter": region.perimeter,
            "equivalent_diameter": region.equivalent_diameter,
            "orientation": region.orientation,
            "aspect_ratio": region.major_axis_length / region.minor_axis_length
            if region.minor_axis_length > 0
            else 0,
            "circularity": (4 * np.pi * region.area) / (region.perimeter**2)
            if region.perimeter > 0
            else 0,
            "solidity": region.solidity,
        }

        # Classify the cell's morphology
        metrics["morphology_class"] = classify_morphology(metrics)

        # Add cell information to the mapping
        cell_id = len(cell_mapping) + 1
        cell_mapping[cell_id] = {
            "bbox": (x1, y1, x2, y2),
            "metrics": metrics,
        }

    return cell_mapping


def annotate_image(image, cell_mapping):
    """
    Annotate the original image with bounding boxes and IDs for detected cells.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The input image is not a valid numpy array.")
    print(f"Annotating image of shape: {image.shape}")  # Debugging

    # Ensure it's in RGB format
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for cell_id, data in cell_mapping.items():
        x1, y1, x2, y2 = data["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, str(cell_id), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated


def annotate_binary_mask(segmented_image, cell_mapping):
    """
    Annotate the binary segmented mask with bounding boxes and morphology class color codes.

    Parameters:
    -----------
    segmented_image : np.ndarray
        The binary segmented mask (black and white).
    cell_mapping : dict
        Cell ID mapping with metrics and bounding boxes.

    Returns:
    --------
    annotated : np.ndarray
        Annotated binary mask with bounding boxes and labels.
    """
    # Ensure input is grayscale
    if len(segmented_image.shape) == 3:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to RGB for annotations
    annotated = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)

    # Define color mapping for morphology classes
    morphology_colors = {
        "Small": (0, 0, 255),  # Blue
        "Round": (255, 0, 0),  # Red
        "Normal": (0, 255, 0),  # Green
        "Elongated": (255, 255, 0),  # Yellow
        "Deformed": (255, 0, 255),  # Magenta
    }

    for cell_id, data in cell_mapping.items():
        y1, x1, y2, x2 = data["bbox"]

        # Get the morphology class and corresponding color
        morphology_class = data["metrics"].get("morphology_class", "Normal")
        color = morphology_colors.get(
            morphology_class, (255, 255, 255))  # Default to white

        # Draw bounding box with morphology-specific color
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Add text label for cell ID and class
        label = f"{cell_id}: {morphology_class}"
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def extract_cell_morphologies(binary_image: np.array) -> pd.DataFrame:
    """
    Extracts cell morphologies from a binarized image with robust error handling.

    Parameters:
    binary_image (numpy.ndarray): Binarized image where cells are white (255) and background is black (0).

    Returns:
    pd.DataFrame: DataFrame containing morphology properties for each cell.
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    morphologies = []

    for contour in contours:
        try:
            area = cv2.contourArea(contour)
            if area < 5:  # Ignore noise
                continue

            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)

            if h == 0 or w == 0:
                continue

            aspect_ratio = float(w) / h
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            equivalent_diameter = np.sqrt(4 * area / np.pi)

            if len(contour) >= 5:
                (_, _), (_, _), angle = cv2.fitEllipse(contour)
            else:
                angle = np.nan

            metrics = {
                "area": area,
                "perimeter": perimeter,
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "solidity": solidity,
                "equivalent_diameter": equivalent_diameter,
                "orientation": angle,
            }

            # Add classification
            metrics["morphology_class"] = classify_morphology(metrics)
            morphologies.append(metrics)

        except Exception as e:
            print(f"Error processing contour: {e}")
            continue

    if morphologies:
        return pd.DataFrame(morphologies)
    else:
        return pd.DataFrame(
            columns=[
                "area",
                "perimeter",
                "aspect_ratio",
                "extent",
                "solidity",
                "equivalent_diameter",
                "orientation",
                "morphology_class"])


def extract_cell_morphologies_time(
        segmented_imgs: np.array,
        **kwargs) -> pd.DataFrame:
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
