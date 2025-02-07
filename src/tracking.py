import btrack
import numpy as np
import os

def track_cells(segmented_images):
    """
    Tracks segmented cells over time using BayesianTracker (btrack).

    Parameters:
    -----------
    segmented_images : np.ndarray
        3D array (time, height, width) of labeled segmented images (each cell has a unique label).

    Returns:
    --------
    list
        A list of tracked cell objects.
    """
    FEATURES = ["area", "major_axis_length", "minor_axis_length", "orientation", "solidity"]

    # Validate input
    if segmented_images is None or not isinstance(segmented_images, np.ndarray) or segmented_images.ndim != 3:
        raise ValueError("Segmented images must be a 3D NumPy array (time, height, width).")

    if np.isnan(segmented_images).any() or np.isinf(segmented_images).any():
        raise ValueError("Segmented images contain NaN or Inf values.")

    # Convert segmented images to btrack objects
    try:
        print("Converting segmented images to objects...")
        objects = btrack.utils.segmentation_to_objects(
            segmented_images,
            properties=tuple(FEATURES),
            num_workers=4,
        )
        print(f"Number of objects detected: {len(objects)}")
    
        # Debugging the first few objects to check structure
        if len(objects) > 0:
            print("Sample object structure:", objects[0])

    except Exception as e:
        raise RuntimeError(f"Failed to convert segmentation to objects: {e}")

    if not objects:
        raise ValueError("No objects detected in the segmentation. Ensure your segmentation produces labeled regions.")

    # Define config file path
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'btrack_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Initialize and run the tracker
    try:
        with btrack.BayesianTracker() as tracker:
            print("Configuring tracker...")
            tracker.configure(config_path)
            tracker.max_search_radius = 50  # Adjust based on cell movement
            tracker.tracking_updates = ["MOTION", "VISUAL"]
            tracker.features = FEATURES

            print("Appending objects to tracker...")
            tracker.append(objects)
            
            # Debug volume dimensions
            print(f"Tracker volume dimensions: ((0, {segmented_images.shape[2]}), (0, {segmented_images.shape[1]}), (0, 1))")
            tracker.volume = ((0, segmented_images.shape[2]), (0, segmented_images.shape[1]), (0, 1))

            print("Starting tracking process...")
            tracker.track()  # Use track() instead of deprecated track_interactive()

            tracks = tracker.tracks
            print(f"Tracking complete. Total tracks found: {len(tracks)}")

    except Exception as e:
        raise RuntimeError(f"Failed to track cells: {e}")
    
    
    print("Tracks returned from track_cells:", tracks)
    print("Type of returned tracks:", type(tracks))
    return tracks  # Make sure it’s returning exactly what you expect

