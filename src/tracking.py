import btrack
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb


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
    FEATURES = [
        "area",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "solidity"]

    # Validate input
    if segmented_images is None or not isinstance(
            segmented_images, np.ndarray) or segmented_images.ndim != 3:
        raise ValueError(
            "Segmented images must be a 3D NumPy array (time, height, width).")

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
        raise ValueError(
            "No objects detected in the segmentation. Ensure your segmentation produces labeled regions.")

    # Define config file path
    config_path = os.path.join(
        os.path.dirname(__file__),
        'config',
        'btrack_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}")

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
            print(
                f"Tracker volume dimensions: ((0, {segmented_images.shape[2]}), (0, {segmented_images.shape[1]}), (0, 1))")
            tracker.volume = (
                (0, segmented_images.shape[2]), (0, segmented_images.shape[1]), (0, 1))

            print("Starting tracking process...")
            tracker.track()  # Use track() instead of deprecated track_interactive()

            tracks = tracker.tracks
            print(f"Tracking complete. Total tracks found: {len(tracks)}")

    except Exception as e:
        raise RuntimeError(f"Failed to track cells: {e}")

    print("Tracks returned from track_cells:", tracks)
    print("Type of returned tracks:", type(tracks))
    return tracks  # Make sure it’s returning exactly what you expect


def overlay_tracks_on_images(segmented_images, tracks, save_video=False, output_path="tracked_cells.mp4"):
    """
    Overlays tracking trajectories on the original segmented images.

    Parameters:
    segmented_images (np.ndarray): 3D array (time, height, width) of segmented images.
    tracks (list): List of tracked cell objects from btrack.
    save_video (bool): If True, saves the output as a video.
    output_path (str): Path to save the output video.
    """
    height, width = segmented_images.shape[1:]
    colors = {}  # Dictionary to store track colors

    # Create an output video writer if required
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 5  # Adjust as needed
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for t in range(segmented_images.shape[0]):
        frame = segmented_images[t].copy()
        frame_rgb = label2rgb(frame, bg_label=0)  # Convert labels to colors
        frame_rgb = (frame_rgb * 255).astype(np.uint8)

        for track in tracks:
            track_id = track['ID']
            x_coords, y_coords, times = track['x'], track['y'], track['t']

            if track_id not in colors:
                colors[track_id] = tuple(
                    map(int, np.random.randint(0, 255, 3)))  # Assign random color

            # Draw trajectory for current time point
            for i in range(len(times) - 1):
                if times[i] < t <= times[i + 1]:
                    cv2.line(frame_rgb, (int(x_coords[i]), int(y_coords[i])),
                             (int(x_coords[i + 1]), int(y_coords[i + 1])), tuple(map(int, colors[track_id])), 2)

            # Draw the latest point if it's in the current frame
            if t in times:
                idx = times.index(t)
                cv2.circle(frame_rgb, (int(x_coords[idx]), int(
                    y_coords[idx])), 3, tuple(map(int, colors[track_id])), -1)
                cv2.putText(frame_rgb, str(track_id), (int(x_coords[idx]), int(y_coords[idx])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, tuple(map(int, colors[track_id])), 1)

        plt.imshow(frame_rgb)
        plt.title(f"Tracked Cells - Frame {t}")
        plt.axis("off")
        plt.show()

        if save_video:
            out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    if save_video:
        out.release()
        print(f"Video saved at {output_path}")

# Usage example:
# overlay_tracks_on_images(segmented_images, tracked_cells, save_video=True)


def visualize_cell_tracks(segmented_images, tracks):
    """
    Overlays cell tracks on segmented images to visualize movement over time.

    Parameters:
    -----------
    segmented_images : np.ndarray
        3D NumPy array (time, height, width) containing segmented images.
    tracks : list
        List of tracked cell objects containing their trajectories.

    Returns:
    --------
    None (Displays the visualization).
    """
    if segmented_images is None or tracks is None or len(tracks) == 0:
        print("No valid tracking data available.")
        return

    num_frames, img_height, img_width = segmented_images.shape

    # Create an RGB overlay
    overlay = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Assign random colors to tracks
    np.random.seed(42)
    track_colors = {
        track.ID: tuple(np.random.randint(0, 255, size=3).tolist())
        for track in tracks
    }

    for track in tracks:
        track_id = track.ID
        track_color = track_colors[track_id]

        # Get the trajectory of the track
        trajectory = np.array(list(zip(track.x, track.y)))

        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                pt1 = tuple(trajectory[i].astype(int))
                pt2 = tuple(trajectory[i + 1].astype(int))

                # Draw a line between consecutive points
                cv2.line(overlay, pt1, pt2, track_color, thickness=2)

    # Display the overlaid image
    plt.figure(figsize=(10, 6))
    plt.imshow(overlay)
    plt.title("Cell Tracking Visualization")
    plt.axis("off")
    plt.show()
