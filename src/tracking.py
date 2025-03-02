import btrack
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import label


def track_cells(segmented_images):
    """
    Tracks segmented cells over time using BayesianTracker (btrack).

    Parameters:
    -----------
    segmented_images : np.ndarray
        3D array (time, height, width) of labeled segmented images (each cell should have a unique label).

    Returns:
    --------
    list, dict
        A list of track dictionaries and a lineage graph.
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

    # Check if images are binary and convert to labeled if needed
    if set(np.unique(segmented_images)).issubset({0, 255}):
        print("Converting binary masks to labeled images...")
        from skimage.measure import label
        labeled_images = np.zeros_like(segmented_images)
        for i in range(segmented_images.shape[0]):
            labeled_images[i] = label(segmented_images[i] > 0)
        segmented_images = labeled_images

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
            h, w = segmented_images.shape[1:3]
            print(f"Tracker volume dimensions: ((0, {w}), (0, {h}))")
            tracker.volume = ((0, w), (0, h))  # 2D tracking - no z dimension

            print("Starting tracking process...")
            # Track with step_size for progress updates
            tracker.track(step_size=100)

            # Optimize tracks (resolves track hypotheses)
            print("Optimizing tracks...")
            tracker.optimize()

            # Get the tracks and graph data for visualization
            data, properties, graph = tracker.to_napari()

            # Get raw tracks
            tracks = tracker.tracks

            # Print statistics about the tracks
            track_lengths = [len(track.x) for track in tracks]
            avg_length = sum(track_lengths) / \
                len(track_lengths) if track_lengths else 0
            max_length = max(track_lengths) if track_lengths else 0

            print(f"Total tracks: {len(tracks)}")
            print(f"Average track length: {avg_length:.2f} frames")
            print(f"Longest track: {max_length} frames")

            # Count tracks by length
            short_tracks = sum(1 for length in track_lengths if length < 5)
            medium_tracks = sum(
                1 for length in track_lengths if 5 <= length < 15)
            long_tracks = sum(1 for length in track_lengths if length >= 15)

            print(f"Short tracks (<5 frames): {short_tracks}")
            print(f"Medium tracks (5-14 frames): {medium_tracks}")
            print(f"Long tracks (≥15 frames): {long_tracks}")

            # Analyze division events if present
            division_events = 0
            for track in tracks:
                if hasattr(track, 'children') and track.children:
                    division_events += 1

            print(f"Cell division events: {division_events}")

    except Exception as e:
        raise RuntimeError(f"Failed to track cells: {e}")

    # Convert tracks to dictionary format for compatibility with visualization
    dict_tracks = []
    for track in tracks:
        # Extract track data
        track_dict = {
            'ID': track.ID,
            'x': track.x,
            'y': track.y,
            't': track.t if hasattr(track, 't') else list(range(len(track.x)))
        }

        # Add lineage information with validation
        try:
            # Only set parent if it's different from own ID
            if hasattr(track, 'parent') and track.parent != track.ID:
                track_dict['parent'] = track.parent
            else:
                track_dict['parent'] = None

            if hasattr(track, 'children') and track.children and len(track.children) > 0:
                track_dict['children'] = track.children.copy()
            else:
                track_dict['children'] = []
        except Exception as e:
            print(
                f"Warning: Could not extract lineage for track {track.ID}: {e}")
            track_dict['parent'] = None
            track_dict['children'] = []

        dict_tracks.append(track_dict)

    print(f"Converted {len(dict_tracks)} tracks to dictionary format")
    return dict_tracks, graph


def optimize_tracking_parameters(segmented_images, test_frames=None):
    """
    Helper function to find optimal tracking parameters for a specific dataset.

    Parameters:
    -----------
    segmented_images : np.ndarray
        3D array of segmented images to analyze
    test_frames : tuple
        Optional (start, end) tuple to limit analysis to a subset of frames

    Returns:
    --------
    dict
        Dictionary of suggested parameters
    """
    import numpy as np
    from skimage.measure import regionprops

    print("Analyzing dataset to suggest optimal tracking parameters...")

    # Focus on a subset of frames if specified
    if test_frames is not None:
        start, end = test_frames
        frames_to_analyze = segmented_images[start:end]
    else:
        frames_to_analyze = segmented_images

    # Calculate cell density
    density_values = []
    for t in range(min(10, frames_to_analyze.shape[0])):
        # Subtract 1 for background
        num_cells = len(np.unique(frames_to_analyze[t])) - 1
        frame_size = frames_to_analyze[t].shape[0] * \
            frames_to_analyze[t].shape[1]
        density = num_cells / frame_size
        density_values.append(density * 10000)  # Scale for readability

    avg_density = np.mean(density_values)
    print(f"Average cell density: {avg_density:.2f} cells per 10k pixels")

    # Calculate average cell size
    cell_sizes = []
    for t in range(min(3, frames_to_analyze.shape[0])):
        props = regionprops(frames_to_analyze[t])
        for prop in props:
            cell_sizes.append(prop.area)

    avg_cell_size = np.mean(cell_sizes) if cell_sizes else 0
    print(f"Average cell size: {avg_cell_size:.1f} pixels")

    # Analyze movement between frames (if multiple frames available)
    movement_distances = []

    if frames_to_analyze.shape[0] >= 2:
        # Extract centroids from first 5 frames (or fewer if not available)
        max_frames = min(5, frames_to_analyze.shape[0]-1)

        for t in range(max_frames):
            current_centroids = {}
            next_centroids = {}

            # Get current frame centroids
            props = regionprops(frames_to_analyze[t])
            for prop in props:
                current_centroids[prop.label] = prop.centroid

            # Get next frame centroids
            props = regionprops(frames_to_analyze[t+1])
            for prop in props:
                next_centroids[prop.label] = prop.centroid

            # For demonstration - in real tracking we'd need to match cells between frames
            # This is just to get a rough estimate of movement
            if len(current_centroids) > 0 and len(next_centroids) > 0:
                curr_points = np.array(list(current_centroids.values()))
                next_points = np.array(list(next_centroids.values()))

                # For each current centroid, find closest in next frame
                from scipy.spatial.distance import cdist
                dist_matrix = cdist(curr_points, next_points)
                min_distances = np.min(dist_matrix, axis=1)
                movement_distances.extend(min_distances)

    avg_movement = np.mean(movement_distances) if movement_distances else 0
    print(f"Average movement between frames: {avg_movement:.2f} pixels")

    # Determine optimal parameters based on analysis
    suggested_params = {}

    # 1. Determine search radius based on cell movement
    if avg_movement > 0:
        # Set search radius to average movement + buffer
        suggested_params['max_search_radius'] = int(
            min(max(avg_movement * 2, 15), 50))
    else:
        # Default to conservative value
        suggested_params['max_search_radius'] = 25

    # 2. Determine optimization level based on density
    if avg_density > 100:  # Extremely dense
        suggested_params['optimization_level'] = 3
        suggested_params['max_lost_frames'] = 3
    elif avg_density > 50:  # Very dense
        suggested_params['optimization_level'] = 2
        suggested_params['max_lost_frames'] = 4
    elif avg_density > 20:  # Moderately dense
        suggested_params['optimization_level'] = 1
        suggested_params['max_lost_frames'] = 5
    else:  # Sparse
        suggested_params['optimization_level'] = 0
        suggested_params['max_lost_frames'] = 7

    # 3. Set minimum track length based on cell size
    if avg_cell_size > 0:
        # Smaller cells tend to need longer tracks to filter noise
        if avg_cell_size < 30:
            suggested_params['min_track_length'] = 4
        else:
            suggested_params['min_track_length'] = 3
    else:
        suggested_params['min_track_length'] = 3

    # 4. Set distance threshold based on movement and density
    if avg_movement > 0:
        suggested_params['max_distance_threshold'] = max(avg_movement * 3, 20)
    else:
        suggested_params['max_distance_threshold'] = 30

    print("\nSuggested tracking parameters for this dataset:")
    for param, value in suggested_params.items():
        print(f"  {param}: {value}")

    return suggested_params


def overlay_tracks_on_images(segmented_images, tracks, save_video=True, output_path="tracked_cells.mp4", show_frames=False,
                             max_tracks=None, progress_callback=None):
    """
    Overlays tracking trajectories on segmented images and creates a video.

    Parameters:
    -----------
    segmented_images : np.ndarray
        3D array (time, height, width) of segmented images.
    tracks : list
        List of tracked cell dictionaries.
    save_video : bool
        If True, saves the output as a video.
    output_path : str
        Path to save the output video.
    show_frames : bool
        If True, displays each frame with matplotlib.
    max_tracks : int or None
        Maximum number of tracks to display. If None, shows all tracks.
    progress_callback : function or None
        Callback function to report progress (takes a value from 0-100).
    """
    if len(segmented_images) == 0 or len(tracks) == 0:
        print("No segmented images or tracks to visualize.")
        return

    height, width = segmented_images.shape[1:]

    # Filter tracks if needed
    if max_tracks is not None and max_tracks < len(tracks):
        # Sort by track length and take the longest ones
        sorted_tracks = sorted(
            tracks, key=lambda track: len(track['x']), reverse=True)
        tracks = sorted_tracks[:max_tracks]

    # Generate consistent colors for tracks
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', min(20, len(tracks)))

    colors = {}
    for i, track in enumerate(tracks):
        track_id = track['ID']
        # Convert matplotlib color (0-1 range) to OpenCV color (0-255 range)
        color = tuple(int(255 * x) for x in cmap(i % 20)[:3])
        # OpenCV uses BGR format
        color = (color[2], color[1], color[0])
        colors[track_id] = color

    # Setup video writer if needed
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 5  # Adjust as needed
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    for t in range(segmented_images.shape[0]):
        # Report progress if callback is provided
        if progress_callback:
            progress_percentage = int((t / segmented_images.shape[0]) * 100)
            progress_callback(progress_percentage)

        # Convert frame to RGB
        frame = segmented_images[t].copy()

        # Use label2rgb to get colored segmentation
        # Convert binary segmentation to labeled regions if needed
        if np.max(frame) <= 1:
            frame = label(frame)

        frame_rgb = label2rgb(frame, bg_label=0)
        frame_rgb = (frame_rgb * 255).astype(np.uint8)

        # Draw trails for each track
        for track in tracks:
            track_id = track['ID']
            x_coords, y_coords = track['x'], track['y']
            times = track['t'] if 't' in track else list(range(len(x_coords)))

            color = colors[track_id]

            # Draw full trajectory up to current time
            for i in range(len(times) - 1):
                if times[i+1] <= t:  # Only draw up to current time
                    pt1 = (int(x_coords[i]), int(y_coords[i]))
                    pt2 = (int(x_coords[i+1]), int(y_coords[i+1]))
                    cv2.line(frame_rgb, pt1, pt2, color, 1)

            # Draw the current position if the track exists at this time
            current_points = [(i, x, y) for i, (x, y, tm) in enumerate(
                zip(x_coords, y_coords, times)) if tm == t]

            for idx, x, y in current_points:
                # Mark current position with larger circle
                cv2.circle(frame_rgb, (int(x), int(y)), 4, color, -1)

                # Add track ID label
                cv2.putText(frame_rgb, str(track_id), (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Mark division events
                if 'children' in track and track['children'] and t == times[-1]:
                    # Draw division marker (star or 'X')
                    cv2.drawMarker(frame_rgb, (int(x), int(y)),
                                   (255, 255, 0), markerType=cv2.MARKER_STAR,
                                   markerSize=10, thickness=2)

        # Add frame number
        cv2.putText(frame_rgb, f"Frame: {t}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show frame if requested
        if show_frames:
            plt.figure(figsize=(10, 8))
            plt.imshow(frame_rgb)
            plt.title(f"Tracked Cells - Frame {t}")
            plt.axis("off")
            plt.show()

        # Write frame to video if saving
        if save_video and out is not None:
            out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # Release the video writer
    if save_video and out is not None:
        out.release()
        print(f"Video saved at {output_path}")

    # Complete progress
    if progress_callback:
        progress_callback(100)


def visualize_lineage_tree(tracks, output_path=None, min_track_length=5, progress_callback=None):
    """
    Creates a lineage tree visualization showing cell divisions.

    Parameters:
    -----------
    tracks : list
        List of track dictionaries with lineage information.
    output_path : str or None
        If provided, saves the visualization to this path.
    min_track_length : int
        Minimum track length to include in the visualization.
    progress_callback : function
        Optional callback to report progress (takes a value from 0-100).
    """
    try:
        import networkx as nx
        from matplotlib.collections import LineCollection
    except ImportError:
        print("Error: networkx or matplotlib not installed. Please install with:")
        print("pip install networkx matplotlib")
        return

    if progress_callback:
        progress_callback(10)

    # Filter tracks by length
    filtered_tracks = [t for t in tracks if len(t['x']) >= min_track_length]
    print(f"Visualizing {len(filtered_tracks)} tracks after filtering.")

    # Create a directed graph
    G = nx.DiGraph()

    # Organize tracks by start time
    for track in filtered_tracks:
        track_id = track['ID']
        start_time = track['t'][0] if track['t'] else 0
        track_length = len(track['x'])

        # Add node with attributes
        G.add_node(track_id, start_time=start_time,
                   length=track_length, track=track)

        # Add edge from parent to this track if available
        if track['parent'] is not None:
            G.add_edge(track['parent'], track_id)

    if progress_callback:
        progress_callback(30)

    # Plot settings
    plt.figure(figsize=(14, 10))

    # Position nodes based on start time (y-axis) and a layout algorithm (x-axis)
    pos = {}

    # Use networkx to generate initial x positions
    try:
        base_pos = nx.spring_layout(G, seed=42)
    except Exception as e:
        print(f"Warning: Error during layout generation: {e}")
        print("Using random positions instead")
        import random
        base_pos = {node: (random.random(), random.random())
                    for node in G.nodes()}

    if progress_callback:
        progress_callback(50)

    # Adjust positions: y-axis is start time, preserve x from layout
    for node in G.nodes():
        start_time = G.nodes[node]['start_time']
        # Negative to make time flow downward
        pos[node] = (base_pos[node][0], -start_time)

    # Draw nodes
    node_sizes = [max(100, G.nodes[n]['length'] * 5) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='skyblue', alpha=0.8)

    if progress_callback:
        progress_callback(70)

    # Draw edges with arrows showing parent-child relationships
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.8,
                           arrows=True, arrowstyle='-|>', arrowsize=15)

    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    if progress_callback:
        progress_callback(90)

    # Add title and labels
    plt.title("Cell Lineage Tree")
    plt.xlabel("Cell Divisions")
    plt.ylabel("Time (frames)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().invert_yaxis()  # Invert y-axis to have time flowing downward

    # Add stats
    total_tracks = len(filtered_tracks)
    division_events = sum(1 for t in filtered_tracks if t.get('children', []))

    stats_text = f"Total Tracks: {total_tracks}\nDivision Events: {division_events}"
    plt.figtext(0.02, 0.02, stats_text, wrap=True, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Lineage tree saved to {output_path}")
    else:
        plt.show()

    if progress_callback:
        progress_callback(100)


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


def tracks_to_dataframe(tracks, features=None):
    """
    Convert tracks to a pandas DataFrame.

    Parameters:
    -----------
    tracks : list
        List of track objects from btrack.
    features : list, optional
        List of feature names to include.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing track data.
    """
    import pandas as pd

    if features is None:
        features = [
            "area",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "solidity"
        ]

    data = []

    for track in tracks:
        # Get track data
        track_id = track.ID

        # Process each point in the track
        for i in range(len(track.t)):
            # Create a row for each timepoint
            row = {
                'ID': track_id,
                't': track.t[i],
                'x': track.x[i],
                'y': track.y[i],
                'z': 0.0,  # Most 2D tracking doesn't use z
                'parent': track.parent if hasattr(track, 'parent') else None,
                'root': track.root if hasattr(track, 'root') else None,
                'state': track.state if hasattr(track, 'state') else None,
                'generation': track.generation if hasattr(track, 'generation') else None
            }

            # Add any features the track might have
            if hasattr(track, 'features') and track.features is not None:
                for feature in features:
                    if i < len(track.features) and feature in track.features[i]:
                        row[feature] = track.features[i][feature]

            data.append(row)

    # Create DataFrame and sort by ID and t
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(['ID', 't']).reset_index(drop=True)

    return df
