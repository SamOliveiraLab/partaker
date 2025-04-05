import btrack
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import label
from scipy import stats
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Rectangle


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

            if hasattr(
                    track, 'children') and track.children and len(
                    track.children) > 0:
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
        max_frames = min(5, frames_to_analyze.shape[0] - 1)

        for t in range(max_frames):
            current_centroids = {}
            next_centroids = {}

            # Get current frame centroids
            props = regionprops(frames_to_analyze[t])
            for prop in props:
                current_centroids[prop.label] = prop.centroid

            # Get next frame centroids
            props = regionprops(frames_to_analyze[t + 1])
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


def overlay_tracks_on_images(
        segmented_images,
        tracks,
        save_video=True,
        output_path="tracked_cells.mp4",
        show_frames=False,
        max_tracks=None,
        progress_callback=None):
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
                if times[i + 1] <= t:  # Only draw up to current time
                    pt1 = (int(x_coords[i]), int(y_coords[i]))
                    pt2 = (int(x_coords[i + 1]), int(y_coords[i + 1]))
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


def visualize_lineage_tree(
        tracks,
        output_path=None,
        min_track_length=5,
        progress_callback=None):
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

    # Position nodes based on start time (y-axis) and a layout algorithm
    # (x-axis)
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
                    if i < len(
                            track.features) and feature in track.features[i]:
                        row[feature] = track.features[i][feature]

            data.append(row)

    # Create DataFrame and sort by ID and t
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(['ID', 't']).reset_index(drop=True)

    return df


def calculate_cell_motility(track):
    """
    Calculate motility metrics for a single cell track.

    Parameters:
    track: dict with keys 'x', 'y', 't' containing position and time data

    Returns:
    dict of motility metrics
    """
    # Extract coordinates and times
    x_coords = np.array(track['x'])
    y_coords = np.array(track['y'])
    times = np.array(track['t']) if 't' in track else np.arange(len(x_coords))

    # Check if track has enough points
    if len(x_coords) < 2:
        return {
            "path_length": 0,
            "net_displacement": 0,
            "tortuosity": 1,
            "avg_velocity": 0,
            "max_velocity": 0,
            "min_velocity": 0,
            "median_velocity": 0,
            "direction_angle": 0
        }

    # Calculate distances between consecutive points
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    dt = np.diff(times)
    dt = np.where(dt == 0, 1, dt)  # Avoid division by zero
    distances = np.sqrt(dx**2 + dy**2)

    # Calculate metrics
    path_length = np.sum(distances)

    # Net displacement (start to end)
    net_displacement = np.sqrt(
        (x_coords[-1] - x_coords[0])**2 +
        (y_coords[-1] - y_coords[0])**2
    )

    # Tortuosity (path length / net displacement)
    tortuosity = path_length / net_displacement if net_displacement > 0 else 1

    # Average velocity
    total_time = times[-1] - times[0] if len(times) > 1 else 1
    total_time = max(1, total_time)  # Avoid division by zero
    avg_velocity = path_length / total_time

    # Instantaneous velocities
    inst_velocities = distances / dt

    # Calculate direction angle (in radians)
    if net_displacement > 0:
        direction_angle = np.arctan2(
            y_coords[-1] - y_coords[0],
            x_coords[-1] - x_coords[0]
        )
    else:
        direction_angle = 0

    return {
        "path_length": path_length,
        "net_displacement": net_displacement,
        "tortuosity": tortuosity,
        "avg_velocity": avg_velocity,
        "max_velocity": np.max(inst_velocities) if len(inst_velocities) > 0 else 0,
        "min_velocity": np.min(inst_velocities) if len(inst_velocities) > 0 else 0,
        "median_velocity": np.median(inst_velocities) if len(inst_velocities) > 0 else 0,
        "direction_angle": direction_angle}


def calculate_motility_index(tracks):
    """
    Calculate a motility index for an entire population of cells.

    Parameters:
    tracks: list of track dictionaries with x, y, t positions

    Returns:
    float: motility index value, dict: detailed metrics
    """
    if not tracks:
        return 0, {}

    # Calculate individual motility metrics for each track
    motility_metrics = [calculate_cell_motility(track) for track in tracks]

    # Calculate aggregate statistics
    displacements = [m["net_displacement"] for m in motility_metrics]
    velocities = [m["avg_velocity"] for m in motility_metrics]
    tortuosities = [m["tortuosity"] for m in motility_metrics]
    path_lengths = [m["path_length"] for m in motility_metrics]

    avg_displacement = np.mean(displacements)
    avg_velocity = np.mean(velocities)
    avg_tortuosity = np.mean(tortuosities)
    avg_path_length = np.mean(path_lengths)

    # Calculate directional coherence (how aligned the cell movements are)
    angles = [m["direction_angle"] for m in motility_metrics]
    x_components = np.cos(angles)
    y_components = np.sin(angles)

    # Average the vectors
    mean_x = np.mean(x_components)
    mean_y = np.mean(y_components)

    # Length of resulting vector (0-1)
    directional_coherence = np.sqrt(mean_x**2 + mean_y**2)

    # Calculate variances (for measuring population heterogeneity)
    displacement_variance = np.var(displacements)
    velocity_variance = np.var(velocities)

    # Calculate coefficient of variation (CV) for displacement and velocity
    cv_displacement = (np.std(displacements) /
                       avg_displacement) if avg_displacement > 0 else 0
    cv_velocity = (
        np.std(velocities) /
        avg_velocity) if avg_velocity > 0 else 0

    # Calculate the motility index
    # Normalize each component to 0-1 scale (with reasonable ranges)
    # Assuming 200 pixels is high
    norm_displacement = min(1.0, avg_displacement / 200)
    # Assuming 20 pixels/frame is high
    norm_velocity = min(1.0, avg_velocity / 20)

    # Inverse of tortuosity (higher value = more direct movement)
    directness = min(1.0, 1.0 / avg_tortuosity)

    # Combine metrics with weights
    motility_index = (
        0.35 * norm_displacement +
        0.35 * norm_velocity +
        0.15 * directness +
        0.15 * directional_coherence
    )

    # Scale to 0-100 for easier interpretation
    motility_index *= 100

    # Detailed metrics for deeper analysis
    detailed_metrics = {
        "average_displacement": avg_displacement,
        "average_velocity": avg_velocity,
        "average_tortuosity": avg_tortuosity,
        "average_path_length": avg_path_length,
        "directional_coherence": directional_coherence,
        "displacement_variance": displacement_variance,
        "velocity_variance": velocity_variance,
        "cv_displacement": cv_displacement,
        "cv_velocity": cv_velocity,
        "noise_displacement": cv_displacement**2,  # Noise = CV²
        "noise_velocity": cv_velocity**2,
        "individual_metrics": motility_metrics
    }

    return motility_index, detailed_metrics


def plot_motility_gauge(ax, motility_index):
    """
    Create a gauge-like visualization for the motility index.

    Parameters:
    ax: matplotlib axis
    motility_index: float, the calculated motility index value
    """
    # Set up the gauge
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw gauge background
    theta = np.linspace(0.75 * np.pi, 0.25 * np.pi, 100)
    r_inner = 3
    r_outer = 4

    # Gauge background
    for r in np.linspace(r_inner, r_outer, 10):
        x = 5 + r * np.cos(theta)
        y = 5 + r * np.sin(theta)
        ax.plot(x, y, color='lightgray', linewidth=1)

    # Color bands (red to green)
    colors = ['#ff3232', '#ff7f32', '#ffcb32', '#cbff32', '#7fff32', '#32ff32']
    for i, color in enumerate(colors):
        theta_band = np.linspace(0.75 *
                                 np.pi -
                                 i *
                                 0.5 *
                                 np.pi /
                                 len(colors), 0.75 *
                                 np.pi -
                                 (i +
                                  1) *
                                 0.5 *
                                 np.pi /
                                 len(colors), 50)
        x_outer = 5 + r_outer * np.cos(theta_band)
        y_outer = 5 + r_outer * np.sin(theta_band)
        x_inner = 5 + r_inner * np.cos(theta_band)
        y_inner = 5 + r_inner * np.sin(theta_band)

        # Combine the points to form a polygon
        x = np.concatenate([x_outer, x_inner[::-1]])
        y = np.concatenate([y_outer, y_inner[::-1]])
        ax.fill(x, y, color=color, alpha=0.7)

    # Scale markers and labels
    for i, label in enumerate(['0', '20', '40', '60', '80', '100']):
        angle = 0.75 * np.pi - i * 0.1 * np.pi
        x = 5 + 4.5 * np.cos(angle)
        y = 5 + 4.5 * np.sin(angle)
        ax.text(x, y, label, ha='center', va='center', fontsize=8)

    # Needle
    needle_angle = 0.75 * np.pi - (motility_index / 100) * 0.5 * np.pi
    ax.plot([5, 5 + 4.2 * np.cos(needle_angle)],
            [5, 5 + 4.2 * np.sin(needle_angle)],
            color='black', linewidth=2)

    # Center circle
    circle = plt.Circle((5, 5), 0.3, color='darkgray')
    ax.add_patch(circle)

    # Display motility index value
    ax.text(5, 3, f"Motility Index", ha='center', fontsize=12)
    ax.text(
        5,
        2,
        f"{motility_index:.1f}",
        ha='center',
        fontsize=16,
        fontweight='bold')



def visualize_cell_regions(tracks, chamber_dimensions=(1392, 1040)):
    """Visualize cell positions with color-coded regions."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    
    # Extract all x,y positions from tracks
    all_x = []
    all_y = []
    for track in tracks:
        all_x.extend(track['x'])
        all_y.extend(track['y'])
    
    # Create figure
    plt.figure(figsize=(12, 9))
    
    # Define region boundaries
    width, height = chamber_dimensions
    edge_margin = 50
    corner_size = 100
    right_channel_x = 1300
    right_channel_width = 50
    
    # Create background for regions
    # Corner regions
    corner_color = 'mistyrose'
    plt.gca().add_patch(Rectangle((0, 0), corner_size, corner_size, 
                                 color=corner_color, alpha=0.3))  # Top-left
    plt.gca().add_patch(Rectangle((0, height-corner_size), corner_size, corner_size, 
                                 color=corner_color, alpha=0.3))  # Bottom-left
    plt.gca().add_patch(Rectangle((width-corner_size, 0), corner_size, corner_size, 
                                 color=corner_color, alpha=0.3))  # Top-right
    plt.gca().add_patch(Rectangle((width-corner_size, height-corner_size), corner_size, corner_size, 
                                 color=corner_color, alpha=0.3))  # Bottom-right
    
    # Edge regions (excluding corners)
    edge_color = 'lightblue'
    plt.gca().add_patch(Rectangle((0, corner_size), edge_margin, height-2*corner_size, 
                                 color=edge_color, alpha=0.3))  # Left edge
    plt.gca().add_patch(Rectangle((width-edge_margin, corner_size), edge_margin, height-2*corner_size, 
                                 color=edge_color, alpha=0.3))  # Right edge
    plt.gca().add_patch(Rectangle((corner_size, 0), width-2*corner_size, edge_margin, 
                                 color=edge_color, alpha=0.3))  # Top edge
    plt.gca().add_patch(Rectangle((corner_size, height-edge_margin), width-2*corner_size, edge_margin, 
                                 color=edge_color, alpha=0.3))  # Bottom edge
    
    # Right channel
    plt.gca().add_patch(Rectangle((right_channel_x, 0), right_channel_width, height, 
                                 color='lightgreen', alpha=0.3))
    
    # Left edge (inlet)
    plt.gca().add_patch(Rectangle((0, 0), edge_margin, height, 
                                 color='lightyellow', alpha=0.3))
    
    # Create scatter plot of cell positions
    plt.scatter(all_x, all_y, alpha=0.3, s=1, color='blue')
    
    # Add labels and title
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Color-Coded Chamber Regions with Cell Positions')
    
    # Add region labels
    plt.text(25, height/2, 'INLET', rotation=90, ha='center', va='center', fontsize=12)
    plt.text(right_channel_x + right_channel_width/2, height/2, 'RIGHT CHANNEL', 
             rotation=90, ha='center', va='center', fontsize=12)
    plt.text(width/2, height/2, 'CENTER', ha='center', va='center', fontsize=14)
    plt.text(corner_size/2, corner_size/2, 'CORNER', ha='center', va='center', fontsize=10)
    plt.text(width-corner_size/2, height-corner_size/2, 'CORNER', ha='center', va='center', fontsize=10)
    
    # Add legend for regions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightyellow', edgecolor='none', alpha=0.3, label='Inlet Region'),
        Patch(facecolor='lightgreen', edgecolor='none', alpha=0.3, label='Right Channel'),
        Patch(facecolor='mistyrose', edgecolor='none', alpha=0.3, label='Corner Regions'),
        Patch(facecolor='lightblue', edgecolor='none', alpha=0.3, label='Edge Regions'),
        Patch(facecolor='white', edgecolor='none', label='Center Region')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=10)
    
    # Add grid and set limits
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, width)
    plt.ylim(0, height)
    
    # Save and show the color-coded plot
    # plt.tight_layout()
    # plt.savefig('color_coded_chamber_regions.png', dpi=300, bbox_inches='tight')
    # plt.show()
    

def visualize_motility_with_chamber_regions(tracks, all_cell_positions, chamber_dimensions, motility_metrics=None):
    """
    Visualize cell motility patterns overlaid on the chamber regions map with all cell positions.
    
    Parameters:
    -----------
    tracks : list
        List of track dictionaries for trajectory analysis
    all_cell_positions : list of tuples or numpy array
        (x, y) coordinates of all detected cells across frames
    chamber_dimensions : tuple
        (width, height) of the chamber
    motility_metrics : dict, optional
        Output from enhanced_motility_index function
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    # Calculate motility metrics if not provided
    if motility_metrics is None:
        from tracking import enhanced_motility_index
        motility_metrics = enhanced_motility_index(tracks, chamber_dimensions)
    
    # Create a lookup of track_id to metrics
    metrics_by_id = {m['track_id']: m for m in motility_metrics['individual_metrics']}
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define chamber regions
    width, height = chamber_dimensions
    edge_margin = 50
    corner_size = 100
    right_channel_x = 1200
    right_channel_width = 70
    
    # Draw chamber regions (with more subtle colors)
    # Corners
    corner_color = 'mistyrose'
    ax.add_patch(plt.Rectangle((0, 0), corner_size, corner_size, 
                              color=corner_color, alpha=0.2))  # Bottom-left
    ax.add_patch(plt.Rectangle((0, height-corner_size), corner_size, corner_size, 
                              color=corner_color, alpha=0.2))  # Top-left
    ax.add_patch(plt.Rectangle((width-corner_size, 0), corner_size, corner_size, 
                              color=corner_color, alpha=0.2))  # Bottom-right
    ax.add_patch(plt.Rectangle((width-corner_size, height-corner_size), corner_size, corner_size, 
                              color=corner_color, alpha=0.2))  # Top-right
    
    # Edges
    edge_color = 'lightblue'
    ax.add_patch(plt.Rectangle((0, corner_size), edge_margin, height-2*corner_size, 
                              color=edge_color, alpha=0.2))  # Left edge
    ax.add_patch(plt.Rectangle((width-edge_margin, corner_size), edge_margin, height-2*corner_size, 
                              color=edge_color, alpha=0.2))  # Right edge
    ax.add_patch(plt.Rectangle((corner_size, 0), width-2*corner_size, edge_margin, 
                              color=edge_color, alpha=0.2))  # Bottom edge
    ax.add_patch(plt.Rectangle((corner_size, height-edge_margin), width-2*corner_size, edge_margin, 
                              color=edge_color, alpha=0.2))  # Top edge
                              
    # Right channel
    ax.add_patch(plt.Rectangle((right_channel_x, 0), right_channel_width, height, 
                             color='lightgreen', alpha=0.2))
    
    # Left edge (inlet)
    ax.add_patch(plt.Rectangle((0, 0), edge_margin, height, 
                             color='lightyellow', alpha=0.2))
    
    # 1. Plot all cell positions as small, semi-transparent dots
    if all_cell_positions:
        # Extract x, y coordinates
        if isinstance(all_cell_positions, list):
            if all_cell_positions and isinstance(all_cell_positions[0], tuple):
                # List of (x,y) tuples
                x_all = [p[0] for p in all_cell_positions]
                y_all = [p[1] for p in all_cell_positions]
            else:
                # Some other format, try to extract positions from tracks
                x_all = []
                y_all = []
                for track in tracks:
                    x_all.extend(track['x'])
                    y_all.extend(track['y'])
        else:
            # Numpy array of points [N, 2]
            x_all = all_cell_positions[:, 0]
            y_all = all_cell_positions[:, 1]
        
        # Plot as a scatter with blue, semi-transparent dots
        plt.scatter(x_all, y_all, s=2, color='blue', alpha=0.1)
    
    # 2. Set up colormap for motility index
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=100)  # Motility index 0-100
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # 3. Plot tracked cell trajectories with color based on motility
    for track in tracks:
        track_id = track.get('ID', -1)
        if track_id in metrics_by_id:
            metrics = metrics_by_id[track_id]
            motility_index = metrics['motility_index']
            
            # Get track coordinates
            x = track['x']
            y = track['y']
            
            # Plot track colored by motility index
            plt.plot(x, y, '-', linewidth=2, alpha=0.8, 
                    color=cmap(norm(motility_index)))
            
            # Mark start and end points
            plt.plot(x[0], y[0], 'o', markersize=4, color=cmap(norm(motility_index)))
            plt.plot(x[-1], y[-1], 's', markersize=4, color=cmap(norm(motility_index)))
    
    # Add region labels
    plt.text(edge_margin/2, height/2, 'INLET', rotation=90, ha='center', va='center', fontsize=12)
    plt.text(right_channel_x + right_channel_width/2, height/2, 'RIGHT CHANNEL', 
             rotation=90, ha='center', va='center', fontsize=12)
    plt.text(width/2, height/2, 'CENTER', ha='center', va='center', fontsize=14)
    plt.text(corner_size/2, corner_size/2, 'CORNER', ha='center', va='center', fontsize=10)
    plt.text(width-corner_size/2, height-corner_size/2, 'CORNER', ha='center', va='center', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Motility Index (0-100)')
    
    # Add legend for regions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightyellow', edgecolor='none', alpha=0.2, label='Inlet Region'),
        Patch(facecolor='lightgreen', edgecolor='none', alpha=0.2, label='Right Channel'),
        Patch(facecolor='mistyrose', edgecolor='none', alpha=0.2, label='Corner Regions'),
        Patch(facecolor='lightblue', edgecolor='none', alpha=0.2, label='Edge Regions'),
        Patch(facecolor='white', edgecolor='none', label='Center Region')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1),
              ncol=3, fontsize=10)
    
    # Add statistical summary
    summary_text = (
        f"Population Metrics:\n"
        f"Avg Motility: {motility_metrics['population_avg_motility']:.1f}\n"
        f"Std Dev: {motility_metrics['population_std_motility']:.1f}\n"
        f"Heterogeneity: {motility_metrics['population_heterogeneity']:.2f}\n"
        f"Sample Size: {motility_metrics['sample_size']} tracks"
    )
    plt.figtext(0.02, 0.02, summary_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    # Set labels and title
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Cell Motility by Chamber Region')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Set axis limits
    plt.xlim(0, width)
    plt.ylim(0, height)
    
    plt.tight_layout()
    return fig, ax    


def enhanced_motility_index(tracks, chamber_dimensions=None):
    """
    Calculate a comprehensive motility index for bacterial cells.
    
    Parameters:
    -----------
    tracks : list
        List of track dictionaries with x, y, t positions
    chamber_dimensions : tuple, optional
        (width, height) of the chamber for normalization
        
    Returns:
    --------
    dict
        Dictionary containing motility index and detailed metrics
    """
    # Initialize containers for metrics
    track_metrics = []
    
    # Process each track
    for track in tracks:
        if len(track.get('x', [])) < 3:  # Skip very short tracks
            continue
            
        # Extract coordinates and times
        x = np.array(track['x'])
        y = np.array(track['y'])
        t = np.array(track['t']) if 't' in track else np.arange(len(x))
        
        # Time intervals between frames
        dt = np.diff(t)
        dt = np.where(dt == 0, 1, dt)  # Prevent division by zero
        
        # Calculate distances between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)
        step_distances = np.sqrt(dx**2 + dy**2)
        
        # 1. Displacement metrics
        start_point = (x[0], y[0])
        end_point = (x[-1], y[-1])
        
        net_displacement = np.sqrt((end_point[0] - start_point[0])**2 + 
                                  (end_point[1] - start_point[1])**2)
        path_length = np.sum(step_distances)
        confinement_ratio = net_displacement / path_length if path_length > 0 else 0
        
        # 2. Velocity metrics
        instantaneous_velocities = step_distances / dt
        avg_velocity = np.mean(instantaneous_velocities)
        max_velocity = np.max(instantaneous_velocities)
        velocity_cv = np.std(instantaneous_velocities) / avg_velocity if avg_velocity > 0 else 0
        
        # 3. Directional metrics
        # Calculate turning angles
        turning_angles = []
        for i in range(len(dx) - 1):
            # Vector 1
            v1 = np.array([dx[i], dy[i]])
            # Vector 2
            v2 = np.array([dx[i+1], dy[i+1]])
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                # Calculate angle between vectors
                cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                turning_angles.append(angle)
        
        # Directional persistence
        mean_turning_angle = np.mean(turning_angles) if turning_angles else np.pi/2
        directional_persistence = 1 - (mean_turning_angle / np.pi)  # 1 = straight, 0 = random
        
        # MSD calculation (simplified)
        # For a full implementation, calculate MSD at multiple time lags
        msd = net_displacement**2 / len(t)
        
        # 4. Regional positioning
        if chamber_dimensions:
            width, height = chamber_dimensions
            # Determine if cell is in center, edge, or corner
            center_x, center_y = width/2, height/2
            distance_from_center = np.sqrt((np.mean(x) - center_x)**2 + (np.mean(y) - center_y)**2)
            normalized_center_distance = distance_from_center / np.sqrt(center_x**2 + center_y**2)
            
            # Higher value means closer to edge
            region_factor = normalized_center_distance
        else:
            region_factor = 0.5  # Neutral if no chamber dimensions provided
        
        # Calculate combined motility index
        # Weights can be adjusted based on importance of each factor
        weights = {
            'confinement': 0.25,
            'velocity': 0.25,
            'persistence': 0.25,
            'region': 0.25
        }
        
        # Normalize each component to 0-1 scale
        norm_confinement = 1 - confinement_ratio  # Invert so higher = more motile
        
        # Normalize velocity (assuming 20 pixels/frame is high motility)
        norm_velocity = min(1.0, avg_velocity / 20) 
        
        motility_index = (
            weights['confinement'] * norm_confinement +
            weights['velocity'] * norm_velocity +
            weights['persistence'] * directional_persistence +
            weights['region'] * region_factor
        ) * 100  # Scale to 0-100
        
        # Store all metrics for this track
        track_metrics.append({
            'track_id': track.get('ID', -1),
            'net_displacement': net_displacement,
            'path_length': path_length,
            'confinement_ratio': confinement_ratio,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'velocity_cv': velocity_cv,
            'directional_persistence': directional_persistence,
            'mean_turning_angle': mean_turning_angle,
            'msd': msd,
            'region_factor': region_factor,
            'track_length': len(t),
            'motility_index': motility_index
        })
    
    # Population-level statistics
    if track_metrics:
        # Calculate population metrics
        all_indices = [t['motility_index'] for t in track_metrics]
        population_avg_index = np.mean(all_indices)
        population_std_index = np.std(all_indices)
        
        # Calculate motility heterogeneity (coefficient of variation)
        heterogeneity = population_std_index / population_avg_index if population_avg_index > 0 else 0
        
        result = {
            'individual_metrics': track_metrics,
            'population_avg_motility': population_avg_index,
            'population_std_motility': population_std_index,
            'population_heterogeneity': heterogeneity,
            'sample_size': len(track_metrics)
        }
    else:
        result = {
            'individual_metrics': [],
            'population_avg_motility': 0,
            'population_std_motility': 0,
            'population_heterogeneity': 0,
            'sample_size': 0
        }
    
    return result

def visualize_motility_map(tracks, chamber_dimensions=None, motility_metrics=None):
    """
    Visualize cell motility patterns in relation to chamber position.
    
    Parameters:
    -----------
    tracks : list
        List of track dictionaries
    chamber_dimensions : tuple, optional
        (width, height) of the chamber
    motility_metrics : dict, optional
        Output from enhanced_motility_index function
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the visualization
    """
    # Calculate motility metrics if not provided
    if motility_metrics is None:
        motility_metrics = enhanced_motility_index(tracks, chamber_dimensions)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define chamber boundaries if provided
    if chamber_dimensions:
        width, height = chamber_dimensions
        plt.xlim(0, width)
        plt.ylim(0, height)
        
        # Draw chamber regions (inlet, edges, corners, etc.)
        edge_margin = 50
        corner_size = 100
        
        # Corners
        corner_color = 'mistyrose'
        ax.add_patch(plt.Rectangle((0, 0), corner_size, corner_size, 
                                  color=corner_color, alpha=0.3))  # Top-left
        ax.add_patch(plt.Rectangle((0, height-corner_size), corner_size, corner_size, 
                                  color=corner_color, alpha=0.3))  # Bottom-left
        ax.add_patch(plt.Rectangle((width-corner_size, 0), corner_size, corner_size, 
                                  color=corner_color, alpha=0.3))  # Top-right
        ax.add_patch(plt.Rectangle((width-corner_size, height-corner_size), corner_size, corner_size, 
                                  color=corner_color, alpha=0.3))  # Bottom-right
        
        # Edges
        edge_color = 'lightblue'
        ax.add_patch(plt.Rectangle((0, corner_size), edge_margin, height-2*corner_size, 
                                  color=edge_color, alpha=0.3))  # Left edge
        ax.add_patch(plt.Rectangle((width-edge_margin, corner_size), edge_margin, height-2*corner_size, 
                                  color=edge_color, alpha=0.3))  # Right edge
        ax.add_patch(plt.Rectangle((corner_size, 0), width-2*corner_size, edge_margin, 
                                  color=edge_color, alpha=0.3))  # Top edge
        ax.add_patch(plt.Rectangle((corner_size, height-edge_margin), width-2*corner_size, edge_margin, 
                                  color=edge_color, alpha=0.3))  # Bottom edge
    
    # Create a lookup of track_id to metrics
    metrics_by_id = {m['track_id']: m for m in motility_metrics['individual_metrics']}
    
    # Set up colormap for motility index
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=100)  # Motility index 0-100
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Plot each track with color based on motility
    for track in tracks:
        track_id = track.get('ID', -1)
        if track_id in metrics_by_id:
            metrics = metrics_by_id[track_id]
            motility_index = metrics['motility_index']
            
            # Get track coordinates
            x = track['x']
            y = track['y']
            
            # Plot track colored by motility index
            plt.plot(x, y, '-', linewidth=1.5, alpha=0.7, 
                    color=cmap(norm(motility_index)))
            
            # Mark start and end points
            plt.plot(x[0], y[0], 'o', markersize=4, color=cmap(norm(motility_index)))
            plt.plot(x[-1], y[-1], 's', markersize=4, color=cmap(norm(motility_index)))
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Motility Index (0-100)')
    
    # Add statistical summary
    summary_text = (
        f"Population Metrics:\n"
        f"Avg Motility: {motility_metrics['population_avg_motility']:.1f}\n"
        f"Std Dev: {motility_metrics['population_std_motility']:.1f}\n"
        f"Heterogeneity: {motility_metrics['population_heterogeneity']:.2f}\n"
        f"Sample Size: {motility_metrics['sample_size']} tracks"
    )
    plt.figtext(0.02, 0.02, summary_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    # Set labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Cell Motility Map: Trajectories Colored by Motility Index')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def visualize_motility_metrics(motility_metrics):
    """
    Create detailed visualizations of motility metrics.
    
    Parameters:
    -----------
    motility_metrics : dict
        Output from enhanced_motility_index function
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the visualizations
    """
    # Extract individual metrics
    metrics = motility_metrics['individual_metrics']
    
    if not metrics:
        print("No metrics available to visualize")
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Motility Index Distribution
    ax1 = fig.add_subplot(221)
    motility_indices = [m['motility_index'] for m in metrics]
    ax1.hist(motility_indices, bins=20, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Motility Index')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Distribution of Motility Indices')
    
    # Add population mean line
    ax1.axvline(motility_metrics['population_avg_motility'], color='red', 
               linestyle='--', label=f"Mean: {motility_metrics['population_avg_motility']:.1f}")
    ax1.legend()
    
    # 2. Confinement Ratio vs Average Velocity
    ax2 = fig.add_subplot(222)
    cr_values = [m['confinement_ratio'] for m in metrics]
    vel_values = [m['avg_velocity'] for m in metrics]
    mi_values = [m['motility_index'] for m in metrics]
    
    scatter = ax2.scatter(cr_values, vel_values, c=mi_values, cmap='viridis', 
                        alpha=0.7, s=50, edgecolor='white')
    ax2.set_xlabel('Confinement Ratio')
    ax2.set_ylabel('Average Velocity (pixels/frame)')
    ax2.set_title('Velocity vs. Confinement')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Motility Index')
    
    # 3. Path Length vs Directional Persistence
    ax3 = fig.add_subplot(223)
    path_values = [m['path_length'] for m in metrics]
    dp_values = [m['directional_persistence'] for m in metrics]
    
    scatter = ax3.scatter(path_values, dp_values, c=mi_values, cmap='viridis', 
                        alpha=0.7, s=50, edgecolor='white')
    ax3.set_xlabel('Path Length (pixels)')
    ax3.set_ylabel('Directional Persistence')
    ax3.set_title('Path Length vs. Persistence')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Motility Index')
    
    # 4. Relationship between metrics
    ax4 = fig.add_subplot(224)
    
    # Group cells by motility quartiles
    sorted_indices = np.argsort(motility_indices)
    quartile_size = len(sorted_indices) // 4
    
    quartiles = {
        'Q1 (Low)': sorted_indices[:quartile_size],
        'Q2': sorted_indices[quartile_size:2*quartile_size],
        'Q3': sorted_indices[2*quartile_size:3*quartile_size],
        'Q4 (High)': sorted_indices[3*quartile_size:]
    }
    
    metrics_to_plot = ['confinement_ratio', 'avg_velocity', 
                       'directional_persistence', 'region_factor']
    
    colors = ['#FF9999', '#FFCC99', '#CCFF99', '#99CCFF']
    
    # Calculate means for each quartile and metric
    quartile_means = {}
    
    for q_name, q_indices in quartiles.items():
        quartile_means[q_name] = {}
        for metric in metrics_to_plot:
            metric_values = [metrics[i][metric] for i in q_indices]
            quartile_means[q_name][metric] = np.mean(metric_values)
    
    # Normalize values between 0 and 1 for radar chart
    normalized_values = {}
    
    for metric in metrics_to_plot:
        all_values = [m[metric] for m in metrics]
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val if max_val > min_val else 1
        
        for q_name in quartiles.keys():
            if metric not in normalized_values:
                normalized_values[metric] = {}
            normalized_values[q_name] = normalized_values.get(q_name, {})
            normalized_values[q_name][metric] = (quartile_means[q_name][metric] - min_val) / range_val
    
    # Plot radar chart
    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D
    
    def radar_factory(num_vars, frame='circle'):
        # Based on matplotlib examples
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        class RadarAxes(PolarAxes):
            name = 'radar'
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')
                
            def fill(self, *args, **kwargs):
                return super().fill_between(np.linspace(0, 2*np.pi, len(args[0])+1),
                                           np.hstack((args[0], args[0][0])), **kwargs)
                
            def plot(self, *args, **kwargs):
                lines = super().plot(*args, **kwargs)
                return lines
        
        register_projection(RadarAxes)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
        
        # Fix axis to go from 0 to 1
        ax.set_ylim(0, 1)
        
        # Set labels
        ax.set_yticklabels([])
        
        # Draw circles and lines
        ax.grid(True)
        
        return fig, ax
    
    # Create radar chart
    labels = ['Confinement', 'Velocity', 'Persistence', 'Edge Proximity']
    
    fig_radar, ax_radar = radar_factory(len(labels))
    ax4.set_visible(False)  # Hide the original ax4
    
    for i, (q_name, q_indices) in enumerate(quartiles.items()):
        values = [normalized_values[q_name][m] for m in metrics_to_plot]
        ax_radar.plot(np.linspace(0, 2*np.pi, len(values), endpoint=False), 
                     values, label=q_name, color=colors[i], linewidth=2)
        ax_radar.fill(values, alpha=0.1, color=colors[i])
    
    
    # Set radar chart labels manually
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
        
    # Set labels at the correct angles
    for i, label in enumerate(labels):
        angle = angles[i]
        # Adjust label position slightly outside the chart
        x = 1.1 * np.cos(angle)
        y = 1.1 * np.sin(angle)
        
        # Handle label alignment based on position
        ha, va = 'center', 'center'
        if np.cos(angle) < -0.1:
            ha = 'right'
        elif np.cos(angle) > 0.1:
            ha = 'left'
        if np.sin(angle) < -0.1:
            va = 'top'
        elif np.sin(angle) > 0.1:
            va = 'bottom'
            
        ax_radar.text(x, y, label, ha=ha, va=va)
    
    ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax_radar.set_title('Motility Components by Quartile')
    
    plt.tight_layout()
    return fig

def analyze_motility_by_region(tracks, chamber_dimensions, motility_metrics=None):
    """
    Analyze how motility varies across different regions of the chamber.
    
    Parameters:
    -----------
    tracks : list
        List of track dictionaries
    chamber_dimensions : tuple
        (width, height) of the chamber
    motility_metrics : dict, optional
        Output from enhanced_motility_index function
        
    Returns:
    --------
    dict
        Dictionary containing regional motility analysis
    matplotlib.figure.Figure
        The figure containing the regional visualization
    """
    # Calculate motility metrics if not provided
    if motility_metrics is None:
        motility_metrics = enhanced_motility_index(tracks, chamber_dimensions)
    
    width, height = chamber_dimensions
    edge_margin = 50
    corner_size = 100
    
    # Define regions
    regions = {
        'center': (corner_size + edge_margin, edge_margin, 
                  width - 2*(corner_size + edge_margin), height - 2*edge_margin),
        'left_edge': (corner_size, edge_margin, edge_margin, height - 2*edge_margin),
        'right_edge': (width - corner_size - edge_margin, edge_margin, 
                      edge_margin, height - 2*edge_margin),
        'top_edge': (corner_size, 0, width - 2*corner_size, edge_margin),
        'bottom_edge': (corner_size, height - edge_margin, 
                       width - 2*corner_size, edge_margin),
        'top_left': (0, 0, corner_size, corner_size),
        'top_right': (width - corner_size, 0, corner_size, corner_size),
        'bottom_left': (0, height - corner_size, corner_size, corner_size),
        'bottom_right': (width - corner_size, height - corner_size, 
                        corner_size, corner_size)
    }
    
    # Function to check if a point is in a region
    def point_in_region(x, y, region):
        rx, ry, rw, rh = region
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    # Store metrics by region
    metrics_by_region = {region: [] for region in regions}
    
    # Assign tracks to regions based on their average position
    for metric in motility_metrics['individual_metrics']:
        track_id = metric['track_id']
        track = next((t for t in tracks if t.get('ID', -1) == track_id), None)
        
        if track:
            # Calculate average position
            avg_x = np.mean(track['x'])
            avg_y = np.mean(track['y'])
            
            # Determine region
            for region_name, region_coords in regions.items():
                if point_in_region(avg_x, avg_y, region_coords):
                    metrics_by_region[region_name].append(metric)
                    break
    
    # Calculate average motility index for each region
    regional_metrics = {}
    
    for region, metrics in metrics_by_region.items():
        if metrics:
            motility_indices = [m['motility_index'] for m in metrics]
            regional_metrics[region] = {
                'avg_motility': np.mean(motility_indices),
                'std_motility': np.std(motility_indices),
                'cell_count': len(metrics),
                'component_averages': {
                    'confinement_ratio': np.mean([m['confinement_ratio'] for m in metrics]),
                    'avg_velocity': np.mean([m['avg_velocity'] for m in metrics]),
                    'directional_persistence': np.mean([m['directional_persistence'] for m in metrics])
                }
            }
        else:
            regional_metrics[region] = {
                'avg_motility': 0,
                'std_motility': 0,
                'cell_count': 0,
                'component_averages': {
                    'confinement_ratio': 0,
                    'avg_velocity': 0,
                    'directional_persistence': 0
                }
            }
    
    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw regions with color based on average motility
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=100)  # Motility index 0-100
    
    for region_name, region_coords in regions.items():
        x, y, w, h = region_coords
        avg_motility = regional_metrics[region_name]['avg_motility']
        cell_count = regional_metrics[region_name]['cell_count']
        
        rect = plt.Rectangle((x, y), w, h, 
                           color=cmap(norm(avg_motility)), 
                           alpha=0.7 if cell_count > 0 else 0.3)
        ax.add_patch(rect)
        
        # Add text for cell count and average motility
        if cell_count > 0:
            plt.text(x + w/2, y + h/2, 
                   f"{region_name}\n{cell_count} cells\nMI: {avg_motility:.1f}", 
                   ha='center', va='center', 
                   color='white' if avg_motility > 50 else 'black',
                   fontsize=9, fontweight='bold')
    
    # Set limits
    plt.xlim(0, width)
    plt.ylim(0, height)
    
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average Motility Index (0-100)')
    
    # Add title and labels
    plt.title('Regional Motility Analysis')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    regional_analysis = {
        'metrics_by_region': regional_metrics,
        'regions': regions
    }
    
    return regional_analysis, fig