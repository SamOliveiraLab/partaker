# tracking_widget.py
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressDialog,
                               QMessageBox, QProgressBar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pubsub import pub
from skimage.measure import label

from nd2_analyzer.analysis.metrics_service import MetricsService


class TrackingWidget(QWidget):
    """
    Widget for basic cell tracking functionality.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_service = MetricsService()

        # Initialize state variables
        self.tracked_cells = None
        self.lineage_tracks = None
        self.has_channels = False
        self.image_data = None

        # Initialize UI components
        self.init_ui()

        # Subscribe to relevant messages
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Create buttons layout
        buttons_layout = QHBoxLayout()

        # Track cells button
        self.track_button = QPushButton("Track Cells")
        self.track_button.clicked.connect(self.track_cells)
        buttons_layout.addWidget(self.track_button)

        # Show lineage tree button
        self.lineage_button = QPushButton("Show Lineage Trees")
        self.lineage_button.clicked.connect(self.show_lineage_dialog)
        self.lineage_button.setEnabled(False)
        buttons_layout.addWidget(self.lineage_button)

        # Motility analysis button
        self.motility_button = QPushButton("Analyze Motility")
        self.motility_button.clicked.connect(self.analyze_motility)
        self.motility_button.setEnabled(False)
        buttons_layout.addWidget(self.motility_button)

        layout.addLayout(buttons_layout)

        # Add visualization area
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def on_image_data_loaded(self, image_data):
        """Handle new image data loading"""
        self.image_data = image_data

        # Reset tracking data
        self.tracked_cells = None
        self.lineage_tracks = None

        # Reset UI
        self.lineage_button.setEnabled(False)
        self.motility_button.setEnabled(False)

        # Determine if image has channels
        shape = image_data.data.shape
        if len(shape) == 5:  # T, P, C, Y, X format
            self.has_channels = True
        else:
            self.has_channels = False

        # Clear visualization
        self.figure.clear()
        self.canvas.draw()

    def track_cells(self):
        """Process cell tracking with lineage detection - using segmentation cache like the old architecture"""
        print("\n======= track_cells method called =======")

        # Check if we already have tracking data
        if hasattr(self, "lineage_tracks") and self.lineage_tracks:
            print(f"TRACKING DATA EXISTS: Found {len(self.lineage_tracks)} existing lineage tracks")

            # If we have lineage_tracks but no tracked_cells, generate them
            if not hasattr(self, "tracked_cells") or not self.tracked_cells:
                print("Regenerating tracked_cells from lineage_tracks")
                # Filter tracks by length
                MIN_TRACK_LENGTH = 2  # Using a smaller value since 5 might be too restrictive
                filtered_tracks = [track for track in self.lineage_tracks if
                                   'x' in track and len(track['x']) >= MIN_TRACK_LENGTH]
                filtered_tracks.sort(
                    key=lambda track: len(track['x']), reverse=True)

                MAX_TRACKS_TO_DISPLAY = 100
                self.tracked_cells = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]
                print(f"Generated {len(self.tracked_cells)} tracked_cells from lineage data")
            else:
                print(f"TRACKED CELLS EXIST: Found {len(self.tracked_cells)} tracked cells")

            # Enable UI elements
            self.lineage_button.setEnabled(True)
            self.motility_button.setEnabled(True)

            # Visualize existing tracks
            print("Visualizing existing tracked cells")
            self.visualize_tracks()

            # Show information message
            QMessageBox.information(
                self, "Using Existing Tracking Data",
                f"Using existing tracking data with {len(self.lineage_tracks)} tracks."
            )
            print("Returning from track_cells without reprocessing")
            return

        # If we get here, we need to run tracking
        print("Continuing with tracking process...")

        if not self.image_data or not self.image_data.is_nd2:
            print("Error: Tracking requires an ND2 dataset")
            QMessageBox.warning(
                self, "Error", "Tracking requires an ND2 dataset.")
            return

        # Get current position and channel with proper handling of None values
        p = pub.sendMessage("get_current_p", default=0)
        if p is None:
            p = 0
            print(f"Current position was None, defaulting to position {p}")

        c = pub.sendMessage("get_current_c", default=0)
        if c is None:
            c = 0
            print(f"Current channel was None, defaulting to channel {c}")

        print(f"Using position={p}, channel={c}")

        try:
            # Get shape from image data
            shape = self.image_data.data.shape
            t_max = shape[0]  # First dimension should be time

            progress = QProgressDialog(
                "Preparing frames for tracking...", "Cancel", 0, t_max, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Set up segmentation cache with current model (if this attribute/method exists)
            segmentation_cache_available = False
            if hasattr(self.image_data, 'segmentation_cache'):
                if hasattr(self.image_data.segmentation_cache, 'with_model'):
                    # Get current model name - adapt this to how your UI stores the current model
                    current_model = "unet"  # Default value
                    if hasattr(self, 'model_dropdown') and hasattr(self.model_dropdown, 'currentText'):
                        current_model = self.model_dropdown.currentText()

                    # Set up the segmentation cache with the current model
                    try:
                        self.image_data.segmentation_cache.with_model(current_model)
                        segmentation_cache_available = True
                    except Exception as e:
                        print(f"Error setting up segmentation cache: {str(e)}")
                else:
                    # Cache exists but no with_model method - assume it's already configured
                    segmentation_cache_available = True

            if not segmentation_cache_available:
                print("Segmentation cache not available, will use metrics service for all frames")

            # Prepare frames for tracking
            labeled_frames = []

            for i in range(t_max):
                if progress.wasCanceled():
                    return
                progress.setValue(i)

                # Try to use segmentation cache first (like old architecture)
                segmented = None

                if segmentation_cache_available:
                    try:
                        # Get segmented image from cache - exactly like old architecture
                        segmented = self.image_data.segmentation_cache[i, p, c]
                    except Exception as cache_error:
                        print(f"Error accessing segmentation cache for frame {i}: {str(cache_error)}")
                        segmented = None

                if segmented is not None:
                    # Apply connected component labeling - exactly like old architecture
                    labeled = label(segmented)
                    num_objects = np.max(labeled)
                    labeled_frames.append(labeled)
                else:
                    # Fall back to metrics-based approach if segmentation cache not available
                    print(f"Frame {i}: Segmentation cache not available, falling back to metrics")

                    metrics_df = self.metrics_service.query(time=i, position=p, channel=c)

                    if not metrics_df.is_empty():
                        # Create frame dimensions
                        frame_shape = (shape[2], shape[3]) if len(shape) == 4 else (shape[3], shape[4])

                        # Create binary mask from bounding boxes
                        binary_mask = np.zeros(frame_shape, dtype=bool)

                        for row in metrics_df.to_pandas().itertuples():
                            y1, x1, y2, x2 = row.y1, row.x1, row.y2, row.x2
                            binary_mask[y1:y2, x1:x2] = True

                        # Apply connected component labeling
                        labeled_frame = label(binary_mask)
                        num_objects = np.max(labeled_frame)
                        print(f"Frame {i}: Using metrics fallback - {num_objects} objects detected")

                        labeled_frames.append(labeled_frame)
                    else:
                        # If no metrics for this frame, add an empty frame
                        frame_shape = (shape[2], shape[3]) if len(shape) == 4 else (shape[3], shape[4])
                        labeled_frames.append(np.zeros(frame_shape, dtype=np.int32))
                        print(f"Frame {i}: No data found, adding empty frame")

            progress.setValue(t_max)

            if not labeled_frames:
                QMessageBox.warning(
                    self, "Error", "No data found for tracking.")
                return

            labeled_frames = np.array(labeled_frames)
            print(f"Prepared {len(labeled_frames)} frames for tracking")

            # Print object statistics
            total_objects = sum(np.max(frame) for frame in labeled_frames)
            print(f"Total objects across all frames: {total_objects}")

            # Perform tracking
            progress.setLabelText("Running cell tracking with lineage detection...")
            progress.setValue(0)
            progress.setMaximum(100)

            from nd2_analyzer.analysis.tracking import track_cells
            all_tracks, _ = track_cells(labeled_frames)
            self.lineage_tracks = all_tracks

            # Filter tracks by length for display
            MIN_TRACK_LENGTH = 5
            filtered_tracks = [track for track in all_tracks if len(
                track['x']) >= MIN_TRACK_LENGTH]
            filtered_tracks.sort(
                key=lambda track: len(track['x']), reverse=True)

            MAX_TRACKS_TO_DISPLAY = 100
            self.tracked_cells = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]

            # Update UI
            self.lineage_button.setEnabled(True)
            self.motility_button.setEnabled(True)

            # Notify other components about tracking data
            pub.sendMessage("tracking_data_available",
                            lineage_tracks=self.lineage_tracks)

            # Visualize tracks
            self.visualize_tracks()

            # Show success message with detailed stats
            total_tracks = len(all_tracks)
            long_tracks = len(filtered_tracks)
            displayed_tracks = len(self.tracked_cells)

            QMessageBox.information(
                self, "Tracking Complete",
                f"Cell tracking completed successfully.\n\n"
                f"Total tracks detected: {total_tracks}\n"
                f"Tracks spanning {MIN_TRACK_LENGTH}+ frames: {long_tracks}\n"
                f"Tracks displayed: {displayed_tracks}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self, "Error", f"Failed to track cells: {str(e)}")

    def visualize_tracks(self):
        """Visualize tracked cell trajectories with statistics"""
        if not self.tracked_cells:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20', min(20, len(self.tracked_cells)))

        # Calculate displacement statistics
        displacements = []
        for track in self.tracked_cells:
            if len(track['x']) >= 2:  # Need at least start and end points
                # Calculate displacement (distance from start to end)
                start_x, start_y = track['x'][0], track['y'][0]
                end_x, end_y = track['x'][-1], track['y'][-1]
                displacement = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                displacements.append(displacement)

        # Calculate statistics
        avg_displacement = np.mean(displacements) if displacements else 0
        max_displacement = np.max(displacements) if displacements else 0

        # Plot each track
        for i, track in enumerate(self.tracked_cells):
            color = cmap(i % 20)
            ax.plot(track['x'], track['y'], '-', color=color,
                    linewidth=1, alpha=0.7, label=f"Track {track['ID']}")

            # Mark start and end points
            ax.plot(track['x'][0], track['y'][0],
                    'o', color=color, markersize=5)
            ax.plot(track['x'][-1], track['y'][-1],
                    's', color=color, markersize=5)

        # Add statistics box
        stats_text = f"Displaying top {len(self.tracked_cells)} tracks\n"
        stats_text += f"Avg displacement: {avg_displacement:.1f}px\n"
        stats_text += f"Max displacement: {max_displacement:.1f}px"

        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', bbox=props)

        # Set title and labels
        ax.set_title('Cell Trajectories')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        self.figure.tight_layout()
        self.canvas.draw()

    def show_lineage_dialog(self):
        """Open the lineage visualization dialog"""
        if not self.lineage_tracks:
            QMessageBox.warning(self, "Error", "No lineage data available.")
            return

        # Open the LineageDialog
        pub.sendMessage("show_lineage_dialog_request",
                        lineage_tracks=self.lineage_tracks)

    def save_tracking_data(self, folder_path):
        """Save tracking data to a file in the specified folder"""
        try:
            # Ensure the folder exists
            os.makedirs(folder_path, exist_ok=True)

            # Prepare tracking data dictionary
            tracking_data = {}

            if hasattr(self, "tracked_cells") and self.tracked_cells is not None:
                tracking_data["tracked_cells"] = self.tracked_cells
                print(f"Saving {len(self.tracked_cells)} tracked cells")
            else:
                print("No tracked_cells to save")

            if hasattr(self, "lineage_tracks") and self.lineage_tracks is not None:
                tracking_data["lineage_tracks"] = self.lineage_tracks
                print(f"Saving {len(self.lineage_tracks)} lineage tracks")
            else:
                print("No lineage_tracks to save")

            # Save data if we have any
            if tracking_data:
                tracking_path = os.path.join(folder_path, "tracking_data.pkl")
                with open(tracking_path, 'wb') as f:
                    pickle.dump(tracking_data, f)
                print(f"Tracking data saved to {tracking_path}")
                return True
            else:
                print("No tracking data to save")
                return False

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error saving tracking data: {str(e)}")
            return False

    def load_tracking_data(self, folder_path):
        """
        Load tracking data from a file in the specified folder.

        Args:
            folder_path: Path to the folder containing the data

        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        try:
            tracking_path = os.path.join(folder_path, "tracking_data.pkl")

            if not os.path.exists(tracking_path):
                print(f"No tracking data found at {tracking_path}")
                return False

            with open(tracking_path, 'rb') as f:
                tracking_data = pickle.load(f)

            # Load tracked cells if available
            if "tracked_cells" in tracking_data and tracking_data["tracked_cells"]:
                self.tracked_cells = tracking_data["tracked_cells"]
                print(f"Loaded {len(self.tracked_cells)} tracked cells")

            # Load lineage tracks if available
            if "lineage_tracks" in tracking_data and tracking_data["lineage_tracks"]:
                self.lineage_tracks = tracking_data["lineage_tracks"]
                print(f"Loaded {len(self.lineage_tracks)} lineage tracks")

            # Update UI based on loaded data
            if self.lineage_tracks:
                self.lineage_button.setEnabled(True)
                self.motility_button.setEnabled(True)

                # Visualize tracks if we have tracked_cells
                if self.tracked_cells:
                    self.visualize_tracks()

            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading tracking data: {str(e)}")
            return False

    def analyze_motility(self):
        """Open the motility analysis dialog"""
        if not self.lineage_tracks:
            QMessageBox.warning(self, "Error", "No tracking data available.")
            return

        # Open the MotilityDialog
        pub.sendMessage("show_motility_dialog_request",
                        tracked_cells=self.tracked_cells,
                        lineage_tracks=self.lineage_tracks,
                        image_data=self.image_data)
