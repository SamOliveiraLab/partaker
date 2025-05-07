# tracking_widget.py
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressDialog,
                               QMessageBox, QProgressBar)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage.measure import label
from pubsub import pub
from metrics_service import MetricsService
import numpy as np
import os
import pickle


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
        """Process cell tracking with lineage detection"""
        print("\n======= track_cells method called =======")

        # Check if we already have tracking data
        if hasattr(self, "lineage_tracks") and self.lineage_tracks:
            print(
                f"TRACKING DATA EXISTS: Found {len(self.lineage_tracks)} existing lineage tracks")

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
                print(
                    f"Generated {len(self.tracked_cells)} tracked_cells from lineage data")
            else:
                print(
                    f"TRACKED CELLS EXIST: Found {len(self.tracked_cells)} tracked cells")

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

        # Get current position and channel
        p = pub.sendMessage("get_current_p", default=0)
        c = pub.sendMessage("get_current_c", default=0)
        print(f"Using position={p}, channel={c}")

        try:
            # Get shape from image data
            shape = self.image_data.data.shape
            t_max = shape[0]  # First dimension should be time

            progress = QProgressDialog(
                "Preparing frames for tracking...", "Cancel", 0, t_max, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Prepare data for tracking using metrics service
            labeled_frames = []
            for i in range(t_max):
                if progress.wasCanceled():
                    return
                progress.setValue(i)

                # Get metrics for this time point, position and channel
                metrics_df = self.metrics_service.query(
                    time=i, position=p, channel=c)

                if not metrics_df.is_empty():
                    # Create a blank labeled frame
                    frame_shape = (shape[2], shape[3]) if len(
                        shape) == 4 else (shape[3], shape[4])
                    labeled_frame = np.zeros(frame_shape, dtype=np.int32)

                    # Fill in the cells based on bounding boxes and cell IDs
                    for row in metrics_df.to_pandas().itertuples():
                        y1, x1, y2, x2 = row.y1, row.x1, row.y2, row.x2
                        cell_id = row.cell_id

                        # Create a simple mask for this cell based on its bounding box
                        labeled_frame[y1:y2, x1:x2] = cell_id

                    labeled_frames.append(labeled_frame)
                else:
                    # If no metrics for this frame, add an empty frame
                    frame_shape = (shape[2], shape[3]) if len(
                        shape) == 4 else (shape[3], shape[4])
                    labeled_frames.append(
                        np.zeros(frame_shape, dtype=np.int32))

            progress.setValue(t_max)

            if not labeled_frames:
                QMessageBox.warning(
                    self, "Error", "No data found for tracking.")
                return

            labeled_frames = np.array(labeled_frames)

            # Perform tracking (delegated to LineageAnalysis module)
            progress.setLabelText("Running cell tracking...")
            progress.setValue(0)
            progress.setMaximum(100)

            from tracking import track_cells
            all_tracks, _ = track_cells(labeled_frames)
            self.lineage_tracks = all_tracks

            # The rest of your existing code stays the same...
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

            # Show success message
            QMessageBox.information(
                self, "Tracking Complete",
                f"Cell tracking completed successfully.\n"
                f"Total tracks: {len(all_tracks)}\n"
                f"Long tracks: {len(filtered_tracks)}\n"
                f"Displayed tracks: {len(self.tracked_cells)}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self, "Error", f"Failed to track cells: {str(e)}")

    def visualize_tracks(self):
        """Visualize tracked cell trajectories"""
        if not self.tracked_cells:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20', min(20, len(self.tracked_cells)))

        for i, track in enumerate(self.tracked_cells):
            color = cmap(i % 20)
            ax.plot(track['x'], track['y'], '-', color=color,
                    linewidth=1, alpha=0.7, label=f"Track {track['ID']}")

            # Mark start and end points
            ax.plot(track['x'][0], track['y'][0],
                    'o', color=color, markersize=5)
            ax.plot(track['x'][-1], track['y'][-1],
                    's', color=color, markersize=5)

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
                        lineage_tracks=self.lineage_tracks)
