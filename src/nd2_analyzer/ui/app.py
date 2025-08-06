from pathlib import Path
import sys
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np
import pandas as pd

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from skimage.measure import label, regionprops

from nd2_analyzer.data.image_data import ImageData

from nd2_analyzer.analysis.morphology.morphology import (
    annotate_binary_mask,
    extract_cells_and_metrics,
)
from nd2_analyzer.analysis.tracking.tracking import track_cells, visualize_cell_regions, enhanced_motility_index, visualize_motility_map, visualize_motility_with_chamber_regions
from .roisel import PolygonROISelector

from .dialogs import AboutDialog, ExperimentDialog
from .widgets import ViewAreaWidget, PopulationWidget, SegmentationWidget, MorphologyWidget, TrackingManager

from pubsub import pub

from nd2_analyzer.data.experiment import Experiment
from nd2_analyzer.analysis.metrics_service import MetricsService

class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Partaker 3 - GUI")
        self.setGeometry(100, 100, 1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Initialize services
        self.metrics_service = MetricsService()

        # Create and add the MorphologyWidget with metrics service
        self.morphology_widget = MorphologyWidget(
            metrics_service=self.metrics_service)

        # Create ViewArea widget
        self.viewArea = ViewAreaWidget()
        self.layout.addWidget(self.viewArea)

        # Initialize other UI components
        self.tab_widget = QTabWidget()

        # Complete initialization
        self.segmentation_tab = SegmentationWidget()
        self.populationTab = PopulationWidget()
        self.morphologyTab = MorphologyWidget()

        self.morphologyVisualizationTab = QWidget()
        self.tracking_manager = TrackingManager()

        # Add tabs to the QTabWidget
        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        self.tab_widget.addTab(self.populationTab, "Population")
        self.tab_widget.addTab(self.morphology_widget, "Morphology")
        self.tab_widget.addTab(self.tracking_manager,
                               "Tracking - Lineage Tree")
        self.initMenuBar()
        self.layout.addWidget(self.tab_widget)

        # Subscribe to events
        pub.subscribe(self.on_exp_loaded, "experiment_loaded")
        pub.subscribe(self.on_image_request, "image_request")
        pub.subscribe(self.on_segmentation_request, "segmentation_request")
        pub.subscribe(self.on_draw_cell_bounding_boxes,
                      "draw_cell_bounding_boxes")
        pub.subscribe(self.highlight_cell, "highlight_cell_requested")
        
        pub.subscribe(self.provide_image_data, "get_image_data")

    def provide_image_data(self, callback):
        """Provide the image_data object through the callback"""
        if hasattr(self, "image_data"):
            print("Providing image_data to callback")
            callback(self.image_data)
        else:
            print("No image_data available")
            callback(None)

    def on_exp_loaded(self, experiment: Experiment):
        self.curr_experiment = experiment
        self.image_data = ImageData.load_nd2(experiment.nd2_files)
        pub.sendMessage("image_data_loaded", image_data=self.image_data)

    def on_image_request(self, time, position, channel):
        """Handle requests for raw image data"""
        if not self.image_data:
            return

        # Retrieve the image data
        try:
            if self.image_data.is_nd2:
                if self.has_channels:
                    image = self.image_data.data[time, position, channel]
                else:
                    image = self.image_data.data[time, position]
            else:
                image = self.image_data.data[time]

            # Convert to NumPy array if needed
            image = np.array(image)

            # Publish the image
            pub.sendMessage("image_ready",
                            image=image,
                            time=time,
                            position=position,
                            channel=channel)
        except Exception as e:
            print(f"Error retrieving image: {e}")

    def on_segmentation_request(self, time, position, channel, model_name):
        """Handle requests for segmentation data"""
        if not self.image_data:
            return

        try:
            # Get the appropriate segmentation model
            if model_name:
                self.image_data.segmentation_cache.with_model(model_name)
            else:
                self.image_data.segmentation_cache.with_model(
                    self.model_dropdown.currentText())

            # Get the segmentation
            segmented_image = self.image_data.segmentation_cache[time,
                                                                 position, channel]

            # Publish the segmentation result
            pub.sendMessage("segmentation_ready",
                            segmented_image=segmented_image,
                            time=time,
                            position=position,
                            channel=channel)
        except Exception as e:
            print(f"Error retrieving segmentation: {e}")

    def open_roi_selector(self):
        # Get the image to use for ROI selection
        image_data = self.current_image

        # Create and show the ROI selector dialog
        roi_dialog = PolygonROISelector(image_data)
        roi_dialog.roi_selected.connect(self.handle_roi_result)
        roi_dialog.exec_()  # Use exec_ to make it modal

    def handle_roi_result(self, mask):
        # Store and apply mask
        self.roi_mask = mask
        self.image_data.segmentation_cache.set_binary_mask(self.roi_mask)
        # Update UI or perform other actions with the new mask
        print(f"ROI mask created with shape: {self.roi_mask.shape}")

    def initMorphologyTimeTab(self):

        # Create a horizontal layout for the tracking buttons
        tracking_buttons_layout = QHBoxLayout()

        # Button for lineage tracking
        self.lineage_button = QPushButton("Visualize Lineage Tree")
        self.lineage_button.clicked.connect(self.track_cells_with_lineage)
        tracking_buttons_layout.addWidget(self.lineage_button)

        # Button to create tracking video
        self.overlay_video_button = QPushButton("Tracking Video")
        self.overlay_video_button.clicked.connect(
            self.overlay_tracks_on_images)
        tracking_buttons_layout.addWidget(self.overlay_video_button)

        self.motility_button = QPushButton("Analyze Cell Motility")
        self.motility_button.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;")
        self.motility_button.clicked.connect(self.analyze_cell_motility)
        tracking_buttons_layout.addWidget(self.motility_button)

        # Inner Tabs for Plots
        self.plot_tabs = QTabWidget()
        self.morphology_fractions_tab = QWidget()

        # Plot for Morphology Fractions
        morphology_fractions_layout = QVBoxLayout(
            self.morphology_fractions_tab)
        self.figure_morphology_fractions = plt.figure()
        self.canvas_morphology_fractions = FigureCanvas(
            self.figure_morphology_fractions)
        morphology_fractions_layout.addWidget(self.canvas_morphology_fractions)

        # Progress Bar
        self.progress_bar = QProgressBar()

    def on_draw_cell_bounding_boxes(self, time, position, channel, cell_mapping):
        """Handle request to draw cell bounding boxes"""
        # Get the segmentation using the same model as the segmentation cache
        if not hasattr(self, "image_data") or not self.image_data:
            print("No image data available")
            return

        # Get the current model from the cache
        current_model = self.image_data.segmentation_cache.model_name
        if not current_model:
            # Default to a standard model if none set
            current_model = "bact_phase_cp3"  # This is CELLPOSE_BACT_PHASE
            self.image_data.segmentation_cache.with_model(current_model)

        # Get the segmentation data
        segmented_image = self.image_data.segmentation_cache[time,
                                                             position, channel]

        if segmented_image is None:
            print(
                f"No segmentation available for T={time}, P={position}, C={channel}")
            return

        # Create annotated image
        annotated_image = annotate_binary_mask(segmented_image, cell_mapping)

        # Display on the view area's image label
        height, width = annotated_image.shape[:2]
        qimage = QImage(annotated_image.data, width, height,
                        annotated_image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.viewArea.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.viewArea.image_label.setPixmap(pixmap)

        # Store the annotated image
        if hasattr(self, "annotated_image"):
            self.annotated_image = annotated_image

        self.current_cell_mapping = cell_mapping

    def visualize_tracking(self, tracks):
        """
        Visualizes the tracked cells as trajectories over time.

        Parameters:
        -----------
        tracks : list
            List of tracked cell dictionaries with x, y, and ID information.
        """
        if not tracks:
            print("No valid tracking data to visualize.")
            return

        self.figure_morphology_fractions.clear()
        ax = self.figure_morphology_fractions.add_subplot(111)

        # Create a colormap for the tracks
        import matplotlib.cm as cm

        # Use a colormap that's easier to distinguish
        cmap = cm.get_cmap('tab20', min(20, len(tracks)))

        # Calculate displacement for each track
        track_displacements = []
        for track in tracks:
            if len(track['x']) >= 2:
                # Calculate total displacement (Euclidean distance from start
                # to end)
                start_x, start_y = track['x'][0], track['y'][0]
                end_x, end_y = track['x'][-1], track['y'][-1]
                displacement = np.sqrt(
                    (end_x - start_x)**2 + (end_y - start_y)**2)
                track_displacements.append(displacement)
            else:
                track_displacements.append(0)

        # Sort tracks by displacement (most movement first)
        sorted_indices = np.argsort(track_displacements)[::-1]
        sorted_tracks = [tracks[i] for i in sorted_indices]

        # Plot each track with its own color
        for i, track in enumerate(sorted_tracks):
            # Get track data
            x_coords = track['x']
            y_coords = track['y']
            track_id = track['ID']

            # Get color from colormap
            color = cmap(i % 20)

            # Plot trajectory
            ax.plot(x_coords, y_coords, marker='.', markersize=3,
                    linewidth=1, color=color, label=f'Track {track_id}')

            # Mark start and end points
            ax.plot(x_coords[0], y_coords[0], 'o',
                    color=color, markersize=6)  # Start
            ax.plot(x_coords[-1], y_coords[-1], 's',
                    color=color, markersize=6)  # End

        # Add labels and title
        ax.set_title('Cell Trajectories Over Time')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        # Add legend with a reasonable number of entries
        if len(tracks) > 10:
            # For lots of tracks, show just a subset in the legend
            ax.legend(ncol=2, fontsize='small', loc='upper right')
        else:
            ax.legend(loc='best')

        # Add statistics to the plot
        stats_text = f"Displaying top {len(tracks)} tracks\n"

        # Calculate some statistics
        avg_displacement = np.mean(track_displacements[:len(tracks)])
        max_displacement = np.max(track_displacements[:len(tracks)])

        stats_text += f"Avg displacement: {avg_displacement:.1f}px\n"
        stats_text += f"Max displacement: {max_displacement:.1f}px"

        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, bbox=dict(
            facecolor='white', alpha=0.7), verticalalignment='bottom')

        # Draw the plot
        self.canvas_morphology_fractions.draw()

    def overlay_tracks_on_images(self):
        """
        Overlays tracking trajectories on the segmented images and creates a video.
        """
        if not hasattr(self, "tracked_cells") or not self.tracked_cells:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run cell tracking first.")
            return

        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Get segmented images
        t = self.dimensions["T"]  # Get total number of frames
        segmented_images = []

        # Show progress dialog
        progress = QProgressDialog(
            "Processing frames for video...", "Cancel", 0, t, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # Get segmented frames
        for i in range(t):
            progress.setValue(i)
            if progress.wasCanceled():
                return

            segmented = self.image_data.segmentation_cache[i, p, c]
            if segmented is not None:
                segmented_images.append(segmented)

        if not segmented_images:
            QMessageBox.warning(self, "Error", "No segmented images found.")
            return

        segmented_images = np.array(segmented_images)

        # Ask user for save location
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Tracked Video", "", "MP4 Files (*.mp4)")
        if not output_path:
            return

        # Update progress dialog for video creation
        progress.setLabelText("Creating tracking video...")
        progress.setValue(0)
        progress.setMaximum(100)

        # Create a progress callback for the overlay function
        def update_progress(value):
            progress.setValue(value)

        # Import the overlay function
        from nd2_analyzer.analysis.tracking.tracking import overlay_tracks_on_images as create_tracking_video

        # Create the video
        try:
            create_tracking_video(
                segmented_images,
                self.tracked_cells,
                save_video=True,
                output_path=output_path,
                show_frames=False,  # Don't show frames in matplotlib
                max_tracks=100,      # Limit to 30 tracks
                progress_callback=update_progress
            )

            QMessageBox.information(
                self,
                "Video Created",
                f"Tracking visualization saved to {output_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self, "Error", f"Failed to create tracking video: {str(e)}")

    def track_cells_with_lineage(self):
        # Check if lineage data is already loaded
        if hasattr(self, "lineage_tracks") and self.lineage_tracks is not None:
            # Skip tracking and go straight to visualization
            print("Using previously loaded tracking data")

            reply = QMessageBox.question(
                self, "Lineage Analysis",
                "Would you like to visualize the cell lineage tree?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.show_lineage_dialog()
            return

        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None
        if not self.image_data.is_nd2:
            QMessageBox.warning(
                self, "Error", "Tracking requires an ND2 dataset.")
            return
        try:
            t = self.dimensions["T"]
            progress = QProgressDialog(
                "Preparing frames for tracking...", "Cancel", 0, t, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            self.image_data.segmentation_cache.with_model(
                self.model_dropdown.currentText())
            labeled_frames = []
            for i in range(t):
                if progress.wasCanceled():
                    return
                progress.setValue(i)
                segmented = self.image_data.segmentation_cache[i, p, c]
                if segmented is not None:
                    labeled = label(segmented)
                    labeled_frames.append(labeled)
            progress.setValue(t)
            if not labeled_frames:
                QMessageBox.warning(
                    self, "Error", "No segmented frames found for tracking.")
                return
            labeled_frames = np.array(labeled_frames)
            print(f"Prepared {len(labeled_frames)} frames for tracking")
            progress.setLabelText(
                "Running cell tracking with lineage detection...")
            progress.setValue(0)
            progress.setMaximum(100)
            all_tracks, _ = track_cells(labeled_frames)
            visualize_cell_regions(all_tracks)
            self.lineage_tracks = all_tracks  # Store all tracks
            MIN_TRACK_LENGTH = 5
            filtered_tracks = [track for track in all_tracks if len(
                track['x']) >= MIN_TRACK_LENGTH]
            filtered_tracks.sort(
                key=lambda track: len(track['x']), reverse=True)
            MAX_TRACKS_TO_DISPLAY = 100
            display_tracks = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]
            total_tracks = len(all_tracks)
            long_tracks = len(filtered_tracks)
            displayed_tracks = len(display_tracks)
            self.tracked_cells = display_tracks
            QMessageBox.information(
                self, "Tracking Complete",
                f"Cell tracking completed successfully.\n\n"
                f"Total tracks detected: {total_tracks}\n"
                f"Tracks spanning {MIN_TRACK_LENGTH}+ frames: {long_tracks}\n"
                f"Tracks displayed: {displayed_tracks}"
            )
            self.visualize_tracking(self.tracked_cells)
            reply = QMessageBox.question(
                self, "Lineage Analysis",
                "Would you like to visualize the cell lineage tree?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.show_lineage_dialog()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Tracking Error",
                f"Failed to track cells with lineage: {str(e)}")

    def show_timepoint_lineage_comparison(self):
        """
        Display both time zero and time last lineage trees side by side for comparison,
        with a separate tab for growth and division analysis.
        """
        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run tracking with lineage first.")
            return

        # Create a dialog for the comparison
        dialog = QDialog(self)
        dialog.setWindowTitle("Lineage Tree Time Comparison")
        dialog.setMinimumWidth(1200)
        dialog.setMinimumHeight(800)
        layout = QVBoxLayout(dialog)

        # Create tab widget for the two different analyses
        tab_widget = QTabWidget()

        # First tab - Time comparison visualization
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)

        # Cell selection options for time comparison
        selection_layout = QHBoxLayout()
        option_group = QButtonGroup(dialog)

        top_radio = QRadioButton("Top Largest Lineage Tree")
        top_radio.setChecked(True)
        option_group.addButton(top_radio)
        selection_layout.addWidget(top_radio)

        cell_radio = QRadioButton("Specific Cell Lineage:")
        option_group.addButton(cell_radio)
        selection_layout.addWidget(cell_radio)

        cell_combo = QComboBox()
        cell_combo.setEnabled(False)
        selection_layout.addWidget(cell_combo)

        # Find dividing cells for the combo box
        dividing_cells = [track['ID']
                          for track in self.lineage_tracks if track.get('children', [])]
        dividing_cells.sort()
        for cell_id in dividing_cells:
            cell_combo.addItem(f"Cell {cell_id}")

        # Enable/disable combo box based on radio selection
        def update_combo_state():
            cell_combo.setEnabled(cell_radio.isChecked())

        top_radio.toggled.connect(update_combo_state)
        cell_radio.toggled.connect(update_combo_state)

        comparison_layout.addLayout(selection_layout)

        # Create a horizontal layout for the two trees
        trees_layout = QHBoxLayout()

        # Time zero tree
        time_zero_widget = QWidget()
        time_zero_layout = QVBoxLayout(time_zero_widget)
        time_zero_label = QLabel("Time Zero (First Appearance)")
        time_zero_label.setAlignment(Qt.AlignCenter)
        time_zero_layout.addWidget(time_zero_label)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        time_zero_figure = Figure(figsize=(5, 8), tight_layout=True)
        time_zero_canvas = FigureCanvas(time_zero_figure)
        time_zero_layout.addWidget(time_zero_canvas)

        # Time last tree
        time_last_widget = QWidget()
        time_last_layout = QVBoxLayout(time_last_widget)
        time_last_label = QLabel("Time Last (Before Division)")
        time_last_label.setAlignment(Qt.AlignCenter)
        time_last_layout.addWidget(time_last_label)

        time_last_figure = Figure(figsize=(5, 8), tight_layout=True)
        time_last_canvas = FigureCanvas(time_last_figure)
        time_last_layout.addWidget(time_last_canvas)

        # Add widgets to the trees layout
        trees_layout.addWidget(time_zero_widget)
        trees_layout.addWidget(time_last_widget)

        # Add trees layout to comparison tab layout
        comparison_layout.addLayout(trees_layout)

        # Create navigation area for previous/next tree
        nav_layout = QHBoxLayout()
        prev_button = QPushButton("Previous Tree")
        tree_counter_label = QLabel("Tree 1/1")
        tree_counter_label.setAlignment(Qt.AlignCenter)
        next_button = QPushButton("Next Tree")

        nav_layout.addWidget(prev_button)
        nav_layout.addWidget(tree_counter_label)
        nav_layout.addWidget(next_button)
        comparison_layout.addLayout(nav_layout)

        # Add the comparison tab to the tab widget
        tab_widget.addTab(comparison_tab, "Time Comparison")

        # Second tab - Growth & Division analysis
        growth_tab = QWidget()
        growth_layout = QVBoxLayout(growth_tab)

        try:
            # Calculate growth metrics if not already available
            if not hasattr(self, "growth_metrics"):
                progress = QProgressDialog(
                    "Calculating growth metrics...", "Cancel", 0, 100, dialog)
                progress.setWindowModality(Qt.WindowModal)
                progress.setValue(10)
                progress.show()
                QApplication.processEvents()

                self.growth_metrics = self.lineage_visualizer.calculate_growth_and_division_metrics(
                    self.lineage_tracks)

                progress.setValue(50)
                progress.setLabelText("Creating visualization...")
                QApplication.processEvents()

            # Create the growth figure with more space
            from matplotlib.figure import Figure
            # Increase figure size to fit the available space
            figure = Figure(figsize=(12, 10), dpi=100)

            # Set up the subplots with more space between them
            gs = figure.add_gridspec(2, 2, hspace=0.4, wspace=0.4)

            # Add the subplots
            ax1 = figure.add_subplot(gs[0, 0])  # Division Time Distribution
            ax2 = figure.add_subplot(gs[0, 1])  # Growth Rate Distribution
            # Division Time: Parent vs. Child
            ax3 = figure.add_subplot(gs[1, 0])
            ax4 = figure.add_subplot(gs[1, 1])  # Summary statistics

            # 1. Histogram of division times
            ax1.hist(self.growth_metrics['division_times'],
                     bins=20, color='skyblue', edgecolor='black')
            ax1.set_title('Division Time Distribution',
                          fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time (frames)', fontsize=12)
            ax1.set_ylabel('Cell Count', fontsize=12)
            ax1.axvline(self.growth_metrics['avg_division_time'], color='red',
                        linestyle='--', label=f"Mean: {self.growth_metrics['avg_division_time']:.1f}")
            ax1.axvline(self.growth_metrics['median_division_time'], color='green',
                        linestyle='--', label=f"Median: {self.growth_metrics['median_division_time']:.1f}")
            ax1.legend(fontsize=11)

            # 2. Histogram of growth rates
            ax2.hist(self.growth_metrics['growth_rates'],
                     bins=20, color='lightgreen', edgecolor='black')
            ax2.set_title('Growth Rate Distribution',
                          fontsize=14, fontweight='bold')
            ax2.set_xlabel('Growth Rate (ln(2)/division time)', fontsize=12)
            ax2.set_ylabel('Cell Count', fontsize=12)
            ax2.axvline(self.growth_metrics['avg_growth_rate'], color='red',
                        linestyle='--', label=f"Mean: {self.growth_metrics['avg_growth_rate']:.4f}")
            ax2.legend(fontsize=11)

            # 3. Parent vs child division times
            # Calculate parent-child division time pairs
            division_time_by_id = {}
            for track in self.lineage_tracks:
                if 'children' in track and track['children'] and 't' in track and len(track['t']) > 0:
                    dt = track['t'][-1] - track['t'][0]
                    if dt > 0:
                        division_time_by_id[track['ID']] = dt

            parent_child_division_pairs = []
            for track in self.lineage_tracks:
                if 'children' in track and track['children'] and track['ID'] in division_time_by_id:
                    parent_dt = division_time_by_id[track['ID']]

                    for child_id in track['children']:
                        if child_id in division_time_by_id:
                            child_dt = division_time_by_id[child_id]
                            parent_child_division_pairs.append(
                                (parent_dt, child_dt))

            if parent_child_division_pairs:
                parent_dts, child_dts = zip(*parent_child_division_pairs)
                ax3.scatter(parent_dts, child_dts, alpha=0.7, color='blue')
                ax3.set_title('Division Time: Parent vs. Child',
                              fontsize=14, fontweight='bold')
                ax3.set_xlabel('Parent Division Time', fontsize=12)
                ax3.set_ylabel('Child Division Time', fontsize=12)

                # Add y=x reference line
                min_val = min(min(parent_dts), min(child_dts))
                max_val = max(max(parent_dts), max(child_dts))
                ax3.plot([min_val, max_val], [
                         min_val, max_val], 'k--', alpha=0.5)

                # Calculate correlation
                correlation = np.corrcoef(parent_dts, child_dts)[0, 1]
                ax3.text(0.05, 0.95, f"Correlation: {correlation:.2f}",
                         transform=ax3.transAxes,
                         verticalalignment='top',
                         fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.8))

            ax4.axis('off')

            # Format the summary text with clear spacing
            summary = (
                f"Growth & Division Summary\n\n"
                f"Total dividing cells: {self.growth_metrics['total_dividing_cells']}\n\n"
                f"Division Time:\n"
                f"  Mean:    {self.growth_metrics['avg_division_time']:.1f} frames\n"
                f"  Std Dev: {self.growth_metrics['std_division_time']:.1f} frames\n"
                f"  Median:  {self.growth_metrics['median_division_time']:.1f} frames\n\n"
                f"Growth Rate:\n"
                f"  Mean:    {self.growth_metrics['avg_growth_rate']:.4f}\n"
                f"  Std Dev: {self.growth_metrics['std_growth_rate']:.4f}\n"
            )

            if parent_child_division_pairs:
                summary += f"\nParent-Child Division Time\nCorrelation: {correlation:.2f}"

            # Create a visible background for the summary text
            summary_text = ax4.text(0.05, 0.95, summary,
                                    transform=ax4.transAxes,
                                    verticalalignment='top',
                                    horizontalalignment='left',
                                    fontfamily='monospace',
                                    fontsize=12,
                                    bbox=dict(facecolor='white', alpha=0.9,
                                              boxstyle='round,pad=1.0',
                                              edgecolor='gray'))

            # Create the canvas and add to layout
            canvas = FigureCanvas(figure)
            growth_layout.addWidget(canvas)

            # Adjust plot spacing
            figure.subplots_adjust(
                hspace=0.35, wspace=0.35, bottom=0.1, top=0.95, left=0.1, right=0.95)

            # Store the figure for saving later
            growth_fig = figure

            progress.setValue(100)
            progress.close()

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_label = QLabel(
                f"Error creating growth visualization: {str(e)}")
            error_label.setStyleSheet("color: red")
            growth_layout.addWidget(error_label)

        # Add the growth tab to the tab widget
        tab_widget.addTab(growth_tab, "Growth & Division")

        # Add tab widget to main layout
        layout.addWidget(tab_widget)

        # Add control buttons at the bottom
        button_layout = QHBoxLayout()
        view_button = QPushButton("Generate Visualization")
        save_button = QPushButton("Save Images")
        close_button = QPushButton("Close")

        button_layout.addWidget(view_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)

        # Add export button to the buttons layout
        export_data_button = QPushButton("Export Classification Data")
        export_data_button.clicked.connect(
            lambda: self.lineage_visualizer.export_morphology_classifications(dialog))
        button_layout.addWidget(export_data_button)

        layout.addLayout(button_layout)

        # Store state for tree navigation
        current_tree_index = [0]
        available_trees = []

        # Function to generate the visualizations
        def generate_visualizations():
            # Get selected cell ID if applicable
            selected_cell = None
            if cell_radio.isChecked() and cell_combo.currentText():
                selected_cell = int(
                    cell_combo.currentText().replace("Cell ", ""))
                current_tree_index[0] = 0  # Reset index for specific cell

            # Show progress while generating
            progress = QProgressDialog(
                "Generating visualizations...", "Cancel", 0, 100, dialog)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            try:
                # First, precompute all morphology classifications
                progress.setLabelText(
                    "Precomputing morphology classifications...")
                progress.setValue(10)
                QApplication.processEvents()

                if hasattr(self, "cell_mapping") and self.cell_mapping:
                    self.lineage_visualizer.cell_mapping = self.cell_mapping
                    print(
                        f"Transferring cell mapping with {len(self.cell_mapping)} entries to lineage visualizer")

                # Precompute morphology for consistent classification
                self.lineage_visualizer.precompute_morphology_classifications(
                    self.lineage_tracks)

                progress.setValue(100)
                QApplication.processEvents()

                # If using top trees mode, find all connected components
                if top_radio.isChecked():
                    # Create a graph to find connected components
                    import networkx as nx
                    G = nx.DiGraph()

                    # Add nodes and edges
                    for track in self.lineage_tracks:
                        track_id = track['ID']
                        G.add_node(track_id)
                        if 'children' in track and track['children']:
                            for child_id in track['children']:
                                G.add_edge(track_id, child_id)

                    # Find all connected components
                    components = list(nx.weakly_connected_components(G))
                    # Sort by size (largest first)
                    available_trees.clear()
                    available_trees.extend(sorted(components, key=len, reverse=True)[
                                           :5])  # Top 5 largest

                    # Make sure current index is valid
                    if not available_trees:
                        QMessageBox.warning(
                            dialog, "Error", "No valid lineage trees found")
                        progress.close()
                        return

                    if current_tree_index[0] >= len(available_trees):
                        current_tree_index[0] = 0

                    # Get root of current tree for visualization
                    tree_nodes = list(available_trees[current_tree_index[0]])

                    # Find the root node of this tree (node with no parent in this tree)
                    root_candidates = []
                    for node in tree_nodes:
                        is_root = True
                        for track in self.lineage_tracks:
                            if 'children' in track and node in track['children']:
                                # Only consider parents in same tree
                                if track['ID'] in tree_nodes:
                                    is_root = False
                                    break
                        if is_root:
                            root_candidates.append(node)

                    # Use first root found or smallest ID if no clear root
                    root_cell_id = root_candidates[0] if root_candidates else min(
                        tree_nodes)

                    # Update tree counter
                    tree_counter_label.setText(
                        f"Tree {current_tree_index[0]+1}/{len(available_trees)}")

                    # Enable navigation buttons if we have multiple trees
                    prev_button.setEnabled(len(available_trees) > 1)
                    next_button.setEnabled(len(available_trees) > 1)
                else:
                    # Specific cell selected - use that as root
                    root_cell_id = selected_cell
                    # Disable navigation for specific cell mode
                    prev_button.setEnabled(False)
                    next_button.setEnabled(False)
                    tree_counter_label.setText("Custom Tree")

                # Generate time zero tree with cartoony style
                progress.setLabelText("Generating Time Zero visualization...")
                progress.setValue(50)
                QApplication.processEvents()

                self.lineage_visualizer.create_cartoony_lineage_comparison(
                    self.lineage_tracks, time_zero_canvas,
                    root_cell_id=root_cell_id, time_point="first")

                # Generate time last tree with cartoony style
                progress.setLabelText("Generating Time Last visualization...")
                progress.setValue(70)
                QApplication.processEvents()

                self.lineage_visualizer.create_cartoony_lineage_comparison(
                    self.lineage_tracks, time_last_canvas,
                    root_cell_id=root_cell_id, time_point="last")

                # Calculate diversity metrics
                progress.setLabelText("Calculating diversity metrics...")
                progress.setValue(90)
                QApplication.processEvents()

                metrics = self.lineage_visualizer.calculate_diversity_metrics(
                    self.lineage_tracks)

                progress.setValue(100)

            except Exception as e:
                QMessageBox.warning(
                    dialog, "Error", f"Error generating visualization: {str(e)}")
                print(f"Visualization error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                progress.close()

        # Navigation functions
        def go_to_next_tree():
            if not available_trees or len(available_trees) <= 1:
                return

            # Move to next tree
            current_tree_index[0] = (
                current_tree_index[0] + 1) % len(available_trees)
            generate_visualizations()

        def go_to_previous_tree():
            if not available_trees or len(available_trees) <= 1:
                return

            # Move to previous tree
            current_tree_index[0] = (
                current_tree_index[0] - 1) % len(available_trees)
            generate_visualizations()

        # Connect button signals
        view_button.clicked.connect(generate_visualizations)
        save_button.clicked.connect(
            lambda: save_images(tab_widget.currentIndex()))
        close_button.clicked.connect(dialog.close)
        prev_button.clicked.connect(go_to_previous_tree)
        next_button.clicked.connect(go_to_next_tree)

        # Function to save images depending on which tab is active
        def save_images(tab_index):
            if tab_index == 0:  # Time Comparison tab
                save_path, _ = QFileDialog.getSaveFileName(
                    dialog, "Save Visualization", "", "PNG Files (*.png)")
                if save_path:
                    # Extract base path without extension
                    base_path = save_path.replace(".png", "")

                    # Get current tree info for filename
                    tree_info = ""
                    if top_radio.isChecked():
                        tree_info = f"tree{current_tree_index[0]+1}"
                    else:
                        tree_info = f"cell{cell_combo.currentText().replace('Cell ', '')}"

                    # Save time zero tree
                    time_zero_path = f"{base_path}_{tree_info}_time_zero.png"
                    time_zero_figure.savefig(
                        time_zero_path, dpi=300, bbox_inches='tight')

                    # Save time last tree
                    time_last_path = f"{base_path}_{tree_info}_time_last.png"
                    time_last_figure.savefig(
                        time_last_path, dpi=300, bbox_inches='tight')

                    QMessageBox.information(
                        dialog, "Save Complete",
                        f"Images saved as:\n{time_zero_path}\n{time_last_path}")

            elif tab_index == 1:  # Growth & Division tab
                save_path, _ = QFileDialog.getSaveFileName(
                    dialog, "Save Growth Analysis", "", "PNG Files (*.png)")
                if save_path:
                    growth_fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    QMessageBox.information(
                        dialog, "Save Complete",
                        f"Growth analysis saved as:\n{save_path}")

        # Initial generation
        generate_visualizations()

        # Show the dialog
        dialog.exec_()

    def show_lineage_dialog(self):
        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run tracking with lineage first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Lineage Tree Visualization")
        dialog.setMinimumWidth(800)
        dialog.setMinimumHeight(700)
        layout = QVBoxLayout(dialog)

        # Add visualization type selection
        viz_layout = QHBoxLayout()
        viz_label = QLabel("Visualization Style:")
        viz_layout.addWidget(viz_label)

        viz_type = QComboBox()
        viz_type.addItems(
            ["Standard Lineage Tree", "Morphology-Enhanced Tree"])
        viz_layout.addWidget(viz_type)
        layout.addLayout(viz_layout)

        # Cell selection options (same as original)
        selection_layout = QHBoxLayout()
        option_group = QButtonGroup(dialog)
        top_radio = QRadioButton("Top 5 Largest Lineage Trees")
        top_radio.setChecked(True)
        option_group.addButton(top_radio)
        selection_layout.addWidget(top_radio)
        cell_radio = QRadioButton("Specific Cell Lineage:")
        option_group.addButton(cell_radio)
        selection_layout.addWidget(cell_radio)
        cell_combo = QComboBox()
        cell_combo.setEnabled(False)
        selection_layout.addWidget(cell_combo)
        dividing_cells = [
            track['ID'] for track in self.lineage_tracks if track.get(
                'children', [])]
        dividing_cells.sort()
        for cell_id in dividing_cells:
            cell_combo.addItem(f"Cell {cell_id}")
        layout.addLayout(selection_layout)

        def update_combo_state():
            cell_combo.setEnabled(cell_radio.isChecked())
        top_radio.toggled.connect(update_combo_state)
        cell_radio.toggled.connect(update_combo_state)

        # Canvas for the visualization (same as original)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        figure = Figure(figsize=(9, 6), tight_layout=True)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        # NEW: Navigation layout for previous/next buttons
        nav_layout = QHBoxLayout()
        prev_button = QPushButton("Previous Tree")
        next_button = QPushButton("Next Tree")

        # Add a label to show current tree number
        tree_counter_label = QLabel("Tree 1/1")
        tree_counter_label.setAlignment(Qt.AlignCenter)

        nav_layout.addWidget(prev_button)
        nav_layout.addWidget(tree_counter_label)
        nav_layout.addWidget(next_button)
        layout.addLayout(nav_layout)

        # Add variables to track current tree index and available trees
        # Use list to allow modification inside closures
        current_tree_index = [0]
        available_trees = []  # Will store the list of trees

        # Original buttons
        button_layout = QHBoxLayout()
        view_button = QPushButton("Visualize")
        save_button = QPushButton("Save")
        close_button = QPushButton("Close")
        button_layout.addWidget(view_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)

        # Add this button to your existing lineage dialog
        comparison_button = QPushButton("Compare Time Zero vs Time Last")
        comparison_button.clicked.connect(
            self.show_timepoint_lineage_comparison)
        button_layout.addWidget(comparison_button)

        layout.addLayout(button_layout)

        def create_visualization():
            selected_cell = None
            if cell_radio.isChecked() and cell_combo.currentText():
                selected_cell = int(
                    cell_combo.currentText().replace("Cell ", ""))
                # Reset index when showing specific cell
                current_tree_index[0] = 0

            # Handle tree identification - this is common code for both visualization types
            if top_radio.isChecked():
                # Get all trees by analyzing connected components
                import networkx as nx
                G = nx.DiGraph()

                # Build the graph
                for track in self.lineage_tracks:
                    G.add_node(track['ID'])
                    if 'children' in track and track['children']:
                        for child_id in track['children']:
                            G.add_edge(track['ID'], child_id)

                # Find connected components (these are our trees)
                connected_components = list(nx.weakly_connected_components(G))
                # Sort by size (largest first)
                available_trees.clear()
                available_trees.extend(
                    sorted(connected_components, key=len, reverse=True)[:5])

                # Make sure current index is valid
                if not available_trees:
                    current_tree_index[0] = 0
                elif current_tree_index[0] >= len(available_trees):
                    current_tree_index[0] = 0

                # Get root of current tree
                if available_trees:
                    tree_nodes = list(available_trees[current_tree_index[0]])

                    # Find root nodes (no parents in this tree)
                    root_candidates = []
                    for node in tree_nodes:
                        is_root = True
                        for _, child in G.in_edges(node):
                            if child in tree_nodes:
                                is_root = False
                                break
                        if is_root:
                            root_candidates.append(node)

                    # If no clear root, use the earliest appearing cell (lowest ID typically)
                    if root_candidates:
                        root_cell_id = min(root_candidates)
                    else:
                        root_cell_id = min(tree_nodes)

                    # Update counter display
                    tree_counter_label.setText(
                        f"Tree {current_tree_index[0]+1}/{len(available_trees)}")
                else:
                    root_cell_id = None
                    tree_counter_label.setText("Tree 0/0")

                # Enable/disable navigation buttons
                prev_button.setEnabled(len(available_trees) > 1)
                next_button.setEnabled(len(available_trees) > 1)
            else:
                # Specific cell mode
                root_cell_id = selected_cell
                available_trees.clear()
                tree_counter_label.setText("Cell Lineage")

                # Disable navigation buttons in cell-specific mode
                prev_button.setEnabled(False)
                next_button.setEnabled(False)

            # Choose visualization based on combo box selection
            if viz_type.currentText() == "Morphology-Enhanced Tree":
                # Use the morphology visualization with the selected root
                self.lineage_visualizer.visualize_morphology_lineage_tree(
                    self.lineage_tracks, canvas, root_cell_id)
            else:
                # Use the standard visualization with the selected root
                self.lineage_visualizer.create_lineage_tree(
                    self.lineage_tracks, canvas, root_cell_id=root_cell_id)

        def go_to_next_tree():
            if not available_trees or len(available_trees) <= 1:
                return

            # Move to next tree
            current_tree_index[0] = (
                current_tree_index[0] + 1) % len(available_trees)

            # Show activity indicator
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                create_visualization()
            finally:
                QApplication.restoreOverrideCursor()

        def go_to_previous_tree():
            if not available_trees or len(available_trees) <= 1:
                return

            # Move to previous tree
            current_tree_index[0] = (
                current_tree_index[0] - 1) % len(available_trees)

            # Show activity indicator
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                create_visualization()
            finally:
                QApplication.restoreOverrideCursor()

        def save_visualization():
            output_path, _ = QFileDialog.getSaveFileName(
                dialog, "Save Lineage Tree", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)")
            if output_path:
                # Use a higher DPI for better image quality
                figure.savefig(output_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(
                    dialog, "Success", f"Lineage tree saved to {output_path}")

        def maximize_dialog():
            """Toggle between normal and maximized window state"""
            if dialog.isMaximized():
                dialog.showNormal()
            else:
                dialog.showMaximized()

        # Add a maximize button
        maximize_button = QPushButton("Maximize Window")
        maximize_button.clicked.connect(maximize_dialog)
        button_layout.addWidget(maximize_button)

        # Connect signals
        view_button.clicked.connect(create_visualization)
        save_button.clicked.connect(save_visualization)
        close_button.clicked.connect(dialog.close)
        viz_type.currentIndexChanged.connect(create_visualization)

        # Connect navigation buttons
        next_button.clicked.connect(go_to_next_tree)
        prev_button.clicked.connect(go_to_previous_tree)

        # Initial visualization
        create_visualization()
        dialog.exec_()

    def visualize_lineage(self):
        """
        Visualize the lineage tree from tracking data, focusing on a single cell.
        """
        if not hasattr(self, "tracked_cells") or not self.tracked_cells:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run cell tracking first.")
            return

        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            # Ask user if they want to pick a specific cell or have one
            # selected automatically
            reply = QMessageBox.question(
                self,
                "Lineage Visualization",
                "Would you like to select a specific cell to view its lineage?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)

            # Get list of cell IDs
            cell_ids = [track['ID'] for track in self.tracked_cells]

            selected_cell = None
            if reply == QMessageBox.Yes:
                # Create dialog for selection
                dialog = QDialog(self)
                dialog.setWindowTitle("Select Cell")
                layout = QVBoxLayout(dialog)

                label = QLabel("Select a cell to visualize:")
                layout.addWidget(label)

                combo = QComboBox()
                combo.addItems([str(cell_id) for cell_id in cell_ids])
                layout.addWidget(combo)

                buttons = QDialogButtonBox(
                    QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                layout.addWidget(buttons)

                if dialog.exec_() == QDialog.Accepted:
                    selected_cell = int(combo.currentText())
                else:
                    return
            else:
                # Pick the cell with the longest track
                longest_track = max(self.tracked_cells,
                                    key=lambda t: len(t['x']))
                selected_cell = longest_track['ID']

            # Create a graph to represent the lineage
            G = nx.DiGraph()

            # Add the selected cell as the root node
            track = next(
                (t for t in self.tracked_cells if t['ID'] == selected_cell), None)
            if not track:
                QMessageBox.warning(
                    self, "Error", f"Cell {selected_cell} not found.")
                return

            # Add metadata to the root node
            G.add_node(selected_cell,
                       first_frame=track['t'][0] if track['t'] else 0,
                       frames=len(track['t']) if track['t'] else len(
                           track['x']),
                       track_data=track)

            # Ask for save location
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Lineage Tree", "", "PNG Files (*.png);;All Files (*)")

            # Find temporal relationships (potential divisions)
            # For each track, find tracks that start when it ends
            end_frame_map = {}  # Maps end frame to list of tracks ending at that frame
            start_frame_map = {}  # Maps start frame to list of tracks starting at that frame

            for t in self.tracked_cells:
                if 't' in t and t['t']:
                    start_frame = t['t'][0]
                    end_frame = t['t'][-1]

                    if end_frame not in end_frame_map:
                        end_frame_map[end_frame] = []
                    end_frame_map[end_frame].append(t)

                    if start_frame not in start_frame_map:
                        start_frame_map[start_frame] = []
                    start_frame_map[start_frame].append(t)

            # Recursively build the tree from the selected cell
            def build_tree(cell_id, current_depth=0, max_depth=3):
                if current_depth >= max_depth:
                    return

                # Get the track for this cell
                parent_track = next(
                    (t for t in self.tracked_cells if t['ID'] == cell_id), None)
                if not parent_track or 't' not in parent_track or not parent_track['t']:
                    return

                # Get the end frame for this track
                end_frame = parent_track['t'][-1]

                # Get potential children (tracks that start right after this
                # one ends)
                children_candidates = []

                # Look at the next few frames for potential children
                for frame in range(end_frame, end_frame + 3):
                    if frame in start_frame_map:
                        # Get tracks that start at this frame
                        for child_track in start_frame_map[frame]:
                            # Skip if it's the parent itself
                            if child_track['ID'] == parent_track['ID']:
                                continue

                            # Calculate proximity between end of parent and
                            # start of child
                            if len(
                                    parent_track['x']) > 0 and len(
                                    child_track['x']) > 0:
                                parent_end_x = parent_track['x'][-1]
                                parent_end_y = parent_track['y'][-1]
                                child_start_x = child_track['x'][0]
                                child_start_y = child_track['y'][0]

                                # Calculate distance
                                distance = ((parent_end_x - child_start_x)**2 +
                                            (parent_end_y - child_start_y)**2)**0.5

                                # If close enough, consider it a potential
                                # child
                                if distance < 30:  # Adjust threshold as needed
                                    children_candidates.append(
                                        (child_track['ID'], distance))

                # Sort by distance and take up to 2 closest as children (for
                # division)
                children_candidates.sort(key=lambda x: x[1])
                children = [c[0] for c in children_candidates[:2]]

                # Add edges to the graph
                for child_id in children:
                    if child_id not in G:
                        # Get child track
                        child_track = next(
                            (t for t in self.tracked_cells if t['ID'] == child_id), None)
                        if child_track:
                            # Add child to graph
                            G.add_node(
                                child_id,
                                first_frame=child_track['t'][0] if child_track['t'] else 0,
                                frames=len(
                                    child_track['t']) if child_track['t'] else len(
                                    child_track['x']),
                                track_data=child_track)
                            # Add edge
                            G.add_edge(cell_id, child_id)
                            # Recursively build tree for this child
                            build_tree(child_id, current_depth + 1, max_depth)

            # Start building the tree
            build_tree(selected_cell)

            # Visualization
            plt.figure(figsize=(10, 8))

            # Use hierarchical layout for tree
            pos = None
            try:
                import numpy as np
                # Since we don't have GraphViz, create a custom tree layout
                pos = {}
                for node in G.nodes():
                    # Get node depth (how many steps from root)
                    try:
                        path_length = len(nx.shortest_path(
                            G, selected_cell, node)) - 1
                    except nx.NetworkXNoPath:
                        path_length = 0

                    # Get all nodes at this depth
                    nodes_at_depth = [
                        n for n in G.nodes() if len(
                            nx.shortest_path(
                                G,
                                selected_cell,
                                n)) - 1 == path_length]

                    # Calculate x position based on position among siblings
                    index = nodes_at_depth.index(node)
                    num_nodes_at_depth = len(nodes_at_depth)

                    if num_nodes_at_depth > 1:
                        x = index / (num_nodes_at_depth - 1)
                    else:
                        x = 0.5

                    # Adjust to spread out the tree
                    x = (x - 0.5) * (1 + path_length) + 0.5

                    # Y coordinate is negative depth (to grow downward)
                    y = -path_length

                    pos[node] = (x, y)
            except Exception as e:
                print(f"Error creating layout: {e}")
                # Fallback to spring layout
                pos = nx.spring_layout(G)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700,
                                   node_color='lightblue', alpha=0.8)

            # Draw edges with arrows
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=2,
                                   arrows=True, arrowstyle='-|>', arrowsize=20)

            # Draw labels with track info
            labels = {
                n: f"ID: {n}\nFrames: {G.nodes[n]['frames']}" for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

            # Title and styling
            plt.title(f"Cell Lineage Tree (Root: Cell {selected_cell})")
            plt.grid(False)
            plt.axis('off')

            # Add stats
            node_count = G.number_of_nodes()
            edge_count = G.number_of_edges()
            stats_text = f"Total Cells: {node_count}\nDivision Events: {edge_count}\nRoot Cell: {selected_cell}"
            plt.figtext(0.02, 0.02, stats_text, bbox=dict(
                facecolor='white', alpha=0.8))

            # Save if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Visualization Error",
                f"Failed to visualize lineage tree: {str(e)}"
            )

    def analyze_cell_motility(self):
        """
        Analyze cell motility using the enhanced model and display visualizations.
        """

        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            QMessageBox.warning(
                self,
                "Error",
                "No tracking data available. Run cell tracking first.")
            return

        # Ask user which set of tracks to use - with custom button text
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Motility Analysis Options")
        msg_box.setText("Which tracks would you like to analyze?")
        msg_box.setInformativeText(
            "• Filtered Tracks: Uses only the longest, most reliable tracks (recommended)\n"
            "• All Tracks: Uses all detected cell tracks for a complete population analysis")

        # Create custom buttons
        filtered_button = msg_box.addButton(
            "Filtered Tracks", QMessageBox.ActionRole)
        all_button = msg_box.addButton("All Tracks", QMessageBox.ActionRole)
        cancel_button = msg_box.addButton(QMessageBox.Cancel)

        msg_box.exec()

        # Handle user choice
        if msg_box.clickedButton() == filtered_button:
            tracks_to_analyze = self.tracked_cells
            track_type = "filtered"
        elif msg_box.clickedButton() == all_button:
            tracks_to_analyze = self.lineage_tracks
            track_type = "all"
        else:
            # User clicked Cancel
            return

        # Check if selected tracks exist
        if not tracks_to_analyze:
            QMessageBox.warning(
                self, "Error", f"No {track_type} tracks available.")
            return

        # Show progress dialog
        progress = QProgressDialog(
            "Analyzing cell motility...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # Get chamber dimensions if available
        chamber_dimensions = None
        try:
            if hasattr(self, "image_data") and hasattr(self.image_data, "data"):
                if self.image_data.is_nd2:
                    if len(self.image_data.data.shape) >= 4:
                        height = self.image_data.data.shape[-2]
                        width = self.image_data.data.shape[-1]
                        chamber_dimensions = (width, height)
                else:
                    height, width = self.image_data.data.shape[1:3]
                    chamber_dimensions = (width, height)

                if chamber_dimensions and (chamber_dimensions[0] < 10 or chamber_dimensions[1] < 10):
                    print(
                        f"Invalid chamber dimensions: {chamber_dimensions}, using defaults")
                    chamber_dimensions = (1392, 1040)
                else:
                    print(f"Using chamber dimensions: {chamber_dimensions}")
            else:
                chamber_dimensions = (1392, 1040)
                print(
                    f"Using default chamber dimensions: {chamber_dimensions}")
        except Exception as e:
            print(f"Error determining chamber dimensions: {e}")
            chamber_dimensions = (1392, 1040)

        try:
            # Calculate enhanced motility metrics
            progress.setValue(20)
            progress.setLabelText("Calculating motility metrics...")
            QApplication.processEvents()

            motility_metrics = enhanced_motility_index(
                tracks_to_analyze, chamber_dimensions)

            progress.setValue(50)
            progress.setLabelText(
                "Collecting cell positions for visualization...")
            QApplication.processEvents()

            # Collect all cell positions from segmentation data
            all_cell_positions = []
            p = self.slider_p.value()
            c = self.slider_c.value() if self.has_channels else None

            try:
                for t in range(min(20, self.dimensions.get("T", 1))):
                    binary_image = self.image_data.segmentation_cache[t, p, c]
                    if binary_image is not None:
                        labeled_image = label(binary_image)
                        for region in regionprops(labeled_image):
                            y, x = region.centroid
                            all_cell_positions.append((x, y))
            except Exception as e:
                print(f"Error collecting cell positions: {e}")
                all_cell_positions = []
                for track in tracks_to_analyze:
                    all_cell_positions.extend(
                        list(zip(track['x'], track['y'])))

            print(
                f"Collected {len(all_cell_positions)} cell positions for visualization")

            # Create combined visualization tab
            progress.setValue(60)
            progress.setLabelText("Creating combined visualization...")
            QApplication.processEvents()

            combined_tab = QWidget()
            combined_layout = QVBoxLayout(combined_tab)
            combined_fig, _ = visualize_motility_with_chamber_regions(
                tracks_to_analyze, all_cell_positions, chamber_dimensions, motility_metrics)
            combined_canvas = FigureCanvas(combined_fig)
            combined_layout.addWidget(combined_canvas)

            # Create dialog and tab widget
            dialog = QDialog(self)
            dialog.setWindowTitle("Cell Motility Analysis")
            dialog.setMinimumWidth(1200)
            dialog.setMinimumHeight(800)
            layout = QVBoxLayout(dialog)
            tab_widget = QTabWidget()

            # Add combined tab as first tab
            tab_widget.insertTab(0, combined_tab, "Motility by Region")
            tab_widget.setCurrentIndex(0)

            # Motility Map Tab
            progress.setValue(40)
            progress.setLabelText("Creating motility visualizations...")
            QApplication.processEvents()

            map_tab = QWidget()
            map_layout = QVBoxLayout(map_tab)
            map_fig, _ = visualize_motility_map(
                tracks_to_analyze, chamber_dimensions, motility_metrics)
            map_canvas = FigureCanvas(map_fig)
            map_layout.addWidget(map_canvas)

            # Detailed Metrics Tab
            metrics_tab = QWidget()
            metrics_layout = QVBoxLayout(metrics_tab)
            metrics_fig = visualize_motility_metrics(motility_metrics)
            metrics_canvas = FigureCanvas(metrics_fig)
            metrics_layout.addWidget(metrics_canvas)

            # Regional Analysis Tab
            region_tab = QWidget()
            region_layout = QVBoxLayout(region_tab)
            if chamber_dimensions:
                progress.setValue(70)
                progress.setLabelText("Analyzing regional variations...")
                QApplication.processEvents()
                regional_analysis, region_fig = analyze_motility_by_region(
                    tracks_to_analyze, chamber_dimensions, motility_metrics)
                region_canvas = FigureCanvas(region_fig)
                region_layout.addWidget(region_canvas)
            else:
                region_label = QLabel(
                    "Chamber dimensions not available for regional analysis.")
                region_label.setAlignment(Qt.AlignCenter)
                region_layout.addWidget(region_label)

            # Add tabs to tab widget
            tab_widget.addTab(map_tab, "Motility Map")
            tab_widget.addTab(metrics_tab, "Detailed Metrics")
            tab_widget.addTab(region_tab, "Regional Analysis")

            # Summary
            summary_text = (
                f"<h3>Motility Analysis Summary</h3>"
                f"<p><b>Population Average Motility Index:</b> {motility_metrics['population_avg_motility']:.1f}/100</p>"
                f"<p><b>Motility Heterogeneity:</b> {motility_metrics['population_heterogeneity']:.2f}</p>"
                f"<p><b>Sample Size:</b> {motility_metrics['sample_size']} cells</p>"
                f"<p>Analysis based on {track_type} tracks.</p>"
            )
            summary_label = QLabel(summary_text)
            summary_label.setTextFormat(Qt.RichText)
            summary_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(summary_label)
            layout.addWidget(tab_widget)

            # Buttons
            button_layout = QHBoxLayout()
            export_button = QPushButton("Export Results")
            close_button = QPushButton("Close")
            button_layout.addWidget(export_button)
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)

            # Export function
            def export_results():
                export_dialog = QDialog(dialog)
                export_dialog.setWindowTitle("Export Options")
                export_layout = QVBoxLayout(export_dialog)
                export_label = QLabel("Select export options:")
                export_layout.addWidget(export_label)

                export_map = QCheckBox("Export Motility Map")
                export_map.setChecked(True)
                export_layout.addWidget(export_map)

                export_metrics = QCheckBox("Export Detailed Metrics Plot")
                export_metrics.setChecked(True)
                export_layout.addWidget(export_metrics)

                export_regional = QCheckBox("Export Regional Analysis")
                export_regional.setChecked(chamber_dimensions is not None)
                export_regional.setEnabled(chamber_dimensions is not None)
                export_layout.addWidget(export_regional)

                export_csv = QCheckBox("Export Metrics as CSV")
                export_csv.setChecked(True)
                export_layout.addWidget(export_csv)

                export_buttons = QDialogButtonBox(
                    QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                export_buttons.accepted.connect(export_dialog.accept)
                export_buttons.rejected.connect(export_dialog.reject)
                export_layout.addWidget(export_buttons)

                if export_dialog.exec() == QDialog.Accepted:
                    save_path, _ = QFileDialog.getSaveFileName(
                        export_dialog, "Save Results", "", "All Files (*)")
                    if save_path:
                        base_path = save_path.replace(
                            ".png", "").replace(".csv", "")
                        if export_map.isChecked():
                            map_fig.savefig(
                                f"{base_path}_motility_map.png", dpi=300, bbox_inches='tight')
                        if export_metrics.isChecked():
                            metrics_fig.savefig(
                                f"{base_path}_detailed_metrics.png", dpi=300, bbox_inches='tight')
                        if export_regional.isChecked() and chamber_dimensions:
                            region_fig.savefig(
                                f"{base_path}_regional_analysis.png", dpi=300, bbox_inches='tight')
                        if export_csv.isChecked():
                            metrics_df = pd.DataFrame(
                                motility_metrics['individual_metrics'])
                            metrics_df.to_csv(
                                f"{base_path}_motility_metrics.csv", index=False)
                        QMessageBox.information(export_dialog, "Export Complete",
                                                f"Results exported to {base_path}_*.png/csv")

            export_button.clicked.connect(export_results)
            close_button.clicked.connect(dialog.close)

            progress.setValue(100)
            progress.close()

            # Store results
            self.motility_results = {
                "motility_metrics": motility_metrics, "track_type": track_type}

            # Update main UI plot
            self.figure_morphology_fractions.clear()
            ax = self.figure_morphology_fractions.add_subplot(111)
            for track in tracks_to_analyze:
                track_id = track.get('ID', -1)
                metric = next((m for m in motility_metrics['individual_metrics']
                               if m['track_id'] == track_id), None)
                if metric:
                    mi = metric['motility_index']
                    color = plt.cm.coolwarm(mi/100)
                    ax.plot(track['x'], track['y'], '-',
                            color=color, linewidth=1, alpha=0.7)

            ax.set_title(
                f"Cell Motility Map (Population Avg: {motility_metrics['population_avg_motility']:.1f})")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.coolwarm, norm=plt.Normalize(0, 100))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Motility Index")
            self.canvas_morphology_fractions.draw()

            dialog.exec()

        except Exception as e:
            import traceback
            traceback.print_exc()
            progress.close()
            QMessageBox.warning(self, "Analysis Error",
                                f"Error analyzing motility: {str(e)}")

    def handle_results(self, results):
        if not results:
            QMessageBox.warning(
                self,
                "Error",
                "No valid results received. Please check the input data.")
            return

        # Get all unique morphology classes across all frames
        all_morphologies = set()
        for frame_data in results.values():
            all_morphologies.update(frame_data["fractions"].keys())

        # Initialize dictionaries for both fractions and counts
        morphology_fractions = {morphology: []
                                for morphology in all_morphologies}
        morphology_counts = {morphology: [] for morphology in all_morphologies}
        total_cells_per_frame = []

        # Get the maximum time point
        max_time = max(results.keys())

        # Fill in the data for each time point
        for t in range(max_time + 1):
            if t in results:
                frame_data = results[t]["fractions"]

                # Get raw counts from the metrics table when available
                if "metrics" in results[t]:
                    metrics_df = results[t]["metrics"]
                    class_counts = metrics_df["morphology_class"].value_counts(
                    ).to_dict()
                    total_cells = len(metrics_df)

                    for morph_class, count in class_counts.items():
                        print(
                            f"  {morph_class}: {count} cells ({count/total_cells*100:.1f}%)")
                else:
                    class_counts = {morph: 0 for morph in all_morphologies}
                    total_cells = 0

                # Store total cell count for this frame
                total_cells_per_frame.append(total_cells)

                for morphology in all_morphologies:
                    # Get the fraction if present, otherwise use 0.0
                    fraction = frame_data.get(morphology, 0.0)
                    morphology_fractions[morphology].append(fraction)

                    # Store the raw count
                    count = class_counts.get(morphology, 0)
                    morphology_counts[morphology].append(count)
            else:
                # For frames with no data, append 0.0 for all morphologies
                total_cells_per_frame.append(0)
                for morphology in all_morphologies:
                    morphology_fractions[morphology].append(0.0)
                    morphology_counts[morphology].append(0)

        # Create a figure with two subplots - fractions and counts
        self.figure_morphology_fractions.clear()
        fig = self.figure_morphology_fractions

        # First subplot - fractions (as before)
        ax1 = fig.add_subplot(2, 1, 1)
        for morphology, fractions in morphology_fractions.items():
            color = self.morphology_colors_rgb.get(
                morphology, (0.5, 0.5, 0.5))  # Default to gray if color not found
            ax1.plot(
                range(len(fractions)),
                fractions,
                marker="o",
                label=morphology,
                color=color)
        ax1.set_title("Morphology Fractions Over Time")
        ax1.set_ylabel("Fraction")
        ax1.legend()

        # Second subplot - raw counts
        ax2 = fig.add_subplot(2, 1, 2)
        for morphology, counts in morphology_counts.items():
            color = self.morphology_colors_rgb.get(
                morphology, (0.5, 0.5, 0.5))  # Default to gray if color not found
            ax2.plot(
                range(len(counts)),
                counts,
                marker="o",
                label=morphology,
                color=color)
        ax2.set_title("Cell Counts By Morphology Over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Count")

        # Plot total cell count on a separate axis
        ax3 = ax2.twinx()
        ax3.plot(range(len(total_cells_per_frame)), total_cells_per_frame,
                 color='black', linestyle='--', label='Total Cells')
        ax3.set_ylabel("Total Cell Count")

        # Add combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        fig.tight_layout()
        self.canvas_morphology_fractions.draw()

    def handle_error(self, error_message):
        print(f"Error: {error_message}")
        QMessageBox.warning(self, "Processing Error", error_message)
        raise Exception(error_message)

    def update_plot(self):
        selected_metric = self.metric_dropdown.currentText()

        if not hasattr(self, "morphologies_over_time"):
            QMessageBox.warning(
                self, "Error", "No data to plot. Please process the frames first.")
            return

        if selected_metric not in self.morphologies_over_time.columns:
            QMessageBox.warning(
                self, "Error", f"Metric {selected_metric} not found in results.")
            return

        metric_data = self.morphologies_over_time[selected_metric]
        if metric_data.empty:
            QMessageBox.warning(
                self, "Error", f"No valid data available for {selected_metric}.")
            return

        self.figure_time_series.clear()
        ax = self.figure_time_series.add_subplot(111)
        ax.plot(metric_data, marker="o")
        ax.set_title(f"{selected_metric.capitalize()} Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel(selected_metric.capitalize())
        self.canvas_time_series.draw()

    def annotate_cells(self):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Ensure segmentation model is set correctly in segmentation_cache
        selected_model = self.model_dropdown.currentText()
        self.image_data.segmentation_cache.with_model(selected_model)

        # Retrieve segmentation from segmentation_cache
        segmented_image = self.image_data.segmentation_cache[t, p, c]

        if segmented_image is None:
            print(f"[ERROR] Segmentation failed for T={t}, P={p}, C={c}")
            QMessageBox.warning(self, "Segmentation Error",
                                "Segmentation failed.")
            return

        # Extract cell metrics and bounding boxes
        self.cell_mapping = extract_cells_and_metrics(
            self.image_data.data[t, p, c], segmented_image)

        if not self.cell_mapping:
            QMessageBox.warning(
                self, "No Cells", "No cells detected in the current frame.")
            return

        # Debugging
        # print(f"✅ Stored Cell Mapping: {list(self.cell_mapping.keys())}")

        # Annotate the binary segmented image
        self.annotated_image = annotate_binary_mask(
            segmented_image, self.cell_mapping)

        # Display the annotated image on the main image display
        # Convert annotated image to QImage
        height, width = self.annotated_image.shape[:2]
        qimage = QImage(
            self.annotated_image.data,
            width,
            height,
            self.annotated_image.strides[0],
            QImage.Format_RGB888,
        )

        # Convert to QPixmap and set to QLabel
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        self.update_annotation_scatter()

    def show_context_menu(self, position):
        context_menu = QMenu(self)

        save_action = context_menu.addAction("Save Annotated Image")
        save_action.triggered.connect(self.save_annotated_image)

        context_menu.exec_(self.image_label.mapToGlobal(position))

    def save_annotated_image(self):
        if not hasattr(
                self,
                "annotated_image") or self.annotated_image is None:
            QMessageBox.warning(self, "Error", "No annotated image to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotated Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")

        if file_path:
            cv2.imwrite(file_path, self.annotated_image)
            QMessageBox.information(
                self, "Success", f"Annotated image saved to {file_path}")
        else:
            QMessageBox.warning(self, "Error", "No file selected.")

    def export_images(self):
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "", "TIFF Files (*.tif);;All Files (*)"
        )

        if not save_path:
            QMessageBox.warning(self, "Export", "No file selected.")
            return

        # Extract the directory and base name from the selected path
        folder_path = Path(save_path).parent
        custom_base_name = Path(save_path).stem

        max_t_value = self.slider_t.value()
        max_p_value = self.slider_p.value()

        for t in range(max_t_value + 1):
            for p in range(max_p_value + 1):
                # Retrieve the specific frame for time t and position p
                if self.image_data.is_nd2:
                    export_image = (
                        self.image_data.data[t, p].compute()
                        if hasattr(self.image_data.data, "compute")
                        else self.image_data.data[t, p]
                    )
                else:
                    export_image = self.image_data.data[t]

                img_to_save = np.array(export_image)

                # Construct the export path with the custom name and dimensions
                file_path = folder_path / f"{custom_base_name}_P{p}_T{t}.tif"
                cv2.imwrite(str(file_path), img_to_save)

        QMessageBox.information(
            self, "Export", f"Images exported successfully to {folder_path}")

    def initMenuBar(self):
        # Create the menu bar
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        # New Experiment
        new_experiment_action = QAction("Experiment", self)
        new_experiment_action.setShortcut("Ctrl+E")
        new_experiment_action.triggered.connect(
            self.show_new_experiment_dialog)
        file_menu.addAction(new_experiment_action)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_to_folder)
        file_menu.addAction(save_action)

        load_action = QAction("Load", self)
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self.load_from_folder)
        file_menu.addAction(load_action)

        help_menu = menu_bar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        if sys.platform == "darwin":
            about_action_mac = QAction("About", self)
            about_action_mac.triggered.connect(self.show_about_dialog)
            self.menuBar().addAction(about_action_mac)

    def show_new_experiment_dialog(self):
        experiment = ExperimentDialog()
        experiment.exec_()

    def show_about_dialog(self):
        about_dialog = AboutDialog()
        about_dialog.exec_()

    def save_to_folder(self):
        """Save the current project to a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder_path:
            # Create directory if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Log segmentation cache state before saving
            if hasattr(self, "image_data") and hasattr(self.image_data, "segmentation_cache"):
                cache = self.image_data.segmentation_cache
                current_model = cache.model_name
                print(f"DEBUG: Saving project with segmentation cache")
                print(f"DEBUG: Current model: {current_model}")

                if current_model and current_model in cache.mmap_arrays_idx:
                    _, indices = cache.mmap_arrays_idx[current_model]
                    print(f"DEBUG: Cache contains {len(indices)} segmented frames for model {current_model}")
                    print(f"DEBUG: Sample indices: {list(indices)[:5] if indices else 'None'}")
                else:
                    print(f"DEBUG: No cached frames for model {current_model}")

            # Save image data
            try:
                print(f"DEBUG: Saving image data to {folder_path}")
                self.image_data.save(folder_path)
                print(f"DEBUG: Image data saved successfully")
            except Exception as e:
                print(f"ERROR: Failed to save image data: {str(e)}")
                import traceback
                traceback.print_exc()

            # Save metrics
            metrics_service = MetricsService()
            try:
                print(f"DEBUG: Saving metrics to {folder_path}")
                metrics_saved = metrics_service.save_to_file(folder_path)
                print(f"DEBUG: Metrics saved: {metrics_saved}")
            except Exception as e:
                print(f"ERROR: Failed to save metrics: {str(e)}")
                import traceback
                traceback.print_exc()

            population_saved = False
            if hasattr(self, "populationTab"):
                population_saved = self.populationTab.save_population_data(folder_path)
                print(f"Population data saved: {population_saved}")

            # Save tracking data
            tracking_saved = self.tracking_manager.tracking_widget.save_tracking_data(
                folder_path)
            print(f"DEBUG: Tracking data saved: {tracking_saved}")

            # Show success message
            QMessageBox.information(
                self, "Save Complete",
                f"Project saved to {folder_path}" +
                ("\nIncludes tracking data" if tracking_saved else "")
            )

    def load_from_folder(self):
        """Load a project from a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Project Folder", "", QFileDialog.ShowDirsOnly
        )

        if folder_path:
            print(f"Project loaded from folder: {folder_path}")

            try:
                # Load image data
                self.image_data = ImageData.load(folder_path)

                # Load metrics data
                metrics_service = MetricsService()
                metrics_loaded = metrics_service.load_from_file(folder_path)

                # Update UI based on loaded image data
                if hasattr(self, "viewArea"):
                    pub.sendMessage("image_data_loaded",
                                    image_data=self.image_data)

                population_loaded = False
                if hasattr(self, "populationTab"):
                    population_loaded = self.populationTab.load_population_data(folder_path)
                    print(f"Population data loaded: {population_loaded}")

                # Load tracking data if available
                tracking_loaded = False
                if hasattr(self, "tracking_manager") and hasattr(self.tracking_manager, "tracking_widget"):
                    tracking_loaded = self.tracking_manager.tracking_widget.load_tracking_data(
                        folder_path)

                # Show success message
                message = f"Project loaded from {folder_path}"
                if metrics_loaded:
                    message += "\nMetrics data loaded successfully"
                if tracking_loaded:
                    message += "\nTracking data loaded successfully"

                QMessageBox.information(self, "Project Loaded", message)

            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.warning(
                    self, "Error", f"Failed to load project: {str(e)}")

    def highlight_cell(self, cell_id):
        """Highlight a specific cell when clicked on PCA plot"""
        print(f"App: Highlighting cell {cell_id}")

        # Ensure cell ID is an integer
        cell_id = int(cell_id)

        # Check if we have the cell mapping
        if not hasattr(self, "current_cell_mapping") or cell_id not in self.current_cell_mapping:
            print(f"Cell {cell_id} not found in current mapping")
            return

        # Get current frame parameters from ViewAreaWidget
        t = self.viewArea.current_t
        p = self.viewArea.current_p
        c = self.viewArea.current_c

        try:
            # Get the segmentation
            segmented_image = self.image_data.segmentation_cache[t, p, c]

            if segmented_image is None:
                print(f"No segmentation available for highlighting")
                return

            # Create single-cell mapping for the selected cell
            single_cell_mapping = {cell_id: self.current_cell_mapping[cell_id]}

            # Import the morphology function
            from nd2_analyzer.analysis.morphology.morphology import annotate_binary_mask

            # Create highlighted image with just the one cell
            highlighted_image = annotate_binary_mask(
                segmented_image, single_cell_mapping)

            # Display on the view area's image label
            height, width = highlighted_image.shape[:2]
            qimage = QImage(highlighted_image.data, width, height,
                            highlighted_image.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(
                self.viewArea.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.viewArea.image_label.setPixmap(pixmap)

        except Exception as e:
            import traceback
            print(f"Error highlighting cell: {e}")
            traceback.print_exc()
