import os
import sys

import numpy as np
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvas
from pubsub import pub

from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.analysis.morphology.morphology import (
    annotate_binary_mask,
)
from nd2_analyzer.data.experiment import Experiment
from nd2_analyzer.data.image_data import ImageData
from .dialogs import AboutDialog, ExperimentDialog
from nd2_analyzer.ui.dialogs.roisel import PolygonROISelector
from .widgets import ViewAreaWidget, PopulationWidget, SegmentationWidget, MorphologyWidget, TrackingManager
from ..data.appstate import ApplicationState


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.appstate = ApplicationState.create_instance()

        self.setWindowTitle("Partaker 3 - GUI")
        self.setGeometry(100, 100, 1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.viewArea = ViewAreaWidget()
        self.layout.addWidget(self.viewArea)

        self.segmentation_tab = SegmentationWidget()
        self.populationTab = PopulationWidget()
        self.morphologyTab = MorphologyWidget()
        self.trackingTab = TrackingManager()

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        self.tab_widget.addTab(self.populationTab, "Population")
        self.tab_widget.addTab(self.morphologyTab, "Morphology")
        self.tab_widget.addTab(self.trackingTab,
                               "Tracking - Lineage Tree")
        self.layout.addWidget(self.tab_widget)

        self.initMenuBar()

        # TODO: do we need to subscribe here?
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
            callback(self.application_state.image_data)
        else:
            print("No image_data available")
            callback(None)

    def on_exp_loaded(self, experiment: Experiment):
        ImageData.load_nd2(experiment.nd2_files)
        pub.sendMessage("image_data_loaded", ImageData.get_instance())

    def on_image_request(self, time, position, channel):
        """Handle requests for raw image data"""
        if not self.application_state.image_data:
            return

        # Retrieve the image data
        try:
            if self.application_state.image_data.is_nd2:
                if self.has_channels:
                    image = self.application_state.image_data.data[time, position, channel]
                else:
                    image = self.application_state.image_data.data[time, position]
            else:
                image = self.application_state.image_data.data[time]

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
        if not self.application_state.image_data:
            return

        try:
            # Get the appropriate segmentation model
            if model_name:
                self.application_state.image_data.segmentation_cache.with_model(model_name)
            else:
                self.application_state.image_data.segmentation_cache.with_model(
                    self.model_dropdown.currentText())

            # Get the segmentation
            segmented_image = self.application_state.image_data.segmentation_cache[time,
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
        roi_dialog = PolygonROISelector()
        roi_dialog.exec_()  # Use exec_ to make it modal

    def on_draw_cell_bounding_boxes(self, time, position, channel, cell_mapping):
        """Handle request to draw cell bounding boxes"""
        # Get the segmentation using the same model as the segmentation cache
        if not hasattr(self, "image_data") or not self.application_state.image_data:
            print("No image data available")
            return

        # Get the current model from the cache
        current_model = self.application_state.image_data.segmentation_cache.model_name
        if not current_model:
            # Default to a standard model if none set
            current_model = "bact_phase_cp3"  # This is CELLPOSE_BACT_PHASE
            self.application_state.image_data.segmentation_cache.with_model(current_model)

        # Get the segmentation data
        segmented_image = self.application_state.image_data.segmentation_cache[time,
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

        image_menu = menu_bar.addMenu("Image")
        roi_action = QAction("ROI", self)
        roi_action.setShortcut("Ctrl+R")
        roi_action.triggered.connect(self.open_roi_selector)
        image_menu.addAction(roi_action)

        test_menu = menu_bar.addMenu("Test")
        test_action = QAction("Test", self)
        test_action.setShortcut("Ctrl+T")
        test_action.triggered.connect(self.hhln_test)
        test_menu.addAction(test_action)

        if sys.platform == "darwin":
            about_action_mac = QAction("About", self)
            about_action_mac.triggered.connect(self.show_about_dialog)
            self.menuBar().addAction(about_action_mac)

    def hhln_test(self):
        ImageData.load_nd2(
            "/Users/hiram/Documents/EVERYTHING/20-29 Research/22 OliveiraLab/22.12 ND2 analyzer/nd2-analyzer/SR_1_5_2h_Pre-C_3h_IPTG_After10h_05_MC.nd2")

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
            if hasattr(self, "image_data") and hasattr(self.application_state.image_data, "segmentation_cache"):
                cache = self.application_state.image_data.segmentation_cache
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
                self.application_state.image_data.save(folder_path)
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
            tracking_saved = self.trackingTab.tracking_widget.save_tracking_data(
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
                self.application_state.image_data = ImageData.load(folder_path)

                # Load metrics data
                metrics_service = MetricsService()
                metrics_loaded = metrics_service.load_from_file(folder_path)

                # Update UI based on loaded image data
                if hasattr(self, "viewArea"):
                    pub.sendMessage("image_data_loaded",
                                    image_data=self.application_state.image_data)

                population_loaded = False
                if hasattr(self, "populationTab"):
                    population_loaded = self.populationTab.load_population_data(folder_path)
                    print(f"Population data loaded: {population_loaded}")

                # Load tracking data if available
                tracking_loaded = False
                if hasattr(self, "tracking_manager") and hasattr(self.trackingTab, "tracking_widget"):
                    tracking_loaded = self.trackingTab.tracking_widget.load_tracking_data(
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
            segmented_image = self.application_state.image_data.segmentation_cache[t, p, c]

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
