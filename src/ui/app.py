import sys
import os
from types import new_class

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QButtonGroup,
    QTabWidget,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QSlider,
    QHBoxLayout,
    QCheckBox,
    QMessageBox,
    QRadioButton,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QFormLayout,
    QDialog,
    QSizePolicy, QComboBox, QLabel, QProgressBar, QDialogButtonBox, QProgressDialog
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot
import PySide6.QtAsyncio as QtAsyncio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from cellpose import models, utils

# import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt
import nd2
import pandas as pd
import numpy as np
import cv2
import imageio.v3 as iio
import tifffile
from morphology import annotate_image, extract_cells_and_metrics, annotate_binary_mask, extract_cell_morphologies, extract_cell_morphologies_time, classify_morphology

from segmentation.segmentation_models import SegmentationModels
from segmentation.segmentation_cache import SegmentationCache

from image_functions import remove_stage_jitter_MAE
from PySide6.QtCore import QThread, Signal, QObject

# import pims
from matplotlib.backends.backend_qt5agg import FigureCanvas

import seaborn as sns

from population import get_fluorescence_all_experiments, get_fluorescence_single_experiment

from fluorescence.sc import analyze_fluorescence_singlecell, analyze_fluorescence_total
from fluorescence.rpu import AVAIL_RPUS

from tracking import track_cells, optimize_tracking_parameters

from tqdm import tqdm

import matplotlib.pyplot as plt

from skimage.measure import label, regionprops
from image_data import ImageData


class MorphologyWorker(QObject):
    progress = Signal(int)  # Progress updates
    finished = Signal(object)  # Finished with results
    error = Signal(str)  # Emit error message

    def __init__(
            self,
            image_data,
            image_frames,
            num_frames,
            position,
            channel):
        super().__init__()
        self.image_data = image_data
        self.image_frames = image_frames
        self.num_frames = num_frames
        self.position = position
        self.channel = channel
        # self.metrics_table.itemClicked.connect(self.on_table_item_click)
        self.cell_mapping = {}

    def run(self):
        results = {}
        try:
            for t in range(self.num_frames):

                current_frame = self.image_frames[t]
                # Skip empty/invalid frames
                if np.mean(current_frame) == 0 or np.std(
                        current_frame) == 0:
                    print(f"Skipping empty frame T={t}")
                    self.progress.emit(t + 1)
                    continue

                t, p, c = (t, self.position, self.channel)

                binary_image = self.image_data.seg_cache[t, p, c]

                # Validate segmentation result
                if binary_image is None or binary_image.sum() == 0:
                    print(f"Frame {t}: No valid segmentation")
                    self.progress.emit(t + 1)
                    continue

                # Extract morphology metrics
                cell_mapping = extract_cells_and_metrics(
                    self.image_frames[t], binary_image)

                # Then convert the cell_mapping to a metrics dataframe
                metrics_list = [data["metrics"] for data in cell_mapping.values()]
                metrics = pd.DataFrame(metrics_list)

                if not metrics.empty:
                    total_cells = len(metrics)

                    # Calculate Morphology Fractions
                    morphology_counts = metrics["morphology_class"].value_counts(normalize=True)
                    fractions = morphology_counts.to_dict()

                    # Save results for this frame, including the raw metrics
                    results[t] = {
                        "fractions": fractions,
                        "metrics": metrics  # Include the full metrics dataframe
                    }
                else:
                    print(
                        f"Frame {t}: Metrics computation returned no valid data.")

                self.progress.emit(t + 1)  # Update progress bar

            if results:
                self.finished.emit(results)  # Emit processed results
            else:
                self.error.emit("No valid results found in any frame.")
        except Exception as e:
            raise e
            self.error.emit(str(e))


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.morphology_colors = {
            "Small": (255, 0, 0),       # Blue
            "Round": (0, 0, 255),       # Red
            "Normal": (0, 255, 0),      # Green
            "Elongated": (0, 255, 255),  # Yellow
            "Deformed": (255, 0, 255),  # Magenta
        }

        self.morphology_colors_rgb = {
            key: (color[2] / 255, color[1] / 255, color[0] / 255)
            for key, color in self.morphology_colors.items()
        }

        # Initialize the processed_images list to store images for export
        self.processed_images = []

        self.setWindowTitle("Partaker 3 - GUI")
        self.setGeometry(100, 100, 1000, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        self.tab_widget = QTabWidget()

        # Initialize other tabs and UI components
        self.importTab = QWidget()
        self.viewArea = QWidget()
        self.layout.addWidget(self.viewArea)
        self.exportTab = QWidget()
        self.populationTab = QWidget()
        self.morphologyTab = QWidget()
        self.morphologyTimeTab = QWidget()

        self.initUI()
        self.layout.addWidget(self.tab_widget)

    def load_from_folder(self, folder_path):
        p = Path(folder_path)

        images = p.iterdir()
        # images = filter(lambda x : x.name.lower().endswith(('.tif')), images)
        img_filelist = sorted(images, key=lambda x: int(x.stem))

        def preproc_img(img): return img  # Placeholder for now
        loaded = np.array([preproc_img(cv2.imread(str(_img)))
                          for _img in img_filelist])

        self.image_data = ImageData(loaded, is_nd2=False)

        print(f"Loaded dataset: {self.image_data.data.shape}")
        self.info_label.setText(f"Dataset size: {self.image_data.data.shape}")
        QMessageBox.about(
            self, "Import", f"Loaded {self.image_data.data.shape[0]} pictures"
        )

        self.image_data.phc_path = folder_path

        self.image_data.segmentation_cache.clear()  # Clear segmentation cache
        print("Segmentation cache cleared.")

    def load_nd2_file(self, file_path):

        self.file_path = file_path
        with nd2.ND2File(file_path) as nd2_file:
            self.nd2_file = nd2_file
            self.dimensions = nd2_file.sizes
            info_text = f"Number of dimensions: {nd2_file.sizes}\n"

            for dim, size in self.dimensions.items():
                info_text += f"{dim}: {size}\n"

            if "C" in self.dimensions.keys():
                self.has_channels = True
                self.channel_number = self.dimensions["C"]
                self.slider_c.setMinimum(0)
                self.slider_c.setMaximum(self.channel_number - 1)
            else:
                self.has_channels = False

            self.info_label.setText(info_text)
            self.image_data = ImageData(nd2.imread(
                file_path, dask=True), is_nd2=True)

            # Set the slider range for position (P) immediately based on
            # dimensions
            max_position = self.dimensions.get("P", 1) - 1
            self.slider_p.setMaximum(max_position)
            # Update population tab slider
            self.slider_p_5.setMaximum(max_position)
            self.update_controls()
            self.display_image()

            self.image_data.segmentation_cache.clear()  # Clear segmentation cache
            print("Segmentation cache cleared.")

    def display_file_info(self, file_path):
        info_text = f"Number of dimensions: {len(self.dimensions)}\n"
        for dim, size in self.dimensions.items():
            info_text += f"{dim}: {size}\n"
        self.info_label.setText(info_text)

    def update_controls(self):
        # Set max values for sliders based on ND2 dimensions
        t_max = self.dimensions.get("T", 1) - 1
        p_max = self.dimensions.get("P", 1) - 1

        # Initialize sliders with full ranges
        self.slider_t.setMaximum(t_max)
        self.slider_p.setMaximum(p_max)

    def update_slider_range(self):
        self.slider_t.setMaximum(self.dimensions.get("T", 1) - 1)

        max_position = self.dimensions.get("P", 1) - 1
        self.slider_p.setMaximum(max_position)
        self.slider_p_5.setMaximum(max_position)

        # Population tab
        self.time_max_box.setMaximum(self.dimensions.get("T", 1) - 1)

    def show_cell_area(self, img):
        from skimage import measure
        import seaborn as sns

        # Check if the image type is CV_32FC1 and convert to CV_8UC1
        if img.dtype == np.float32 or img.dtype == np.int32 or img.dtype == np.int64:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

        # Binarize the image using Otsu's thresholding
        _, bw_image = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bw_image, connectivity=8)

        # Extract pixel counts for each component (ignore background)
        # Skip the first label (background)
        pixel_counts = stats[1:, cv2.CC_STAT_AREA]

    def overlay_labels_on_segmentation(self, segmented_images):
        """
        Overlay cell IDs on segmented images.

        Parameters:
        segmented_images (list of np.ndarray): List of segmented images (labeled masks).

        Returns:
        list of np.ndarray: Images with overlaid cell IDs.
        """
        labeled_images = []

        for mask in segmented_images:
            # Convert to RGB for color text overlay
            labeled_image = cv2.cvtColor(
                mask.astype(
                    np.uint8) * 255,
                cv2.COLOR_GRAY2BGR)

            # Get unique cell IDs (ignore background 0)
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids != 0]

            for cell_id in unique_ids:
                # Find coordinates of the current cell
                coords = np.column_stack(np.where(mask == cell_id))

                # Calculate centroid of the cell
                centroid_y, centroid_x = coords.mean(axis=0).astype(int)

                # Overlay cell ID text at the centroid
                cv2.putText(
                    labeled_image,
                    str(cell_id),
                    (centroid_x, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,  # Font size
                    (255, 255, 255),  # White color
                    1,  # Thickness
                    cv2.LINE_AA
                )

            labeled_images.append(labeled_image)

        return labeled_images

    def display_image(self):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Retrieve the current frame
        image_data = self.image_data.data
        if self.image_data.is_nd2:
            image_data = image_data[t, p,
                                    c] if self.has_channels else image_data[t, p]
        else:
            image_data = image_data[t]

        image_data = np.array(image_data)  # Ensure it's a NumPy array

        # Apply thresholding or segmentation if selected
        if self.radio_thresholding.isChecked():
            threshold = self.threshold_slider.value()
            image_data = cv2.threshold(
                image_data, threshold, 255, cv2.THRESH_BINARY)[1]
            if hasattr(image_data, 'compute'):
                image_data = image_data.compute()

        elif self.radio_labeled_segmentation.isChecked():
            # Ensure segmentation model is set correctly in seg_cache
            selected_model = self.model_dropdown.currentText()
            self.image_data.seg_cache.with_model(selected_model)

            # Retrieve segmentation from seg_cache
            segmented = self.image_data.seg_cache[t, p, c]

            if segmented is None:
                print(f"[ERROR] Segmentation failed for T={t}, P={p}, C={c}")
                QMessageBox.warning(
                    self, "Segmentation Error", "Segmentation failed.")
                return

            # Relabel the segmented regions
            labeled_cells = label(segmented)

            # Ensure relabeling created valid labels
            max_label = labeled_cells.max()
            if max_label == 0:
                print("[ERROR] No valid labeled regions found.")
                QMessageBox.warning(
                    self, "Labeled Segmentation Error", "No valid labeled regions found.")
                return

            # Convert labels to color image
            labeled_image = plt.cm.nipy_spectral(
                labeled_cells.astype(float) / max_label)
            labeled_image = (labeled_image[:, :, :3] * 255).astype(np.uint8)

            # Overlay Cell IDs
            props = regionprops(labeled_cells)
            for prop in props:
                y, x = prop.centroid  # Get centroid coordinates
                cell_id = prop.label  # Get cell ID
                cv2.putText(labeled_image, str(cell_id), (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the labeled image
            height, width, _ = labeled_image.shape
            qimage = QImage(
                labeled_image.data,
                width,
                height,
                3 * width,
                QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

            return

        elif self.radio_segmented.isChecked():
            cache_key = (t, p, c)
            # if cache_key in self.image_data.segmentation_cache:
            #     print(
            #         f"[CACHE HIT] Using cached segmentation for T={t}, P={p}, C={c}")
            #     image_data = self.image_data.segmentation_cache[cache_key]
            # else:
            #     if self.model_dropdown.currentText() == SegmentationModels.CELLPOSE:
            #         model_type = 'bact_phase_cp3' if c in (
            #             0, None) else 'bact_fluor_cp3'
            #         print(f"Using model type: {model_type} for channel {c}")
            #     else:
            #         model_type = None

            #     frame = self.image_data.data[t, p,
            # c] if self.image_data.is_nd2 else self.image_data.data[t]

            #     # Perform segmentation
            #     segmented = SegmentationModels().segment_images(np.array(
            #         [frame]), self.model_dropdown.currentText(), model_type=model_type)[0]

            #     # Relabel if output is binary mask
            #     if segmented.max() == 255 and len(np.unique(segmented)) <= 2:

            #         segmented = label(segmented).astype(
            #             np.uint8) * 255  # Ensure correct type for OpenCV
            #         segmented_display = (segmented > 0).astype(np.uint8) * 255

            #     self.image_data.segmentation_cache[cache_key] = segmented_display
            #     image_data = segmented_display

            self.image_data.seg_cache.with_model(
                self.model_dropdown.currentText())  # Setting the model we want
            image_data = self.image_data.seg_cache[t, p, c]

            # Extract and cache morphology metrics if not already cached
            metrics_key = cache_key + ('metrics',)
            if metrics_key not in self.image_data.segmentation_cache:
                # print(f"Extracting morphology metrics for T={t}, P={p}, C={c}")
                metrics = extract_cell_morphologies(image_data)
                self.image_data.segmentation_cache[metrics_key] = metrics

        else:  # Normal view or overlay
            if self.radio_overlay_outlines.isChecked():
                # Ensure segmentation model is set correctly in seg_cache
                selected_model = self.model_dropdown.currentText()
                self.image_data.seg_cache.with_model(selected_model)

                # Retrieve segmentation from seg_cache
                segmented_image = self.image_data.seg_cache[t, p, c]

                if segmented_image is None:
                    print(
                        f"[ERROR] Segmentation failed for T={t}, P={p}, C={c}")
                    QMessageBox.warning(
                        self, "Segmentation Error", "Segmentation failed.")
                    return

                print(
                    f"[SUCCESS] Retrieved segmentation for overlay - T={t}, P={p}, C={c}")

                # Extract outlines
                outlines = utils.masks_to_outlines(segmented_image)
                overlay = image_data.copy()

                # Verify dimensions match before applying overlay
                if outlines.shape == overlay.shape:
                    overlay[outlines] = overlay.max()
                else:
                    print(
                        f"Dimension mismatch - Outline shape: {outlines.shape}, Image shape: {overlay.shape}")
                    outlines = cv2.resize(
                        outlines.astype(np.uint8),
                        (overlay.shape[1], overlay.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay[outlines] = overlay.max()

                image_data = overlay

            # Normalize and apply color for normal/overlay views
            image_data = cv2.normalize(
                image_data,
                None,
                0,
                255,
                cv2.NORM_MINMAX).astype(np.uint8)

            # Apply color based on channel for non-binary images
            if self.has_channels:
                colored_image = np.zeros(
                    (image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
                if c == 0:  # Phase contrast - grayscale
                    colored_image = cv2.cvtColor(
                        image_data, cv2.COLOR_GRAY2BGR)
                elif c == 1:  # mCherry - red
                    colored_image[:, :, 2] = image_data  # Red channel
                elif c == 2:  # YFP - yellow/green
                    colored_image[:, :, 1] = image_data  # Green channel
                    colored_image[:, :, 2] = image_data * \
                        0.5  # Add red to make it yellow
                image_data = colored_image
                image_format = QImage.Format_RGB888
            else:
                image_format = QImage.Format_Grayscale8

        # Normalize the image safely for grayscale images only
        if len(image_data.shape) == 2:  # Grayscale
            if image_data.max() > 0:
                image_data = (
                    image_data.astype(
                        np.float32) /
                    image_data.max() *
                    65535).astype(
                    np.uint16)
            else:
                image_data = np.zeros_like(image_data, dtype=np.uint16)

        # Determine format based on image type
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            image_format = QImage.Format_RGB888
            height, width, _ = image_data.shape
        else:
            image_format = QImage.Format_Grayscale16
            height, width = image_data.shape[:2]

        # Display image
        image = QImage(
            image_data.data,
            width,
            height,
            image_data.strides[0],
            image_format)
        pixmap = QPixmap.fromImage(image).scaled(
            self.image_label.size(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        # Store this processed image for export
        self.processed_images.append(image_data)

    def initImportTab(self):
        def importFile():
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName()
            if file_path:
                self.load_nd2_file(file_path)

        def importFolder():
            file_dialog = QFileDialog()
            _path = file_dialog.getExistingDirectory()
            self.load_from_folder(_path)

        def slice_and_export():
            if not hasattr(self, "image_data") or not self.image_data.is_nd2:
                QMessageBox.warning(
                    self, "Error", "No ND2 file loaded to slice.")
                return

            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Sliced Data", "", "TIFF Files (*.tif);;All Files (*)")

            if not save_path:
                QMessageBox.warning(
                    self, "Error", "No save location selected.")
                return

            try:
                sliced_data = self.image_data.data[0:4, 0, :, :].compute()

                tifffile.imwrite(save_path, np.array(sliced_data), imagej=True)
                QMessageBox.information(
                    self, "Success", f"Sliced data saved to {save_path}"
                )
            except Exception as e:
                QMessageBox.warning(
                    self, "Error", f"Failed to slice and export: {e}")

        layout = QVBoxLayout(self.importTab)

        slice_button = QPushButton("Slice and Export")
        slice_button.clicked.connect(slice_and_export)
        layout.addWidget(slice_button)

        button = QPushButton("Select File / Folder")
        button.clicked.connect(
            lambda: (
                importFile()
                if not self.is_folder_checkbox.isChecked()
                else importFolder()
            )
        )
        layout.addWidget(button)

        self.is_folder_checkbox = QCheckBox("Load from folder?")
        layout.addWidget(self.is_folder_checkbox)

        self.filename_label = QLabel("Filename will be shown here.")
        layout.addWidget(self.filename_label)

        self.info_label = QLabel("File info will be shown here.")
        layout.addWidget(self.info_label)

    def initMorphologyTimeTab(self):
        layout = QVBoxLayout(self.morphologyTimeTab)

        # Process button
        self.segment_button = QPushButton("Process Morphology Over Time")
        layout.addWidget(self.segment_button)

        # Create a horizontal layout for the tracking buttons
        tracking_buttons_layout = QHBoxLayout()

        # Button to track cells over time
        self.track_button = QPushButton("Track Cells")
        self.track_button.clicked.connect(self.track_cells_over_time)
        tracking_buttons_layout.addWidget(self.track_button)

        # Button for lineage tracking
        self.lineage_button = QPushButton("Visualize Lineage Tree")
        self.lineage_button.clicked.connect(self.track_cells_with_lineage)
        tracking_buttons_layout.addWidget(self.lineage_button)

        # Button to create tracking video
        self.overlay_video_button = QPushButton("Tracking Video")
        self.overlay_video_button.clicked.connect(
            self.overlay_tracks_on_images)
        tracking_buttons_layout.addWidget(self.overlay_video_button)

        # Add the horizontal button layout to the main layout
        layout.addLayout(tracking_buttons_layout)

        # Inner Tabs for Plots
        self.plot_tabs = QTabWidget()
        self.morphology_fractions_tab = QWidget()

        self.plot_tabs.addTab(
            self.morphology_fractions_tab,
            "Morphology Fractions")
        layout.addWidget(self.plot_tabs)

        # Plot for Morphology Fractions
        morphology_fractions_layout = QVBoxLayout(
            self.morphology_fractions_tab)
        self.figure_morphology_fractions = plt.figure()
        self.canvas_morphology_fractions = FigureCanvas(
            self.figure_morphology_fractions)
        morphology_fractions_layout.addWidget(self.canvas_morphology_fractions)

        # Progress Bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Connect Process Button
        self.segment_button.clicked.connect(
            self.process_morphology_time_series)

    
    def track_cells_over_time(self):
        """
        Tracks cells over time using BayesianTracker (btrack) and visualizes the results.
        """
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None
        
        if not self.image_data.is_nd2:
            QMessageBox.warning(self, "Error", "Tracking requires an ND2 dataset.")
            return
        
        try:
            t = self.dimensions["T"]  # Get total number of frames
            
            # Create a progress dialog
            progress = QProgressDialog("Preparing frames for tracking...", "Cancel", 0, t, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Use seg_cache for consistency with the rest of the application
            self.image_data.seg_cache.with_model(self.model_dropdown.currentText())
            
            # Create an array to store labeled images
            labeled_frames = []
            
            # Process each frame to create labeled images
            for i in range(t):
                if progress.wasCanceled():
                    return
                    
                progress.setValue(i)
                
                # Get segmentation from cache
                segmented = self.image_data.seg_cache[i, p, c]
                
                if segmented is not None:
                    # Convert binary segmentation to labeled regions
                    from skimage.measure import label
                    labeled = label(segmented)
                    labeled_frames.append(labeled)
            
            progress.setValue(t)
            
            if not labeled_frames:
                QMessageBox.warning(self, "Error", "No segmented frames found for tracking.")
                return
                
            # Convert to numpy array
            labeled_frames = np.array(labeled_frames)
            print(f"Prepared {len(labeled_frames)} frames for tracking")
            
            # Update progress dialog
            progress.setLabelText("Running cell tracking...")
            progress.setValue(0)
            progress.setMaximum(100)
            
            # Run cell tracking
            all_tracks = track_cells(labeled_frames)
            
            # The tracks are already dictionaries, so we don't need to convert them
            
            # Filter tracks - only keep those spanning a minimum number of frames
            MIN_TRACK_LENGTH = 5  # Only keep tracks that span at least 5 frames
            filtered_tracks = [track for track in all_tracks if len(track['x']) >= MIN_TRACK_LENGTH]
            
            # Sort tracks by length (longest first)
            filtered_tracks.sort(key=lambda track: len(track['x']), reverse=True)
            
            # Keep only the top N longest tracks for visualization
            MAX_TRACKS_TO_DISPLAY = 30
            display_tracks = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]
            
            # Calculate statistics for reporting
            total_tracks = len(all_tracks)
            long_tracks = len(filtered_tracks)
            displayed_tracks = len(display_tracks)
            
            # Store the filtered tracks for visualization
            self.tracked_cells = display_tracks
            
            # Show information about the tracking results
            QMessageBox.information(
                self, 
                "Tracking Complete",
                f"Cell tracking completed successfully.\n\n"
                f"Total tracks detected: {total_tracks}\n"
                f"Tracks spanning {MIN_TRACK_LENGTH}+ frames: {long_tracks}\n"
                f"Tracks displayed: {displayed_tracks}"
            )
            
            # Visualize the tracking results
            self.visualize_tracking(self.tracked_cells)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Tracking Error",
                f"Failed to track cells: {str(e)}"
            )
    
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
        from matplotlib.colors import Normalize
        
        # Use a colormap that's easier to distinguish
        cmap = cm.get_cmap('tab20', min(20, len(tracks)))
        
        # Calculate displacement for each track
        track_displacements = []
        for track in tracks:
            if len(track['x']) >= 2:
                # Calculate total displacement (Euclidean distance from start to end)
                start_x, start_y = track['x'][0], track['y'][0]
                end_x, end_y = track['x'][-1], track['y'][-1]
                displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
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
            ax.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=6)  # Start
            ax.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=6)  # End
        
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
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7), verticalalignment='bottom')
        
        # Draw the plot
        self.canvas_morphology_fractions.draw()
    
    def overlay_tracks_on_images(self):
        """
        Overlays tracking trajectories on the segmented images and creates a video.
        """
        if not hasattr(self, "tracked_cells") or not self.tracked_cells:
            QMessageBox.warning(
                self, "Error", "No tracking data available. Run cell tracking first.")
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

            segmented = self.image_data.seg_cache[i, p, c]
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
        from tracking import overlay_tracks_on_images as create_tracking_video

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
                self, "Video Created", f"Tracking visualization saved to {output_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self, "Error", f"Failed to create tracking video: {str(e)}")


    def track_cells_with_lineage(self):
        """
        Track cells over time with lineage detection and visualize both
        tracking results and lineage tree.
        """
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        if not self.image_data.is_nd2:
            QMessageBox.warning(
                self, "Error", "Tracking requires an ND2 dataset.")
            return

        try:
            t = self.dimensions["T"]  # Get total number of frames

            # Create a progress dialog
            progress = QProgressDialog(
                "Preparing frames for tracking...", "Cancel", 0, t, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Use seg_cache for consistency with the rest of the application
            self.image_data.seg_cache.with_model(
                self.model_dropdown.currentText())

            # Create an array to store labeled images
            labeled_frames = []

            # Process each frame to create labeled images
            for i in range(t):
                if progress.wasCanceled():
                    return

                progress.setValue(i)

                # Get segmentation from cache
                segmented = self.image_data.seg_cache[i, p, c]

                if segmented is not None:
                    # Convert binary segmentation to labeled regions
                    from skimage.measure import label
                    labeled = label(segmented)
                    labeled_frames.append(labeled)

            progress.setValue(t)

            if not labeled_frames:
                QMessageBox.warning(
                    self, "Error", "No segmented frames found for tracking.")
                return

            # Convert to numpy array
            labeled_frames = np.array(labeled_frames)
            print(f"Prepared {len(labeled_frames)} frames for tracking")

            # Update progress dialog
            progress.setLabelText(
                "Running cell tracking with lineage detection...")
            progress.setValue(0)
            progress.setMaximum(100)

            # Call the tracking function without extra parameters
            from tracking import track_cells
            all_tracks = track_cells(labeled_frames)

            # Filter tracks - only keep those spanning a minimum number of frames
            MIN_TRACK_LENGTH = 5  # Only keep tracks that span at least 5 frames
            filtered_tracks = [track for track in all_tracks if len(
                track['x']) >= MIN_TRACK_LENGTH]

            # Sort tracks by length (longest first)
            filtered_tracks.sort(
                key=lambda track: len(track['x']), reverse=True)

            # Keep only the top N longest tracks for visualization
            MAX_TRACKS_TO_DISPLAY = 30
            display_tracks = filtered_tracks[:MAX_TRACKS_TO_DISPLAY]

            # Calculate statistics for reporting
            total_tracks = len(all_tracks)
            long_tracks = len(filtered_tracks)
            displayed_tracks = len(display_tracks)

            # Count division events
            division_events = sum(1 for track in all_tracks if track.get(
                'children') and len(track['children']) > 0)

            # Store the filtered tracks for visualization
            self.tracked_cells = display_tracks

            # Store full lineage information
            self.lineage_tracks = all_tracks

            # Show information about the tracking results
            QMessageBox.information(
                self,
                "Tracking Complete",
                f"Cell tracking with lineage detection completed.\n\n"
                f"Total tracks detected: {total_tracks}\n"
                f"Tracks spanning {MIN_TRACK_LENGTH}+ frames: {long_tracks}\n"
                f"Cell division events detected: {division_events}\n"
                f"Tracks displayed: {displayed_tracks}"
            )

            # Create visualization showing cell trajectories
            self.visualize_tracking(self.tracked_cells)

            # Ask if user wants to visualize lineage tree
            reply = QMessageBox.question(
                self,
                "Lineage Analysis",
                "Would you like to visualize the cell lineage tree?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                self.show_lineage_dialog()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Tracking Error",
                f"Failed to track cells with lineage: {str(e)}"
            )
    
    def show_lineage_dialog(self):
        """
        Show dialog with options to visualize lineage trees
        """
        # Check if tracking has been performed
        if not hasattr(self, "lineage_tracks") or not self.lineage_tracks:
            QMessageBox.warning(
                self, "Error", "No tracking data available. Run tracking with lineage first.")
            return
        
        # Create a dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Lineage Tree Visualization")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(700)
        layout = QVBoxLayout(dialog)
        
        # Add description
        info_label = QLabel(
            "Select a visualization option below. Cell lineage trees show parent-child relationships.\n"
            "Red nodes indicate cells that divide further, blue nodes are cells that don't divide."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Create a horizontal layout for selection options
        selection_layout = QHBoxLayout()
        
        # Option group
        option_group = QButtonGroup(dialog)
        
        # Top trees option
        top_radio = QRadioButton("Top 5 Largest Lineage Trees")
        top_radio.setChecked(True)
        option_group.addButton(top_radio)
        selection_layout.addWidget(top_radio)
        
        # Individual cell option
        cell_radio = QRadioButton("Specific Cell Lineage:")
        option_group.addButton(cell_radio)
        selection_layout.addWidget(cell_radio)
        
        # Dropdown for cell selection
        cell_combo = QComboBox()
        cell_combo.setEnabled(False)
        selection_layout.addWidget(cell_combo)
        
        # Find dividing cells and add to dropdown
        dividing_cells = []
        for track in self.lineage_tracks:
            if 'children' in track and track['children']:
                dividing_cells.append(track['ID'])
        
        # Sort by ID for easier selection
        dividing_cells.sort()
        
        # Add to dropdown
        for cell_id in dividing_cells:
            cell_combo.addItem(f"Cell {cell_id}")
        
        # Add selection layout
        layout.addLayout(selection_layout)
        
        # Enable/disable combo box based on radio selection
        def update_combo_state():
            cell_combo.setEnabled(cell_radio.isChecked())
        
        top_radio.toggled.connect(update_combo_state)
        cell_radio.toggled.connect(update_combo_state)
        
        # Matplotlib figure for preview
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        figure = Figure(figsize=(9, 6), tight_layout=True)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        view_button = QPushButton("Visualize")
        save_button = QPushButton("Save")
        close_button = QPushButton("Close")
        
        button_layout.addWidget(view_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # Function to create visualization
        def create_visualization():
            import networkx as nx
            figure.clear()
            
            if top_radio.isChecked():
                # Create visualization of top 5 lineage trees
                G = nx.DiGraph()
                
                # Add nodes and edges
                for track in self.lineage_tracks:
                    track_id = track['ID']
                    
                    # Add node with timing info
                    G.add_node(track_id, 
                            start_time=track['t'][0] if track['t'] else 0,
                            end_time=track['t'][-1] if track['t'] else 0,
                            duration=len(track['t']) if track['t'] else 0,
                            divides=bool(track.get('children', [])))
                    
                    # Add edges for children
                    if track.get('children'):
                        for child_id in track['children']:
                            G.add_edge(track_id, child_id)
                
                # Find connected components (separate lineage trees)
                components = list(nx.weakly_connected_components(G))
                print(f"Found {len(components)} separate lineage trees")
                
                # Get top 5 largest components
                largest_components = sorted(components, key=len, reverse=True)[:5]
                
                # Create subplots
                if largest_components:
                    num_components = len(largest_components)
                    for i, component in enumerate(largest_components):
                        if num_components > 1:
                            ax = figure.add_subplot(1, num_components, i+1)
                        else:
                            ax = figure.add_subplot(111)
                        
                        # Get subgraph for this component
                        subgraph = G.subgraph(component)
                        
                        # Find root(s)
                        roots = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0]
                        if not roots:
                            ax.text(0.5, 0.5, "No root node found", 
                                    horizontalalignment='center', 
                                    verticalalignment='center',
                                    transform=ax.transAxes)
                            continue
                        
                        # Create hierarchical layout
                        try:
                            # Try for a "tree" layout
                            pos = hierarchy_pos(subgraph, roots[0])
                        except:
                            # Fall back to spring layout
                            pos = nx.spring_layout(subgraph, seed=42)
                        
                        # Draw nodes with color based on whether they divide
                        node_colors = ['red' if subgraph.nodes[n]['divides'] else 'blue' 
                                    for n in subgraph.nodes()]
                        
                        # Size nodes based on track duration
                        node_sizes = [max(100, subgraph.nodes[n]['duration'] * 10) 
                                    for n in subgraph.nodes()]
                        
                        # Draw the graph
                        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                                            node_color=node_colors, alpha=0.8, ax=ax)
                        nx.draw_networkx_edges(subgraph, pos, edge_color='gray',
                                            arrows=True, arrowstyle='-|>', ax=ax)
                        nx.draw_networkx_labels(subgraph, pos, font_size=8, ax=ax)
                        
                        ax.set_title(f"Lineage Tree {i+1}\n({len(component)} cells)")
                        ax.axis('off')
                else:
                    ax = figure.add_subplot(111)
                    ax.text(0.5, 0.5, "No lineage trees found", 
                            horizontalalignment='center', 
                            verticalalignment='center',
                            transform=ax.transAxes)
            
            else:  # Specific cell selected
                if not cell_combo.currentText():
                    return
                    
                # Get selected cell ID
                cell_id = int(cell_combo.currentText().replace("Cell ", ""))
                
                # Create a subgraph for this cell's lineage
                G = nx.DiGraph()
                
                # Find this cell and all its descendants
                def add_descendants(cell_id):
                    track = next((t for t in self.lineage_tracks if t['ID'] == cell_id), None)
                    if not track:
                        return
                    
                    # Add this cell
                    G.add_node(cell_id, 
                            start_time=track['t'][0] if track['t'] else 0,
                            end_time=track['t'][-1] if track['t'] else 0,
                            duration=len(track['t']) if track['t'] else 0,
                            divides=bool(track.get('children', [])))
                    
                    # Add all children recursively
                    if track.get('children'):
                        for child_id in track['children']:
                            G.add_edge(cell_id, child_id)
                            add_descendants(child_id)
                
                # Start with the selected cell
                add_descendants(cell_id)
                
                # Create a single plot
                ax = figure.add_subplot(111)
                
                if len(G.nodes()) > 0:
                    # Create hierarchical layout
                    try:
                        pos = hierarchy_pos(G, cell_id)
                    except:
                        pos = nx.spring_layout(G, seed=42)
                    
                    # Draw nodes with color based on whether they divide
                    node_colors = ['red' if G.nodes[n]['divides'] else 'blue' 
                                for n in G.nodes()]
                    
                    # Size nodes based on track duration
                    node_sizes = [max(100, G.nodes[n]['duration'] * 10) 
                                for n in G.nodes()]
                    
                    # Draw the graph
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                        node_color=node_colors, alpha=0.8, ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color='gray',
                                        arrows=True, arrowstyle='-|>', ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                    
                    ax.set_title(f"Lineage Tree for Cell {cell_id}\n({len(G.nodes())} cells)")
                    
                    # Add statistics
                    division_count = sum(1 for n in G.nodes() if G.nodes[n]['divides'])
                    stats_text = f"Start Time: {G.nodes[cell_id]['start_time']}\n" \
                                f"Divisions: {division_count}\n" \
                                f"Generations: {nx.dag_longest_path_length(G)}"
                                
                    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.7))
                else:
                    ax.text(0.5, 0.5, f"No lineage data found for cell {cell_id}", 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        transform=ax.transAxes)
                
                ax.axis('off')
            
            canvas.draw()
        
        # Function to save visualization
        def save_visualization():
            output_path, _ = QFileDialog.getSaveFileName(
                dialog, "Save Lineage Tree", "", "PNG Files (*.png);;All Files (*)")
            
            if output_path:
                figure.savefig(output_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(
                    dialog, "Success", f"Lineage tree saved to {output_path}")
        
        # Connect buttons
        view_button.clicked.connect(create_visualization)
        save_button.clicked.connect(save_visualization)
        close_button.clicked.connect(dialog.close)
        
        # Create initial visualization
        create_visualization()
        
        # Show dialog
        dialog.exec_()

    def hierarchy_pos(G, root, width=1., vert_gap=0.1, vert_loc=0, xcenter=0.5):
        """
        Position nodes in a hierarchical layout.
        """
        def _hierarchy_pos(G, root, width=1., vert_gap=0.1, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not children:
                return pos
            dx = width / len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                    vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, 
                                    parent=root, parsed=parsed)
            return pos
        
        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    
    def visualize_lineage(self):
        """
        Visualize the lineage tree from tracking data, focusing on a single cell.
        """
        if not hasattr(self, "tracked_cells") or not self.tracked_cells:
            QMessageBox.warning(self, "Error", "No tracking data available. Run cell tracking first.")
            return
        
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            # Ask user if they want to pick a specific cell or have one selected automatically
            reply = QMessageBox.question(
                self, 
                "Lineage Visualization", 
                "Would you like to select a specific cell to view its lineage?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            
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
                
                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                layout.addWidget(buttons)
                
                if dialog.exec_() == QDialog.Accepted:
                    selected_cell = int(combo.currentText())
                else:
                    return
            else:
                # Pick the cell with the longest track
                longest_track = max(self.tracked_cells, key=lambda t: len(t['x']))
                selected_cell = longest_track['ID']
            
            # Create a graph to represent the lineage
            G = nx.DiGraph()
            
            # Add the selected cell as the root node
            track = next((t for t in self.tracked_cells if t['ID'] == selected_cell), None)
            if not track:
                QMessageBox.warning(self, "Error", f"Cell {selected_cell} not found.")
                return
                
            # Add metadata to the root node
            G.add_node(selected_cell, 
                    first_frame=track['t'][0] if track['t'] else 0,
                    frames=len(track['t']) if track['t'] else len(track['x']),
                    track_data=track)
            
            # Ask for save location
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Lineage Tree", "", "PNG Files (*.png);;All Files (*)"
            )
            
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
                parent_track = next((t for t in self.tracked_cells if t['ID'] == cell_id), None)
                if not parent_track or 't' not in parent_track or not parent_track['t']:
                    return
                    
                # Get the end frame for this track
                end_frame = parent_track['t'][-1]
                
                # Get potential children (tracks that start right after this one ends)
                children_candidates = []
                
                # Look at the next few frames for potential children
                for frame in range(end_frame, end_frame + 3):
                    if frame in start_frame_map:
                        # Get tracks that start at this frame
                        for child_track in start_frame_map[frame]:
                            # Skip if it's the parent itself
                            if child_track['ID'] == parent_track['ID']:
                                continue
                                
                            # Calculate proximity between end of parent and start of child
                            if len(parent_track['x']) > 0 and len(child_track['x']) > 0:
                                parent_end_x = parent_track['x'][-1]
                                parent_end_y = parent_track['y'][-1]
                                child_start_x = child_track['x'][0]
                                child_start_y = child_track['y'][0]
                                
                                # Calculate distance
                                distance = ((parent_end_x - child_start_x)**2 + 
                                            (parent_end_y - child_start_y)**2)**0.5
                                
                                # If close enough, consider it a potential child
                                if distance < 30:  # Adjust threshold as needed
                                    children_candidates.append((child_track['ID'], distance))
                
                # Sort by distance and take up to 2 closest as children (for division)
                children_candidates.sort(key=lambda x: x[1])
                children = [c[0] for c in children_candidates[:2]]
                
                # Add edges to the graph
                for child_id in children:
                    if child_id not in G:
                        # Get child track
                        child_track = next((t for t in self.tracked_cells if t['ID'] == child_id), None)
                        if child_track:
                            # Add child to graph
                            G.add_node(child_id,
                                    first_frame=child_track['t'][0] if child_track['t'] else 0,
                                    frames=len(child_track['t']) if child_track['t'] else len(child_track['x']),
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
                        path_length = len(nx.shortest_path(G, selected_cell, node)) - 1
                    except nx.NetworkXNoPath:
                        path_length = 0
                    
                    # Get all nodes at this depth
                    nodes_at_depth = [n for n in G.nodes() if 
                                    len(nx.shortest_path(G, selected_cell, n)) - 1 == path_length]
                    
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
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
            
            # Draw edges with arrows
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, 
                                arrows=True, arrowstyle='-|>', arrowsize=20)
            
            # Draw labels with track info
            labels = {n: f"ID: {n}\nFrames: {G.nodes[n]['frames']}" for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
            
            # Title and styling
            plt.title(f"Cell Lineage Tree (Root: Cell {selected_cell})")
            plt.grid(False)
            plt.axis('off')
            
            # Add stats
            node_count = G.number_of_nodes()
            edge_count = G.number_of_edges()
            stats_text = f"Total Cells: {node_count}\nDivision Events: {edge_count}\nRoot Cell: {selected_cell}"
            plt.figtext(0.02, 0.02, stats_text, bbox=dict(facecolor='white', alpha=0.8))
            
            # Save if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Saved lineage tree to {output_path}")
            
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
        
        
        
    def get_all_descendants(self, cell_id):
        """
        Recursively get all descendants of a cell.
        """
        descendants = []

        # Find the track for this cell
        parent_track = None
        for track in self.lineage_tracks:
            if track['ID'] == cell_id:
                parent_track = track
                break

        if parent_track and 'children' in parent_track and parent_track['children']:
            # Add immediate children
            descendants.extend(parent_track['children'])

            # Recursively add descendants of each child
            for child_id in parent_track['children']:
                descendants.extend(self.get_all_descendants(child_id))

        return descendants

    def visualize_focused_lineage_tree(self, root_cell_id, output_path=None, progress_callback=None):
        """
        Create a focused lineage tree visualization starting from a specific cell.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        if progress_callback:
            progress_callback(20)

        # Get the cell and all its descendants
        descendants = self.get_all_descendants(root_cell_id)

        # Add the root cell itself
        relevant_cell_ids = [root_cell_id] + descendants

        # Create the graph
        G = nx.DiGraph()

        # Add nodes and edges
        for track in self.lineage_tracks:
            if track['ID'] in relevant_cell_ids:
                # Add this node
                G.add_node(track['ID'],
                           start_time=track['t'][0] if track['t'] else 0,
                           track=track)

                # Add edges from parent to children
                if 'children' in track and track['children']:
                    for child_id in track['children']:
                        if child_id in relevant_cell_ids:
                            G.add_edge(track['ID'], child_id)

        if progress_callback:
            progress_callback(40)

        # Set up the plot
        plt.figure(figsize=(12, 10))

        # Use hierarchical layout for tree
        try:
            # First try a hierarchical layout (best for trees)
            pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        except:
            try:
                # Fallback to a different hierarchical layout
                pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                # Last resort - spring layout with time on y-axis
                pos = {}
                for node in G.nodes():
                    # Use random x position and time-based y position
                    time = G.nodes[node].get('start_time', 0)
                    import random
                    # Negative for top-down orientation
                    pos[node] = (random.random(), -time)

        if progress_callback:
            progress_callback(60)

        # Draw the nodes
        node_sizes = [300 for _ in G.nodes()]
        node_colors = ['skyblue' for _ in G.nodes()]

        # Highlight the root cell
        node_colors = ['skyblue' if node != root_cell_id else 'red'
                       for node in G.nodes()]

        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8
        )

        # Draw the edges with arrows
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=1.5,
            alpha=0.8,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15
        )

        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10)

        if progress_callback:
            progress_callback(80)

        # Add title and labels
        plt.title(f"Cell Lineage Tree (Root Cell: {root_cell_id})")
        plt.grid(True, linestyle='--', alpha=0.3)

        # Remove axis
        plt.axis('off')

        # Add info text
        info_text = f"Root Cell: {root_cell_id}\n"
        info_text += f"Total Descendants: {len(descendants)}\n"
        info_text += f"Division Events: {len([t for t in self.lineage_tracks if t['ID'] in relevant_cell_ids and t.get('children')])}"

        plt.figtext(0.02, 0.02, info_text,
                    bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Lineage tree saved to {output_path}")

        plt.tight_layout()
        plt.show()

        if progress_callback:
            progress_callback(100)

    def process_morphology_time_series(self):
        p = self.slider_p.value()
        c = self.slider_c.value() if "C" in self.dimensions else None  # Default C to None

        if not self.image_data.is_nd2:
            QMessageBox.warning(
                self, "Error", "This feature only supports ND2 datasets.")
            return

        try:
            # Extract image data for all time points
            t = self.dimensions["T"]  # Get the total number of time points
            self.image_data.seg_cache.with_model(self.model_dropdown.currentText())
            if "C" in self.dimensions:
                image_data = np.array(
                    self.image_data.data[0:t, p, c, :, :].compute()
                    if hasattr(self.image_data.data[0:t, p, c, :, :], "compute")
                    else self.image_data.data[0:t, p, c, :, :]
                )
            else:
                image_data = np.array(
                    self.image_data.data[0:t, p, :, :].compute()
                    if hasattr(self.image_data.data[0:t, p, :, :], "compute")
                    else self.image_data.data[0:t, p, :, :]
                )

            if image_data.size == 0:
                QMessageBox.warning(
                    self,
                    "Error",
                    "No valid data found for the selected position and channel.",
                )
                return
        except Exception as e:
            QMessageBox.warning(
                self, "Data Error", f"Failed to extract image data: {e}"
            )
            return

        num_frames = image_data.shape[0]
        self.progress_bar.setMaximum(num_frames)
        self.progress_bar.setValue(0)

        # Disable the button while the worker is running
        self.segment_button.setEnabled(False)

        # Create the worker and thread
        self.worker = MorphologyWorker(
            self.image_data, image_data, num_frames, p, c
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # Connect worker signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.handle_results)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.thread.quit)

        # Cleanup
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Re-enable button when thread finishes
        self.thread.finished.connect(
            lambda: self.segment_button.setEnabled(True))

        self.thread.start()

    
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
        morphology_fractions = {morphology: [] for morphology in all_morphologies}
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
                    class_counts = metrics_df["morphology_class"].value_counts().to_dict()
                    total_cells = len(metrics_df)
                    
                    # Print diagnostics for this frame
                    print(f"Frame {t}: Total cells = {total_cells}")
                    for morph_class, count in class_counts.items():
                        print(f"  {morph_class}: {count} cells ({count/total_cells*100:.1f}%)")
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

    def initViewArea(self):
        layout = QVBoxLayout(self.viewArea)

        self.image_label = QLabel()
        # Allow the label to scale the image
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)

        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(
            self.show_context_menu)

        annotate_button = QPushButton("Classify Cells")
        annotate_button.clicked.connect(self.annotate_cells)
        layout.addWidget(annotate_button)

        segment_button = QPushButton("Segment This Position")
        segment_button.clicked.connect(self.segment_this_p)
        layout.addWidget(segment_button)

        self.segmentation_progress_bar = QProgressBar()
        layout.addWidget(self.segmentation_progress_bar)

        # T controls
        t_layout = QHBoxLayout()
        t_label = QLabel("T: 0")
        t_layout.addWidget(t_label)
        self.t_left_button = QPushButton("<")
        self.t_left_button.clicked.connect(
            lambda: self.slider_t.setValue(self.slider_t.value() - 1)
        )
        t_layout.addWidget(self.t_left_button)

        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.valueChanged.connect(self.display_image)
        self.slider_t.valueChanged.connect(
            lambda value: t_label.setText(f"T: {value}"))

        t_layout.addWidget(self.slider_t)

        self.t_right_button = QPushButton(">")
        self.t_right_button.clicked.connect(
            lambda: self.slider_t.setValue(self.slider_t.value() + 1)
        )
        t_layout.addWidget(self.t_right_button)

        layout.addLayout(t_layout)

        # P controls
        p_layout = QHBoxLayout()
        p_label = QLabel("P: 0")
        p_layout.addWidget(p_label)
        self.p_left_button = QPushButton("<")
        self.p_left_button.clicked.connect(
            lambda: self.slider_p.setValue(self.slider_p.value() - 1)
        )
        p_layout.addWidget(self.p_left_button)

        self.slider_p = QSlider(Qt.Horizontal)
        self.slider_p.valueChanged.connect(self.display_image)
        self.slider_p.valueChanged.connect(
            lambda value: p_label.setText(f"P: {value}"))
        p_layout.addWidget(self.slider_p)

        self.p_right_button = QPushButton(">")
        self.p_right_button.clicked.connect(
            lambda: self.slider_p.setValue(self.slider_p.value() + 1)
        )
        p_layout.addWidget(self.p_right_button)

        layout.addLayout(p_layout)

        # C control (channel)
        c_layout = QHBoxLayout()
        c_label = QLabel("C: 0")
        c_layout.addWidget(c_label)
        self.c_left_button = QPushButton("<")
        self.c_left_button.clicked.connect(
            lambda: self.slider_c.setValue(self.slider_c.value() - 1)
        )
        c_layout.addWidget(self.c_left_button)

        self.slider_c = QSlider(Qt.Horizontal)
        self.slider_c.valueChanged.connect(self.display_image)
        self.slider_c.valueChanged.connect(
            lambda value: c_label.setText(f"C: {value}"))
        c_layout.addWidget(self.slider_c)

        self.c_right_button = QPushButton(">")
        self.c_right_button.clicked.connect(
            lambda: self.slider_c.setValue(self.slider_c.value() + 1)
        )
        c_layout.addWidget(self.c_right_button)

        layout.addLayout(c_layout)

        # Create a radio button for thresholding, normal and segmented
        self.radio_normal = QRadioButton("Normal")
        self.radio_thresholding = QRadioButton("Thresholding")
        self.radio_segmented = QRadioButton("Segmented")
        self.radio_overlay_outlines = QRadioButton("Overlay with Outlines")
        # Add new radio button for labeled segmentation
        self.radio_labeled_segmentation = QRadioButton("Labeled Segmentation")

        # Create a button group and add the radio buttons to it
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_normal)
        self.button_group.addButton(self.radio_thresholding)
        self.button_group.addButton(self.radio_labeled_segmentation)
        self.button_group.addButton(self.radio_segmented)
        self.button_group.addButton(self.radio_overlay_outlines)
        self.button_group.buttonClicked.connect(self.display_image)

        # Set default selection
        self.radio_normal.setChecked(True)

        # Add radio buttons to the layout
        layout.addWidget(self.radio_thresholding)
        layout.addWidget(self.radio_normal)
        layout.addWidget(self.radio_segmented)
        layout.addWidget(self.radio_overlay_outlines)
        layout.addWidget(self.radio_labeled_segmentation)

        # Threshold slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.valueChanged.connect(self.display_image)
        layout.addWidget(self.threshold_slider)

        # Segmentation model selection
        model_label = QLabel("Select Segmentation Model:")
        layout.addWidget(model_label)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(
            [
                SegmentationModels.CELLPOSE_BACT_PHASE,
                SegmentationModels.CELLPOSE_BACT_FLUOR,
                SegmentationModels.CELLPOSE,
                SegmentationModels.UNET,
                SegmentationModels.CELLSAM])
        self.model_dropdown.currentIndexChanged.connect(
            lambda: self.image_data.seg_cache.with_model(
                self.model_dropdown.currentText()))
        layout.addWidget(self.model_dropdown)

    def annotate_cells(self):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Ensure segmentation model is set correctly in seg_cache
        selected_model = self.model_dropdown.currentText()
        self.image_data.seg_cache.with_model(selected_model)

        # Retrieve segmentation from seg_cache
        segmented_image = self.image_data.seg_cache[t, p, c]

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
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

        self.update_annotation_scatter()

        print(f"[SUCCESS] Annotated image displayed for T={t}, P={p}, C={c}")

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

    # Initialize the Export tab with the export button
    def initExportTab(self):
        layout = QVBoxLayout(self.exportTab)
        export_button = QPushButton("Export Images")
        export_button.clicked.connect(self.export_images)
        layout.addWidget(export_button)
        label = QLabel("This Tab Exports processed images sequentially.")
        layout.addWidget(label)

    def save_video(self, file_path):
        # Assuming self.image_data is a 4D numpy array with shape (frames,
        # height, width, channels)
        if hasattr(self, "image_data"):
            print(self.image_data.data.shape)

            with iio.imopen(file_path, "w", plugin="pyav") as writer:
                writer.init_video_stream(
                    "libx264", fps=30, pixel_format="yuv444p")

                writer._video_stream.options = {
                    "preset": "veryslow",
                    "qp": "0",
                }  # 'crf': '0',

                writer.write(self.image_data.data)

            # iio.imwrite(file_path, self.image_data.data,
            #             # plugin="pyav",
            #             plugin="ffmpeg",
            #             fps=30,
            #             codec='libx264',
            #             output_params=['-crf', '0',
            #                             '-preset', 'veryslow',
            #                             '-qp', '0'],
            #             pixelformat='yuv444p')

    def initUI(self):
        # Initialize tabs as QWidget
        self.importTab = QWidget()
        self.exportTab = QWidget()
        self.populationTab = QWidget()
        self.morphologyTab = QWidget()
        self.morphologyTimeTab = QWidget()
        self.morphologyVisualizationTab = QWidget()

        # Add tabs to the QTabWidget
        self.tab_widget.addTab(self.importTab, "Import")
        self.tab_widget.addTab(self.exportTab, "Export")
        self.tab_widget.addTab(self.populationTab, "Population")
        self.tab_widget.addTab(self.morphologyTab, "Morphology")
        self.tab_widget.addTab(self.morphologyTimeTab, "Morphology / Time")
        self.tab_widget.addTab(
            self.morphologyVisualizationTab,
            "Morphology Visualization")

        # Initialize tab layouts and content
        self.initImportTab()
        self.initViewArea()
        self.initExportTab()
        self.initPopulationTab()
        self.initMorphologyTab()
        self.initMorphologyTimeTab()
        self.initMorphologyVisualizationTab()

    def initMorphologyTab(self):
        layout = QVBoxLayout(self.morphologyTab)

        # Create QTabWidget for inner tabs
        inner_tab_widget = QTabWidget()
        self.scatter_tab = QWidget()
        self.table_tab = QWidget()

        # Add tabs to the inner tab widget
        inner_tab_widget.addTab(self.scatter_tab, "PCA Plot")
        inner_tab_widget.addTab(self.table_tab, "Metrics Table")

        # Scatter plot tab layout (PCA)
        scatter_layout = QVBoxLayout(self.scatter_tab)

        # Annotated image display (adjusted size)
        self.annotated_image_label = QLabel(
            "Annotated image will be displayed here.")
        self.annotated_image_label.setFixedSize(
            300, 300)  # Adjust size as needed
        self.annotated_image_label.setAlignment(Qt.AlignCenter)
        self.annotated_image_label.setScaledContents(True)
        scatter_layout.addWidget(self.annotated_image_label)

        # Dropdown for selecting metric to color PCA scatter plot
        self.color_dropdown_annot = QComboBox()
        self.color_dropdown_annot.addItems(
            [
                "area",
                "perimeter",
                "aspect_ratio",
                "extent",
                "solidity",
                "equivalent_diameter",
                "orientation",
            ]
        )

        # Add dropdown for coloring
        # dropdown_layout = QHBoxLayout()
        # dropdown_layout.addWidget(QLabel("Color by:"))
        # dropdown_layout.addWidget(self.color_dropdown_annot)
        # scatter_layout.addLayout(dropdown_layout)

        # PCA scatter plot display
        self.figure_annot_scatter = plt.figure()
        self.canvas_annot_scatter = FigureCanvas(self.figure_annot_scatter)
        scatter_layout.addWidget(self.canvas_annot_scatter)

        # Connect dropdown change to PCA plot update
        self.color_dropdown_annot.currentTextChanged.connect(
            self.update_annotation_scatter)

        # Table tab layout (Metrics Table)
        table_layout = QVBoxLayout(self.table_tab)
        # Add the Export Button at the top of the table layout
        self.export_button = QPushButton("Export to CSV")
        self.export_button.setStyleSheet(
            "background-color: white; color: black; font-size: 14px;")
        table_layout.addWidget(self.export_button)

        # Connect the button to the export function (use annotation or define
        # it here)
        self.export_button.clicked.connect(self.export_metrics_to_csv)

        self.metrics_table = QTableWidget()  # Create the table widget
        # Connect the table item click signal to the handler
        self.metrics_table.itemClicked.connect(self.on_table_item_click)
        table_layout.addWidget(self.metrics_table)

        # Add the inner tab widget to the annotated tab layout
        layout.addWidget(inner_tab_widget)

    def initMorphologyVisualizationTab(self):
        layout = QVBoxLayout(self.morphologyVisualizationTab)

        # Button layout for actions
        button_layout = QHBoxLayout()

        # Add the ideal examples button
        ideal_button = QPushButton("Select Ideal Cells")
        ideal_button.clicked.connect(self.select_ideal_examples)
        button_layout.addWidget(ideal_button)

        # Add similarity analysis button
        similarity_button = QPushButton("Analyze Similarity to Ideals")
        similarity_button.clicked.connect(self.calculate_similarity_to_ideals)
        button_layout.addWidget(similarity_button)

        # Add optimization button
        optimize_button = QPushButton("Optimize Classification")
        optimize_button.clicked.connect(
            self.optimize_classification_parameters)
        button_layout.addWidget(optimize_button)

        layout.addLayout(button_layout)

        # Matplotlib canvas for plotting
        self.figure_morphology_metrics = plt.figure()
        self.canvas_morphology_metrics = FigureCanvas(
            self.figure_morphology_metrics)
        layout.addWidget(self.canvas_morphology_metrics)

    def select_ideal_examples(self):
        """
        Allow the user to select ideal examples of each morphology class.
        """
        # Initialize a dictionary to store ideal examples
        if not hasattr(self, "ideal_examples"):
            self.ideal_examples = {
                "Small": None,
                "Round": None,
                "Normal": None,
                "Elongated": None,
                "Deformed": None
            }

        # Create a dialog to select ideal cells
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Ideal Cell Examples")
        dialog.setMinimumWidth(600)

        layout = QVBoxLayout(dialog)

        # Instructions
        instructions = QLabel(
            "Select one ideal example for each morphology class.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Create a split layout
        main_layout = QHBoxLayout()

        # Form layout for selection
        form_layout = QFormLayout()
        selection_widget = QWidget()
        selection_widget.setLayout(form_layout)

        # Preview area
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_label = QLabel("Cell Preview")
        preview_layout.addWidget(preview_label)
        self.preview_image = QLabel()
        self.preview_image.setMinimumSize(200, 200)
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setScaledContents(True)
        preview_layout.addWidget(self.preview_image)

        main_layout.addWidget(selection_widget)
        main_layout.addWidget(preview_widget)
        layout.addLayout(main_layout)

        # Create dropdown selectors for each class
        self.ideal_selectors = {}

        # Get all cell IDs and their classifications
        cell_ids = []
        if hasattr(self, "cell_mapping") and self.cell_mapping:
            for cell_id, data in self.cell_mapping.items():
                if "metrics" in data and "morphology_class" in data["metrics"]:
                    cell_class = data["metrics"]["morphology_class"]
                    cell_ids.append((cell_id, cell_class))

        # Sort by class and ID
        cell_ids.sort(key=lambda x: (x[1], x[0]))

        # Create dropdowns for each class
        for class_name in self.ideal_examples.keys():
            combo = QComboBox()
            # Add all cells of this class
            class_cells = [str(cell_id) for cell_id,
                           cell_class in cell_ids if cell_class == class_name]

            if class_cells:
                combo.addItems(class_cells)

                # Set current selection if already defined
                if self.ideal_examples[class_name] is not None:
                    idx = combo.findText(str(self.ideal_examples[class_name]))
                    if idx >= 0:
                        combo.setCurrentIndex(idx)

                # Connect the currentIndexChanged signal
                combo.currentIndexChanged.connect(
                    lambda idx, cn=class_name: self.update_preview(cn))
            else:
                combo.addItem("No cells of this class")
                combo.setEnabled(False)

            self.ideal_selectors[class_name] = combo
            form_layout.addRow(f"Ideal {class_name}:", combo)

        # Buttons
        button_box = QHBoxLayout()
        save_button = QPushButton("Save Ideal Examples")
        save_button.clicked.connect(lambda: self.save_ideal_examples(dialog))
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)

        button_box.addWidget(save_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        # Initially update the preview for the first class with cells
        for class_name in self.ideal_examples.keys():
            if self.ideal_selectors[class_name].isEnabled():
                self.update_preview(class_name)
                break

        # Show the dialog
        dialog.exec_()

    def update_preview(self, class_name):
        """
        Update the preview image for the selected cell.
        """
        combo = self.ideal_selectors[class_name]
        if combo.isEnabled() and combo.currentText() != "No cells of this class":
            try:
                cell_id = int(combo.currentText())
                # Get the bounding box for this cell
                y1, x1, y2, x2 = self.cell_mapping[cell_id]["bbox"]

                # Get the current frame
                t = self.slider_t.value()
                p = self.slider_p.value()
                c = self.slider_c.value() if self.has_channels else None

                # Extract the cell region from the segmented image
                segmented_image = self.image_data.seg_cache[t, p, c]

                # Create a visualization focusing on just this cell
                # Make a local crop of the segmented image around the cell
                padding = 10  # Extra pixels around the bounding box
                y_min = max(0, y1 - padding)
                y_max = min(segmented_image.shape[0], y2 + padding)
                x_min = max(0, x1 - padding)
                x_max = min(segmented_image.shape[1], x2 + padding)

                # Crop the segmented region
                cropped_seg = segmented_image[y_min:y_max, x_min:x_max]

                # Convert to RGB and highlight the cell
                cropped_rgb = cv2.cvtColor((cropped_seg > 0).astype(
                    np.uint8) * 255, cv2.COLOR_GRAY2BGR)

                # Create a mask for the target cell
                cell_mask = np.zeros_like(cropped_seg)
                # Adjust bounding box coordinates for the crop
                local_y1, local_x1 = y1 - y_min, x1 - x_min
                local_y2, local_x2 = y2 - y_min, x2 - x_min

                # Use connected components to find the cell within the bounding box
                roi = cropped_seg[max(0, local_y1):min(cropped_seg.shape[0], local_y2),
                                  max(0, local_x1):min(cropped_seg.shape[1], local_x2)]

                if roi.max() > 0:
                    # Use connected components
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        roi, connectivity=8)

                    # Find largest component
                    largest_label = 1
                    largest_area = 0
                    for label in range(1, num_labels):
                        area = stats[label, cv2.CC_STAT_AREA]
                        if area > largest_area:
                            largest_area = area
                            largest_label = label

                    # Create mask for largest component
                    component_mask = (labels == largest_label).astype(
                        np.uint8) * 255

                    # Place it in the full mask at the right position
                    cell_mask[max(0, local_y1):min(cropped_seg.shape[0], local_y2),
                              max(0, local_x1):min(cropped_seg.shape[1], local_x2)] = component_mask

                # Highlight the cell in the appropriate morphology color
                # Default to red if color not found
                color = self.morphology_colors.get(class_name, (0, 0, 255))
                cropped_rgb[cell_mask > 0] = color

                # Draw bounding box
                cv2.rectangle(cropped_rgb,
                              (max(0, local_x1), max(0, local_y1)),
                              (min(cropped_seg.shape[1]-1, local_x2),
                               min(cropped_seg.shape[0]-1, local_y2)),
                              (255, 0, 0), 1)

                # Add text
                cv2.putText(cropped_rgb, f"ID: {cell_id}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Convert to QImage and display in preview
                height, width = cropped_rgb.shape[:2]
                bytes_per_line = 3 * width
                qimage = QImage(cropped_rgb.data, width, height,
                                bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.preview_image.setPixmap(pixmap)

            except Exception as e:
                print(f"Error updating preview: {e}")
                self.preview_image.clear()
                self.preview_image.setText("Preview not available")

    def save_ideal_examples(self, dialog):
        """
        Save the selected ideal examples.
        """
        # Update the ideal examples dictionary
        for class_name, combo in self.ideal_selectors.items():
            if combo.isEnabled() and combo.currentText() != "No cells of this class":
                self.ideal_examples[class_name] = int(combo.currentText())

        # Store the metrics for each ideal example
        self.ideal_metrics = {}
        for class_name, cell_id in self.ideal_examples.items():
            if cell_id is not None:
                self.ideal_metrics[class_name] = self.cell_mapping[cell_id]["metrics"]

        # Print the ideal metrics for debugging
        print("Ideal Metrics:")
        for class_name, metrics in self.ideal_metrics.items():
            print(f"{class_name}: {metrics}")

        # Close the dialog
        QMessageBox.information(
            dialog, "Success", "Ideal examples saved successfully.")
        dialog.accept()

    def calculate_similarity_to_ideals(self):
        """
        Calculate how similar each cell is to the ideal examples of each class.
        """
        if not hasattr(self, "ideal_metrics") or not self.ideal_metrics:
            QMessageBox.warning(
                self, "Error", "No ideal metrics defined. Please select ideal examples first.")
            return

        # Get all cells from current frame
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Make sure cell mapping exists
        if not hasattr(self, "cell_mapping") or not self.cell_mapping:
            QMessageBox.warning(
                self, "Error", "No cell data available. Please classify cells first.")
            return

        # Metrics to compare (exclude morphology_class)
        metrics_to_compare = ["area", "perimeter", "equivalent_diameter",
                              "orientation", "aspect_ratio", "circularity", "solidity"]

        # Store results
        similarity_results = []

        # For each cell, calculate similarity to each ideal
        for cell_id, cell_data in self.cell_mapping.items():
            cell_metrics = cell_data["metrics"]
            current_class = cell_metrics.get("morphology_class", "Unknown")

            cell_result = {
                "cell_id": cell_id,
                "current_class": current_class
            }

            # Calculate similarity to each ideal
            best_similarity = 0
            best_class = None

            for class_name, ideal in self.ideal_metrics.items():
                if not ideal:  # Skip if no ideal defined for this class
                    continue

                # Calculate Euclidean distance in feature space
                # First normalize each feature to prevent any single feature from dominating
                squared_diff_sum = 0
                valid_metrics = 0

                for metric in metrics_to_compare:
                    if metric in cell_metrics and metric in ideal:
                        # Retrieve values
                        cell_value = cell_metrics[metric]
                        ideal_value = ideal[metric]

                        # Skip if either value is None
                        if cell_value is None or ideal_value is None:
                            continue

                        # Normalize based on ideal value to get relative difference
                        if ideal_value != 0:
                            normalized_diff = (
                                cell_value - ideal_value) / ideal_value
                            squared_diff_sum += normalized_diff ** 2
                            valid_metrics += 1

                # Calculate similarity (invert distance to get similarity)
                if valid_metrics > 0:
                    distance = (squared_diff_sum / valid_metrics) ** 0.5
                    # Convert to similarity (0-1 scale)
                    similarity = 1 / (1 + distance)

                    cell_result[f"similarity_{class_name}"] = similarity

                    # Keep track of best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_class = class_name

            # Record best match
            cell_result["best_match_class"] = best_class
            cell_result["best_match_similarity"] = best_similarity

            # Calculate if classification matches best similarity
            cell_result["matches_best"] = (current_class == best_class)

            similarity_results.append(cell_result)

        # Convert to DataFrame for easier analysis
        import pandas as pd
        self.similarity_df = pd.DataFrame(similarity_results)

        # Calculate overall statistics
        total_cells = len(similarity_results)
        matching_cells = sum(
            1 for result in similarity_results if result["matches_best"])
        match_percentage = (matching_cells / total_cells *
                            100) if total_cells > 0 else 0

        # Display results
        self.display_similarity_results(match_percentage)

        return similarity_results

    def display_similarity_results(self, match_percentage):
        """
        Display the similarity analysis results.
        """
        # Clear the existing figure
        self.figure_morphology_metrics.clear()

        # Create a figure with subplots
        gridspec = self.figure_morphology_metrics.add_gridspec(2, 2)

        # 1. Pie chart of current classifications
        ax1 = self.figure_morphology_metrics.add_subplot(gridspec[0, 0])
        class_counts = self.similarity_df["current_class"].value_counts()
        ax1.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
        ax1.set_title("Current Classification Distribution")

        # 2. Pie chart of best match classifications
        ax2 = self.figure_morphology_metrics.add_subplot(gridspec[0, 1])
        best_counts = self.similarity_df["best_match_class"].value_counts()
        ax2.pie(best_counts, labels=best_counts.index, autopct='%1.1f%%')
        ax2.set_title("Best Match Classification Distribution")

        # 3. Bar chart of match percentage by class
        ax3 = self.figure_morphology_metrics.add_subplot(gridspec[1, 0])
        match_by_class = self.similarity_df.groupby(
            "current_class")["matches_best"].mean() * 100
        match_by_class.plot(kind='bar', ax=ax3)
        ax3.set_title("Match Percentage by Class")
        ax3.set_ylabel("Match Percentage (%)")
        ax3.set_ylim(0, 100)

        # 4. Text summary
        ax4 = self.figure_morphology_metrics.add_subplot(gridspec[1, 1])
        ax4.axis('off')
        summary_text = f"""
        Classification Analysis Summary:
        
        Total Cells: {len(self.similarity_df)}
        Matching Current to Best: {match_percentage:.1f}%
        
        Ideal Examples Used:
        """

        for class_name, cell_id in self.ideal_examples.items():
            if cell_id is not None:
                summary_text += f"\n{class_name}: Cell #{cell_id}"

        ax4.text(0, 0.5, summary_text, va='center')

        # Adjust layout and draw
        self.figure_morphology_metrics.tight_layout()
        self.canvas_morphology_metrics.draw()

        # Show result in a message box
        QMessageBox.information(
            self,
            "Similarity Analysis",
            f"Classification to Ideal Similarity Analysis Complete\n\n"
            f"Overall match percentage: {match_percentage:.1f}%\n\n"
            f"This indicates how well your current classification matches the 'ideal' examples."
        )

    def optimize_classification_parameters(self):
        """
        Optimize classification thresholds to maximize similarity to ideal examples.
        """
        if not hasattr(self, "ideal_metrics") or not self.ideal_metrics:
            QMessageBox.warning(
                self, "Error", "No ideal metrics defined. Please select ideal examples first.")
            return

        # Debug: Print original parameters
        print("\n=== ORIGINAL DEFAULT PARAMETERS ===")
        default_params = {
            # Small cell parameters
            "small_max_area": 1000,
            "small_max_perimeter": 150,
            "small_max_aspect_ratio": 3.0,

            # Round cell parameters
            "round_min_circularity": 0.8,
            "round_max_aspect_ratio": 1.5,
            "round_min_solidity": 0.9,

            # Normal cell parameters
            "normal_min_circularity": 0.6,
            "normal_max_circularity": 0.8,
            "normal_min_aspect_ratio": 1.5,
            "normal_max_aspect_ratio": 3.0,
            "normal_min_solidity": 0.85,

            # Elongated cell parameters
            "elongated_min_area": 3000,
            "elongated_min_aspect_ratio": 5.0,
            "elongated_max_circularity": 0.4,

            # Deformed cell parameters (these are inverse of normal)
            "deformed_max_circularity": 0.602,
            "deformed_max_solidity": 0.731
        }

        for key, value in default_params.items():
            print(f"{key}: {value}")

        # Get current classification distribution
        self.original_classification = {}
        for cell_id, cell_data in self.cell_mapping.items():
            if "metrics" in cell_data and "morphology_class" in cell_data["metrics"]:
                self.original_classification[cell_id] = cell_data["metrics"]["morphology_class"]

        # Debug: Print current classification distribution
        class_counts = {}
        for cls in self.original_classification.values():
            class_counts[cls] = class_counts.get(cls, 0) + 1
        print("\n=== CURRENT CLASSIFICATION DISTRIBUTION ===")
        for cls, count in class_counts.items():
            print(f"{cls}: {count} cells")

        # Make sure cell mapping exists
        if not hasattr(self, "cell_mapping") or not self.cell_mapping:
            QMessageBox.warning(
                self, "Error", "No cell data available. Please classify cells first.")
            return

        # Create progress dialog
        progress = QProgressDialog(
            "Optimizing classification parameters...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # Number of random parameter sets to try
        num_trials = 100

        # Parameter ranges to search
        param_ranges = {
            # Small cell parameters
            "small_max_area": (500, 2000),
            "small_max_perimeter": (50, 300),
            "small_max_aspect_ratio": (2.0, 4.0),

            # Round cell parameters
            "round_min_circularity": (0.7, 0.9),
            "round_max_aspect_ratio": (1.1, 2.0),
            "round_min_solidity": (0.8, 1.0),

            # Normal cell parameters
            "normal_min_circularity": (0.4, 0.7),
            "normal_max_circularity": (0.7, 0.9),
            "normal_min_aspect_ratio": (1.2, 2.0),
            "normal_max_aspect_ratio": (2.0, 4.0),
            "normal_min_solidity": (0.8, 0.95),

            # Elongated cell parameters
            "elongated_min_area": (2000, 4000),
            "elongated_min_aspect_ratio": (4.0, 7.0),
            "elongated_max_circularity": (0.2, 0.5),

            # Deformed cell parameters
            "deformed_max_circularity": (0.4, 0.7),
            "deformed_max_solidity": (0.7, 0.9)
        }

        best_match_percentage = 0
        best_parameters = None
        best_parameter_trial = -1

        import random
        import numpy as np

        # Try multiple random parameter sets
        for trial in range(num_trials):
            # Update progress
            progress.setValue(int((trial / num_trials) * 100))
            if progress.wasCanceled():
                break

            # Generate random parameters within ranges
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                params[param_name] = random.uniform(min_val, max_val)

            # Make sure max > min for paired parameters
            if params["normal_min_circularity"] > params["normal_max_circularity"]:
                params["normal_min_circularity"], params["normal_max_circularity"] = \
                    params["normal_max_circularity"], params["normal_min_circularity"]

            if params["normal_min_aspect_ratio"] > params["normal_max_aspect_ratio"]:
                params["normal_min_aspect_ratio"], params["normal_max_aspect_ratio"] = \
                    params["normal_max_aspect_ratio"], params["normal_min_aspect_ratio"]

            # Debug for first few trials
            if trial < 3:
                print(f"\n=== TRIAL {trial+1} PARAMETERS ===")
                for key, value in params.items():
                    print(f"{key}: {value:.2f}")

            # Test parameters on all cells
            matches = 0
            total_cells = 0
            classification_counts = {"Small": 0, "Round": 0,
                                     "Normal": 0, "Elongated": 0, "Deformed": 0}

            for cell_id, cell_data in self.cell_mapping.items():
                total_cells += 1

                # Get cell metrics
                cell_metrics = cell_data["metrics"]

                # Find which ideal example this cell is closest to
                best_match_class = None
                best_match_similarity = 0

                for class_name, ideal_metrics in self.ideal_metrics.items():
                    if not ideal_metrics:  # Skip if no ideal for this class
                        continue

                    # Calculate similarity to this ideal
                    metrics_to_compare = ["area", "perimeter", "equivalent_diameter",
                                          "aspect_ratio", "circularity", "solidity"]

                    squared_diff_sum = 0
                    valid_metrics = 0

                    for metric in metrics_to_compare:
                        if metric in cell_metrics and metric in ideal_metrics:
                            cell_value = cell_metrics[metric]
                            ideal_value = ideal_metrics[metric]

                            if cell_value is not None and ideal_value is not None and ideal_value != 0:
                                normalized_diff = (
                                    cell_value - ideal_value) / ideal_value
                                squared_diff_sum += normalized_diff ** 2
                                valid_metrics += 1

                    if valid_metrics > 0:
                        distance = (squared_diff_sum / valid_metrics) ** 0.5
                        similarity = 1 / (1 + distance)

                        if similarity > best_match_similarity:
                            best_match_similarity = similarity
                            best_match_class = class_name

                # Classify using current parameter set
                classification = classify_morphology(cell_metrics, params)
                classification_counts[classification] = classification_counts.get(
                    classification, 0) + 1

                # Check if classification matches best similarity
                if classification == best_match_class:
                    matches += 1

            # Calculate match percentage
            match_percentage = (matches / total_cells *
                                100) if total_cells > 0 else 0

            # Debug for a few trials
            if trial < 3 or match_percentage > best_match_percentage:
                print(f"\nTrial {trial+1} results:")
                print(f"Match percentage: {match_percentage:.2f}%")
                print("Classification distribution:")
                for cls, count in classification_counts.items():
                    print(f"  {cls}: {count} cells")

            # Update best if this is better
            if match_percentage > best_match_percentage:
                best_match_percentage = match_percentage
                best_parameters = params.copy()
                best_parameter_trial = trial + 1

        # Close progress dialog
        progress.setValue(100)

        # Debug: Print the best parameters
        print("\n=== BEST PARAMETERS (Trial #{}) ===".format(best_parameter_trial))
        if best_parameters:
            for key, value in best_parameters.items():
                print(f"{key}: {value:.3f}")

        # Store best parameters
        self.best_classification_parameters = best_parameters

        # Simulate reclassification to see differences
        new_classifications = {}
        for cell_id, cell_data in self.cell_mapping.items():
            cell_metrics = cell_data["metrics"]
            new_class = classify_morphology(cell_metrics, best_parameters)
            new_classifications[cell_id] = new_class

        # Compare original vs new classifications
        changes = 0
        change_matrix = {}  # From -> To counts
        for cell_id in self.original_classification:
            original = self.original_classification[cell_id]
            new = new_classifications[cell_id]

            if original != new:
                changes += 1
                key = f"{original} -> {new}"
                change_matrix[key] = change_matrix.get(key, 0) + 1

        print(f"\n=== CLASSIFICATION CHANGES ===")
        print(
            f"Total cells that would change: {changes} out of {len(self.original_classification)}")
        print("Change details:")
        for change, count in change_matrix.items():
            print(f"  {change}: {count} cells")

        # Display results with detailed information
        result_message = (
            f"Optimization complete!\n\n"
            f"Best match percentage: {best_match_percentage:.1f}%\n\n"
            f"Changes if applied: {changes} of {len(self.original_classification)} cells\n"
        )

        # Add detail on most significant changes
        if changes > 0:
            result_message += "\nSignificant changes:\n"
            for change, count in sorted(change_matrix.items(), key=lambda x: x[1], reverse=True)[:3]:
                result_message += f"• {change}: {count} cells\n"

        result_message += "\nWould you like to apply these optimized parameters?"

        reply = QMessageBox.question(self, "Optimization Results", result_message,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            # Apply the optimized parameters and reclassify cells
            self.apply_optimized_parameters(new_classifications)

        # Return best parameters
        return best_parameters, best_match_percentage

    def apply_optimized_parameters(self, predicted_classifications=None):
        """
        Apply the optimized classification parameters to all cells.

        Args:
            predicted_classifications: Pre-calculated new classifications (optional)
        """
        if not hasattr(self, "best_classification_parameters"):
            QMessageBox.warning(
                self, "Error", "No optimized parameters available.")
            return

        # For tracking changes
        change_count = 0
        old_class_count = {"Small": 0, "Round": 0,
                           "Normal": 0, "Elongated": 0, "Deformed": 0}
        new_class_count = {"Small": 0, "Round": 0,
                           "Normal": 0, "Elongated": 0, "Deformed": 0}

        # Reclassify all cells using optimized parameters
        for cell_id, cell_data in self.cell_mapping.items():
            try:
                # Get current class
                old_class = cell_data["metrics"].get(
                    "morphology_class", "Unknown")
                old_class_count[old_class] = old_class_count.get(
                    old_class, 0) + 1

                # Get or calculate new class
                if predicted_classifications and cell_id in predicted_classifications:
                    new_class = predicted_classifications[cell_id]
                else:
                    cell_metrics = cell_data["metrics"]
                    new_class = classify_morphology(
                        cell_metrics, self.best_classification_parameters)

                # Count change if different
                if old_class != new_class:
                    change_count += 1
                    print(f"Cell {cell_id}: {old_class} -> {new_class}")

                # Update classification
                cell_data["metrics"]["morphology_class"] = new_class
                new_class_count[new_class] = new_class_count.get(
                    new_class, 0) + 1

            except Exception as e:
                print(f"Error reclassifying cell {cell_id}: {e}")

        # Print summary of changes
        print("\n=== CLASSIFICATION CHANGES APPLIED ===")
        print(
            f"Changed classification for {change_count} cells out of {len(self.cell_mapping)}")
        print("\nBefore counts:")
        for cls, count in old_class_count.items():
            print(f"  {cls}: {count}")
        print("\nAfter counts:")
        for cls, count in new_class_count.items():
            print(f"  {cls}: {count}")

        # Update the metrics table if it's visible
        if hasattr(self, "populate_metrics_table"):
            print("Updating metrics table...")
            self.populate_metrics_table()

        # Update any visualization
        if hasattr(self, "update_annotation_scatter"):
            print("Updating annotation scatter...")
            self.update_annotation_scatter()

        # Display a detailed before/after comparison
        self.display_classification_results(old_class_count, new_class_count)

        QMessageBox.information(self, "Reclassification Complete",
                                f"All cells have been reclassified using optimized parameters.\n\n"
                                f"Changed classification for {change_count} cells out of {len(self.cell_mapping)}.")

    def display_classification_results(self, before_counts, after_counts):
        """
        Display a chart comparing classification before and after optimization.

        Args:
            before_counts: Dictionary with counts before optimization
            after_counts: Dictionary with counts after optimization
        """
        self.figure_morphology_metrics.clear()
        ax = self.figure_morphology_metrics.add_subplot(111)

        # Get all class labels
        all_classes = sorted(
            set(list(before_counts.keys()) + list(after_counts.keys())))

        # Set up for bar chart
        x = range(len(all_classes))
        width = 0.35

        # Create the bars
        before_values = [before_counts.get(cls, 0) for cls in all_classes]
        after_values = [after_counts.get(cls, 0) for cls in all_classes]

        # Plot the bars
        before_bars = ax.bar([i - width/2 for i in x],
                             before_values, width, label='Before Optimization')
        after_bars = ax.bar([i + width/2 for i in x],
                            after_values, width, label='After Optimization')

        # Add labels and titles
        ax.set_xlabel('Morphology Class')
        ax.set_ylabel('Cell Count')
        ax.set_title('Cell Classification Before vs After Optimization')
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes)
        ax.legend()

        # Add value labels on the bars
        for i, v in enumerate(before_values):
            ax.text(i - width/2, v + 0.5, str(v), ha='center')

        for i, v in enumerate(after_values):
            ax.text(i + width/2, v + 0.5, str(v), ha='center')

        # Highlight differences
        for i, (before, after) in enumerate(zip(before_values, after_values)):
            if before != after:
                diff = after - before
                color = 'green' if diff > 0 else 'red'
                ax.text(i, max(before, after) + 5, f"{'+' if diff > 0 else ''}{diff}",
                        ha='center', color=color, fontweight='bold')

        # Draw the plot
        self.canvas_morphology_metrics.draw()

    def export_metrics_to_csv(self):
        """
        Exports the metrics table data to a CSV file.
        """
        try:
            if not self.cell_mapping:
                QMessageBox.warning(self, "Error", "No cell data available.")
                return

            metrics_data = [
                {**{"ID": cell_id}, **data["metrics"]}
                for cell_id, data in self.cell_mapping.items()
            ]
            metrics_df = pd.DataFrame(metrics_data)

            if metrics_df.empty:
                QMessageBox.warning(
                    self, "Error", "No data available to export.")
                return

            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Metrics Data", "", "CSV Files (*.csv);;All Files (*)")
            if save_path:
                metrics_df.to_csv(save_path, index=False)
                QMessageBox.information(
                    self, "Success", f"Metrics data exported to {save_path}"
                )
            else:
                QMessageBox.warning(self, "Cancelled", "Export cancelled.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def update_annotation_scatter(self):
        try:
            # Extract current frame and segmentation
            t = self.slider_t.value()
            p = self.slider_p.value()
            c = self.slider_c.value() if self.has_channels else None
            frame = self.get_current_frame(t, p, c)

            segmented_image = self.image_data.seg_cache[t, p, c]

            # Extract cell metrics
            self.cell_mapping = extract_cells_and_metrics(
                frame, segmented_image)
            self.populate_metrics_table()

            # Prepare DataFrame
            metrics_data = [
                {**{"ID": cell_id}, **data["metrics"], **
                    {"Class": data["metrics"]["morphology_class"]}}
                for cell_id, data in self.cell_mapping.items()
            ]
            morphology_df = pd.DataFrame(metrics_data)

            # Select numeric features for PCA
            numeric_features = [
                'area',
                'perimeter',
                'equivalent_diameter',
                'orientation',
                'aspect_ratio',
                'circularity',
                'solidity']
            X = morphology_df[numeric_features].values

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)

            # Store PCA results
            pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
            pca_df['Class'] = morphology_df['Class']
            pca_df['ID'] = morphology_df['ID']

            # Plot PCA scatter
            self.figure_annot_scatter.clear()
            ax = self.figure_annot_scatter.add_subplot(111)
            scatter = ax.scatter(
                pca_df['PC1'],
                pca_df['PC2'],
                c=[
                    self.morphology_colors_rgb[class_] for class_ in pca_df['Class']],
                s=50,
                edgecolor='w',
                picker=True)

            ax.set_title("PCA Scatter Plot")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

            # Enable interactive annotations and highlighting
            self.annotate_scatter_points(ax, scatter, pca_df)

            self.canvas_annot_scatter.draw()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")

    def segment_this_p(self):
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Create list to store segmented results
        segmented_results = []
        total_frames = self.image_data.data.shape[0]

        for t in tqdm(range(total_frames), desc="Segmenting frames"):
            # cache_key = (t, p, c)

            # # Use cached segmentation if available
            # if cache_key in self.image_data.segmentation_cache:
            #     print(
            #         f"[CACHE HIT] Using cached segmentation for T={t}, P={p}, C={c}")
            #     segmented = self.image_data.segmentation_cache[cache_key]
            # else:
            #     print(f"[CACHE MISS] Segmenting T={t}, P={p}, C={c}")

            #     frame = self.image_data.data[t, p, c]

            #     # Determine model type ONLY if Cellpose is selected
            #     model_type = None
            #     if self.model_dropdown.currentText() == SegmentationModels.CELLPOSE:
            #         model_type = 'bact_phase_cp3' if c in (
            #             0, None) else 'bact_fluor_cp3'

            #     # Perform segmentation
            #     segmented = SegmentationModels().segment_images(np.array(
            #         [frame]), self.model_dropdown.currentText(), model_type=model_type)[0]

            #     # Rescale the image to the base resolution. Reverse because
            #     # it's opencv
            #     base_resolution = (
            #         self.image_data.data.shape[4],
            #         self.image_data.data.shape[3])
            #     segmented = cv2.resize(
            #         segmented,
            #         base_resolution,
            #         interpolation=cv2.INTER_NEAREST)

            #     # Store result in cache
            #     self.image_data.segmentation_cache[cache_key] = segmented

            self.image_data.seg_cache.with_model(
                self.model_dropdown.currentText())  # Setting the model we want
            segmented = self.image_data.seg_cache[t, p, c]

            # Label segmented objects (Assign unique label to each object)
            labeled_cells = label(segmented)

            # Visualize labeled segmentation
            plt.figure(figsize=(5, 5))
            # Color-coded labels
            plt.imshow(labeled_cells, cmap='nipy_spectral')
            plt.title(f'Labeled Segmentation - Frame {t}')
            plt.axis('off')
            # plt.show()

            segmented_results.append(labeled_cells)

        # Convert list to numpy array
        self.segmented_time_series = np.array(segmented_results)
        # Set to false when new segmentation is needed
        self.is_time_series_segmented = True

        QMessageBox.information(
            self, "Segmentation Complete",
            f"Segmentation for all time points is complete. Shape: {self.segmented_time_series.shape}"
        )

    def get_current_frame(self, t, p, c=None):
        """
        Retrieve the current frame based on slider values for time, position, and channel.
        """
        if self.image_data.is_nd2:
            if self.has_channels:
                frame = self.image_data.data[t, p, c]
            else:
                frame = self.image_data.data[t, p]
        else:
            frame = self.image_data.data[t]

        # Convert to NumPy array if needed
        return np.array(frame)

    def annotate_scatter_points(self, ax, scatter, pca_df):
        """
        Adds interactive hover annotations and click event to highlight a selected cell.

        Parameters:
        ax : matplotlib.axes.Axes
            The axes object for the scatter plot.
        scatter : matplotlib.collections.PathCollection
            The scatter plot object.
        pca_df : pd.DataFrame
            DataFrame containing PCA results with cell IDs and classes.
        """
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            ha="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)

        def update_annot(ind):
            """Update annotation text and position based on hovered point."""
            index = ind["ind"][0]
            pos = scatter.get_offsets()[index]
            annot.xy = pos
            selected_id = int(pca_df.iloc[index]["ID"])
            cell_class = pca_df.iloc[index]["Class"]
            annot.set_text(f"ID: {selected_id}\nClass: {cell_class}")
            annot.get_bbox_patch().set_alpha(0.8)

        def on_hover(event):
            """Handle hover events to show annotations."""
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    self.canvas_annot_scatter.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        self.canvas_annot_scatter.draw_idle()

        def on_click(event):
            """Handle click events to highlight the selected cell in the segmented image."""
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    index = ind["ind"][0]
                    selected_id = int(pca_df.iloc[index]["ID"])
                    cell_class = pca_df.iloc[index]["Class"]
                    self.highlight_cell_in_image(selected_id)

        self.canvas_annot_scatter.mpl_connect("motion_notify_event", on_hover)
        self.canvas_annot_scatter.mpl_connect("button_press_event", on_click)

        # Add legend using self.morphology_colors_rgb
        handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                color=color,
                label=key,
                markersize=8,
                linestyle='',
            )
            for key, color in self.morphology_colors_rgb.items()
        ]
        ax.legend(
            handles=handles,
            title="Morphology Class",
            loc='best',
        )

    def on_table_item_click(self, item):
        row = item.row()

        cell_id = self.metrics_table.item(row, 0).text()

        # print(f"Row {row} clicked, Cell ID: {cell_id}")

        self.highlight_cell_in_image(cell_id)

    def highlight_cell_in_image(self, cell_id):
        # print(f"🔍 Highlighting cell with ID: {cell_id}")

        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Get the binary segmentation
        segmented_image = self.image_data.seg_cache[t, p, c]

        if segmented_image is None:
            QMessageBox.warning(self, "Error", "Segmented image not found.")
            return

        # Debug info
        unique_labels = np.unique(segmented_image)
        # print(f"🔍 Unique labels in segmented image: {unique_labels}")

        # Ensure cell ID is an integer
        cell_id = int(cell_id)

        # Ensure stored cell mappings exist
        if not hasattr(self, "cell_mapping") or not self.cell_mapping:
            QMessageBox.warning(
                self, "Error", "No stored cell mappings found. Did you classify cells first?")
            return

        available_ids = list(map(int, self.cell_mapping.keys()))
        # print(f"📝 Available Segmentation Cell IDs: {available_ids}")

        if cell_id not in available_ids:
            QMessageBox.warning(
                self, "Error", f"Cell ID {cell_id} not found in segmentation. Available IDs: {available_ids}")
            return

        # Get the bounding box coordinates for the selected cell
        y1, x1, y2, x2 = self.cell_mapping[cell_id]["bbox"]

        # Create a visualization of the segmented image
        # Convert binary segmentation to RGB for visualization
        segmented_rgb = cv2.cvtColor((segmented_image > 0).astype(
            np.uint8) * 255, cv2.COLOR_GRAY2BGR)

        # Create a mask for just this cell based on the bounding box
        cell_mask = np.zeros_like(segmented_image, dtype=np.uint8)

        # Extract the region of interest from the segmentation
        roi = segmented_image[y1:y2, x1:x2]

        # If there are cells in the ROI, isolate the main one
        if roi.max() > 0:
            # Use connected components to find distinct objects in the ROI
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                roi, connectivity=8)

            # Find the largest component (excluding background)
            largest_label = 1  # Default to first label
            largest_area = 0

            for label in range(1, num_labels):  # Skip background (0)
                area = stats[label, cv2.CC_STAT_AREA]
                if area > largest_area:
                    largest_area = area
                    largest_label = label

            # Create mask for the largest component
            roi_mask = (labels == largest_label).astype(np.uint8) * 255

            # Place the ROI mask back in the full image mask
            cell_mask[y1:y2, x1:x2] = roi_mask

        # Highlight the cell in red on the segmented image
        segmented_rgb[cell_mask > 0] = [0, 0, 255]  # BGR format - Red

        # Also draw the bounding box in blue
        cv2.rectangle(segmented_rgb, (x1, y1), (x2, y2),
                      (255, 0, 0), 1)  # Blue rectangle

        # Add cell ID text
        cv2.putText(segmented_rgb, str(cell_id), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Green text

        # Convert to QImage and display
        height, width = segmented_rgb.shape[:2]
        bytes_per_line = 3 * width

        qimage = QImage(segmented_rgb.data, width, height,
                        bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        # print(
        #     f"✅ Successfully highlighted cell {cell_id} at bounding box {(y1, x1, y2, x2)}")

    def highlight_selected_cell(self, cell_id, cache_key):
        """
        Highlights a selected cell on the segmented image when a point on the scatter plot is clicked.

        Parameters:
        -----------
        cell_id : int
            ID of the cell to highlight.
        cache_key : tuple
            Key to retrieve cached segmentation and cell mapping.
        """
        # Ensure cell mapping exists
        if not hasattr(self, "cell_mapping") or not self.cell_mapping:
            QMessageBox.warning(self, "Error", "Cell mapping not initialized.")
            return

        # Retrieve bounding box for the selected cell
        if cell_id not in self.cell_mapping:
            QMessageBox.warning(self, "Error", f"Cell ID {cell_id} not found.")
            return

        bbox = self.cell_mapping[cell_id]["bbox"]
        y1, x1, y2, x2 = bbox  # Correct order (y1, x1, y2, x2)

        # Create a copy of the annotated image to avoid overwriting
        highlighted_image = self.annotated_image.copy()
        cv2.rectangle(highlighted_image, (x1, y1), (x2, y2),
                      (255, 0, 0), 2)  # Highlight with red box

        # Display the highlighted image in the scatter plot tab
        height, width = highlighted_image.shape[:2]
        qimage = QImage(
            highlighted_image.data,
            width,
            height,
            highlighted_image.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.annotated_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.annotated_image_label.setPixmap(pixmap)

    def generate_morphology_data(self):
        # Generate morphological data for the annotated tab
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Get segmented image
        segmented_image = self.get_segmented_data(t, p, c)

        # Extract morphology
        self.morphology_data = extract_cell_morphologies(segmented_image)

        # Automatically plot default metrics
        self.update_annotation_scatter()

    # TODO: remove
    def generate_annotations_and_scatter(self):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Extract the current frame
        image_data = self.image_data.data
        if self.image_data.is_nd2:
            if self.has_channels:
                image_data = image_data[t, p, c]
            else:
                image_data = image_data[t, p]
        else:
            image_data = image_data[t]

        # Ensure image_data is a numpy array
        image_data = np.array(image_data)

        # Perform segmentation
        segmented_image = segment_this_image(image_data)

        # Extract cells and their metrics
        self.cell_mapping = extract_cells_and_metrics(
            image_data, segmented_image)

        if not self.cell_mapping:
            QMessageBox.warning(
                self, "No Cells", "No cells detected in the current frame."
            )
            return

        # Populate the metrics table
        self.populate_metrics_table()

        # Annotate the original image
        try:
            annotated_image = annotate_image(image_data, self.cell_mapping)
        except ValueError as e:
            print(f"Annotation Error: {e}")
            QMessageBox.warning(self, "Annotation Error", str(e))
            return

        # Display the annotated image
        height, width = annotated_image.shape[:2]
        qimage = QImage(
            annotated_image.data,
            width,
            height,
            annotated_image.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.annotated_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.annotated_image_label.setPixmap(pixmap)

        # Generate scatter plot
        self.generate_scatter_plot()

    def populate_metrics_table(self):
        if not self.cell_mapping:
            QMessageBox.warning(
                self, "Error", "No cell data available for metrics table.")
            return

        # Convert cell mapping to a DataFrame
        metrics_data = [
            {**{"ID": cell_id}, **data["metrics"]}
            for cell_id, data in self.cell_mapping.items()
        ]
        metrics_df = pd.DataFrame(metrics_data)

        # Round numerical columns to 2 decimal places
        numeric_columns = [
            'area',
            'perimeter',
            'equivalent_diameter',
            'orientation',
            'aspect_ratio',
            'circularity',
            'solidity']  # Adjust based on actual column names
        metrics_df[numeric_columns] = metrics_df[numeric_columns].round(2)

        # Update QTableWidget
        self.metrics_table.setRowCount(metrics_df.shape[0])
        self.metrics_table.setColumnCount(metrics_df.shape[1])
        self.metrics_table.setHorizontalHeaderLabels(metrics_df.columns)

        for row in range(metrics_df.shape[0]):
            for col, value in enumerate(metrics_df.iloc[row]):
                self.metrics_table.setItem(
                    row, col, QTableWidgetItem(str(value)))

    def generate_scatter_plot(self):
        areas = [data["metrics"]["area"]
                 for data in self.cell_mapping.values()]
        perimeters = [
            data["metrics"]["perimeter"] for data in self.cell_mapping.values()
        ]
        ids = list(self.cell_mapping.keys())

        self.figure_scatter_plot.clear()
        ax = self.figure_scatter_plot.add_subplot(111)

        # Create scatter plot with interactivity
        scatter = ax.scatter(
            areas,
            perimeters,
            c=areas,
            cmap="viridis",
            picker=True)
        ax.set_title("Area vs Perimeter")
        ax.set_xlabel("Area")
        ax.set_ylabel("Perimeter")

        # Annotate scatter points with IDs
        for i, txt in enumerate(ids):
            ax.annotate(txt, (areas[i], perimeters[i]))

        # Add click event handling
        self.figure_scatter_plot.canvas.mpl_connect(
            "pick_event", lambda event: self.on_scatter_click(event)
        )

        self.canvas_scatter_plot.draw()

    def on_scatter_click(self, event):
        # Get the index of the clicked point
        ind = event.ind[0]  # Index of the clicked point
        cell_id = list(self.cell_mapping.keys())[ind]

        print(f"Clicked on scatter point: ID {cell_id}")

        # Extract the specific image frame
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        if self.image_data.is_nd2:
            if self.has_channels:
                image_data = self.image_data.data[t, p, c]
            else:
                image_data = self.image_data.data[t, p]
        else:
            image_data = self.image_data.data[t]

        image_data = np.array(image_data)  # Ensure it's a NumPy array

        # Highlight the corresponding cell in the annotated image
        annotated_image = annotate_image(
            image_data, {cell_id: self.cell_mapping[cell_id]}
        )

        # Update the annotated image display
        height, width = annotated_image.shape[:2]
        qimage = QImage(
            annotated_image.data,
            width,
            height,
            annotated_image.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.annotated_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.annotated_image_label.setPixmap(pixmap)

    def initPopulationTab(self):
        layout = QVBoxLayout(self.populationTab)
        label = QLabel("Average Pixel Intensity")
        layout.addWidget(label)

        self.population_figure = plt.figure()
        self.population_canvas = FigureCanvas(self.population_figure)
        layout.addWidget(self.population_canvas)

        # Channel control
        channel_choice_layout = QHBoxLayout()
        channel_combo = QComboBox()
        channel_combo.addItem('0')
        channel_combo.addItem('1')
        channel_combo.addItem('2')
        # channel_combo.valueChanged.connect(self.plot_fluorescence_signal)
        channel_choice_layout.addWidget(QLabel("Cannel selection: "))
        channel_choice_layout.addWidget(channel_combo)
        self.channel_combo = channel_combo

        # P controls
        p_layout = QHBoxLayout()
        p_label = QLabel("P: 0")
        p_layout.addWidget(p_label)

        # Set slider range based on loaded dimensions, or default to 0 if not
        # loaded
        max_p = (
            self.dimensions.get("P", 1) - 1
            if hasattr(self, "dimensions") and "P" in self.dimensions
            else 0
        )
        self.slider_p_5 = QSlider(Qt.Horizontal)
        self.slider_p_5.setMinimum(0)
        self.slider_p_5.setMaximum(max_p)
        self.slider_p_5.setValue(0)
        self.slider_p_5.valueChanged.connect(
            lambda value: p_label.setText(f'P: {value}'))

        p_layout.addWidget(self.slider_p_5)

        # Checkbox for single cell analysis
        self.single_cell_checkbox = QCheckBox("Single Cell Analysis")
        layout.addWidget(self.single_cell_checkbox)

        # Button to manually plot
        plot_fluo_btn = QPushButton("Plot Fluorescence")
        plot_fluo_btn.clicked.connect(self.plot_fluorescence_signal)

        channel_choice_layout.addWidget(plot_fluo_btn)

        layout.addLayout(p_layout)
        layout.addLayout(channel_choice_layout)

        # Time range controls
        time_range_layout = QHBoxLayout()
        time_range_layout.addWidget(QLabel("Time Range:"))

        self.time_min_box = QSpinBox()
        time_range_layout.addWidget(self.time_min_box)

        self.time_max_box = QSpinBox()
        time_range_layout.addWidget(self.time_max_box)

        layout.addLayout(time_range_layout)

        # Create the combobox and populate it with the dictionary keys
        self.rpu_params_combo = QComboBox()
        for key in AVAIL_RPUS.keys():
            self.rpu_params_combo.addItem(key)

        hb = QHBoxLayout()
        hb.addWidget(QLabel("Select RPU Parameters:"))
        hb.addWidget(self.rpu_params_combo)
        layout.addLayout(hb)

        # Only attempt to plot if image_data has been loaded
        if hasattr(self, 'image_data') and self.image_data is not None:
            self.plot_fluorescente_signal()

    def plot_fluorescence_signal(self):
        if not hasattr(self, 'image_data'):
            return

        p = self.slider_p.value()
        c = int(self.channel_combo.currentText())
        rpu = AVAIL_RPUS[self.rpu_params_combo.currentText()]
        t_s, t_e = self.time_min_box.value(), self.time_max_box.value()  # Time range

        fluo, timestamp = analyze_fluorescence_singlecell(
            self.segmented_time_series[t_s:t_e], self.image_data.data[t_s:t_e, p, c], rpu)

        self.population_figure.clear()
        ax = self.population_figure.add_subplot(111)

        plot_timestamp = []
        plot_fluo = []
        fluo_mean = []
        fluo_std = []

        # Each fluo is an array of the fluorescence under each bacteria
        #
        # timestamp = [0, 1, 2]
        # fluo = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # 1 2 3
        #           1    2    3
        #                     1    2   3

        # Copy timestamp for each fluorescence observation
        for t, fluo_data in zip(timestamp, fluo):
            fluo_mean.append(np.mean(fluo_data))
            fluo_std.append(np.std(fluo_data))
            for f in fluo_data:
                plot_timestamp.append(t)
                plot_fluo.append(f)

        fluo_mean = np.array(fluo_mean)
        fluo_std = np.array(fluo_std)

        # Randomly select up to 30 timepoints from plot_fluo and plot_timestamp
        points = np.array(list(zip(plot_timestamp, plot_fluo)))
        if len(points) > 200:
            points = points[np.random.choice(
                points.shape[0], 200, replace=False)]
        plot_timestamp, plot_fluo = zip(*points)

        ax.scatter(
            plot_timestamp,
            plot_fluo,
            color='blue',
            alpha=0.5,
            marker='+')
        ax.plot(timestamp, fluo_mean, color='red', label='Mean')
        ax.fill_between(
            timestamp,
            fluo_mean -
            fluo_std,
            fluo_mean +
            fluo_std,
            color='red',
            alpha=0.2,
            label='Std Dev')
        ax.set_title(f'Fluorescence signal for Position P={p}')
        ax.set_xlabel('T')
        ax.set_ylabel('Cell activity in RPUs')
        self.population_canvas.draw()
