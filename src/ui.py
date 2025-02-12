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
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot
from PySide6.QtWidgets import QSizePolicy, QComboBox, QLabel, QProgressBar
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
from morphology import annotate_image, extract_cells_and_metrics, annotate_binary_mask, extract_cell_morphologies, extract_cell_morphologies_time

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

from tracking import track_cells

from tqdm import tqdm

import matplotlib.pyplot as plt

from skimage.measure import label, regionprops

"""
Can hold either an ND2 file or a series of images
"""


class ImageData:
    def __init__(self, data, is_nd2=False):
        self.data = data
        self.processed_images = []
        self.is_nd2 = is_nd2
        self.segmentation_cache = {}
        self.seg_cache = SegmentationCache(data)


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

    def run(self):
        results = {}
        try:
            for t in range(self.num_frames):
                cache_key = (t, self.position, self.channel)

                # Check if segmentation is already cached
                if cache_key in self.image_data.segmentation_cache:
                    print(
                        f"[CACHE HIT] T={t}, P={self.position}, C={self.channel}")
                    binary_image = self.image_data.segmentation_cache[cache_key]
                else:
                    print(
                        f"[CACHE MISS] Segmenting T={t}, P={self.position}, C={self.channel}")
                    current_frame = self.image_frames[t]

                    # Skip empty/invalid frames
                    if np.mean(current_frame) == 0 or np.std(
                            current_frame) == 0:
                        print(f"Skipping empty frame T={t}")
                        self.progress.emit(t + 1)
                        continue

                    try:
                        # Get model type based on channel
                        model_type = 'bact_phase_cp3' if self.channel in (
                            0, None) else 'bact_fluor_cp3'
                        seg_result = SegmentationModels().segment_images(
                            [current_frame],
                            SegmentationModels.CELLPOSE,
                            model_type=model_type
                        )

                        if seg_result is None or len(seg_result) == 0:
                            raise ValueError("Empty segmentation result")

                        binary_image = seg_result[0]
                        self.image_data.segmentation_cache[cache_key] = binary_image
                    except Exception as seg_error:
                        print(
                            f"Segmentation failed for T={t}: {str(seg_error)}")
                        self.progress.emit(t + 1)
                        continue

                # Validate segmentation result
                if binary_image is None or binary_image.sum() == 0:
                    print(f"Frame {t}: No valid segmentation")
                    self.progress.emit(t + 1)
                    continue

                # Extract morphology metrics
                metrics = extract_cell_morphologies(binary_image)

                if not metrics.empty:
                    total_cells = len(metrics)

                    # Calculate Morphology Fractions
                    morphology_counts = metrics["morphology_class"].value_counts(
                        normalize=True)
                    fractions = morphology_counts.to_dict()

                    # Save results for this frame
                    results[t] = {"fractions": fractions}
                else:
                    print(
                        f"Frame {t}: Metrics computation returned no valid data.")

                self.progress.emit(t + 1)  # Update progress bar

            if results:
                self.finished.emit(results)  # Emit processed results
            else:
                self.error.emit("No valid results found in any frame.")
        except Exception as e:
            self.error.emit(str(e))


class TabWidgetApp(QMainWindow):
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

        # # TODO: de-comment
        # # Create a histogram of pixel counts using Seaborn
        # plt.figure(figsize=(10, 6))
        # sns.histplot(pixel_counts, bins=30, kde=False, color="blue", alpha=0.7)
        # plt.title("Histogram of Pixel Counts of Connected Components")
        # plt.xlabel("Pixel Count")
        # plt.ylabel("Number of Components")
        # plt.grid(True)
        # plt.show()

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
            cache_key = (t, p, c, "labeled")

            # Check if labeled segmentation is cached
            if cache_key in self.image_data.segmentation_cache:
                print(
                    f"[CACHE HIT] Using cached labeled segmentation for T={t}, P={p}, C={c}")
                labeled_image = self.image_data.segmentation_cache[cache_key]
            else:
                print(
                    f"[CACHE MISS] Generating labeled segmentation for T={t}, P={p}, C={c}")

                seg_cache_key = (t, p, c)

                # Check if raw segmentation is cached
                if seg_cache_key in self.image_data.segmentation_cache:
                    print(
                        f"[CACHE HIT] Using cached raw segmentation for T={t}, P={p}, C={c}")
                    segmented = self.image_data.segmentation_cache[seg_cache_key]
                else:
                    print(f"[CACHE MISS] Segmenting T={t}, P={p}, C={c}")

                    frame = self.image_data.data[t, p, c]

                    model_type = None
                    if self.model_dropdown.currentText() == SegmentationModels.CELLPOSE:
                        model_type = 'bact_phase_cp3' if c in (
                            0, None) else 'bact_fluor_cp3'
                        print(
                            f"Using model type: {model_type} for channel {c}")

                    # Perform segmentation
                    segmented = SegmentationModels().segment_images(np.array(
                        [frame]), self.model_dropdown.currentText(), model_type=model_type)[0]

                    self.image_data.segmentation_cache[seg_cache_key] = segmented

                # Label the segmented regions
                labeled_cells = label(segmented)

                # Convert labels to color image
                labeled_image = plt.cm.nipy_spectral(
                    labeled_cells.astype(float) / labeled_cells.max())
                labeled_image = (
                    labeled_image[:, :, :3] * 255).astype(np.uint8)

                # Overlay Cell IDs
                props = regionprops(labeled_cells)
                for prop in props:
                    y, x = prop.centroid  # Get centroid coordinates
                    cell_id = prop.label  # Get cell ID
                    cv2.putText(labeled_image, str(cell_id), (int(x), int(
                        y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # Cache the labeled image
                self.image_data.segmentation_cache[cache_key] = labeled_image

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
            return  # Prevent further processing for labeled segmentation

        elif self.radio_segmented.isChecked():
            cache_key = (t, p, c)
            if cache_key in self.image_data.segmentation_cache:
                print(
                    f"[CACHE HIT] Using cached segmentation for T={t}, P={p}, C={c}")
                image_data = self.image_data.segmentation_cache[cache_key]
            else:
                if self.model_dropdown.currentText() == SegmentationModels.CELLPOSE:
                    model_type = 'bact_phase_cp3' if c in (
                        0, None) else 'bact_fluor_cp3'
                    print(f"Using model type: {model_type} for channel {c}")
                else:
                    model_type = None

                frame = self.image_data.data[t, p, c] if self.image_data.is_nd2 else self.image_data.data[t]
                
                # Perform segmentation
                segmented = SegmentationModels().segment_images(
                    np.array([frame]), self.model_dropdown.currentText(), model_type=model_type
                )[0]
                

                # Relabel if output is binary mask
                if segmented.max() == 255 and len(np.unique(segmented)) <= 2:
                    
                    segmented = label(segmented).astype(np.uint8)* 255  # Ensure correct type for OpenCV
                    segmented_display = (segmented > 0).astype(np.uint8) * 255


                self.image_data.segmentation_cache[cache_key] = segmented_display
                image_data = segmented_display

            # Extract and cache morphology metrics if not already cached
            metrics_key = cache_key + ('metrics',)
            if metrics_key not in self.image_data.segmentation_cache:
                print(f"Extracting morphology metrics for T={t}, P={p}, C={c}")
                metrics = extract_cell_morphologies(image_data)
                self.image_data.segmentation_cache[metrics_key] = metrics
        
            
        else:  # Normal view or overlay
            if self.radio_overlay_outlines.isChecked():
                cache_key = (t, p, c)
                if cache_key in self.image_data.segmentation_cache:
                    segmented_image = self.image_data.segmentation_cache[cache_key]
                else:
                    segmented_image = SegmentationModels().segment_images(
                        [image_data],
                        SegmentationModels.CELLPOSE,
                        model_type=model_type
                    )[0]
                    self.image_data.segmentation_cache[cache_key] = segmented_image

                # Ensure segmented image has same dimensions as input image
                if segmented_image.shape != image_data.shape:
                    segmented_image = cv2.resize(
                        segmented_image,
                        (image_data.shape[1],
                         image_data.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

                outlines = utils.masks_to_outlines(segmented_image)
                overlay = image_data.copy()

                # Verify dimensions match before applying overlay
                if outlines.shape == overlay.shape:
                    overlay[outlines] = overlay.max()
                else:
                    print(
                        f"Dimension mismatch - Outline shape: {outlines.shape}, Image shape: {overlay.shape}")
                    outlines = cv2.resize(
                        outlines.astype(
                            np.uint8),
                        (overlay.shape[1],
                         overlay.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay[outlines] = overlay.max()

                image_data = overlay

            # Normalize and apply color for normal/overlay views
            image_data = cv2.normalize(
                image_data,
                None,
                0,
                255,
                cv2.NORM_MINMAX).astype(
                np.uint8)

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
                    # Add some red to make it more yellow
                    colored_image[:, :, 2] = image_data * 0.5
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

        # Button to track cells over time
        self.track_button = QPushButton("Track Cells")
        self.track_button.clicked.connect(self.track_cells_over_time)
        layout.addWidget(self.track_button)

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
        Tracks segmented cells over time and visualizes trajectories.
        """
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        if not self.image_data.is_nd2:
            QMessageBox.warning(
                self, "Error", "Tracking requires an ND2 dataset.")
            return

        try:
            t = self.dimensions["T"]  # Get total number of frames

            # Extract segmented images for all time points
            segmented_imgs = np.array([
                self.image_data.segmentation_cache[(i, p, c)] for i in range(t)
                if (i, p, c) in self.image_data.segmentation_cache
            ])

            print(f"Segmented Images Shape: {segmented_imgs.shape}")

            if segmented_imgs.size == 0:
                QMessageBox.warning(
                    self, "Error", "No segmented images found for tracking.")
                return

            # Ensure the segmented images are binary
            segmented_imgs = (segmented_imgs > 0).astype(np.uint8) * 255

            # Run cell tracking
            self.tracked_cells = track_cells(segmented_imgs)

            # Debug: Check structure of tracked cells
            for track in self.tracked_cells:
                print(
                    f"Track ID: {track['ID']}, X: {track['x']}, Y: {track['y']}")

            QMessageBox.information(
                self,
                "Tracking Complete",
                "Cell tracking completed successfully.")

            # Plot tracking results
            self.visualize_tracking(self.tracked_cells)

        except Exception as e:
            QMessageBox.warning(
                self,
                "Tracking Error",
                f"Failed to track cells: {e}")

    def visualize_tracking(self, tracks):
        """
        Visualizes the tracked cells as trajectories over time.

        Parameters:
        -----------
        tracks : list
            List of tracked cell objects (OrderedDict) with tracking information.
        """
        self.figure_morphology_fractions.clear()
        ax = self.figure_morphology_fractions.add_subplot(111)

        for track in tracks:
            x_coords = track['x']
            y_coords = track['y']
            track_id = track['ID']

            # Plot trajectory
            ax.plot(x_coords, y_coords, marker='o', label=f'Track {track_id}')

        ax.set_title('Cell Trajectories Over Time')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()

        # Draw the plot
        self.canvas_morphology_fractions.draw()

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

        print("Results received successfully:", results)

        # Get all unique morphology classes across all frames
        all_morphologies = set()
        for frame_data in results.values():
            all_morphologies.update(frame_data["fractions"].keys())

        # Initialize the dictionary with all possible morphology classes
        morphology_fractions = {morphology: []
                                for morphology in all_morphologies}

        # Get the maximum time point
        max_time = max(results.keys())

        # Fill in the fractions for each time point, using 0.0 for missing
        # classes
        for t in range(max_time + 1):
            if t in results:
                frame_data = results[t]["fractions"]
                for morphology in all_morphologies:
                    # Get the fraction if present, otherwise use 0.0
                    fraction = frame_data.get(morphology, 0.0)
                    morphology_fractions[morphology].append(fraction)
            else:
                # For frames with no data, append 0.0 for all morphologies
                for morphology in all_morphologies:
                    morphology_fractions[morphology].append(0.0)

        self.figure_morphology_fractions.clear()
        ax2 = self.figure_morphology_fractions.add_subplot(111)

        # Plot each morphology class with its corresponding color from
        # self.morphology_colors_rgb
        for morphology, fractions in morphology_fractions.items():
            color = self.morphology_colors_rgb.get(
                morphology, (0.5, 0.5, 0.5))  # Default to gray if color not found
            ax2.plot(
                range(
                    len(fractions)),
                fractions,
                marker="o",
                label=morphology,
                color=color)

        ax2.set_title("Morphology Fractions Over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Fraction")
        ax2.legend()
        self.canvas_morphology_fractions.draw()

    def handle_error(self, error_message):
        print(f"Error: {error_message}")
        QMessageBox.warning(self, "Processing Error", error_message)

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
            [SegmentationModels.CELLPOSE, SegmentationModels.UNET, SegmentationModels.CELLSAM])
        self.model_dropdown.currentIndexChanged.connect(
            lambda: self.image_data.seg_cache.with_model(
                self.model_dropdown.currentText()))
        layout.addWidget(self.model_dropdown)

    def annotate_cells(self):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Extract the current frame
        image_data = self.image_data.data
        if self.image_data.is_nd2:
            frame = image_data[t, p,
                               c] if self.has_channels else image_data[t, p]
        else:
            frame = image_data[t]

        frame = np.array(frame)  # Ensure it's a NumPy array

        # Perform segmentation
        cache_key = (t, p, c)
        if cache_key in self.image_data.segmentation_cache:
            print(
                f"[CACHE HIT] Using cached segmentation for T={t}, P={p}, C={c}")
            segmented_image = self.image_data.segmentation_cache[cache_key]
        else:
            print(f"[CACHE MISS] Segmenting T={t}, P={p}, C={c}")
            segmented_image = SegmentationModels().segment_images(
                [frame], self.model_dropdown.currentText())[0]
            self.image_data.segmentation_cache[cache_key] = segmented_image

        # Extract cell metrics and bounding boxes
        cell_mapping = extract_cells_and_metrics(frame, segmented_image)

        if not cell_mapping:
            QMessageBox.warning(
                self,
                "No Cells",
                "No cells detected in the current frame.")
            return

        # Annotate the binary segmented image
        annotated_binary_mask = annotate_binary_mask(
            segmented_image, cell_mapping)

        # Set the annotated image for saving
        self.annotated_image = annotated_binary_mask  # <-- Ensure this is set

        # **Display the annotated image on the main image display**
        height, width = annotated_binary_mask.shape[:2]
        qimage = QImage(
            annotated_binary_mask.data,
            width,
            height,
            annotated_binary_mask.strides[0],
            QImage.Format_RGB888,
        )
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

        # Dropdowns to select X and Y metrics
        self.x_metric_dropdown = QComboBox()
        self.y_metric_dropdown = QComboBox()
        metrics_list = [
            "area", "perimeter", "aspect_ratio", "circularity", "solidity",
            "equivalent_diameter", "orientation"
        ]
        self.x_metric_dropdown.addItems(metrics_list)
        self.y_metric_dropdown.addItems(metrics_list)

        # Layout for dropdowns
        dropdown_layout = QHBoxLayout()
        dropdown_layout.addWidget(QLabel("X-axis Metric:"))
        dropdown_layout.addWidget(self.x_metric_dropdown)
        dropdown_layout.addWidget(QLabel("Y-axis Metric:"))
        dropdown_layout.addWidget(self.y_metric_dropdown)

        layout.addLayout(dropdown_layout)

        # Button to trigger plotting
        plot_button = QPushButton("Plot Metrics")
        plot_button.clicked.connect(self.plot_morphology_metrics)
        layout.addWidget(plot_button)

        # Matplotlib canvas for plotting
        self.figure_morphology_metrics = plt.figure()
        self.canvas_morphology_metrics = FigureCanvas(
            self.figure_morphology_metrics)
        layout.addWidget(self.canvas_morphology_metrics)

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
            cache_key = (t, p, c)

            # Perform segmentation if not cached
            if cache_key not in self.image_data.segmentation_cache:
                segmented_image = segment_this_image(frame)
                self.image_data.segmentation_cache[cache_key] = segmented_image
            else:
                segmented_image = self.image_data.segmentation_cache[cache_key]

            # Extract cell metrics
            self.cell_mapping = extract_cells_and_metrics(frame, segmented_image)
            self.populate_metrics_table()

            # Prepare DataFrame
            metrics_data = [
                {**{"ID": cell_id}, **data["metrics"], **{"Class": data["metrics"]["morphology_class"]}}
                for cell_id, data in self.cell_mapping.items()
            ]
            morphology_df = pd.DataFrame(metrics_data)

            # Select numeric features for PCA
            numeric_features = ['area', 'perimeter', 'equivalent_diameter', 'orientation', 
                                'aspect_ratio', 'circularity', 'solidity']
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
            scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=[self.morphology_colors_rgb[class_] for class_ in pca_df['Class']],
                                s=50, edgecolor='w', picker=True)
            
            ax.set_title("PCA Scatter Plot")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

            # Enable interactive annotations and highlighting
            self.annotate_scatter_points(ax, scatter, pca_df)

            self.canvas_annot_scatter.draw()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")


    
    
    def plot_morphology_metrics(self):
        x_metric = self.x_metric_dropdown.currentText()
        y_metric = self.y_metric_dropdown.currentText()

        # Check if segmentation and metrics are available
        if not hasattr(self, "cell_mapping") or not self.cell_mapping:
            QMessageBox.warning(
                self,
                "Error",
                "No cell data available for plotting. Please classify cells first.")
            return

        # Extract data for selected metrics directly from cell_mapping
        x_data = [data["metrics"].get(x_metric, np.nan)
                  for data in self.cell_mapping.values()]
        y_data = [data["metrics"].get(y_metric, np.nan)
                  for data in self.cell_mapping.values()]

        # Convert data to numeric and handle non-numeric values
        x_data = pd.to_numeric(x_data, errors='coerce')
        y_data = pd.to_numeric(y_data, errors='coerce')

        # Filter out NaNs
        valid_indices = ~(pd.isna(x_data) | pd.isna(y_data))
        x_data = x_data[valid_indices]
        y_data = y_data[valid_indices]

        # Apply KMeans clustering
        if len(x_data) > 0 and len(y_data) > 0:
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(np.column_stack((x_data, y_data)))
            centroids = kmeans.cluster_centers_

            # Plotting the clustered data
            self.figure_morphology_metrics.clear()
            ax = self.figure_morphology_metrics.add_subplot(111)

            scatter = ax.scatter(
                x_data,
                y_data,
                c=clusters,
                cmap='viridis',
                s=50,
                edgecolor='w')
            ax.scatter(centroids[:, 0], centroids[:, 1],
                       c='red', marker='X', s=200, label='Centroids')
            ax.set_title(
                f"{x_metric.capitalize()} vs {y_metric.capitalize()} with Clustering")
            ax.set_xlabel(x_metric.capitalize())
            ax.set_ylabel(y_metric.capitalize())
            ax.legend()

            self.canvas_morphology_metrics.draw()
        else:
            QMessageBox.warning(self, "Error", "No valid data to plot.")

    def segment_this_p(self):
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None

        # Get total number of time points
        # Only process first 10 frames
        total_frames = min(10, self.image_data.data.shape[0])
        print(f"Processing first {total_frames} frames")

        # Create list to store segmented results
        segmented_results = []

        for t in tqdm(range(total_frames), desc="Segmenting frames"):
            cache_key = (t, p, c)

            # Use cached segmentation if available
            if cache_key in self.image_data.segmentation_cache:
                print(
                    f"[CACHE HIT] Using cached segmentation for T={t}, P={p}, C={c}")
                segmented = self.image_data.segmentation_cache[cache_key]
            else:
                print(f"[CACHE MISS] Segmenting T={t}, P={p}, C={c}")

                frame = self.image_data.data[t, p, c]

                # Determine model type ONLY if Cellpose is selected
                model_type = None
                if self.model_dropdown.currentText() == SegmentationModels.CELLPOSE:
                    model_type = 'bact_phase_cp3' if c in (
                        0, None) else 'bact_fluor_cp3'

                # Perform segmentation
                segmented = SegmentationModels().segment_images(np.array(
                    [frame]), self.model_dropdown.currentText(), model_type=model_type)[0]

                # Store result in cache
                self.image_data.segmentation_cache[cache_key] = segmented

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
                    self.highlight_cell_in_image(selected_id, cell_class)

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

        # Ensure we retrieve the correct **cell ID from the table column**
        cell_id_item = self.metrics_table.item(row, 0)  # Assuming column 0 is the ID column
        if cell_id_item is None:
            print(f"Error: No cell ID found at row {row}.")
            return

        cell_id = int(cell_id_item.text())  # Convert to int
        cell_class_item = self.metrics_table.item(row, self.metrics_table.columnCount() - 1)  # Assuming last column has class

        if cell_class_item is None:
            print(f"Error: No cell class found for Cell ID {cell_id}.")
            return

        cell_class = cell_class_item.text()  # Extract class name

        print(f"Row {row} clicked, Cell ID: {cell_id}, Cell Class: {cell_class}")

        self.highlight_cell_in_image(cell_id, cell_class)





    def highlight_cell_in_image(self, cell_id, cell_class):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value() if self.has_channels else None
        cache_key = (t, p, c)

        if cache_key not in self.image_data.segmentation_cache:
            QMessageBox.warning(self, "Error", "Segmented image not found.")
            return

        segmented_image = self.image_data.segmentation_cache[cache_key]

        # Ensure segmentation labels are correctly formatted
        if segmented_image.max() <= 255 and segmented_image.dtype == np.uint8:
            segmented_image = label(segmented_image).astype(np.uint16)
            self.image_data.segmentation_cache[cache_key] = segmented_image

        cell_id = int(cell_id)

        # Create an RGB version of the segmented image where all cells are grayscale
        base_image = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)

        # Normalize grayscale mapping based on total unique IDs
        unique_labels = np.unique(segmented_image)
        max_label = unique_labels.max()
        
        for label_id in unique_labels:
            if label_id == 0:  # Background
                continue
            # Normalize to a grayscale intensity in range 50-200 to make all visible
            intensity = int(50 + (label_id / max_label) * 150)
            base_image[segmented_image == label_id] = (intensity, intensity, intensity)

        # Highlight the selected cell with its corresponding class color
        mask = segmented_image == cell_id
        highlight_color = tuple(int(x * 255) for x in self.morphology_colors_rgb.get(cell_class, (1, 1, 1)))
        base_image[mask] = highlight_color

        # Display the image
        height, width, _ = base_image.shape
        qimage = QImage(base_image.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)


        
    
    
        
     
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

        # Get the current position from the position slider in the population
        # tab
        p = self.slider_p_5.value()
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
        if len(points) > 30 * len(timestamp):
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
