from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QButtonGroup, QTabWidget, QWidget, QLabel, QPushButton, QFileDialog, QScrollArea, QSlider, QHBoxLayout, QCheckBox, QMessageBox, QRadioButton
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QComboBox, QLabel

import sys
import os

# import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt
import nd2
import numpy as np
import cv2
import imageio.v3 as iio

# Local imports
from morphology import extract_cell_morphologies
from segmentation import segment_all_images, segment_this_image
from image_functions import remove_stage_jitter_MAE

# import pims
from matplotlib.backends.backend_qt5agg import FigureCanvas

import seaborn as sns

from population import get_fluorescence_all_experiments, get_fluorescence_single_experiment

"""
Can hold either an ND2 file or a series of images
"""
class ImageData:
    def __init__(self, data, is_nd2=False):
        self.data = data
        self.processed_images = []
        self.is_nd2 = is_nd2

class TabWidgetApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the processed_images list to store images for export
        self.processed_images = []

        self.setWindowTitle('Partaker 3 - GUI')
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
        self.initUI()
        self.layout.addWidget(self.tab_widget)

    def load_from_folder(self, folder_path, aligned_images = False):
        p = Path(folder_path)
        
        images = p.iterdir()
        # images = filter(lambda x : x.name.lower().endswith(('.tif')), images)
        img_filelist = sorted(images, key=lambda x : int(x.stem))
        
        preproc_img = lambda img : img # Placeholder for now
        loaded = np.array([preproc_img(cv2.imread(str(_img))) for _img in img_filelist])

        if not aligned_images:

            self.image_data = ImageData(loaded, is_nd2=False)

            print(f'Loaded dataset: {self.image_data.data.shape}')
            self.info_label.setText(f'Dataset size: {self.image_data.data.shape}')
            QMessageBox.about(self, "Import", f'Loaded {self.image_data.data.shape[0]} pictures')

            self.image_data.phc_path  = folder_path

        else:
            self.image_data.aligned_data = loaded

            print(f'Loaded aligned: {loaded.shape}')
            QMessageBox.about(self, "Import", f'Loaded aligned images. Size: {self.image_data.aligned_data.shape}')

            self.image_data.aligned_phc_path  = folder_path

    def load_nd2_file(self, file_path):

        self.file_path = file_path
        with nd2.ND2File(file_path) as nd2_file:
            self.nd2_file = nd2_file
            self.dimensions = nd2_file.sizes
            info_text = f"Number of dimensions: {nd2_file.sizes}\n"
            
            for dim, size in self.dimensions.items():
                info_text += f"{dim}: {size}\n"
            
            if 'C' in self.dimensions.keys():
                self.has_channels = True
                self.channel_number = self.dimensions['C']
                self.slider_c.setMinimum(0)
                self.slider_c.setMaximum(self.channel_number - 1)
            else:
                self.has_channels = False

            self.info_label.setText(info_text)
            self.image_data = ImageData(nd2.imread(file_path, dask=True), is_nd2=True)

            # Set the slider range for position (P) immediately based on dimensions
            max_position = self.dimensions.get("P", 1) - 1
            self.slider_p.setMaximum(max_position)
            self.slider_p_5.setMaximum(max_position)  # Update population tab slider

            self.update_mapping_dropdowns()
            self.update_controls()
            
            self.mapping_controls["time"].currentIndexChanged.connect(self.update_slider_range)
            self.mapping_controls["position"].currentIndexChanged.connect(self.update_slider_range)

            self.display_image()
            
    def update_mapping_dropdowns(self):
        # Clear all dropdowns before updating
        for dropdown in self.mapping_controls.values():
            dropdown.clear()

        # Populate each dropdown based on its specific dimension
        time_dim = self.dimensions.get("T", 1)
        position_dim = self.dimensions.get("P", 1)
        channel_dim = self.dimensions.get("C", 1) 
        x_dim = self.dimensions.get("X", 1)
        y_dim = self.dimensions.get("Y", 1)


        self.mapping_controls["time"].addItem("Select Time")
        for i in range(time_dim):
            self.mapping_controls["time"].addItem(str(i))

        self.mapping_controls["position"].addItem("Select Position")
        for i in range(position_dim):
            self.mapping_controls["position"].addItem(str(i))

        # Populate Channel dropdown if multiple channels exist
        if "C" in self.dimensions:
            self.mapping_controls["channel"].addItem("Select Channel")
            for i in range(channel_dim):
                self.mapping_controls["channel"].addItem(str(i))

        self.mapping_controls["x_coord"].addItem("Fixed X range: 0 to {}".format(x_dim - 1))
        self.mapping_controls["y_coord"].addItem("Fixed Y range: 0 to {}".format(y_dim - 1))

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
        # Get selected values from dropdowns for time and position
        selected_time = self.mapping_controls["time"].currentText()
        selected_position = self.mapping_controls["position"].currentText()

        if selected_time.isdigit():
            self.slider_t.setMaximum(int(selected_time))
        else:
            self.slider_t.setMaximum(self.dimensions.get("T", 1) - 1)

        if selected_position.isdigit():
            max_position = int(selected_position)
            self.slider_p.setMaximum(max_position)
            self.slider_p_5.setMaximum(max_position) 
        else:
            max_position = self.dimensions.get("P", 1) - 1
            self.slider_p.setMaximum(max_position)
            self.slider_p_5.setMaximum(max_position)
    
    def show_cell_area(self, img):
        from skimage import measure
        import seaborn as sns

        # Binarize the image using Otsu's thresholding
        _, bw_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_image, connectivity=8)

        # Extract pixel counts for each component (ignore background)
        pixel_counts = stats[1:, cv2.CC_STAT_AREA]  # Skip the first label (background)

        # Create a histogram of pixel counts using Seaborn
        plt.figure(figsize=(10, 6))
        sns.histplot(pixel_counts, bins=30, kde=False, color='blue', alpha=0.7)
        plt.title('Histogram of Pixel Counts of Connected Components')
        plt.xlabel('Pixel Count')
        plt.ylabel('Number of Components')
        plt.grid(True)
        plt.show()

        # # Label connected components
        # labeled_image, num_components = measure.label(img, connectivity=2, return_num=True)

        # # Count pixels in each component (ignore background)
        # pixel_counts = np.bincount(labeled_image.ravel())[1:]  # Skip the first element (background)

        # # Create a histogram of pixel counts
        # plt.hist(pixel_counts, bins=30, color='blue', alpha=0.7)
        # plt.title('Histogram of Pixel Counts of Connected Components')
        # plt.xlabel('Pixel Count')
        # plt.ylabel('Number of Components')
        # plt.grid(True)
        # plt.show()

    def display_image(self):
        t = self.slider_t.value()
        p = self.slider_p.value()
        c = self.slider_c.value()

        image_data = self.image_data.data
        if self.image_data.is_nd2:
            if self.has_channels:
                image_data = image_data[t, p, c]
            else:
                image_data = image_data[t, p] 
        else:
            image_data = image_data[t]

        # image_data = image_data.compute().data
        image_data = np.array(image_data)

        # Apply thresholding or segmentation if selected
        if self.radio_thresholding.isChecked():
            threshold = self.threshold_slider.value()
            image_data = cv2.threshold(image_data, threshold, 255, cv2.THRESH_BINARY)[1]
            image_data = image_data.compute() 
        elif self.radio_segmented.isChecked():
            image_data = segment_this_image(image_data)
            self.show_cell_area(image_data)

        # Normalize the image from 0 to 65535
        image_data = (image_data.astype(np.float32) / image_data.max() * 65535).astype(np.uint16)

        # Update image format and display
        image_format = QImage.Format_Grayscale16
        height, width = image_data.shape[:2]
        image = QImage(image_data, width, height, image_format)
        pixmap = QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        # Store this processed image for export
        self.processed_images.append(image_data)
        
    def align_images(self):
        
        # Check if the dataset and phc_path are loaded
        if not hasattr(self.image_data, 'data') or not hasattr(self.image_data, 'phc_path'):
            QMessageBox.warning(self, "Alignment Error", "No phase contrast images loaded or missing path.")
            return

        stage_MAE_scores = remove_stage_jitter_MAE(
            './mat/aligned_phc/',
            self.image_data.phc_path, 
            None, 
            None, 
            None,
            None, 
            10000, 
            -15,
            True,
            False
        )
        
        self.load_from_folder('./mat/aligned_phc/', aligned_images=True)
        
        QMessageBox.about(self, "Alignment", f"Alignment completed successfully. {stage_MAE_scores}")

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

        layout = QVBoxLayout(self.importTab)

        # Select file/folder button
        button = QPushButton("Select File / Folder")
        button.clicked.connect(lambda: importFile() if not self.is_folder_checkbox.isChecked() else importFolder())
        layout.addWidget(button)

        # Checkbox to toggle between file and folder import
        self.is_folder_checkbox = QCheckBox("Load from folder?")
        layout.addWidget(self.is_folder_checkbox)

        # Filename and info labels
        self.filename_label = QLabel("Filename will be shown here.")
        layout.addWidget(self.filename_label)

        self.info_label = QLabel("File info will be shown here.")
        layout.addWidget(self.info_label)

        # Dropdowns for mapping ND2 dimensions
        self.mapping_controls = {}

        mapping_labels = {
            "time": "Time",
            "position": "Position",
            "channel": "Channel",
            "x_coord": "X Coordinate",
            "y_coord": "Y Coordinate"
        }

        for key, label_text in mapping_labels.items():
            label = QLabel(label_text)
            layout.addWidget(label)

            dropdown = QComboBox()
            dropdown.addItem("Select Dimension")  # Placeholder
            layout.addWidget(dropdown)
            
            self.mapping_controls[key] = dropdown

        self.mapping_info_label = QLabel("Load an ND2 file to view and assign dimension mappings.")
        layout.addWidget(self.mapping_info_label)

    def initMorphologyTimeTab(self):
        pass
        # segment_all_images(img)

        # for img in images:
        #     segmented_image = 
        #     morphology_data = extract_cell_morphologies(segmented_image)

    def initMorphologyTab(self):
        def segment_and_plot():
            t = self.slider_t.value()
            p = self.slider_p.value()
            c = self.slider_c.value()

            image_data = self.image_data.data
            if self.image_data.is_nd2:
                if self.has_channels:
                    image_data = image_data[t, p, c]
                else:
                    image_data = image_data[t, p]
            else:
                image_data = image_data[t]

            image_data = np.array(image_data)
            segmented_image = segment_this_image(image_data)
            morphology_data = extract_cell_morphologies(segmented_image)

            x_key = self.x_dropdown.currentText()
            y_key = self.y_dropdown.currentText()

            # Plot centroids
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            # sns.scatterplot(data=morphology_data, x=x_key, y=y_key, hue='area', palette='viridis', ax=ax)
            sns.scatterplot(data=morphology_data, x=x_key, y=y_key, palette='viridis', ax=ax)
            ax.set_title(f'{x_key} vs {y_key}')
            self.canvas.draw()

        layout = QVBoxLayout(self.morphologyTab)

        segment_button = QPushButton("Segment and Plot")
        segment_button.clicked.connect(segment_and_plot)

        """
            morphology = {
                'area': area,
                'perimeter': perimeter,
                'bounding_box': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity,
                'equivalent_diameter': equivalent_diameter,
                'orientation': angle
            }
        """
        labels_layout = QHBoxLayout()

        x_dropdown_w = QVBoxLayout()
        x_dropdown_w.addWidget(QLabel("Select X variable"))
        x_dropdown = QComboBox()
        x_dropdown.addItem('area')
        x_dropdown.addItem('perimeter')
        x_dropdown.addItem('bounding_box')
        x_dropdown.addItem('aspect_ratio')
        x_dropdown.addItem('extent')
        x_dropdown.addItem('solidity')
        x_dropdown.addItem('equivalent_diameter')
        x_dropdown.addItem('orientation')
        x_dropdown_w.addWidget(x_dropdown)
        wid = QWidget()
        wid.setLayout(x_dropdown_w)
        labels_layout.addWidget(wid)

        y_dropdown_w = QVBoxLayout()
        y_dropdown_w.addWidget(QLabel("Select Y variable"))
        y_dropdown = QComboBox()
        y_dropdown.addItem('area')
        y_dropdown.addItem('perimeter')
        y_dropdown.addItem('bounding_box')
        y_dropdown.addItem('aspect_ratio')
        y_dropdown.addItem('extent')
        y_dropdown.addItem('solidity')
        y_dropdown.addItem('equivalent_diameter')
        y_dropdown.addItem('orientation')
        y_dropdown_w.addWidget(y_dropdown)
        wid = QWidget()
        wid.setLayout(y_dropdown_w)
        labels_layout.addWidget(wid)
        layout.addLayout(labels_layout)

        self.x_dropdown = x_dropdown
        self.y_dropdown = y_dropdown

        layout.addWidget(segment_button)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def initViewArea(self):
        layout = QVBoxLayout(self.viewArea)
        # label = QLabel("Content of Tab 2")
        # layout.addWidget(label)

        self.image_label = QLabel()
        self.image_label.setScaledContents(True)  # Allow the label to scale the image
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)
        
        # Another label for aligned images
        # self.aligned_image_label = QLabel()
        # self.aligned_image_label.setScaledContents(True)  # Allow the label to scale the image
        # self.aligned_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # layout.addWidget(self.aligned_image_label)

        # Align button
        # align_button = QPushButton("Align Images")
        # align_button.clicked.connect(self.align_images)
        # layout.addWidget(align_button)

        # T controls
        t_layout = QHBoxLayout()
        t_label = QLabel("T: 0")
        t_layout.addWidget(t_label)
        self.t_left_button = QPushButton("<")
        self.t_left_button.clicked.connect(lambda: self.slider_t.setValue(self.slider_t.value() - 1))
        t_layout.addWidget(self.t_left_button)

        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.valueChanged.connect(self.display_image)
        self.slider_t.valueChanged.connect(lambda value: t_label.setText(f'T: {value}'))
        
        t_layout.addWidget(self.slider_t)

        self.t_right_button = QPushButton(">")
        self.t_right_button.clicked.connect(lambda: self.slider_t.setValue(self.slider_t.value() + 1))
        t_layout.addWidget(self.t_right_button)

        layout.addLayout(t_layout)

        # P controls
        p_layout = QHBoxLayout()
        p_label = QLabel("P: 0")
        p_layout.addWidget(p_label)
        self.p_left_button = QPushButton("<")
        self.p_left_button.clicked.connect(lambda: self.slider_p.setValue(self.slider_p.value() - 1))
        p_layout.addWidget(self.p_left_button)

        self.slider_p = QSlider(Qt.Horizontal)
        self.slider_p.valueChanged.connect(self.display_image)
        self.slider_p.valueChanged.connect(lambda value: p_label.setText(f'P: {value}'))
        p_layout.addWidget(self.slider_p)

        self.p_right_button = QPushButton(">")
        self.p_right_button.clicked.connect(lambda: self.slider_p.setValue(self.slider_p.value() + 1))
        p_layout.addWidget(self.p_right_button)

        layout.addLayout(p_layout)

        # C control (channel)
        c_layout = QHBoxLayout()
        c_label = QLabel("C: 0")
        c_layout.addWidget(c_label)
        self.c_left_button = QPushButton("<")
        self.c_left_button.clicked.connect(lambda: self.slider_c.setValue(self.slider_c.value() - 1))
        c_layout.addWidget(self.c_left_button)

        self.slider_c = QSlider(Qt.Horizontal)
        self.slider_c.valueChanged.connect(self.display_image)
        self.slider_c.valueChanged.connect(lambda value: c_label.setText(f'C: {value}'))
        c_layout.addWidget(self.slider_c)

        self.c_right_button = QPushButton(">")
        self.c_right_button.clicked.connect(lambda: self.slider_c.setValue(self.slider_c.value() + 1))
        c_layout.addWidget(self.c_right_button)

        layout.addLayout(c_layout)

        # Create a radio button for thresholding, normal and segmented
        self.radio_normal = QRadioButton("Normal")
        self.radio_thresholding = QRadioButton("Thresholding")
        self.radio_segmented = QRadioButton("Segmented")

        # Create a button group and add the radio buttons to it
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_normal)
        self.button_group.addButton(self.radio_thresholding)
        self.button_group.addButton(self.radio_segmented)

        # Set default selection
        self.radio_normal.setChecked(True)

        # Add radio buttons to the layout
        layout.addWidget(self.radio_thresholding)
        layout.addWidget(self.radio_normal)
        layout.addWidget(self.radio_segmented)

        # Threshold slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.valueChanged.connect(self.display_image)
        layout.addWidget(self.threshold_slider)

    def export_images(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save As", "", "TIFF Files (*.tif);;All Files (*)")
        
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
                    export_image = self.image_data.data[t, p].compute() if hasattr(self.image_data.data, 'compute') else self.image_data.data[t, p]
                else:
                    export_image = self.image_data.data[t]

                img_to_save = np.array(export_image)

                # Construct the export path with the custom name and dimensions
                file_path = folder_path / f"{custom_base_name}_P{p}_T{t}.tif"
                cv2.imwrite(str(file_path), img_to_save)
        
        QMessageBox.information(self, "Export", f"Images exported successfully to {folder_path}")
    
    # Initialize the Export tab with the export button
    def initExportTab(self):
        layout = QVBoxLayout(self.exportTab)
        export_button = QPushButton("Export Images")
        export_button.clicked.connect(self.export_images)
        layout.addWidget(export_button)
        label = QLabel("This Tab Exports processed images sequentially.")
        layout.addWidget(label)

    def save_video(self, file_path):
        # Assuming self.image_data is a 4D numpy array with shape (frames, height, width, channels)
        if hasattr(self, 'image_data'):
            print(self.image_data.data.shape)

            with iio.imopen(file_path, "w", plugin="pyav") as writer:
                writer.init_video_stream("libx264", fps=30, pixel_format="yuv444p")

                writer._video_stream.options = {'preset': 'veryslow', 'qp': '0'} # 'crf': '0', 

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
        self.tab_widget.addTab(self.importTab, "Import")
        self.tab_widget.addTab(self.exportTab, "Export")
        self.tab_widget.addTab(self.populationTab, "Population")
        self.tab_widget.addTab(self.morphologyTab, "Morphology")

        self.initImportTab()
        self.initViewArea()
        self.initExportTab()
        self.initPopulationTab()
        self.initMorphologyTab()

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

        # Set slider range based on loaded dimensions, or default to 0 if not loaded
        max_p = self.dimensions.get("P", 1) - 1 if hasattr(self, 'dimensions') and "P" in self.dimensions else 0
        self.slider_p_5 = QSlider(Qt.Horizontal)
        self.slider_p_5.setMinimum(0)
        self.slider_p_5.setMaximum(max_p) 
        self.slider_p_5.setValue(0) 
        # self.slider_p_5.valueChanged.connect(self.plot_fluorescente_signal)
        self.slider_p_5.valueChanged.connect(lambda value: p_label.setText(f'P: {value}'))  
        p_layout.addWidget(self.slider_p_5)

        # Button to manually plot
        plot_fluo_btn = QPushButton("Plot Fluorescence")
        plot_fluo_btn.clicked.connect(self.plot_fluorescence_signal)

        channel_choice_layout.addWidget(plot_fluo_btn)

        layout.addLayout(p_layout)
        layout.addLayout(channel_choice_layout)

        # Only attempt to plot if image_data has been loaded
        if hasattr(self, 'image_data') and self.image_data is not None:
            self.plot_fluorescente_signal()

    def ___plot_fluorescente_signal(self):
        if not hasattr(self, 'image_data'):
            return

        selected_time = self.mapping_controls["time"].currentText()
        max_time = int(selected_time) if selected_time.isdigit() else self.dimensions.get("T", 1) - 1

        full_time_range = self.dimensions.get("T", 1) - 1
        x_axis_limit = full_time_range + 2 

        # Get the current position from the position slider in the population tab
        p = self.slider_p_5.value()
        
        chan_sel = int(self.channel_combo.currentText())

        # get intensities
        levels, RPUs, error = get_fluorescence_all_experiments(self.image_data.data, self.dimensions, chan_sel)

        self.population_figure.clear()
        ax = self.population_figure.add_subplot(111)

        for rpu in RPUs:
            ax.plot(rpu, color='gray')
        ax.plot(np.mean(rpu), color='red')
        
        ax.set_xlim(0, x_axis_limit)
        ax.set_title(f'Fluorescence signal for Position P={p}')
        ax.set_xlabel('T')
        ax.set_ylabel('Signal / RPU')
        self.canvas.draw()

    def plot_fluorescence_signal(self):
        if not hasattr(self, 'image_data'):
            return

        selected_time = self.mapping_controls["time"].currentText()
        max_time = int(selected_time) if selected_time.isdigit() else self.dimensions.get("T", 1) - 1

        full_time_range = self.dimensions.get("T", 1) - 1
        x_axis_limit = full_time_range + 2 

        # Get the current position from the position slider in the population tab
        p = self.slider_p_5.value()

        chan_sel = int(self.channel_combo.currentText())
        levels, RPUs, timestamp = get_fluorescence_single_experiment(self.image_data.data, self.dimensions, p, chan_sel)

        # print(levels, RPUs)

        self.population_figure.clear()
        ax = self.population_figure.add_subplot(111)

        ax.plot(timestamp, levels, color='blue')
        
        ax2 = ax.twinx()
        ax2.plot(timestamp, RPUs, color='red')
        
        # ax.set_xlim(0, x_axis_limit)
        ax.set_title(f'Fluorescence signal for Position P={p}')
        ax.set_xlabel('T')
        ax.set_ylabel('Signal / RPU')
        self.population_canvas.draw()

    def __plot_fluorescente_signal(self):
        if not hasattr(self, 'image_data'):
            return

        selected_time = self.mapping_controls["time"].currentText()
        max_time = int(selected_time) if selected_time.isdigit() else self.dimensions.get("T", 1) - 1

        full_time_range = self.dimensions.get("T", 1) - 1
        x_axis_limit = full_time_range + 2 

        # Get the current position from the position slider in the population tab
        p = self.slider_p_5.value()
        average_intensities = []

        # Calculate average intensities only up to max_time
        for t in range(max_time + 1): 
            if self.image_data.data.ndim == 4:
                image_data = self.image_data.data[t, p, :, :]
            elif self.image_data.data.ndim == 3: 
                image_data = self.image_data.data[t, :, :]

            # Convert to grayscale if necessary
            if image_data.ndim == 3 and image_data.shape[-1] in [3, 4]:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

            average_intensity = image_data.mean()
            average_intensities.append(average_intensity)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(average_intensities, marker='o')
        
        ax.set_xlim(0, x_axis_limit)
        ax.set_title(f'Average Pixel Intensity for Position P={p}')
        ax.set_xlabel('T')
        ax.set_ylabel('Intensity')
        self.canvas.draw()
