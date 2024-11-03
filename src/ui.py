from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QButtonGroup, QTabWidget, QWidget, QLabel, QPushButton, QFileDialog, QScrollArea, QSlider, QHBoxLayout, QCheckBox, QMessageBox, QRadioButton
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy

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
from segmentation import segment_this_image
from image_functions import remove_stage_jitter_MAE

# import pims
from matplotlib.backends.backend_qt5agg import FigureCanvas

"""
Can hold either an ND2 file or a series of images
"""
class ImageData:
    def __init__(self, data, is_nd2=False):
        self.data = data
        self.is_nd2 = is_nd2

class TabWidgetApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Partaker 3 - GUI')
        self.setGeometry(100, 100, 1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget()

        # Create tabs
        self.importTab = QWidget()
        self.viewTab = QWidget()
        self.exportTab = QWidget()
        self.populationTab = QWidget()

        # self.align_tab = QWidget()

        self.initUI()

        # Add the tab widget to the layout
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

            self.info_label.setText(info_text)
            self.image_data = ImageData(nd2.imread(file_path, dask=True), is_nd2=True)  # Load the image data once
            
            self.image_data.nd2_dimensions = nd2_file.sizes
            
            self.update_controls()
            self.display_image()  # Display the first image
            self.plot_average_intensity()
            # self.display_thresholded_image()

    def display_file_info(self, file_path):
        info_text = f"Number of dimensions: {len(self.dimensions)}\n"
        for dim, size in self.dimensions.items():
            info_text += f"{dim}: {size}\n"
        self.info_label.setText(info_text)

    def update_controls(self):
        self.slider_t.setMaximum(self.dimensions.get('T', 1) - 1)
        self.slider_p.setMaximum(self.dimensions.get('P', 1) - 1)
        self.slider_p_5.setMaximum(self.dimensions.get('P', 1) - 1)

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
        
        image_data = self.image_data.data

        if self.image_data.is_nd2:
            image_data = image_data[t, p] # Assuming the dimensions are (T, P, Y, X, C)
        else:
            image_data = image_data[t]

        # np conversion    
        image_data = image_data.compute().data

        if self.radio_thresholding.isChecked():
            threshold = self.threshold_slider.value()
            image_data = cv2.threshold(image_data, threshold, 255, cv2.THRESH_BINARY)[1]

            # Convert from dask to numpy
            image_data = image_data.compute()

            # Convert to grayscale if necessary
            if image_data.ndim == 3 and image_data.shape[-1] in [3, 4]:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

            # Apply Otsu's thresholding
            _, thresholded_image = cv2.threshold(image_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.radio_segmented.isChecked():
            image_data = segment_this_image(image_data)
            self.show_cell_area(image_data)

        image_format = QImage.Format_Grayscale16
        height, width = image_data.shape[:2]

        # Convert the image data to QImage
        image = QImage(image_data, width, height, image_format)
        pixmap = QPixmap.fromImage(image)

        # Scale the pixmap to fit the label size
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Display the scaled image in the label
        self.image_label.setPixmap(scaled_pixmap)

    def align_images(self):
        # Check if we have a dataset loaded

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
        
        # Message box when loaded
        QMessageBox.about(self, "Alignment", f"Alignment completed successfully. {stage_MAE_scores}")

        # Load into aligned images

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

        button = QPushButton("Select File / Folder")
        button.clicked.connect(lambda : importFile() if not self.is_folder_checkbox.isChecked() else importFolder())
        layout.addWidget(button)

        checkbox = QCheckBox("Load from folder?")
        layout.addWidget(checkbox)
        self.is_folder_checkbox = checkbox

        self.filename_label = QLabel("Filename will be shown here.")
        layout.addWidget(self.filename_label)

        self.info_label = QLabel("File info will be shown here.")
        layout.addWidget(self.info_label)

        # self.figure = plt.figure()
        # self.canvas = FigureCanvas(self.figure)
        # layout.addWidget(self.canvas)

        # # Just a hint for the user
        # ax = self.figure.add_subplot(111, projection='3d')
        # ax.text(0.5, 0.5, 0.5, "Select file first", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # self.canvas.draw()

    def initViewTab(self):
        layout = QVBoxLayout(self.viewTab)
        # label = QLabel("Content of Tab 2")
        # layout.addWidget(label)

        self.image_label = QLabel()
        self.image_label.setScaledContents(True)  # Allow the label to scale the image
        layout.addWidget(self.image_label)

        # Another label for aligned images
        self.aligned_image_label = QLabel()
        self.aligned_image_label.setScaledContents(True)  # Allow the label to scale the image
        layout.addWidget(self.aligned_image_label)

        # Align button
        align_button = QPushButton("Align Images")
        align_button.clicked.connect(self.align_images)
        layout.addWidget(align_button)

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

    def initExportTab(self):
        def exportData():
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(filter="Video Files (*.mp4 *.avi)")
            if file_path:
                self.save_video(file_path)

        layout = QVBoxLayout(self.exportTab)
        label = QLabel("Coming soon!")
        layout.addWidget(label)

        export_button = QPushButton("Export")
        export_button.clicked.connect(exportData)
        layout.addWidget(export_button)

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
        self.tab_widget.addTab(self.viewTab, "View")

        # self.tab_widget.addTab(self.align_tab, "Align")

        self.tab_widget.addTab(self.exportTab, "Export")
        self.tab_widget.addTab(self.populationTab, "Population")

        self.initImportTab()
        self.initViewTab()
        self.initExportTab()
        self.initPopulationTab()

    def initPopulationTab(self):
        layout = QVBoxLayout(self.populationTab)
        label = QLabel("Average Pixel Intensity")
        layout.addWidget(label)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # P controls
        p_layout = QHBoxLayout()
        p_label = QLabel("P: nan")
        p_layout.addWidget(p_label)
        self.slider_p_5 = QSlider(Qt.Horizontal)
        self.slider_p_5.valueChanged.connect(self.plot_average_intensity)
        # Display the slider value
        self.slider_p_5.valueChanged.connect(lambda value: p_label.setText(f'P: {value}'))

        p_layout.addWidget(self.slider_p_5)

        layout.addLayout(p_layout)

        self.plot_average_intensity()

    def plot_average_intensity(self):
        if not hasattr(self, 'image_data'):
            return

        p = self.slider_p_5.value()
        average_intensities = []
        for t in range(self.dimensions.get('T', 1)):
            if self.image_data.data.ndim == 4:  # Assuming the dimensions are (T, P, Y, X, C)
                image_data = self.image_data.data[t, p, :, :]
            elif self.image_data.data.ndim == 3:  # Assuming the dimensions are (T, Y, X, C)
                image_data = self.image_data.data[t, :, :]

            # Convert to grayscale if necessary
            if image_data.ndim == 3 and image_data.shape[-1] in [3, 4]:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

            average_intensity = image_data.mean()
            average_intensities.append(average_intensity)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(average_intensities, marker='o')
        ax.set_title(f'Average Pixel Intensity for P={p}')
        ax.set_xlabel('T')
        ax.set_ylabel('Intensity')
        self.canvas.draw()
