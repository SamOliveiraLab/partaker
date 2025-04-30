import os
import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QSlider, QGroupBox, QRadioButton, QButtonGroup, 
                              QSizePolicy, QComboBox, QCheckBox)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QImage
import numpy as np
from pubsub import pub
import cv2

from segmentation.segmentation_models import SegmentationModels

class ViewAreaWidget(QWidget):
    """
    A standalone widget for viewing and controlling ND2 image data with segmentation options.
    Uses PyPubSub for communication with other components.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize state variables
        self.current_t = 0
        self.current_p = 0
        self.current_c = 0
        self.current_mode = "normal"  # normal, segmented, or labeled
        self.current_model = None
        
        # Set up the UI
        self.init_ui()
        
        # Subscribe to relevant topics
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_image_ready, "image_ready")
        pub.subscribe(self.highlight_cell, "highlight_cell_requested")
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.image_label)
        
        # ND2 Controls Group
        nd2_controls_group = QGroupBox("ND2 Controls")
        nd2_controls_layout = QVBoxLayout()
        
        # T controls (Time)
        t_layout = QHBoxLayout()
        self.t_label = QLabel("T: 0")
        t_layout.addWidget(self.t_label)
        
        self.t_left_button = QPushButton("<")
        self.t_left_button.clicked.connect(
            lambda: self.slider_t.setValue(self.slider_t.value() - 1)
        )
        t_layout.addWidget(self.t_left_button)
        
        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.setMinimum(0)
        self.slider_t.setMaximum(0)  # Will be updated when experiment is loaded
        self.slider_t.valueChanged.connect(self.on_slider_changed)
        self.slider_t.valueChanged.connect(
            lambda value: self.t_label.setText(f"T: {value}")
        )
        t_layout.addWidget(self.slider_t)
        
        self.t_right_button = QPushButton(">")
        self.t_right_button.clicked.connect(
            lambda: self.slider_t.setValue(self.slider_t.value() + 1)
        )
        t_layout.addWidget(self.t_right_button)
        
        nd2_controls_layout.addLayout(t_layout)
        
        # P controls (Position)
        p_layout = QHBoxLayout()
        self.p_label = QLabel("P: 0")
        p_layout.addWidget(self.p_label)
        
        self.p_left_button = QPushButton("<")
        self.p_left_button.clicked.connect(
            lambda: self.slider_p.setValue(self.slider_p.value() - 1)
        )
        p_layout.addWidget(self.p_left_button)
        
        self.slider_p = QSlider(Qt.Horizontal)
        self.slider_p.setMinimum(0)
        self.slider_p.setMaximum(0)  # Will be updated when experiment is loaded
        self.slider_p.valueChanged.connect(self.on_slider_changed)
        self.slider_p.valueChanged.connect(
            lambda value: self.p_label.setText(f"P: {value}")
        )
        p_layout.addWidget(self.slider_p)
        
        self.p_right_button = QPushButton(">")
        self.p_right_button.clicked.connect(
            lambda: self.slider_p.setValue(self.slider_p.value() + 1)
        )
        p_layout.addWidget(self.p_right_button)
        
        nd2_controls_layout.addLayout(p_layout)
        
        # C controls (Channel)
        c_layout = QHBoxLayout()
        self.c_label = QLabel("C: 0")
        c_layout.addWidget(self.c_label)
        
        self.c_left_button = QPushButton("<")
        self.c_left_button.clicked.connect(
            lambda: self.slider_c.setValue(self.slider_c.value() - 1)
        )
        c_layout.addWidget(self.c_left_button)
        
        self.slider_c = QSlider(Qt.Horizontal)
        self.slider_c.setMinimum(0)
        self.slider_c.setMaximum(0)  # Will be updated when experiment is loaded
        self.slider_c.valueChanged.connect(self.on_slider_changed)
        self.slider_c.valueChanged.connect(
            lambda value: self.c_label.setText(f"C: {value}")
        )
        c_layout.addWidget(self.slider_c)
        
        self.c_right_button = QPushButton(">")
        self.c_right_button.clicked.connect(
            lambda: self.slider_c.setValue(self.slider_c.value() + 1)
        )
        c_layout.addWidget(self.c_right_button)
        nd2_controls_layout.addLayout(c_layout)

        # Add normalization checkbox
        self.normalize_checkbox = QCheckBox("Normalize Image")
        self.normalize_checkbox.setChecked(True)  # Default to checked
        self.normalize_checkbox.stateChanged.connect(self.on_slider_changed)  # Refresh image when changed
        nd2_controls_layout.addWidget(self.normalize_checkbox)
        
        nd2_controls_group.setLayout(nd2_controls_layout)
        layout.addWidget(nd2_controls_group)
        
        # Display Mode Controls
        display_mode_group = QGroupBox("Display Mode")
        display_mode_layout = QVBoxLayout()
        
        self.radio_normal = QRadioButton("Normal")
        self.radio_segmented = QRadioButton("Segmented")
        self.radio_overlay_outlines = QRadioButton("Overlay with Outlines")
        self.radio_labeled_segmentation = QRadioButton("Labeled Segmentation")
        
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_normal)
        self.button_group.addButton(self.radio_segmented)
        self.button_group.addButton(self.radio_overlay_outlines)
        self.button_group.addButton(self.radio_labeled_segmentation)
        self.button_group.buttonClicked.connect(self.on_display_mode_changed)
        
        # Set default selection
        self.radio_normal.setChecked(True)
        
        display_mode_layout.addWidget(self.radio_normal)
        display_mode_layout.addWidget(self.radio_segmented)
        display_mode_layout.addWidget(self.radio_overlay_outlines)
        display_mode_layout.addWidget(self.radio_labeled_segmentation)
        
        display_mode_group.setLayout(display_mode_layout)
        layout.addWidget(display_mode_group)
        
        # Segmentation Model Selection
        model_group = QGroupBox("Segmentation Model")
        model_layout = QVBoxLayout()
        
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([
            SegmentationModels.CELLPOSE_BACT_PHASE,
            SegmentationModels.CELLPOSE_BACT_FLUOR,
            SegmentationModels.CELLPOSE,
            SegmentationModels.UNET
        ])
        self.model_dropdown.currentTextChanged.connect(self.on_model_changed)
        self.current_model = self.model_dropdown.currentText()
        
        model_layout.addWidget(self.model_dropdown)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Set the layout
        self.setLayout(layout)
    
    def show_context_menu(self, position):
        """Show context menu for the image label"""
        # This is a placeholder - implement as needed
        pass

    @Slot()
    def on_slider_changed(self):
        """Handle slider value changes and request appropriate image"""
        # Update current values
        self.current_t = self.slider_t.value()
        self.current_p = self.slider_p.value()
        self.current_c = self.slider_c.value()

        # Send different requests based on display mode
        if self.current_mode == "normal":
            pub.sendMessage("raw_image_request",
                           time=self.current_t,
                           position=self.current_p,
                           channel=self.current_c)
        else:
            pub.sendMessage("segmented_image_request",
                           time=self.current_t,
                           position=self.current_p,
                           channel=self.current_c,
                           mode=self.current_mode,
                           model=self.current_model)

    def on_image_ready(self, image, time, position, channel, mode):
        """Handle both raw and segmented image responses"""
        # Check if this is the image we're currently expecting
        if (time != self.current_t or 
            position != self.current_p or 
            channel != self.current_c or 
            mode != self.current_mode):
            return  # Ignore outdated images

        # Convert and display based on mode
        if mode == "normal":
            self._display_raw_image(image)
        else:
            self._display_processed_image(image, mode)

    def _display_raw_image(self, image):
        """Display raw microscope image"""
        # Make a copy to avoid modifying the original
        self.current_image_data = image.copy()
        
        # Handle grayscale images
        if len(image.shape) == 2:
            # Apply normalization if checkbox is checked
            if self.normalize_checkbox.isChecked():
                if image.dtype != np.uint16:
                    image = (image.astype(np.float32) / image.max() * 65535).astype(np.uint16)
                
                image = cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
            else:
                # If not normalizing, ensure the image is in a suitable format
                if image.dtype != np.uint16:
                    # Scale to 16-bit range while preserving relative values
                    image = image.astype(np.uint16)
            
            height, width = image.shape
            bytes_per_line = image.strides[0]
            fmt = QImage.Format_Grayscale16
        
        else:  # Color
            # Apply normalization if checkbox is checked
            if self.normalize_checkbox.isChecked() and image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            height, width, _ = image.shape
            bytes_per_line = 3 * width
            fmt = QImage.Format_RGB888
        
        q_image = QImage(image.data, width, height, bytes_per_line, fmt)
        self._update_display(q_image)

    def _display_processed_image(self, image, mode):
        """Display processed segmentation results"""
        # Make a copy to avoid modifying the original
        image = image.copy()
        
        if mode == "segmented":
            # Segmented images are typically binary masks (single channel)
            if len(image.shape) == 2:
                # Apply normalization if checkbox is checked
                if self.normalize_checkbox.isChecked():
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    # Ensure image is uint8 for display
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)
                
                height, width = image.shape
                bytes_per_line = width
                fmt = QImage.Format_Grayscale8
            else:
                # If somehow the segmented image is multi-channel, convert to grayscale
                if self.normalize_checkbox.isChecked():
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                height, width = image.shape
                bytes_per_line = width
                fmt = QImage.Format_Grayscale8
        
        elif mode in ["overlay", "labeled"]:
            # Overlay and labeled images should be RGB
            if len(image.shape) == 2:
                # Convert single channel to RGB
                if self.normalize_checkbox.isChecked():
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Ensure image is uint8 RGB
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            height, width, channels = image.shape
            bytes_per_line = channels * width
            fmt = QImage.Format_RGB888
        
        else:
            # Unsupported mode
            return
        
        # Create QImage and display it
        q_image = QImage(image.data, width, height, bytes_per_line, fmt)
        self._update_display(q_image)

    def _update_display(self, q_image):
        """Common display update logic"""
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        
        # Store the current image for highlighting
        if hasattr(self, "current_image_data"):
            self.current_image = self.current_image_data.copy()
        
    
    @Slot(object)
    def on_display_mode_changed(self, button):
        """Handle display mode changes"""
        if button == self.radio_normal:
            self.current_mode = "normal"
        elif button == self.radio_segmented:
            self.current_mode = "segmented"
        elif button == self.radio_overlay_outlines:
            self.current_mode = "overlay"
        elif button == self.radio_labeled_segmentation:
            self.current_mode = "labeled"
        
        # Request new image with updated mode
        self.on_slider_changed()
    
    @Slot(str)
    def on_model_changed(self, model_name):
        """Handle segmentation model changes"""
        self.current_model = model_name
        
        # If in a segmentation mode, request new image with updated model
        if self.current_mode in ["segmented", "overlay", "labeled"]:
            self.on_slider_changed()
    
    def on_image_data_loaded(self, image_data):
        """
        Handle new image_data loading.
        Updates slider ranges based on image_data dimensions.
        
        Args:
            image_data: The loaded image_data object
        """
        # Get dimensions from image_data
        _shape = image_data.data.shape
        t_max = _shape[0] - 1
        p_max = _shape[1] - 1
        
        print(f"Shape: {_shape}, Length: {len(_shape)}")
        
        if len(_shape) == 5:  # T, P, C, Y, X format
            c_max = _shape[2] - 1
            self.has_channels = True
        elif len(_shape) == 4:  # T, P, Y, X format (no channels)
            c_max = 0  # Default to single channel
            self.has_channels = False
        else:
            c_max = 0
            self.has_channels = False
        
        # Update slider ranges
        self.slider_t.setMaximum(max(0, t_max))
        self.slider_p.setMaximum(max(0, p_max))
        self.slider_c.setMaximum(max(0, c_max))
        
        # Reset slider positions
        self.slider_t.setValue(0)
        self.slider_p.setValue(0)
        self.slider_c.setValue(0)
        
        # Request initial image
        self.on_slider_changed()
    
    def apply_label_colormap(self, labeled_image):
        """
        Apply a colormap to a labeled segmentation image.
        
        Args:
            labeled_image: Labeled segmentation image
            
        Returns:
            Colored image suitable for display
        """
        # Normalize to 0-255 range
        if labeled_image.max() > 0:
            normalized = (labeled_image.astype(np.float32) / labeled_image.max() * 255).astype(np.uint8)
        else:
            normalized = labeled_image.astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Set background (label 0) to black
        colored[labeled_image == 0] = [0, 0, 0]
        
        return colored

    # TODO: implement highlight cell
    def highlight_cell(self, cell_id):
        """
        Highlight a specific cell in the current image view
        
        Args:
            cell_id: ID of the cell to highlight
        """
        print(f"ViewAreaWidget: Highlighting cell {cell_id}")
        
        # Get current frame information
        t = self.current_t
        p = self.current_p
        c = self.current_c
        
        # Get the metrics service
        metrics_service = None
        def get_service(service):
            nonlocal metrics_service
            metrics_service = service
        pub.sendMessage("get_metrics_service", callback=get_service)
        
        if not metrics_service:
            print("Metrics service not available")
            return
        
        # Query the metrics service for this cell
        df = metrics_service.query(time=t, position=p, channel=c, cell_id=cell_id)
        
        if df.is_empty():
            print(f"Cell {cell_id} not found in metrics service")
            return
        
        # Convert to pandas for easy access
        cell_data = df.to_pandas()
        
        # Check if we have a current image to work with
        if not hasattr(self, "current_image") or self.current_image is None:
            print("No current image available to highlight cell")
            return
        
        # Make a copy of the current image for highlighting
        highlighted_image = self.current_image.copy()
        
        # Get the bounding box or use centroid to create one
        if all(col in cell_data.columns for col in ['y1', 'x1', 'y2', 'x2']):
            # Use bounding box if available
            y1 = int(cell_data['y1'].iloc[0])
            x1 = int(cell_data['x1'].iloc[0])
            y2 = int(cell_data['y2'].iloc[0])
            x2 = int(cell_data['x2'].iloc[0])
        elif 'centroid_y' in cell_data.columns and 'centroid_x' in cell_data.columns:
            # Create bounding box from centroid
            cy = int(cell_data['centroid_y'].iloc[0])
            cx = int(cell_data['centroid_x'].iloc[0])
            # Create a box around the centroid
            box_size = 20  # Adjust based on your cell size
            y1, x1 = cy - box_size, cx - box_size
            y2, x2 = cy + box_size, cx + box_size
        else:
            print(f"Cell {cell_id} has no position information")
            return
        
        # Draw a highlighted rectangle around the cell
        cv2.rectangle(
            highlighted_image, 
            (x1, y1), 
            (x2, y2), 
            (0, 0, 255),  # Red color (BGR format)
            2             # Line thickness
        )
        
        # Add a text label with the cell ID
        cv2.putText(
            highlighted_image,
            f"Cell {cell_id}",
            (x1, y1 - 5),  # Position above the cell
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,           # Font scale
            (0, 255, 0),   # Green color (BGR)
            1              # Line thickness
        )
        
        # Display the highlighted image
        if len(highlighted_image.shape) == 2:
            # Grayscale image
            height, width = highlighted_image.shape
            bytes_per_line = width
            fmt = QImage.Format_Grayscale8
        else:
            # Color image
            height, width, _ = highlighted_image.shape
            bytes_per_line = 3 * width
            fmt = QImage.Format_RGB888
        
        # Convert to QImage and display
        q_image = QImage(highlighted_image.data, width, height, bytes_per_line, fmt)
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        
        # Store the highlighted image
        self.highlighted_image = highlighted_image