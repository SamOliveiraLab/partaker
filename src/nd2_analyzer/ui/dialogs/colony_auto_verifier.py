from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
                               QWidget, QScrollArea, QFrame, QGroupBox, QSlider)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from nd2_analyzer.ui.biofilms.colony_separator import ColonySeparator

class VerifyColoniesDialog(QDialog):
    colonies_verified = Signal(list)

    def __init__(self, image, colonies, parent=None):
        super().__init__(parent)

        self.image = image
        self.colonies = colonies.copy()
        self.colony_separator = ColonySeparator()
        self.colony_separator.detected_colonies = self.colonies.copy()

        self.setWindowTitle("Verify Colonies")
        self.setMinimumSize(800, 600)
        self.resize(900, 650)

        # build the real UI
        self.init_ui()

        # populate widgets
        self.update_colonies_list()
        self.update_display()
        self.setup_image()

    def init_ui(self):

        # 🔥 MAIN LAYOUT (VERTICAL)
        main_layout = QVBoxLayout(self)

        # Top of the dialog: Image and Colony List
        top_layout = QHBoxLayout()

        # Image Stage
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.setAlignment(Qt.AlignCenter)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        left_layout.addWidget(scroll_area)

        self.status_label = QLabel("Detected colonies shown with bounding boxes")
        self.status_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        left_layout.addWidget(self.status_label)
        top_layout.addWidget(left_widget, 2)

        # Colony List Stage
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setMaximumWidth(300)

        title = QLabel("Detected Colonies")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        right_layout.addWidget(title)
        colonies_group = QFrame()
        colonies_group.setFrameStyle(QFrame.Box)
        colonies_layout = QVBoxLayout(colonies_group)

        self.colonies_count_label = QLabel(f"Colonies: {len(self.colonies)}")
        self.colonies_count_label.setStyleSheet("font-weight: bold;")
        colonies_layout.addWidget(self.colonies_count_label)

        self.colonies_list_widget = QWidget()
        self.colonies_list_layout = QVBoxLayout(self.colonies_list_widget)

        colonies_scroll = QScrollArea()
        colonies_scroll.setWidget(self.colonies_list_widget)
        colonies_scroll.setWidgetResizable(True)
        colonies_scroll.setMaximumHeight(450)

        colonies_layout.addWidget(colonies_scroll)

        right_layout.addWidget(colonies_group)

        top_layout.addWidget(right_widget, 1)

        # add top section
        main_layout.addLayout(top_layout)

        # ---------------- MIDDLE: SLIDERS ----------------
        size_group = QGroupBox("Size Filtering")
        size_layout = QVBoxLayout()
        size_group.setLayout(size_layout)

        # (KEEP YOUR EXISTING SLIDER CODE EXACTLY THE SAME HERE)
        # Min size slider
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("Min Colony Size (px):"))

        self.min_size_slider = QSlider(Qt.Horizontal)
        self.min_size_slider.setMinimum(1)
        self.min_size_slider.setMaximum(100)
        self.min_size_slider.setValue(1)
        self.min_size_slider.valueChanged.connect(self.on_min_size_changed)
        self.min_size_slider.sliderMoved.connect(self.detect_colonies)
        min_size_layout.addWidget(self.min_size_slider)

        self.min_size_label = QLabel(f"{self.min_size_slider.value()}")
        self.min_size_label.setMinimumWidth(50)
        self.min_size_label.setStyleSheet("font-weight: bold;")
        min_size_layout.addWidget(self.min_size_label)
        size_layout.addLayout(min_size_layout)

        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(60)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.threshold_slider.sliderMoved.connect(self.detect_colonies)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel(f"{self.threshold_slider.value()}")
        self.threshold_label.setMinimumWidth(60)
        self.threshold_label.setStyleSheet("font-weight: bold;")
        threshold_layout.addWidget(self.threshold_label)
        size_layout.addLayout(threshold_layout)

        # Kernel slider
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Kernel Size (px):"))

        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(0)
        self.kernel_slider.setMaximum(100)
        self.kernel_slider.setValue(1)
        self.kernel_slider.valueChanged.connect(self.on_kernel_changed)
        self.kernel_slider.sliderMoved.connect(self.detect_colonies)
        kernel_layout.addWidget(self.kernel_slider)

        self.kernel_label = QLabel(f"{self.kernel_slider.value()}")
        self.kernel_label.setMinimumWidth(60)
        self.kernel_label.setStyleSheet("font-weight: bold;")
        kernel_layout.addWidget(self.kernel_label)

        size_layout.addLayout(min_size_layout)
        size_layout.addLayout(threshold_layout)
        size_layout.addLayout(kernel_layout)

        main_layout.addWidget(size_group)

        # ---------------- BOTTOM: BUTTONS ----------------
        button_layout = QHBoxLayout()

        self.ok_btn = QPushButton("Accept Colonies")
        self.ok_btn.clicked.connect(self.accept_colonies)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()  # centers nicely
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

    def on_min_size_changed(self, value):
        """Handle minimum colony size slider change"""
        self.min_size_label.setText(f"{value}%")
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        total_pixels = h * w

        # convert percent → actual pixel area
        min_area = (value / 100.0) * total_pixels * 0.01
        self.colony_separator.update_parameters(min_colony_size=min_area)

    def on_threshold_changed(self, value):
        """Handle maximum colony size slider change"""
        self.threshold_label.setText(f"{value}%")

        # Map slider value to 0-255 range
        threshold = int((value / 100.0) * 255)
        self.colony_separator.update_parameters(intensity_threshold=threshold)

    def on_kernel_changed(self, value):
        """Handle kernel blur size slider change"""
        self.kernel_label.setText(f"{value}%")
        if self.image is None:
            return
        h, w = self.image.shape[:2]

        # Smaller dimensions used for consistency
        base = min(h, w)
        kernel_size = int((value / 100.0) * base)

        # Ensure kernel is odd (OpenCV preference)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.colony_separator.update_parameters(kernel_size=kernel_size)

    def setup_image(self):
        """Setup the image display"""
        if self.image is None:
            self.image_label.setText("No image data")
            return
        # Convert image to displayable format
        if len(self.image.shape) == 2:
            # Grayscale image
            display_image = self.image.copy()
            # Normalize to 0-255
            if display_image.max() > 255 or display_image.dtype != np.uint8:
                display_image = ((display_image - display_image.min()) /
                                 (display_image.max() - display_image.min()) * 255).astype(np.uint8)
        else:
            display_image = self.image_

        self.original_image = display_image
        self.update_display()

    def update_display(self):
        if not hasattr(self, 'original_image'):
            return

        # Start with original image
        display_image = self.original_image.copy()

        # Convert to RGB for overlays
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)

        # Get overlay from colony separator
        overlay = self.colony_separator.create_colony_overlay(display_image.shape[:2])

        # Check if overlay exists and is same size
        if overlay.shape[:2] == display_image.shape[:2]:
            # Blend overlay over the base image
            alpha = 0.6  # transparency factor (0–1)
            display_image = cv2.addWeighted(display_image, 1.0, overlay, alpha, 0)

        # Convert to QPixmap and display
        height, width = display_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)



    def detect_colonies(self):
        """Detect colonies automatically using Otsu thresholding"""
        if self.image is None:
            self.status_label.setText("⚠️ No image loaded. Please view an image first.")
            return
        print(f"Detecting colonies from image shape: {self.image.shape}")

        # Detect colonies using Otsu + Triangle method (openCV)
        colonies = self.colony_separator.detect_colonies_otsu(self.image)
        if len(colonies) > 0:
            self.status_label.setText(f"✅ Detected {len(colonies)} colonies automatically.")

            # Automatically update colony list and display
            self.colonies = colonies
            self.colony_separator.detected_colonies = colonies
            self.update_colonies_list()
            self.update_display()

            # Print colony info
            for colony in colonies:
                print(f"Colony {colony['colony_id']}: Area={colony['area']:.0f}px²")
        else:
            self.status_label.setText("No colonies detected. Try adjusting the threshold.")


    def update_colonies_list(self):
        """Update the colony list widget with the current colonies."""
        for i in reversed(range(self.colonies_list_layout.count())):
            child = self.colonies_list_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        # Set widget for colony list
        for i, colony in enumerate(self.colonies):
            colony_widget = QWidget()
            colony_layout = QHBoxLayout(colony_widget)
            colony_layout.setContentsMargins(5, 2, 5, 2)
            label = QLabel(f"Colony {colony['colony_id']}")

            # Set Delete button widget
            delete_btn = QPushButton("Delete")
            delete_btn.setMaximumWidth(60)
            delete_btn.setStyleSheet(
                "background-color: #f44336; color: white; font-size: 10px;"
            )
            delete_btn.clicked.connect(lambda checked, idx=i: self.delete_colony(idx))

            colony_layout.addWidget(label)
            colony_layout.addWidget(delete_btn)
            self.colonies_list_layout.addWidget(colony_widget)

        self.colonies_count_label.setText(f"Colonies: {len(self.colonies)}")

    def accept_colonies(self):
        """Accept the selected colonies and emit signal"""
        self.colonies_verified.emit(self.colonies)
        self.accept()

    def delete_colony(self, index):
        """Delete colony at given index and update colony IDs"""
        if 0 <= index < len(self.colonies):
            del self.colonies[index]
            for i, colony in enumerate(self.colonies):
                colony["colony_id"] = i + 1

            self.update_colonies_list()
            self.update_display()