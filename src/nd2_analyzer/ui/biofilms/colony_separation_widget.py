"""
Colony Separation Widget - Separate tab for manual and automatic colony detection
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QTabWidget, QGroupBox, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt
from pubsub import pub
import numpy as np

from .colony_separator import ColonySeparator


class ColonySeparationWidget(QWidget):
    """Widget for colony separation with manual and automatic detection modes"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize colony separator
        self.colony_separator = ColonySeparator()
        self.current_raw_image = None
        self.colony_overlay_visible = False

        # Subscribe to image updates
        pub.subscribe(self.on_image_ready, "image_ready")

        self.init_ui()

    def init_ui(self):
        """Initialize the UI with two sub-tabs"""
        main_layout = QVBoxLayout(self)

        # Create tab widget for Manual vs Automatic
        self.tab_widget = QTabWidget()

        # Manual Selection Tab
        manual_tab = self.create_manual_selection_tab()
        self.tab_widget.addTab(manual_tab, "Manual Selection")

        # Automatic Thresholding Tab
        threshold_tab = self.create_threshold_tab()
        self.tab_widget.addTab(threshold_tab, "Automatic Detection")

        main_layout.addWidget(self.tab_widget)

        # Common controls at the bottom
        self.add_common_controls(main_layout)

        main_layout.addStretch()

    def create_manual_selection_tab(self):
        """Create the manual selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header
        header = QLabel("Manual Colony Selection")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Use manual selection to draw polygons around colonies.\n"
            "This is useful for fluorescence channels or complex colony shapes."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(instructions)

        # Manual selection button
        self.start_manual_btn = QPushButton("Start Manual Selection")
        self.start_manual_btn.clicked.connect(self.start_manual_selection)
        self.start_manual_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 10px;"
        )
        layout.addWidget(self.start_manual_btn)

        # Info label
        info_label = QLabel(
            "Click 'Start Manual Selection' to open the ROI selector.\n"
            "Draw polygons around each colony you want to track."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-style: italic; color: #999; padding: 10px;")
        layout.addWidget(info_label)

        layout.addStretch()
        return tab

    def create_threshold_tab(self):
        """Create the automatic thresholding tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header
        header = QLabel("Automatic Colony Detection (Otsu)")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Automatically detect colonies using Otsu thresholding.\n"
            "Works best with fluorescence images (GFP, RFP, etc.)."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(instructions)

        # Morphology controls group
        morphology_group = QGroupBox("Morphological Parameters")
        morphology_layout = QVBoxLayout()

        # Closing radius slider
        closing_layout = QHBoxLayout()
        closing_layout.addWidget(QLabel("Closing Radius:"))

        self.closing_radius_slider = QSlider(Qt.Horizontal)
        self.closing_radius_slider.setMinimum(0)
        self.closing_radius_slider.setMaximum(10)
        self.closing_radius_slider.setValue(1)
        self.closing_radius_slider.valueChanged.connect(self.on_closing_radius_changed)
        closing_layout.addWidget(self.closing_radius_slider)

        self.closing_radius_label = QLabel("1")
        self.closing_radius_label.setMinimumWidth(30)
        self.closing_radius_label.setStyleSheet("font-weight: bold;")
        closing_layout.addWidget(self.closing_radius_label)

        morphology_layout.addLayout(closing_layout)

        # Min object size slider
        min_obj_layout = QHBoxLayout()
        min_obj_layout.addWidget(QLabel("Min Object Size:"))

        self.min_object_slider = QSlider(Qt.Horizontal)
        self.min_object_slider.setMinimum(1)
        self.min_object_slider.setMaximum(100)
        self.min_object_slider.setValue(6)
        self.min_object_slider.valueChanged.connect(self.on_min_object_changed)
        min_obj_layout.addWidget(self.min_object_slider)

        self.min_object_label = QLabel("6")
        self.min_object_label.setMinimumWidth(40)
        self.min_object_label.setStyleSheet("font-weight: bold;")
        min_obj_layout.addWidget(self.min_object_label)

        morphology_layout.addLayout(min_obj_layout)

        morphology_group.setLayout(morphology_layout)
        layout.addWidget(morphology_group)

        # Size filtering group
        size_group = QGroupBox("Size Filtering")
        size_layout = QVBoxLayout()

        # Min size slider
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("Min Colony Size (px):"))

        self.min_size_slider = QSlider(Qt.Horizontal)
        self.min_size_slider.setMinimum(100)
        self.min_size_slider.setMaximum(10000)
        self.min_size_slider.setValue(1000)
        self.min_size_slider.valueChanged.connect(self.on_min_size_changed)
        min_size_layout.addWidget(self.min_size_slider)

        self.min_size_label = QLabel("1000")
        self.min_size_label.setMinimumWidth(50)
        self.min_size_label.setStyleSheet("font-weight: bold;")
        min_size_layout.addWidget(self.min_size_label)

        size_layout.addLayout(min_size_layout)

        # Max size slider
        max_size_layout = QHBoxLayout()
        max_size_layout.addWidget(QLabel("Max Colony Size (px):"))

        self.max_size_slider = QSlider(Qt.Horizontal)
        self.max_size_slider.setMinimum(10000)
        self.max_size_slider.setMaximum(500000)
        self.max_size_slider.setValue(100000)
        self.max_size_slider.valueChanged.connect(self.on_max_size_changed)
        max_size_layout.addWidget(self.max_size_slider)

        self.max_size_label = QLabel("100000")
        self.max_size_label.setMinimumWidth(60)
        self.max_size_label.setStyleSheet("font-weight: bold;")
        max_size_layout.addWidget(self.max_size_label)

        size_layout.addLayout(max_size_layout)
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # Detect button
        self.detect_btn = QPushButton("Detect Colonies")
        self.detect_btn.clicked.connect(self.detect_colonies)
        self.detect_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;"
        )
        layout.addWidget(self.detect_btn)

        layout.addStretch()
        return tab

    def add_common_controls(self, layout):
        """Add common controls that apply to both modes"""
        # Separator line
        separator = QLabel()
        separator.setFrameStyle(QLabel.HLine | QLabel.Sunken)
        layout.addWidget(separator)

        # Colony count display
        self.colony_count_label = QLabel("Colonies: 0")
        self.colony_count_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #2196F3; padding: 10px;"
        )
        layout.addWidget(self.colony_count_label)

        # Control buttons
        button_layout = QHBoxLayout()

        self.show_overlay_btn = QPushButton("Show/Hide Overlay")
        self.show_overlay_btn.clicked.connect(self.toggle_colony_overlay)
        self.show_overlay_btn.setEnabled(False)
        button_layout.addWidget(self.show_overlay_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all_colonies)
        self.clear_btn.setStyleSheet("background-color: #f44336; color: white;")
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("Load an image to begin colony separation.")
        self.status_label.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # === Event Handlers ===

    def on_image_ready(self, image, time, position, channel, mode):
        """Handle image ready event"""
        if mode == "normal":
            # Store the raw image for colony detection
            self.current_raw_image = image
            self.status_label.setText(f"Image loaded: T={time}, P={position}, C={channel}")

    def on_closing_radius_changed(self, value):
        """Handle closing radius slider change"""
        self.closing_radius_label.setText(str(value))
        self.colony_separator.update_parameters(closing_radius=value)

    def on_min_object_changed(self, value):
        """Handle min object size slider change"""
        self.min_object_label.setText(str(value))
        self.colony_separator.update_parameters(min_object_size=value)

    def on_min_size_changed(self, value):
        """Handle minimum colony size slider change"""
        self.min_size_label.setText(str(value))
        self.colony_separator.update_parameters(min_colony_size=value)

    def on_max_size_changed(self, value):
        """Handle maximum colony size slider change"""
        self.max_size_label.setText(str(value))
        self.colony_separator.update_parameters(max_colony_size=value)

    def detect_colonies(self):
        """Detect colonies automatically using Otsu thresholding"""
        if self.current_raw_image is None:
            self.status_label.setText("⚠️ No image loaded. Please view an image first.")
            return

        print(f"Detecting colonies using Otsu from image shape: {self.current_raw_image.shape}")

        # Detect colonies using Otsu method (from notebook)
        colonies = self.colony_separator.detect_colonies_otsu(self.current_raw_image)

        # Update UI
        self.colony_count_label.setText(f"Colonies: {len(colonies)}")
        self.show_overlay_btn.setEnabled(len(colonies) > 0)

        if len(colonies) > 0:
            self.status_label.setText(f"✅ Detected {len(colonies)} colonies automatically.")

            # Automatically show overlay
            self.colony_overlay_visible = True
            self.update_colony_overlay()

            # Print colony info
            for colony in colonies:
                print(f"Colony {colony['colony_id']}: Area={colony['area']:.0f}px²")
        else:
            self.status_label.setText("No colonies detected. Try adjusting the threshold.")

    def start_manual_selection(self):
        """Start manual colony selection"""
        if self.current_raw_image is None:
            self.status_label.setText("⚠️ No image loaded. Please view an image first.")
            return

        from nd2_analyzer.ui.dialogs.colony_roi_selector import ColonyROISelector

        # Get existing colonies
        existing_colonies = []
        for colony in self.colony_separator.get_all_colonies():
            if 'polygon_points' in colony:
                existing_colonies.append({
                    'colony_id': colony['colony_id'],
                    'polygon': colony['polygon_points'],
                    'mask': None
                })

        # Open ROI selector dialog
        roi_dialog = ColonyROISelector(
            self.current_raw_image,
            existing_colonies=existing_colonies,
            parent=self
        )
        roi_dialog.colonies_selected.connect(self.handle_selected_colonies)

        # Update UI
        self.start_manual_btn.setEnabled(False)
        self.status_label.setText("ROI Selector opened. Draw polygons around colonies.")

        # Show dialog
        result = roi_dialog.exec()

        # Reset UI
        self.start_manual_btn.setEnabled(True)

        if result:
            colonies_count = len(self.colony_separator.get_all_colonies())
            self.status_label.setText(f"✅ Manual selection complete: {colonies_count} colonies.")
        else:
            self.status_label.setText("Manual selection cancelled.")

    def handle_selected_colonies(self, colonies_data):
        """Handle colonies selected from ROI selector"""
        print(f"Received {len(colonies_data)} colonies from ROI selector")

        # Clear existing manual additions
        self.colony_separator.manual_additions = []

        # Add each colony
        for colony_data in colonies_data:
            polygon = colony_data['polygon']
            self.colony_separator.add_manual_colony(polygon, self.current_raw_image.shape)

        # Update display
        total_colonies = len(self.colony_separator.get_all_colonies())
        self.colony_count_label.setText(f"Colonies: {total_colonies}")
        self.show_overlay_btn.setEnabled(total_colonies > 0)

        # Show overlay
        self.colony_overlay_visible = True
        self.update_colony_overlay()

    def toggle_colony_overlay(self):
        """Toggle colony overlay visibility"""
        self.colony_overlay_visible = not self.colony_overlay_visible
        self.update_colony_overlay()

    def update_colony_overlay(self):
        """Update colony overlay in the view"""
        colonies = self.colony_separator.get_all_colonies()

        if self.current_raw_image is not None and self.colony_overlay_visible and len(colonies) > 0:
            # Create overlay
            overlay = self.colony_separator.create_colony_overlay(self.current_raw_image.shape)
            pub.sendMessage("show_colony_overlay", overlay=overlay)
        else:
            # Hide overlay
            pub.sendMessage("hide_colony_overlay")

    def clear_all_colonies(self):
        """Clear all detected colonies"""
        self.colony_separator.detected_colonies = []
        self.colony_separator.manual_additions = []
        self.colony_count_label.setText("Colonies: 0")
        self.show_overlay_btn.setEnabled(False)
        self.colony_overlay_visible = False
        self.update_colony_overlay()
        self.status_label.setText("All colonies cleared.")
