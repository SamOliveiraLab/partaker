import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QSizePolicy,
    QComboBox,
    QCheckBox,
)
from pubsub import pub

from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
from nd2_analyzer.data.image_data import ImageData
from nd2_analyzer.utils.image_functions import normalize_image


class ViewAreaWidget(QWidget):
    """
    Widget for viewing and controlling ND2 image data with segmentation options.
    Uses PyPubSub for communication with other components.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize state variables
        self.current_t = 0
        self.current_p = 0
        self.current_c = 0
        self.current_overlay_channel = 0  # Channel for overlay mode
        self.current_mode = "normal"
        self.current_model = None
        self.num_channels = 0
        self.current_image_data = None

        # Set up the UI
        self.init_ui()

        # Subscribe to relevant topics
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_image_ready, "image_ready")
        pub.subscribe(self.highlight_cell, "highlight_cell_requested")
        pub.subscribe(self.provide_current_param, "get_current_t")
        pub.subscribe(self.provide_current_param, "get_current_p")
        pub.subscribe(self.provide_current_param, "get_current_c")
        pub.subscribe(self.on_segmentation_cache_miss, "segmentation_cache_miss")

    # CENTRALIZED IMAGE PROCESSING FUNCTIONS

    def _prepare_image_for_display(self, image, normalize=True):
        """
        Standardize image format for Qt display.

        Returns:
            tuple: (processed_image, width, height, bytes_per_line, qt_format)
        """
        img = image.copy()

        if normalize:
            img = normalize_image(img)

        # Handle grayscale images
        if len(img.shape) == 2:
            if img.dtype != np.uint16:
                if normalize:
                    img = normalize_image(img).astype(np.uint16)
                else:
                    if img.max() <= 255:
                        img = (img.astype(np.float32) * 257).astype(np.uint16)
                    else:
                        img = img.astype(np.uint16)

            height, width = img.shape
            bytes_per_line = width * 2
            qt_format = QImage.Format_Grayscale16

        # Handle color images
        else:
            if img.dtype != np.uint8:
                img = normalize_image(img) if normalize else np.clip(img, 0, 255).astype(np.uint8)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            height, width, channels = img.shape
            bytes_per_line = width * channels
            qt_format = QImage.Format_RGB888

        return img, width, height, bytes_per_line, qt_format

    def _convert_segmentation_to_display(self, image, mode):
        """Convert segmentation results to display format."""
        img = image.copy()

        if mode == "segmented":
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:, :, 0]

            if img.dtype != np.uint8:
                img = normalize_image(img)

            return img

        elif mode in ["overlay", "labeled"]:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            if img.dtype != np.uint8:
                img = normalize_image(img)

            return img

        return img

    def _create_qimage_and_display(self, image):
        """Create QImage from processed array and display it."""
        normalize = self.normalize_checkbox.isChecked()
        processed_img, width, height, bytes_per_line, qt_format = (
            self._prepare_image_for_display(image, normalize)
        )

        q_image = QImage(processed_img.data, width, height, bytes_per_line, qt_format)
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        self.current_image_data = image.copy()

    # EVENT HANDLERS

    def on_image_ready(self, image, time, position, channel, mode):
        """Handle both raw and segmented image responses"""
        if (time != self.current_t or position != self.current_p or
                channel != self.current_c or mode != self.current_mode):
            return

        display_image = image if mode == "normal" else self._convert_segmentation_to_display(image, mode)
        self._create_qimage_and_display(display_image)

    def on_segmentation_cache_miss(self, time, position, channel, model):
        """Handle when cached segmentation is not available"""
        if (time == self.current_t and position == self.current_p and channel == self.current_c):
            print(f"No cached segmentation found for T={time}, P={position}, C={channel}")
            pub.sendMessage(
                "segmented_image_request",
                time=time,
                position=position,
                channel=channel,
                mode=self.current_mode,
                model=self.current_model,
                overlay_channel=self.current_overlay_channel if self.current_mode == "overlay" else None,
                use_cached=False,
            )

    def provide_current_param(self, topic=pub.AUTO_TOPIC, default=0):
        """Provide current parameters to other components"""
        param_map = {
            "get_current_t": self.current_t,
            "get_current_p": self.current_p,
            "get_current_c": self.current_c,
        }
        topic_name = topic.getName()
        value = param_map.get(topic_name, default)
        return value

    # UI INITIALIZATION

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
        layout.addWidget(self._create_nd2_controls())

        # Display Mode Controls
        layout.addWidget(self._create_display_mode_controls())

        # Segmentation Model Selection
        layout.addWidget(self._create_model_selection())

        self.setLayout(layout)

    def _create_nd2_controls(self):
        """Create ND2 navigation controls group"""
        group = QGroupBox("ND2 Controls")
        layout = QVBoxLayout()

        # T controls (Time)
        t_layout = QHBoxLayout()
        self.t_label = QLabel("T: 0")
        t_layout.addWidget(self.t_label)

        self.t_left_button = QPushButton("<")
        self.t_left_button.clicked.connect(lambda: self.slider_t.setValue(self.slider_t.value() - 1))
        t_layout.addWidget(self.t_left_button)

        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.setMinimum(0)
        self.slider_t.setMaximum(0)
        self.slider_t.valueChanged.connect(self.on_slider_changed)
        self.slider_t.valueChanged.connect(lambda value: self.t_label.setText(f"T: {value}"))
        t_layout.addWidget(self.slider_t)

        self.t_right_button = QPushButton(">")
        self.t_right_button.clicked.connect(lambda: self.slider_t.setValue(self.slider_t.value() + 1))
        t_layout.addWidget(self.t_right_button)

        layout.addLayout(t_layout)

        # P controls (Position)
        p_layout = QHBoxLayout()
        self.p_label = QLabel("P: 0")
        p_layout.addWidget(self.p_label)

        self.p_left_button = QPushButton("<")
        self.p_left_button.clicked.connect(lambda: self.slider_p.setValue(self.slider_p.value() - 1))
        p_layout.addWidget(self.p_left_button)

        self.slider_p = QSlider(Qt.Horizontal)
        self.slider_p.setMinimum(0)
        self.slider_p.setMaximum(0)
        self.slider_p.valueChanged.connect(self.on_slider_changed)
        self.slider_p.valueChanged.connect(lambda value: self.p_label.setText(f"P: {value}"))
        p_layout.addWidget(self.slider_p)

        self.p_right_button = QPushButton(">")
        self.p_right_button.clicked.connect(lambda: self.slider_p.setValue(self.slider_p.value() + 1))
        p_layout.addWidget(self.p_right_button)

        layout.addLayout(p_layout)

        # C controls (Channel)
        c_layout = QHBoxLayout()
        self.c_label = QLabel("C: 0")
        c_layout.addWidget(self.c_label)

        self.c_left_button = QPushButton("<")
        self.c_left_button.clicked.connect(lambda: self.slider_c.setValue(self.slider_c.value() - 1))
        c_layout.addWidget(self.c_left_button)

        self.slider_c = QSlider(Qt.Horizontal)
        self.slider_c.setMinimum(0)
        self.slider_c.setMaximum(0)
        self.slider_c.valueChanged.connect(self.on_slider_changed)
        self.slider_c.valueChanged.connect(lambda value: self.c_label.setText(f"C: {value}"))
        c_layout.addWidget(self.slider_c)

        self.c_right_button = QPushButton(">")
        self.c_right_button.clicked.connect(lambda: self.slider_c.setValue(self.slider_c.value() + 1))
        c_layout.addWidget(self.c_right_button)

        layout.addLayout(c_layout)

        # Normalization checkbox
        self.normalize_checkbox = QCheckBox("Normalize Image")
        self.normalize_checkbox.setChecked(True)
        self.normalize_checkbox.stateChanged.connect(self.on_slider_changed)
        layout.addWidget(self.normalize_checkbox)

        group.setLayout(layout)
        return group

    def _create_display_mode_controls(self):
        """Create display mode controls group"""
        group = QGroupBox("Display Mode")
        layout = QVBoxLayout()

        self.radio_normal = QRadioButton("Normal")
        # self.radio_segmented = QRadioButton("Segmented")
        self.radio_overlay_outlines = QRadioButton("Overlay with Outlines")
        self.radio_labeled_segmentation = QRadioButton("Labeled Segmentation")

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_normal)
        # self.button_group.addButton(self.radio_segmented)
        self.button_group.addButton(self.radio_overlay_outlines)
        self.button_group.addButton(self.radio_labeled_segmentation)
        self.button_group.buttonClicked.connect(self.on_display_mode_changed)

        self.radio_normal.setChecked(True)

        layout.addWidget(self.radio_normal)
        # layout.addWidget(self.radio_segmented)
        layout.addWidget(self.radio_overlay_outlines)

        # Overlay channel selection (only visible in overlay mode)
        overlay_channel_layout = QHBoxLayout()
        overlay_channel_label = QLabel("Overlay Channel:")
        self.overlay_channel_dropdown = QComboBox()
        self.overlay_channel_dropdown.currentIndexChanged.connect(self.on_overlay_channel_changed)
        overlay_channel_layout.addWidget(overlay_channel_label)
        overlay_channel_layout.addWidget(self.overlay_channel_dropdown)

        self.overlay_channel_widget = QWidget()
        self.overlay_channel_widget.setLayout(overlay_channel_layout)
        self.overlay_channel_widget.setVisible(False)  # Hidden by default
        layout.addWidget(self.overlay_channel_widget)

        layout.addWidget(self.radio_labeled_segmentation)

        group.setLayout(layout)
        return group

    def _create_model_selection(self):
        """Create segmentation model selection group"""
        group = QGroupBox("Segmentation Model")
        layout = QVBoxLayout()

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(SegmentationModels.available_models)
        self.model_dropdown.currentTextChanged.connect(self.on_model_changed)
        self.current_model = self.model_dropdown.currentText()

        layout.addWidget(self.model_dropdown)
        group.setLayout(layout)
        return group

    def show_context_menu(self, position):
        """Show context menu for the image label"""
        pass

    def check_cache_status(self, t, p, c, model):
        """Check and print the status of the segmentation cache for a specific frame"""
        image_data = None

        def receive_image_data(data):
            nonlocal image_data
            image_data = data

        pub.sendMessage("get_image_data", callback=receive_image_data)

        cache_exists = False
        if image_data and hasattr(image_data, "segmentation_cache"):
            cache = image_data.segmentation_cache

            if model in cache.mmap_arrays_idx:
                array, indices = cache.mmap_arrays_idx[model]
                simple_key = (t, p, c)
                model_key = (t, p, c, model)

                if simple_key in indices or model_key in indices:
                    cache_exists = True

        return cache_exists

    @Slot(object)
    def on_display_mode_changed(self, button):
        """Handle display mode changes"""
        old_mode = self.current_mode

        if button == self.radio_normal:
            self.current_mode = "normal"
        # elif button == self.radio_segmented:
        #     self.current_mode = "segmented"
        elif button == self.radio_overlay_outlines:
            self.current_mode = "overlay"
        elif button == self.radio_labeled_segmentation:
            self.current_mode = "labeled"

        # Toggle overlay channel dropdown visibility
        self.overlay_channel_widget.setVisible(self.current_mode == "overlay")

        print(f"Display mode changed from '{old_mode}' to '{self.current_mode}'")

        if self.current_mode in ["segmented", "overlay", "labeled"]:
            t, p, c = self.current_t, self.current_p, self.current_c
            cache_exists = self.check_cache_status(t, p, c, self.current_model)

            if cache_exists:
                print(f"Using cached segmentation for T={t}, P={p}, C={c}")

        self.on_slider_changed()

    @Slot(int)
    def on_overlay_channel_changed(self, index):
        """Handle overlay channel selection change"""
        self.current_overlay_channel = index
        print(f"Overlay channel changed to: {index}")

        # Refresh display if in overlay mode
        if self.current_mode == "overlay":
            self.on_slider_changed()

    @Slot()
    def on_slider_changed(self):
        """Handle slider value changes and request appropriate image"""
        self.current_t = self.slider_t.value()
        self.current_p = self.slider_p.value()
        self.current_c = self.slider_c.value()
        t, p, c = self.current_t, self.current_p, self.current_c

        pub.sendMessage("view_index_changed", index=(t, p, c))

        if self.current_mode == "normal":
            image = ImageData.get_instance().get(t, p, c)
            self.on_image_ready(image, t, p, c, "normal")
        else:
            pub.sendMessage(
                "segmented_image_request",
                time=t,
                position=p,
                channel=c,
                mode=self.current_mode,
                model=self.current_model,
                overlay_channel=self.current_overlay_channel if self.current_mode == "overlay" else None,
            )

    @Slot(str)
    def on_model_changed(self, model_name):
        """Handle segmentation model changes"""
        self.current_model = model_name

        if self.current_mode in ["segmented", "overlay", "labeled"]:
            self.on_slider_changed()

    def on_image_data_loaded(self, image_data):
        """Handle new image_data loading and update slider ranges"""
        _shape = image_data.data.shape
        t_max = _shape[0] - 1
        p_max = _shape[1] - 1

        if len(_shape) == 5:  # T, P, C, Y, X format
            c_max = _shape[2] - 1
            self.num_channels = _shape[2]
        elif len(_shape) == 4:  # T, P, Y, X format (no channels)
            c_max = 0
            self.num_channels = 1
        else:
            c_max = 0
            self.num_channels = 1

        # Update channel sliders
        self.slider_t.setMaximum(max(0, t_max))
        self.slider_p.setMaximum(max(0, p_max))
        self.slider_c.setMaximum(max(0, c_max))

        # Populate overlay channel dropdown
        self.overlay_channel_dropdown.clear()
        self.overlay_channel_dropdown.addItems([f"Channel {i}" for i in range(self.num_channels)])
        self.current_overlay_channel = 0

        self.slider_t.setValue(0)
        self.slider_p.setValue(0)
        self.slider_c.setValue(0)

        self.on_slider_changed()

    def highlight_cell(self, cell_id):
        """Highlight a specific cell in the current image view"""
        print(f"ViewAreaWidget: Highlighting cell {cell_id}")

        t, p, c = self.current_t, self.current_p, self.current_c
        cell_mapping = None

        def receive_cell_mapping(mapping):
            nonlocal cell_mapping
            cell_mapping = mapping

        pub.sendMessage(
            "get_cell_mapping",
            time=t,
            position=p,
            channel=c,
            cell_id=cell_id,
            callback=receive_cell_mapping,
        )

        if not cell_mapping or cell_id not in cell_mapping:
            print(f"Cell {cell_id} mapping not available")
            return

        if not hasattr(self, "current_image_data") or self.current_image_data is None:
            print("No current image available to highlight cell")
            return

        cell_info = cell_mapping[cell_id]
        highlighted_image = self.current_image_data.copy()

        if len(highlighted_image.shape) == 2:
            highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_GRAY2BGR)

        y1, x1, y2, x2 = cell_info["bbox"]

        cv2.rectangle(highlighted_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            highlighted_image,
            f"Cell {cell_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        if "metrics" in cell_info and "morphology_class" in cell_info["metrics"]:
            morph_class = cell_info["metrics"]["morphology_class"]
            cv2.putText(
                highlighted_image,
                f"Class: {morph_class}",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        self._create_qimage_and_display(highlighted_image)
