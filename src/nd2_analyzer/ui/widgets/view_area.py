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
from skimage import exposure

from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels
from nd2_analyzer.data.image_data import ImageData


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
        pub.subscribe(self.provide_current_param, "get_current_t")
        pub.subscribe(self.provide_current_param, "get_current_p")
        pub.subscribe(self.provide_current_param, "get_current_c")
        pub.subscribe(self.on_segmentation_cache_miss, "segmentation_cache_miss")

    # CENTRALIZED IMAGE PROCESSING FUNCTIONS

    def _normalize_image(self, image):
        """
        Normalize image to appropriate display range based on dtype.

        Args:
            image: Input image array

        Returns:
            Normalized image with appropriate dtype
        """
        if image.dtype == np.uint16:
            # Normalize 16-bit to full range
            return exposure.rescale_intensity(image, out_range=(0, 65535)).astype(
                np.uint16
            )
        elif image.dtype == np.uint8:
            # Already in proper range for 8-bit
            return image
        else:
            # Float or other types - normalize to 8-bit range
            return exposure.rescale_intensity(image, out_range="uint8").astype(np.uint8)

    def _prepare_image_for_display(self, image, normalize=True):
        """
        Standardize image format for Qt display.

        Args:
            image: Input image array
            normalize: Whether to apply normalization

        Returns:
            tuple: (processed_image, width, height, bytes_per_line, qt_format)
        """
        # Work on copy to avoid modifying original
        img = image.copy()

        # Apply normalization if requested
        if normalize:
            img = self._normalize_image(img)

        # Handle grayscale images
        if len(img.shape) == 2:
            # Ensure grayscale is 16-bit for better display quality
            if img.dtype != np.uint16:
                if normalize:
                    img = self._normalize_image(img).astype(np.uint16)
                else:
                    # Scale without normalization
                    if img.max() <= 255:
                        img = (img.astype(np.float32) * 257).astype(np.uint16)
                    else:
                        img = img.astype(np.uint16)

            height, width = img.shape
            bytes_per_line = width * 2  # 2 bytes per 16-bit pixel
            qt_format = QImage.Format_Grayscale16

        # Handle color images
        else:
            # Ensure color images are 8-bit RGB
            if img.dtype != np.uint8:
                if normalize:
                    img = self._normalize_image(img)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

            # Convert grayscale to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA to RGB
                img = img[:, :, :3]

            height, width, channels = img.shape
            bytes_per_line = width * channels
            qt_format = QImage.Format_RGB888

        return img, width, height, bytes_per_line, qt_format

    def _convert_segmentation_to_display(self, image, mode):
        """
        Convert segmentation results to display format.

        Args:
            image: Segmentation result
            mode: Display mode ('segmented', 'overlay', 'labeled')

        Returns:
            RGB image ready for display
        """
        img = image.copy()

        if mode == "segmented":
            # Binary mask - convert to grayscale
            if len(img.shape) > 2:
                img = (
                    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    if img.shape[2] == 3
                    else img[:, :, 0]
                )

            # Ensure proper range
            if img.dtype != np.uint8:
                img = self._normalize_image(img)

            return img

        elif mode in ["overlay", "labeled"]:
            # Should be RGB color image
            if len(img.shape) == 2:
                # Convert grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA to RGB
                img = img[:, :, :3]

            # Ensure uint8
            if img.dtype != np.uint8:
                img = self._normalize_image(img)

            return img

        return img

    def _create_qimage_and_display(self, image):
        """
        Create QImage from processed array and display it.

        Args:
            image: Processed image array ready for display
        """
        normalize = self.normalize_checkbox.isChecked()
        processed_img, width, height, bytes_per_line, qt_format = (
            self._prepare_image_for_display(image, normalize)
        )

        # Create QImage
        q_image = QImage(processed_img.data, width, height, bytes_per_line, qt_format)

        # Scale and display
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

        # Store for highlighting
        self.current_image_data = image.copy()

    # SIMPLIFIED EVENT HANDLERS

    def on_image_ready(self, image, time, position, channel, mode):
        """Handle both raw and segmented image responses"""
        # Check if this is the image we're currently expecting
        if (
            time != self.current_t
            or position != self.current_p
            or channel != self.current_c
            or mode != self.current_mode
        ):
            return  # Ignore outdated images

        # Process based on mode
        if mode == "normal":
            display_image = image
        else:
            display_image = self._convert_segmentation_to_display(image, mode)

        # Display the processed image
        self._create_qimage_and_display(display_image)

    def on_segmentation_cache_miss(self, time, position, channel, model):
        """Handle when cached segmentation is not available"""
        if (
            time == self.current_t
            and position == self.current_p
            and channel == self.current_c
        ):
            print(
                f"No cached segmentation found for T={time}, P={position}, C={channel}"
            )
            pub.sendMessage(
                "segmented_image_request",
                time=time,
                position=position,
                channel=channel,
                mode=self.current_mode,
                model=self.current_model,
                use_cached=False,
            )

    def provide_current_param(self, topic=pub.AUTO_TOPIC, default=0):
        param_map = {
            "get_current_t": self.current_t,
            "get_current_p": self.current_p,
            "get_current_c": self.current_c,
        }
        topic_name = topic.getName()
        value = param_map.get(topic_name, default)
        print(f"ViewAreaWidget: Providing {topic_name}={value}")
        return value

    # UI INITIALIZATION (keeping existing code structure)
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
        self.slider_t.setMaximum(0)
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
        self.slider_p.setMaximum(0)
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
        self.slider_c.setMaximum(0)
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
        self.normalize_checkbox.setChecked(True)
        self.normalize_checkbox.stateChanged.connect(self.on_slider_changed)
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
        self.model_dropdown.addItems(
            SegmentationModels.available_models
        )
        self.model_dropdown.currentTextChanged.connect(self.on_model_changed)
        self.current_model = self.model_dropdown.currentText()

        model_layout.addWidget(self.model_dropdown)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        self.setLayout(layout)

    def show_context_menu(self, position):
        """Show context menu for the image label"""
        pass

    def check_cache_status(self, t, p, c, model):
        """Check and print the status of the segmentation cache for a specific frame."""
        image_data = None

        def receive_image_data(data):
            nonlocal image_data
            image_data = data

        pub.sendMessage("get_image_data", callback=receive_image_data)

        cache_exists = False
        if image_data and hasattr(image_data, "segmentation_cache"):
            cache = image_data.segmentation_cache

            print("CACHE STATUS:")
            print(f"  Current model: {cache.model_name}")

            if model in cache.mmap_arrays_idx:
                array, indices = cache.mmap_arrays_idx[model]
                print(f"  Model {model} has {len(indices)} cached entries")

                simple_key = (t, p, c)
                model_key = (t, p, c, model)

                if simple_key in indices:
                    print(
                        f"  ✓ Frame T={t}, P={p}, C={c} exists with simple key format"
                    )
                    cache_exists = True
                elif model_key in indices:
                    print(f"  ✓ Frame T={t}, P={p}, C={c} exists with model key format")
                    cache_exists = True
                else:
                    sample_keys = list(indices)[:3] if indices else []
                    print(f"  ✗ Frame T={t}, P={p}, C={c} NOT found in cache")
                    print(f"  Sample cached keys: {sample_keys}")
            else:
                print(f"  ✗ Model {model} not found in cache")
                print(f"  Available models: {list(cache.mmap_arrays_idx.keys())}")
        else:
            print("  ✗ No image data or segmentation cache available")

        return cache_exists

    @Slot(object)
    def on_display_mode_changed(self, button):
        """Handle display mode changes"""
        old_mode = self.current_mode

        if button == self.radio_normal:
            self.current_mode = "normal"
        elif button == self.radio_segmented:
            self.current_mode = "segmented"
        elif button == self.radio_overlay_outlines:
            self.current_mode = "overlay"
        elif button == self.radio_labeled_segmentation:
            self.current_mode = "labeled"

        print(f"Display mode changed from '{old_mode}' to '{self.current_mode}'")

        if self.current_mode in ["segmented", "overlay", "labeled"]:
            t, p, c = self.current_t, self.current_p, self.current_c
            cache_exists = self.check_cache_status(t, p, c, self.current_model)

            from nd2_analyzer.analysis.metrics_service import MetricsService

            metrics_service = MetricsService()
            has_metrics = not metrics_service.query_optimized(
                time=t, position=p
            ).is_empty()

            print(
                f"Frame T={t}, P={p}, C={c}: Cache exists={cache_exists}, Metrics exist={has_metrics}"
            )

            if cache_exists:
                print(f"Using cached segmentation for T={t}, P={p}, C={c}")
                self.on_slider_changed()
                return
            elif has_metrics:
                print(
                    f"Using existing metrics for T={t}, P={p}, C={c} - but need to regenerate segmentation"
                )
                self.on_slider_changed()
                return

        self.on_slider_changed()

    @Slot()
    def on_slider_changed(self):
        """Handle slider value changes and request appropriate image"""
        self.current_t = self.slider_t.value()
        self.current_p = self.slider_p.value()
        self.current_c = self.slider_c.value()
        t, p, c = self.current_t, self.current_p, self.current_c

        print(f"DEBUG: Sliders changed to T={t}, P={p}, C={c}")
        pub.sendMessage("view_index_changed", index=(t, p, c))

        if self.current_mode == "normal":
            image = ImageData.get_instance().get(t, p, c)
            self.on_image_ready(image, t, p, c, "normal")
        else:
            print(
                f"DEBUG: Requesting segmented image for T={t}, P={p}, C={c} with mode={self.current_mode}, model={self.current_model}"
            )
            pub.sendMessage(
                "segmented_image_request",
                time=t,
                position=p,
                channel=c,
                mode=self.current_mode,
                model=self.current_model,
            )

    @Slot(str)
    def on_model_changed(self, model_name):
        """Handle segmentation model changes"""
        self.current_model = model_name

        if self.current_mode in ["segmented", "overlay", "labeled"]:
            self.on_slider_changed()

    def on_image_data_loaded(self, image_data):
        """Handle new image_data loading. Updates slider ranges based on image_data dimensions."""
        _shape = image_data.data.shape
        t_max = _shape[0] - 1
        p_max = _shape[1] - 1

        print(f"Shape: {_shape}, Length: {len(_shape)}")

        if len(_shape) == 5:  # T, P, C, Y, X format
            c_max = _shape[2] - 1
            self.has_channels = True
        elif len(_shape) == 4:  # T, P, Y, X format (no channels)
            c_max = 0
            self.has_channels = False
        else:
            c_max = 0
            self.has_channels = False

        self.slider_t.setMaximum(max(0, t_max))
        self.slider_p.setMaximum(max(0, p_max))
        self.slider_c.setMaximum(max(0, c_max))

        self.slider_t.setValue(0)
        self.slider_p.setValue(0)
        self.slider_c.setValue(0)

        self.on_slider_changed()

    def highlight_cell(self, cell_id):
        """Highlight a specific cell in the current image view"""
        print(f"ViewAreaWidget: Highlighting cell {cell_id}")

        # Automatically switch to labeled segmentation mode for highlighting
        if not self.radio_labeled_segmentation.isChecked():
            self.radio_labeled_segmentation.setChecked(True)
            self.on_display_mode_changed(self.radio_labeled_segmentation)

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

        # Request fresh labeled segmentation image to avoid stacking highlights
        fresh_image = None

        def receive_fresh_image(image, time, position, channel, mode):
            nonlocal fresh_image
            if time == t and position == p and channel == c and mode == "labeled":
                fresh_image = image

        # Temporarily subscribe to get the fresh image
        pub.subscribe(receive_fresh_image, "image_ready")

        # Request the fresh labeled segmentation
        pub.sendMessage(
            "segmented_image_request",
            time=t,
            position=p,
            channel=c,
            mode="labeled",
            model=self.current_model,
        )

        # Unsubscribe after getting the image
        pub.unsubscribe(receive_fresh_image, "image_ready")

        if fresh_image is None:
            print("Could not get fresh labeled segmentation image")
            return

        # Convert to display format and use as base for highlighting
        fresh_image = self._convert_segmentation_to_display(fresh_image, "labeled")

        cell_info = cell_mapping[cell_id]
        highlighted_image = fresh_image.copy()

        if len(highlighted_image.shape) == 2:
            highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_GRAY2BGR)

        y1, x1, y2, x2 = cell_info["bbox"]

        # Draw WHITE thick bounding box to highlight the selected cell
        cv2.rectangle(highlighted_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # Add white text label with cell ID
        cv2.putText(
            highlighted_image,
            f"Cell {cell_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if "metrics" in cell_info and "morphology_class" in cell_info["metrics"]:
            morph_class = cell_info["metrics"]["morphology_class"]
            # Add white text for morphology class
            cv2.putText(
                highlighted_image,
                f"Class: {morph_class}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        self._create_qimage_and_display(highlighted_image)
        self.highlighted_image = highlighted_image
