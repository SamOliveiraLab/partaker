from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QSpinBox, QProgressBar, QGroupBox, QCheckBox,
                               QListWidget, QAbstractItemView)
from PySide6.QtCore import Qt, Slot, QTimer
from pubsub import pub
import numpy as np


class SegmentationWidget(QWidget):
    """
    Widget for batch segmentation of time-lapse microscopy images.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize state variables
        self.current_model = None
        self.time_range = (0, 0)
        self.positions = []
        self.channel = 0
        self.mode = "segmented"
        self.is_segmenting = False
        self.queue = []
        self.processed_frames = set()

        # Set up the UI
        self.init_ui()

        # Subscribe to relevant topics
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_image_ready, "image_ready")

        # Timer for handling segmentation requests with a slight delay
        self.request_timer = QTimer(self)
        self.request_timer.setSingleShot(True)
        self.request_timer.timeout.connect(self.process_next_in_queue)

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Model selection
        model_group = QGroupBox("Segmentation Model")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Position selection
        position_group = QGroupBox("Positions")
        position_layout = QVBoxLayout()

        # Position list with checkboxes
        self.position_list = QListWidget()
        self.position_list.setSelectionMode(QAbstractItemView.MultiSelection)
        position_layout.addWidget(self.position_list)

        # Quick selection buttons
        pos_buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_positions)
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_no_positions)
        pos_buttons_layout.addWidget(select_all_btn)
        pos_buttons_layout.addWidget(select_none_btn)
        position_layout.addLayout(pos_buttons_layout)

        position_group.setLayout(position_layout)
        layout.addWidget(position_group)

        # Time range selection
        time_group = QGroupBox("Time Range")
        time_layout = QHBoxLayout()

        time_layout.addWidget(QLabel("From:"))
        self.time_start_spin = QSpinBox()
        self.time_start_spin.setMinimum(0)
        time_layout.addWidget(self.time_start_spin)

        time_layout.addWidget(QLabel("To:"))
        self.time_end_spin = QSpinBox()
        self.time_end_spin.setMinimum(0)
        time_layout.addWidget(self.time_end_spin)

        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Channel and Mode selection
        options_layout = QHBoxLayout()

        # Channel selection
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        channel_layout.addWidget(self.channel_combo)
        options_layout.addLayout(channel_layout)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["segmented", "overlay", "labeled"])
        mode_layout.addWidget(self.mode_combo)
        options_layout.addLayout(mode_layout)

        layout.addLayout(options_layout)

        # Segmentation controls
        controls_layout = QHBoxLayout()
        self.segment_button = QPushButton("Segment Selected")
        self.segment_button.clicked.connect(self.start_segmentation)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_segmentation)
        self.cancel_button.setEnabled(False)
        controls_layout.addWidget(self.segment_button)
        controls_layout.addWidget(self.cancel_button)
        layout.addLayout(controls_layout)

        # Progress display
        self.progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)

        self.progress_label = QLabel("Ready")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)

        self.progress_group.setLayout(progress_layout)
        layout.addWidget(self.progress_group)

        self.setLayout(layout)

    def on_image_data_loaded(self, image_data):
        """Handle new image data loading"""
        # Get dimensions from image_data
        shape = image_data.data.shape
        t_max = shape[0] - 1
        p_max = shape[1] - 1
        c_max = shape[2] - 1

        # Update time range spinboxes
        self.time_start_spin.setMaximum(t_max)
        self.time_end_spin.setMaximum(t_max)
        self.time_end_spin.setValue(t_max)

        # Populate position list
        self.position_list.clear()
        for p in range(p_max + 1):
            self.position_list.addItem(f"Position {p}")

        # Select all positions by default
        self.select_all_positions()

        # Populate channel combo
        self.channel_combo.clear()
        for c in range(c_max + 1):
            self.channel_combo.addItem(f"Channel {c}")

        # Populate model combo
        self.model_combo.clear()
        from segmentation.segmentation_models import SegmentationModels
        self.model_combo.addItems([
            SegmentationModels.OMNIPOSE_BACT_PHASE,
            SegmentationModels.CELLPOSE_BACT_PHASE,
            SegmentationModels.CELLPOSE_BACT_FLUOR,
            SegmentationModels.CELLPOSE,
            SegmentationModels.UNET
        ])

    def select_all_positions(self):
        """Select all positions in the list"""
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(True)

    def select_no_positions(self):
        """Deselect all positions in the list"""
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(False)

    def start_segmentation(self):
        """Start the segmentation process"""
        if self.is_segmenting:
            return

        # Get selected positions
        self.positions = []
        for item in self.position_list.selectedItems():
            pos = int(item.text().split(" ")[1])
            self.positions.append(pos)

        if not self.positions:
            self.progress_label.setText("No positions selected")
            return

        # Get time range
        t_start = self.time_start_spin.value()
        t_end = self.time_end_spin.value()

        if t_end < t_start:
            self.progress_label.setText("Invalid time range")
            return

        self.time_range = (t_start, t_end)

        # Get channel and mode
        self.channel = self.channel_combo.currentIndex()
        self.mode = self.mode_combo.currentText()
        self.current_model = self.model_combo.currentText()

        # Check if metrics data already exists for these parameters
        from metrics_service import MetricsService
        metrics_service = MetricsService()

        # Check if we need to segment anything
        frames_to_segment = []
        for p in self.positions:
            for t in range(t_start, t_end + 1):
                frames_to_segment.append((t, p))
                # TODO: clean this up
                # If metrics don't exist for this frame, add it to the queue
                # if not metrics_service.has_data_for(position=p, time=t, channel=self.channel):
                #    frames_to_segment.append((t, p))

        # If all frames already have metrics, no need to segment
        if not frames_to_segment:
            self.progress_label.setText("All frames already segmented")
            return

        # Build queue of frames to process
        self.queue = frames_to_segment

        # Set up progress tracking
        self.processed_frames.clear()
        total_frames = len(self.queue)
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(0)

        # Update UI
        self.segment_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.is_segmenting = True
        self.progress_label.setText(f"Segmenting {total_frames} frames...")

        # Start processing
        self.process_next_in_queue()

    def process_next_in_queue(self):
        """Process the next frame in the queue"""
        if not self.is_segmenting or not self.queue:
            if self.is_segmenting:
                self._segmentation_finished()
            return

        # Get next frame to process
        time, position = self.queue[0]

        # Request segmentation
        pub.sendMessage("segmented_image_request",
                        time=time,
                        position=position,
                        channel=self.channel,
                        mode=self.mode,
                        model=self.current_model)

        # Update status
        self.progress_label.setText(f"Processing T={time}, P={position}")

    def on_image_ready(self, image, time, position, channel, mode):
        """Handle segmented image ready event"""
        if not self.is_segmenting:
            return

        # Check if this is a response to one of our requests
        if (mode != self.mode or
            channel != self.channel or
            position not in self.positions or
                not (self.time_range[0] <= time <= self.time_range[1])):
            return

        # Create a unique key for this frame
        frame_key = (time, position, channel)

        # Skip if we've already processed this frame
        if frame_key in self.processed_frames:
            return

        # Mark as processed
        self.processed_frames.add(frame_key)

        # Remove from queue if it's the current item
        if self.queue and (time, position) == self.queue[0]:
            self.queue.pop(0)

        # Update progress
        self.progress_bar.setValue(len(self.processed_frames))

        # Schedule next processing with a small delay
        self.request_timer.start(50)  # 50ms delay

    def cancel_segmentation(self):
        """Cancel the segmentation process"""
        self.is_segmenting = False
        self.queue.clear()
        self.segment_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_label.setText("Cancelled")
        pub.sendMessage("segmentation_cancelled")

    def _segmentation_finished(self):
        """Handle completion of all segmentation tasks"""
        self.is_segmenting = False
        self.segment_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_label.setText("Completed")

        # Notify that batch segmentation is complete
        pub.sendMessage("batch_segmentation_completed",
                        time_range=self.time_range,
                        positions=self.positions,
                        model=self.current_model)
