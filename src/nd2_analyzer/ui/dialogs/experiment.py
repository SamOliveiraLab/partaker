import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QListWidget,
    QFileDialog,
    QMessageBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QTabWidget,
    QWidget,
    QLabel,
    QComboBox
)
from pubsub import pub

from nd2_analyzer.data.experiment import Experiment


class ExperimentDialog(QDialog):
    """Dialog for creating or editing microscopy experiments"""

    experimentCreated = Signal(object)

    def __init__(self, parent=None, experiment: Experiment = None):
        """
        Initialize the experiment dialog.

        Args:
            parent: Parent widget
            experiment: Optional existing experiment to edit
        """
        super().__init__(parent)
        self.experiment = experiment
        self.setWindowTitle("Create / Edit Experiment" if experiment is None else f"Edit Experiment: {experiment.name}")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)

        # Default import mode
        self.import_mode = None
        self.file_paths = []
        self.component_intervals = {}
        self.focus_loss_intervals = []
        self.file_map = {}  # (p,t,c) -> file path

        self.create_ui()

        # Load existing experiment data if provided
        if self.experiment:
            self.load_experiment_data()

    def create_ui(self):
        """Create the user interface"""
        main_layout = QVBoxLayout()

        # Create tabbed interface for organization
        tab_widget = QTabWidget()

        # Tab 1: Basic Settings
        basic_tab = self.create_basic_tab()
        tab_widget.addTab(basic_tab, "Basic Settings")

        # Tab 2: Analysis Configuration
        analysis_tab = self.create_analysis_tab()
        tab_widget.addTab(analysis_tab, "Analysis Configuration")

        # Tab 3: Channels
        channels_tab = self.create_channels_tab()
        tab_widget.addTab(channels_tab, "Channels")

        # Tab 4: Components & Focus
        components_tab = self.create_components_tab()
        tab_widget.addTab(components_tab, "Components & Focus")

        main_layout.addWidget(tab_widget)

        # Dialog buttons
        button_layout = QHBoxLayout()
        self.create_button = QPushButton("Save Experiment" if self.experiment else "Create Experiment")
        self.create_button.clicked.connect(self.create_experiment)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.create_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def create_basic_tab(self) -> QWidget:
        """Create the basic settings tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Experiment details group
        details_group = QGroupBox("Experiment Details")
        details_layout = QFormLayout()

        self.name_edit = QLineEdit()
        details_layout.addRow("Experiment Name:", self.name_edit)

        self.time_step_spinbox = QDoubleSpinBox()
        self.time_step_spinbox.setRange(0.001, 3600.0)
        self.time_step_spinbox.setValue(60.0)
        self.time_step_spinbox.setSuffix(" seconds")
        self.time_step_spinbox.setDecimals(3)
        self.time_step_spinbox.valueChanged.connect(self._update_focus_conversion_helper)
        details_layout.addRow("PHC Time Interval:", self.time_step_spinbox)

        self.fluorescence_factor_spinbox = QDoubleSpinBox()
        self.fluorescence_factor_spinbox.setRange(0.1, 100.0)
        self.fluorescence_factor_spinbox.setValue(3.0)
        self.fluorescence_factor_spinbox.setDecimals(1)
        self.fluorescence_factor_spinbox.valueChanged.connect(self._update_focus_conversion_helper)
        details_layout.addRow("Fluorescence Factor:", self.fluorescence_factor_spinbox)

        self.import_mode_combo = QComboBox()
        self.import_mode_combo.addItems(["File", "Directory"])
        self.import_mode_combo.setPlaceholderText("Select import mode...")
        self.import_mode_combo.setCurrentIndex(-1)
        details_layout.addRow("Import Mode:", self.import_mode_combo)
        self.import_mode_combo.currentTextChanged.connect(self.update_import_mode)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        # Files group container
        self.file_group = QGroupBox("Image Files")
        file_layout = QVBoxLayout()
        self.file_list_widget = QListWidget()
        file_layout.addWidget(self.file_list_widget)
        file_buttons_layout = QHBoxLayout()
        self.add_file_button = QPushButton("Add File")
        self.add_file_button.clicked.connect(self.add_file)
        self.remove_file_button = QPushButton("Remove File")
        self.remove_file_button.clicked.connect(self.remove_file)
        file_buttons_layout.addWidget(self.add_file_button)
        file_buttons_layout.addWidget(self.remove_file_button)
        file_layout.addLayout(file_buttons_layout)

        self.file_group.setLayout(file_layout)
        layout.addWidget(self.file_group)
        self.file_group.hide()

        # Directory group container
        self.directory_group = QGroupBox("Image Directory")
        dir_layout = QHBoxLayout()
        self.directory_edit = QLineEdit()
        self.directory_edit.setPlaceholderText("Select directory...")
        self.directory_edit.setReadOnly(True)
        self.directory_button = QPushButton("Browse")
        self.directory_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(self.directory_edit)
        dir_layout.addWidget(self.directory_button)

        self.directory_group.setLayout(dir_layout)
        layout.addWidget(self.directory_group)
        self.directory_group.hide()

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_analysis_tab(self) -> QWidget:
        """Create the analysis configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Analysis parameters group
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QFormLayout()

        self.epsilon_spinbox = QDoubleSpinBox()
        self.epsilon_spinbox.setRange(0.0, 1000.0)
        self.epsilon_spinbox.setValue(0.1)
        self.epsilon_spinbox.setDecimals(3)
        analysis_layout.addRow("Epsilon (min fluorescence):", self.epsilon_spinbox)

        self.time_start_spinbox = QSpinBox()
        self.time_start_spinbox.setRange(0, 100000)
        self.time_start_spinbox.setValue(0)
        analysis_layout.addRow("Time Range Start (frames):", self.time_start_spinbox)

        self.time_end_spinbox = QSpinBox()
        self.time_end_spinbox.setRange(0, 100000)
        self.time_end_spinbox.setValue(5000)
        analysis_layout.addRow("Time Range End (frames):", self.time_end_spinbox)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Positions group
        positions_group = QGroupBox("Selected Positions")
        positions_layout = QVBoxLayout()

        positions_help = QLabel("Enter comma-separated position indices (e.g., 0,1,2,3)")
        positions_help.setStyleSheet("color: gray; font-size: 9pt;")
        positions_layout.addWidget(positions_help)

        self.positions_edit = QLineEdit()
        self.positions_edit.setPlaceholderText("0,1,2,3")
        positions_layout.addWidget(self.positions_edit)

        positions_group.setLayout(positions_layout)
        layout.addWidget(positions_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_channels_tab(self) -> QWidget:
        """Create the channels configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # RPU values group
        rpu_group = QGroupBox("RPU Calibration Values")
        rpu_layout = QFormLayout()

        self.mcherry_rpu_spinbox = QSpinBox()
        self.mcherry_rpu_spinbox.setRange(0, 1000000)
        rpu_layout.addRow("mCherry RPU:", self.mcherry_rpu_spinbox)

        self.yfp_rpu_spinbox = QSpinBox()
        self.yfp_rpu_spinbox.setRange(0, 1000000)
        rpu_layout.addRow("YFP RPU:", self.yfp_rpu_spinbox)

        rpu_group.setLayout(rpu_layout)
        layout.addWidget(rpu_group)

        # Channel colors group
        colors_group = QGroupBox("Channel Colors (Hex Codes)")
        colors_layout = QFormLayout()

        self.mcherry_color_edit = QLineEdit()
        self.mcherry_color_edit.setPlaceholderText("#FF4444")
        colors_layout.addRow("mCherry Color:", self.mcherry_color_edit)

        self.yfp_color_edit = QLineEdit()
        self.yfp_color_edit.setPlaceholderText("#FFB347")
        colors_layout.addRow("YFP Color:", self.yfp_color_edit)

        colors_group.setLayout(colors_layout)
        layout.addWidget(colors_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_components_tab(self) -> QWidget:
        """Create the components/intervals tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Component intervals group
        intervals_group = QGroupBox("Component Activation Intervals")
        intervals_layout = QVBoxLayout()

        help_label = QLabel("Add time intervals (in hours) for component activations")
        help_label.setStyleSheet("color: gray; font-size: 9pt;")
        intervals_layout.addWidget(help_label)

        # Component input
        component_input_layout = QHBoxLayout()
        self.component_name_edit = QLineEdit()
        self.component_name_edit.setPlaceholderText("Component Name (e.g., aTc, IPTG)")

        self.interval_start_spinbox = QDoubleSpinBox()
        self.interval_start_spinbox.setRange(0, 1000)
        self.interval_start_spinbox.setDecimals(2)
        self.interval_start_spinbox.setSuffix(" h")

        self.interval_end_spinbox = QDoubleSpinBox()
        self.interval_end_spinbox.setRange(0, 1000)
        self.interval_end_spinbox.setDecimals(2)
        self.interval_end_spinbox.setSuffix(" h")

        self.add_interval_button = QPushButton("Add Interval")
        self.add_interval_button.clicked.connect(self.add_component_interval)

        component_input_layout.addWidget(self.component_name_edit)
        component_input_layout.addWidget(QLabel("from"))
        component_input_layout.addWidget(self.interval_start_spinbox)
        component_input_layout.addWidget(QLabel("to"))
        component_input_layout.addWidget(self.interval_end_spinbox)
        component_input_layout.addWidget(self.add_interval_button)

        intervals_layout.addLayout(component_input_layout)

        # Component list
        self.components_list_widget = QListWidget()
        intervals_layout.addWidget(self.components_list_widget)

        # Remove button
        remove_comp_button_layout = QHBoxLayout()
        self.remove_interval_button = QPushButton("Remove Selected")
        self.remove_interval_button.clicked.connect(self.remove_component_interval)
        remove_comp_button_layout.addStretch()
        remove_comp_button_layout.addWidget(self.remove_interval_button)
        intervals_layout.addLayout(remove_comp_button_layout)

        intervals_group.setLayout(intervals_layout)
        layout.addWidget(intervals_group)

        # Focus loss intervals group
        focus_group = QGroupBox("Focus Loss Intervals")
        focus_layout = QVBoxLayout()

        focus_help = QLabel("Track time intervals (in hours) where autofocus failed")
        focus_help.setStyleSheet("color: gray; font-size: 9pt;")
        focus_layout.addWidget(focus_help)

        # Add conversion helper (will be updated dynamically)
        self.focus_conversion_help = QLabel()
        self.focus_conversion_help.setStyleSheet("color: #2196F3; font-size: 8pt; font-style: italic;")
        self.focus_conversion_help.setWordWrap(True)
        focus_layout.addWidget(self.focus_conversion_help)

        # Update conversion helper initially
        self._update_focus_conversion_helper()

        # Focus loss input
        focus_input_layout = QHBoxLayout()

        self.focus_start_spinbox = QDoubleSpinBox()
        self.focus_start_spinbox.setRange(0, 1000)
        self.focus_start_spinbox.setDecimals(2)
        self.focus_start_spinbox.setSuffix(" h")

        self.focus_end_spinbox = QDoubleSpinBox()
        self.focus_end_spinbox.setRange(0, 1000)
        self.focus_end_spinbox.setDecimals(2)
        self.focus_end_spinbox.setSuffix(" h")

        self.add_focus_loss_button = QPushButton("Add Focus Loss")
        self.add_focus_loss_button.clicked.connect(self.add_focus_loss_interval)

        focus_input_layout.addWidget(QLabel("Lost focus from"))
        focus_input_layout.addWidget(self.focus_start_spinbox)
        focus_input_layout.addWidget(QLabel("to"))
        focus_input_layout.addWidget(self.focus_end_spinbox)
        focus_input_layout.addWidget(self.add_focus_loss_button)

        focus_layout.addLayout(focus_input_layout)

        # Focus loss list
        self.focus_loss_list_widget = QListWidget()
        focus_layout.addWidget(self.focus_loss_list_widget)

        # Remove button
        remove_focus_button_layout = QHBoxLayout()
        self.remove_focus_loss_button = QPushButton("Remove Selected")
        self.remove_focus_loss_button.clicked.connect(self.remove_focus_loss_interval)
        remove_focus_button_layout.addStretch()
        remove_focus_button_layout.addWidget(self.remove_focus_loss_button)
        focus_layout.addLayout(remove_focus_button_layout)

        focus_group.setLayout(focus_layout)
        layout.addWidget(focus_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def add_file(self):
        """Open the file dialog to add ND2 and OME-TIFF files"""
        # ".tif", ".tiff", ".ome.tif", ".ome.tiff", "ome.btf", "ome.tf2", "ome.tf8")
        file_filter = "ND2 and TIFF Files (*.nd2 *tif *tiff *.ome.tif *ome.tiff *ome.tf2 *ome.tf8 *ome.btf)"
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter(file_filter)
        dialog.setWindowTitle("Select Image File")
        dialog.finished.connect(self._on_file_dialog_finished)
        dialog.open()

    def select_directory(self):
        """Open a directory dialog and set the import path."""
        folder = QFileDialog.getExistingDirectory(self,"Select Image Directory","", QFileDialog.ShowDirsOnly)
        if not folder:
            return
        # Show folder in the line edit
        self.directory_edit.setText(folder)
        self.import_from_directory(folder)


    def import_from_directory(self, folder: str):
        """Import TIFF files and classify them using filename regex."""
        import os
        import re

        self.file_map = {}  # (p,t,c) -> file path

        batch_pattern = re.compile(r"^pos(?P<p>\d+)_t(?P<t>\d+)_(?P<c>\d+)\.(tif|tiff)$", re.IGNORECASE)
        stacked_pattern = re.compile(r"^pos(?P<p>\d+)_(?P<c>\d+)\.(tif|tiff)$", re.IGNORECASE)


        self.file_list_widget.clear()
        self.file_paths = []
        detected_batch = False
        detected_stacked = False

        for root, dirs, files in os.walk(folder):
            for filename in files:
                if not filename.lower().endswith((".tif", ".tiff")):
                    continue

                full_path = os.path.join(root, filename)
                batch_match = batch_pattern.match(filename)
                stacked_match = stacked_pattern.match(filename)

                if batch_match:
                    detected_batch = True
                    p = int(batch_match.group("p"))
                    t = int(batch_match.group("t"))
                    c = int(batch_match.group("c"))
                    print(f"Batch TIFF → pos={p}, time={t}, channel={c}")

                    self.file_map[(p, t, c)] = full_path
                    self.file_paths.append(full_path)
                    self.file_list_widget.addItem(filename)


                elif stacked_match:
                    detected_stacked = True
                    p = int(stacked_match.group("p"))
                    c = int(stacked_match.group("c"))
                    print(f"Stacked TIFF → pos={p}, channel={c}")

                    self.file_map[(p, None, c)] = full_path
                    self.file_paths.append(full_path)
                    self.file_list_widget.addItem(filename)

                else:
                    print(f"Skipped file: {filename}")

        # Determine mode
        if detected_batch and detected_stacked:
            QMessageBox.critical(self,"Import Error","Mixed TIFF formats detected (batch + stacked).")
            self.import_mode = None
            return

        elif detected_batch:
            self.import_mode = "batch_tiff"

        elif detected_stacked:
            self.import_mode = "stacked_tiff"

        else:
            QMessageBox.warning(
                self,
                "Import Error",
                "No valid TIFF files found."
            )
            self.import_mode = None
            return

        print(f"Detected import mode: {self.import_mode}")

    def update_import_mode(self, mode):
        """Updates the status for import mode"""
        self.import_mode = mode
        if mode == "File":
            self.file_group.show()
            self.directory_group.hide()

        elif mode == "Directory":
            self.file_group.hide()
            self.directory_group.show()

    def _on_file_dialog_finished(self, result):
        """Handle file dialog completion"""
        dialog = self.sender()
        if result == QDialog.DialogCode.Accepted:
            selected_files = dialog.selectedFiles()
            for file_path in selected_files:
                if file_path in self.file_paths:
                    QMessageBox.warning(self, "Duplicate File", "This file is already in the list.")
                    continue
                self.file_paths.append(file_path)
                self.file_list_widget.addItem(os.path.basename(file_path))

    def remove_file(self):
        """Remove selected file from the list"""
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            file_name = item.text()
            for file_path in self.file_paths[:]:
                if os.path.basename(file_path) == file_name:
                    self.file_paths.remove(file_path)
                    row = self.file_list_widget.row(item)
                    self.file_list_widget.takeItem(row)
                    break

    def add_component_interval(self):
        """Add a component activation interval"""
        name = self.component_name_edit.text().strip()
        start = self.interval_start_spinbox.value()
        end = self.interval_end_spinbox.value()

        if not name:
            QMessageBox.warning(self, "Input Error", "Component name cannot be empty.")
            return

        if start >= end:
            QMessageBox.warning(self, "Input Error", "Start time must be less than end time.")
            return

        if name not in self.component_intervals:
            self.component_intervals[name] = []

        self.component_intervals[name].append((start, end))
        self.components_list_widget.addItem(f"{name}: {start:.2f}h - {end:.2f}h")

        self.component_name_edit.clear()
        self.interval_start_spinbox.setValue(0.0)
        self.interval_end_spinbox.setValue(0.0)

    def remove_component_interval(self):
        """Remove selected component interval"""
        selected_items = self.components_list_widget.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            text = item.text()
            # Parse the text to remove from component_intervals
            name = text.split(":")[0].strip()
            interval_text = text.split(":")[1].strip()
            start = float(interval_text.split("-")[0].replace("h", "").strip())
            end = float(interval_text.split("-")[1].replace("h", "").strip())

            if name in self.component_intervals:
                try:
                    self.component_intervals[name].remove((start, end))
                    if not self.component_intervals[name]:
                        del self.component_intervals[name]
                except ValueError:
                    pass

            row = self.components_list_widget.row(item)
            self.components_list_widget.takeItem(row)

    def add_focus_loss_interval(self):
        """Add a focus loss interval"""
        start = self.focus_start_spinbox.value()
        end = self.focus_end_spinbox.value()

        if start >= end:
            QMessageBox.warning(self, "Input Error", "Start time must be less than end time.")
            return

        self.focus_loss_intervals.append((start, end))
        # Keep sorted by start time
        self.focus_loss_intervals.sort(key=lambda x: x[0])

        # Refresh the list widget
        self._refresh_focus_loss_list()

        self.focus_start_spinbox.setValue(0.0)
        self.focus_end_spinbox.setValue(0.0)

    def remove_focus_loss_interval(self):
        """Remove selected focus loss interval"""
        selected_items = self.focus_loss_list_widget.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            text = item.text()
            # Parse "Lost focus: 10.50h - 15.25h"
            interval_text = text.split(":", 1)[1].strip()
            start = float(interval_text.split("-")[0].replace("h", "").strip())
            end = float(interval_text.split("-")[1].replace("h", "").strip())

            try:
                self.focus_loss_intervals.remove((start, end))
            except ValueError:
                pass

        self._refresh_focus_loss_list()

    def _refresh_focus_loss_list(self):
        """Refresh the focus loss list widget"""
        self.focus_loss_list_widget.clear()
        for start, end in self.focus_loss_intervals:
            self.focus_loss_list_widget.addItem(f"Lost focus: {start:.2f}h - {end:.2f}h")

    def _update_focus_conversion_helper(self):
        """Update the focus loss conversion helper text based on current time settings"""
        # Calculate time interval per frame in hours
        phc_interval = self.time_step_spinbox.value()  # in seconds
        fluorescence_factor = self.fluorescence_factor_spinbox.value()
        time_interval_hours = (phc_interval * fluorescence_factor) / 3600

        # Create helper text with actual calculated value
        helper_text = (
            f"To convert frames to hours, multiply frame number by {time_interval_hours:.4f}\n"
            f"Example: Frame 10 = 10 × {time_interval_hours:.4f} = {10 * time_interval_hours:.2f} hours"
        )

        self.focus_conversion_help.setText(helper_text)

    def load_experiment_data(self):
        """Load data from existing experiment into the form"""
        if not self.experiment:
            return

        # Basic settings
        self.name_edit.setText(self.experiment.name)
        self.time_step_spinbox.setValue(self.experiment.phc_interval)
        self.fluorescence_factor_spinbox.setValue(self.experiment.fluorescence_factor)

        # Files
        self.file_paths = list(self.experiment.image_files)
        for file_path in self.file_paths:
            self.file_list_widget.addItem(os.path.basename(file_path))

        # Analysis parameters
        self.epsilon_spinbox.setValue(self.experiment.epsilon)
        self.time_start_spinbox.setValue(self.experiment.time_range[0])
        self.time_end_spinbox.setValue(self.experiment.time_range[1])
        self.positions_edit.setText(",".join(map(str, self.experiment.selected_positions)))

        # Channels
        if "mcherry" in self.experiment.rpu_values:
            self.mcherry_rpu_spinbox.setValue(int(self.experiment.rpu_values["mcherry"]))
        if "yfp" in self.experiment.rpu_values:
            self.yfp_rpu_spinbox.setValue(int(self.experiment.rpu_values["yfp"]))

        if "mcherry" in self.experiment.channel_colors:
            self.mcherry_color_edit.setText(self.experiment.channel_colors["mcherry"])
        if "yfp" in self.experiment.channel_colors:
            self.yfp_color_edit.setText(self.experiment.channel_colors["yfp"])

        # Components
        self.component_intervals = dict(self.experiment.component_intervals)
        for name, intervals in self.component_intervals.items():
            for start, end in intervals:
                self.components_list_widget.addItem(f"{name}: {start:.2f}h - {end:.2f}h")

        # Focus loss intervals
        self.focus_loss_intervals = list(self.experiment.focus_loss_intervals)
        self._refresh_focus_loss_list()

    def create_experiment(self):
        """Create or update experiment from form data"""
        name = self.name_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Input Error", "Please enter an experiment name.")
            return

        if not self.file_paths:
            QMessageBox.warning(self, "Input Error", "Please add at least one image file.")
            return

        # Parse positions
        try:
            positions_text = self.positions_edit.text().strip()
            if positions_text:
                selected_positions = [int(p.strip()) for p in positions_text.split(",")]
            else:
                selected_positions = [0, 1, 2, 3]
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid position format. Use comma-separated integers.")
            return

        # Build RPU values
        rpu_values = {}
        if self.mcherry_rpu_spinbox.value() > 0:
            rpu_values["mcherry"] = float(self.mcherry_rpu_spinbox.value())
        if self.yfp_rpu_spinbox.value() > 0:
            rpu_values["yfp"] = float(self.yfp_rpu_spinbox.value())

        # Build channel colors
        channel_colors = {
            "mcherry": "#FF4444",
            "yfp": "#FFB347",
            "1": "#FF4444",
            "2": "#FFB347"
        }
        if self.mcherry_color_edit.text().strip():
            channel_colors["mcherry"] = self.mcherry_color_edit.text().strip()
            channel_colors["1"] = self.mcherry_color_edit.text().strip()
        if self.yfp_color_edit.text().strip():
            channel_colors["yfp"] = self.yfp_color_edit.text().strip()
            channel_colors["2"] = self.yfp_color_edit.text().strip()

        try:
            experiment = Experiment(
                name=name,
                image_files=self.file_paths,
                interval=self.time_step_spinbox.value(),
                fluorescence_factor=self.fluorescence_factor_spinbox.value(),
                epsilon=self.epsilon_spinbox.value(),
                selected_positions=selected_positions,
                time_range=(self.time_start_spinbox.value(), self.time_end_spinbox.value()),
                channel_colors=channel_colors,
                rpu_values=rpu_values,
                component_intervals=self.component_intervals,
                focus_loss_intervals=self.focus_loss_intervals,
                file_map=self.file_map,
                import_mode=self.import_mode
            )

            from nd2_analyzer.data.image_data import ImageData
            # Checks if the user imported a TIFF directory
            if self.import_mode in ["Directory", "batch_tiff", "stacked_tiff"]:
                image_data = ImageData.load_tiff_directory(
                    self.file_map,
                    self.import_mode
                )
            else:
                # Standard ND2 / multi‑file import
                image_data = ImageData.load_nd2(
                    self.file_paths,
                    import_mode = self.import_mode
                )

            self.experimentCreated.emit(experiment)
            pub.sendMessage("experiment_loaded", experiment=experiment)
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def get_import_mode(self):
        return self.import_mode