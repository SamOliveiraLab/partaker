import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                               QLineEdit, QPushButton, QListWidget,
                               QFileDialog, QMessageBox, QDoubleSpinBox, QGroupBox,
                               QFormLayout)
from PySide6.QtWidgets import QSpinBox
from pubsub import pub

from nd2_analyzer.data.experiment import Experiment


class ExperimentDialog(QDialog):
    # Signal that emits an Experiment instance when created
    experimentCreated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Experiment")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        # Create the UI
        self.create_ui()

    def create_ui(self):
        main_layout = QVBoxLayout()

        # Experiment details group
        details_group = QGroupBox("Experiment Details")
        details_layout = QFormLayout()

        # Name input
        self.name_edit = QLineEdit()
        details_layout.addRow("Experiment Name:", self.name_edit)

        # Time step input (in seconds)
        self.time_step_spinbox = QDoubleSpinBox()
        self.time_step_spinbox.setRange(0.001, 3600.0)  # 1ms to 1 hour
        self.time_step_spinbox.setValue(60.0)  # Default: 60 seconds
        self.time_step_spinbox.setSuffix(" seconds")
        self.time_step_spinbox.setDecimals(3)
        details_layout.addRow("Time Step:", self.time_step_spinbox)

        details_group.setLayout(details_layout)
        main_layout.addWidget(details_group)

        # Files group
        files_group = QGroupBox("ND2 Files")
        files_layout = QVBoxLayout()

        # File list
        self.file_list_widget = QListWidget()
        files_layout.addWidget(self.file_list_widget)

        # File buttons
        file_buttons_layout = QHBoxLayout()
        self.add_file_button = QPushButton("Add File")
        self.add_file_button.clicked.connect(self.add_file)
        self.remove_file_button = QPushButton("Remove File")
        self.remove_file_button.clicked.connect(self.remove_file)
        file_buttons_layout.addWidget(self.add_file_button)
        file_buttons_layout.addWidget(self.remove_file_button)
        files_layout.addLayout(file_buttons_layout)

        files_group.setLayout(files_layout)
        main_layout.addWidget(files_group)
        # RPU group
        rpu_group = QGroupBox("RPU Values")
        rpu_layout = QFormLayout()
        self.mcherry_rpu_spinbox = QSpinBox()
        self.mcherry_rpu_spinbox.setRange(0, 1000000)
        rpu_layout.addRow("mCherry RPU:", self.mcherry_rpu_spinbox)
        self.yfp_rpu_spinbox = QSpinBox()
        self.yfp_rpu_spinbox.setRange(0, 1000000)
        rpu_layout.addRow("YFP RPU:", self.yfp_rpu_spinbox)
        rpu_group.setLayout(rpu_layout)
        main_layout.addWidget(rpu_group)

        # Events group
        events_group = QGroupBox("Events")
        events_layout = QVBoxLayout()
        event_input_layout = QHBoxLayout()
        self.event_name_edit = QLineEdit()
        self.event_name_edit.setPlaceholderText("Event Name")
        self.event_time_spinbox = QDoubleSpinBox()
        self.event_time_spinbox.setRange(0, 1e6)
        self.event_time_spinbox.setDecimals(3)
        self.event_time_spinbox.setSuffix(" s")
        self.add_event_button = QPushButton("Add Event")
        self.add_event_button.clicked.connect(self.add_event)
        event_input_layout.addWidget(self.event_name_edit)
        event_input_layout.addWidget(self.event_time_spinbox)
        event_input_layout.addWidget(self.add_event_button)
        events_layout.addLayout(event_input_layout)
        self.events_list_widget = QListWidget()
        events_layout.addWidget(self.events_list_widget)
        events_group.setLayout(events_layout)
        main_layout.addWidget(events_group)

        # Internal list of events (tuples of (name, time))
        self.events = []

        # Dialog buttons
        button_layout = QHBoxLayout()
        self.create_button = QPushButton("Create Experiment")
        self.create_button.clicked.connect(self.create_experiment)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.create_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Internal list of file paths (UI only concern)
        self.file_paths = []

    def add_file(self):
        """Open file dialog to add ND2 files to the experiment"""
        file_filter = "ND2 Files (*.nd2)"

        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter(file_filter)
        dialog.setWindowTitle("Select ND2 File")
        dialog.finished.connect(self._on_file_dialog_finished)
        dialog.open()

    def _on_file_dialog_finished(self, result):
        """Handle file dialog completion"""
        dialog = self.sender()
        if result == QDialog.DialogCode.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files:
                for file_path in selected_files:
                    # Check if file is already in our UI list
                    if file_path in self.file_paths:
                        QMessageBox.warning(self, "Duplicate File", "This file is already in the list.")
                        continue

                    # Add to our internal list and UI
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

    def create_experiment(self):

        """Gather input and attempt to create an Experiment instance"""

        name = self.name_edit.text().strip()
        events = list(self.events)

        if not name:
            QMessageBox.warning(self, "Input Error", "Please enter an experiment name.")
            return

        time_step = self.time_step_spinbox.value()

        if not self.file_paths:
            QMessageBox.warning(self, "Input Error", "Please add at least one ND2 file.")
            return

        try:
            # Let the Experiment class handle all validation and creation

            experiment = Experiment(
                name=name,
                nd2_files=self.file_paths,
                interval=time_step,
                # mcherry_rpu=mcherry_rpu,
                # yfp_rpu=yfp_rpu,
                # events=events
            )

            # If we get here, creation was successful
            # Emit Qt signal for direct connections
            self.experimentCreated.emit(experiment)

            pub.sendMessage("experiment_loaded", experiment=experiment)

            self.accept()

        except FileNotFoundError as e:
            # File not found error from Experiment class
            QMessageBox.critical(self, "File Error", str(e))

        except ValueError as e:
            # Validation error from Experiment class
            QMessageBox.critical(self, "Validation Error", str(e))

        except ImportError as e:
            # Missing dependency
            QMessageBox.critical(self, "Dependency Error", str(e))

        # except Exception as e:
        #     # Any other error
        #     QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")

        pub.sendMessage("experiment", experiment=experiment)

    def add_event(self):
        """Add an event (name, time) to the list"""
        name = self.event_name_edit.text().strip()
        time = self.event_time_spinbox.value()
        if not name:
            QMessageBox.warning(self, "Input Error", "Event name cannot be empty.")
            return
        # Optional: check for duplicates
        for n, t in self.events:
            if n == name and abs(t - time) < 1e-6:
                QMessageBox.warning(self, "Duplicate Event", "This event already exists.")
                return
        self.events.append((name, time))
        self.events_list_widget.addItem(f"{name} @ {time:.3f} s")
        self.event_name_edit.clear()
        self.event_time_spinbox.setValue(0.0)
