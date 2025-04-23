import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QComboBox, QSpinBox, QCheckBox, QRadioButton, QGroupBox,
                              QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from pubsub import pub

class PopulationWidget(QWidget):
    """
    A standalone widget for population-level fluorescence analysis.
    Uses PyPubSub for communication with other components.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize state variables
        self.selected_ps = set()
        
        # Define RPU parameters - this should ideally come from a configuration
        self.AVAIL_RPUS = {
            # This would be populated from a central configuration
            # For now, using placeholder values
            "Default": 1.0,
            "mCherry": 0.8,
            "YFP": 1.2
        }
        
        # Set up the UI
        self.init_ui()
        
        # Subscribe to relevant topics
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_experiment_loaded, "experiment_loaded")
        pub.subscribe(self.on_segmentation_completed, "segmentation_completed")
        pub.subscribe(self.on_position_changed, "position_changed")
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Matplotlib figure for plotting
        self.population_figure = plt.figure()
        self.population_canvas = FigureCanvas(self.population_figure)
        layout.addWidget(self.population_canvas)
        
        # P selection mode radio buttons
        p_mode_group = QGroupBox("P Selection Mode")
        p_mode_layout = QVBoxLayout()
        
        self.use_current_p_radio = QRadioButton("Use current P")
        self.use_current_p_radio.setChecked(True)  # Default selection
        self.select_ps_radio = QRadioButton("Select Ps to aggregate")
        
        p_mode_layout.addWidget(self.use_current_p_radio)
        p_mode_layout.addWidget(self.select_ps_radio)
        p_mode_group.setLayout(p_mode_layout)
        layout.addWidget(p_mode_group)
        
        # Create the multiple P selection widget (initially hidden)
        self.multi_p_widget = QWidget()
        self.multi_p_widget.setVisible(False)  # Hidden by default
        multi_p_layout = QVBoxLayout(self.multi_p_widget)
        
        # Create a table to show selected Ps
        self.selected_ps_table = QTableWidget()
        self.selected_ps_table.setColumnCount(2)  # P value and Remove button
        self.selected_ps_table.setHorizontalHeaderLabels(["P Value", "Action"])
        self.selected_ps_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.selected_ps_table.setSelectionMode(QAbstractItemView.NoSelection)
        multi_p_layout.addWidget(QLabel("Selected Ps:"))
        multi_p_layout.addWidget(self.selected_ps_table)
        
        # Add dropdown and button to add new Ps
        add_p_layout = QHBoxLayout()
        self.p_dropdown = QComboBox()
        
        self.add_p_button = QPushButton("Add P")  # Store as instance variable
        self.add_p_button.clicked.connect(self.add_p_to_selection)
        self.add_p_button.setEnabled(False)  # Initially disabled until dropdown is populated
        
        add_p_layout.addWidget(self.p_dropdown)
        add_p_layout.addWidget(self.add_p_button)
        multi_p_layout.addLayout(add_p_layout)
        
        # Connect dropdown signal to update button state
        self.p_dropdown.currentIndexChanged.connect(self.update_add_button_state)
        
        # Add the multi_p_widget to the main layout
        layout.addWidget(self.multi_p_widget)
        
        # Connect radio buttons to toggle the multi_p_widget visibility
        self.use_current_p_radio.toggled.connect(self.update_p_selection_mode)
        self.select_ps_radio.toggled.connect(self.update_p_selection_mode)
        
        # Checkbox for single cell analysis
        self.single_cell_checkbox = QCheckBox("Single Cell Analysis")
        layout.addWidget(self.single_cell_checkbox)
        
        # Button to manually plot
        plot_fluo_btn = QPushButton("Plot Fluorescence")
        plot_fluo_btn.clicked.connect(self.plot_fluorescence_signal)
        
        # Channel control
        channel_choice_layout = QHBoxLayout()
        channel_combo = QComboBox()
        channel_combo.addItem('0')
        channel_combo.addItem('1')
        channel_combo.addItem('2')
        channel_choice_layout.addWidget(QLabel("Channel selection: "))
        channel_choice_layout.addWidget(channel_combo)
        self.channel_combo = channel_combo
        channel_choice_layout.addWidget(plot_fluo_btn)
        
        layout.addLayout(channel_choice_layout)
        
        # Time range controls
        time_range_layout = QHBoxLayout()
        time_range_layout.addWidget(QLabel("Time Range:"))
        
        self.time_min_box = QSpinBox()
        time_range_layout.addWidget(self.time_min_box)
        
        self.time_max_box = QSpinBox()
        time_range_layout.addWidget(self.time_max_box)
        
        layout.addLayout(time_range_layout)
        
        # Create the combobox and populate it with the dictionary keys
        self.rpu_params_combo = QComboBox()
        for key in self.AVAIL_RPUS.keys():
            self.rpu_params_combo.addItem(key)
        
        hb = QHBoxLayout()
        hb.addWidget(QLabel("Select RPU Parameters:"))
        hb.addWidget(self.rpu_params_combo)
        layout.addLayout(hb)
        
        # Set the layout
        self.setLayout(layout)
        
        # Initialize current position
        self.current_position = 0

    def update_add_button_state(self):
        """Enable or disable the Add P button based on dropdown state"""
        self.add_p_button.setEnabled(self.p_dropdown.count() > 0)

    
    def update_p_selection_mode(self):
        """Show or hide the multiple P selection widget based on radio button selection"""
        if self.select_ps_radio.isChecked():
            self.multi_p_widget.setVisible(True)
        else:
            self.multi_p_widget.setVisible(False)
    
    def add_p_to_selection(self):
        """Add a P value to the selection table"""
        try:
            p_value = int(self.p_dropdown.currentText())
        except Exception:
            return
        
        # Check if this P is already in the selection
        if p_value in self.selected_ps:
            return
        
        # Add to our set of selected Ps
        self.selected_ps.add(p_value)
        
        # Update the table
        row_position = self.selected_ps_table.rowCount()
        self.selected_ps_table.insertRow(row_position)
        
        # Add P value
        self.selected_ps_table.setItem(
            row_position, 0, QTableWidgetItem(str(p_value)))
        
        # Add remove button
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(
            lambda: self.remove_p_from_selection(p_value))
        self.selected_ps_table.setCellWidget(row_position, 1, remove_button)
        
        # Update dropdown to remove this P
        current_index = self.p_dropdown.currentIndex()
        self.p_dropdown.removeItem(current_index)
    
    def remove_p_from_selection(self, p_value):
        """Remove a P value from the selection"""
        if p_value in self.selected_ps:
            self.selected_ps.remove(p_value)
            
            # Find and remove the row from the table
            for row in range(self.selected_ps_table.rowCount()):
                if int(self.selected_ps_table.item(row, 0).text()) == p_value:
                    self.selected_ps_table.removeRow(row)
                    break
            
            # Add the P value back to the dropdown
            # Sort the items to keep them in numerical order
            self.p_dropdown.addItem(str(p_value))
            items = [self.p_dropdown.itemText(i) for i in range(self.p_dropdown.count())]
            items = sorted(items, key=int)
            
            self.p_dropdown.clear()
            for item in items:
                self.p_dropdown.addItem(item)
    
    def get_selected_ps(self):
        """Return the selected P values based on the current mode"""
        if self.use_current_p_radio.isChecked():
            # Return current P from the application state
            return [self.current_position]
        else:
            # Multiple P mode - return the set of selected Ps
            return list(self.selected_ps)
    
    def plot_fluorescence_signal(self):
        """Request fluorescence data and plot it"""
        selected_ps = self.get_selected_ps()
        channel = int(self.channel_combo.currentText())
        rpu_key = self.rpu_params_combo.currentText()
        rpu = self.AVAIL_RPUS[rpu_key]
        t_start = self.time_min_box.value()
        t_end = self.time_max_box.value()
        single_cell = self.single_cell_checkbox.isChecked()
        
        # Request fluorescence data via PyPubSub
        pub.sendMessage("request_fluorescence_data", 
                       positions=selected_ps,
                       channel=channel,
                       rpu=rpu,
                       t_start=t_start,
                       t_end=t_end,
                       single_cell=single_cell)
    
    def on_fluorescence_data_ready(self, combined_fluo, combined_timestamp):
        """
        Handle fluorescence data when it's ready.
        
        Args:
            combined_fluo: List of fluorescence data for each position
            combined_timestamp: List of timestamps for each position
        """
        # Handle combined_fluo as a list of lists
        all_fluo_data = []
        all_timestamp_data = []
        
        # Iterate through each position's data
        for pos_idx, (fluo_list, timestamp_list) in enumerate(zip(combined_fluo, combined_timestamp)):
            for t_idx, (t, fluo_values) in enumerate(zip(timestamp_list, fluo_list)):
                for f in fluo_values:
                    all_fluo_data.append(f)
                    all_timestamp_data.append(t)
        
        # Convert to numpy arrays for efficient processing
        all_fluo_data = np.array(all_fluo_data)
        all_timestamp_data = np.array(all_timestamp_data)
        
        self.population_figure.clear()
        ax = self.population_figure.add_subplot(111)
        
        plot_timestamp = []
        plot_fluo = []
        fluo_mean = []
        fluo_std = []
        
        # Calculate mean and std for each timestamp
        unique_timestamps = np.unique(all_timestamp_data)
        for t in unique_timestamps:
            fluo_data = all_fluo_data[all_timestamp_data == t]
            fluo_mean.append(np.mean(fluo_data))
            fluo_std.append(np.std(fluo_data))
            for f in fluo_data:
                plot_timestamp.append(t)
                plot_fluo.append(f)
        
        fluo_mean = np.array(fluo_mean)
        fluo_std = np.array(fluo_std)
        
        npoints = 500
        # Randomly select up to npoints points for plotting
        points = np.array(list(zip(plot_timestamp, plot_fluo)))
        if len(points) > npoints:
            points = points[np.random.choice(
                points.shape[0], npoints, replace=False)]
            plot_timestamp, plot_fluo = zip(*points)
        
        ax.scatter(
            plot_timestamp,
            plot_fluo,
            color='blue',
            alpha=0.5,
            marker='+')
        ax.plot(unique_timestamps, fluo_mean, color='red', label='Mean')
        ax.fill_between(
            unique_timestamps,
            fluo_mean - fluo_std,
            fluo_mean + fluo_std,
            color='red',
            alpha=0.2,
            label='Std Dev')
        
        selected_ps = self.get_selected_ps()
        ax.set_title(f'Fluorescence signal for Positions {selected_ps}')
        ax.set_xlabel('T')
        ax.set_ylabel('Cell activity in RPUs')
        ax.legend()
        
        self.population_canvas.draw()
    
    def on_experiment_loaded(self, experiment):
        # Update RPU values if available in the experiment
        if hasattr(experiment, 'rpu_values') and experiment.rpu_values:
            self.AVAIL_RPUS = experiment.rpu_values
            self.rpu_params_combo.clear()
            for key in self.AVAIL_RPUS.keys():
                self.rpu_params_combo.addItem(key)

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
        c_max = _shape[2] - 1
        
        # Update time range spinboxes
        self.time_min_box.setRange(0, t_max)
        self.time_max_box.setRange(0, t_max)
        self.time_max_box.setValue(t_max)  # Set max to the last frame
        
        # Update P dropdown with available positions
        self.p_dropdown.clear()
        self.selected_ps.clear()
        for p in range(p_max + 1):
            self.p_dropdown.addItem(str(p))
        
        # Clear the selected Ps table
        self.selected_ps_table.setRowCount(0)
    
    def on_segmentation_completed(self, position, time_range):
        """
        Handle segmentation completion notification.
        
        Args:
            position: Position that was segmented
            time_range: Time range that was segmented
        """
        # If the segmented position is one we're interested in, we might want to update
        # For now, just log it
        print(f"Segmentation completed for position {position}, time range {time_range}")
    
    def on_position_changed(self, position):
        """
        Handle position change notification.
        
        Args:
            position: New current position
        """
        self.current_position = position
