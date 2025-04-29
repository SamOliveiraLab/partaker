import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QListWidget, QListWidgetItem
)
from pubsub import pub
import polars as pl  # Import Polars

# Assume MetricsService is a singleton with a .df attribute (Polars DataFrame)
from metrics_service import MetricsService
from experiment import Experiment

class PopulationWidget(QWidget):
    """
    Widget for plotting population-level fluorescence over time.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_service = MetricsService()  # Singleton instance
        self.init_ui()
        self.experiment : Experiment = None

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Matplotlib figure
        self.population_figure = plt.figure()
        self.population_canvas = FigureCanvas(self.population_figure)
        layout.addWidget(self.population_canvas)

        # Position selection
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Select Positions:"))
        self.position_list = QListWidget()
        self.position_list.setSelectionMode(QListWidget.MultiSelection)
        pos_layout.addWidget(self.position_list)
        layout.addLayout(pos_layout)

        # Channel selection
        ch_layout = QHBoxLayout()
        ch_layout.addWidget(QLabel("Fluorescence Channel:"))
        self.channel_combo = QComboBox()
        ch_layout.addWidget(self.channel_combo)
        layout.addLayout(ch_layout)

        # Metric selection
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItem("Mean Intensity")
        self.metric_combo.addItem("Integrated Intensity")
        self.metric_combo.addItem("Normalized Intensity")
        metric_layout.addWidget(self.metric_combo)
        layout.addLayout(metric_layout)

        # Time range selection
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time Range:"))
        self.time_min_box = QSpinBox()
        self.time_max_box = QSpinBox()
        time_layout.addWidget(self.time_min_box)
        time_layout.addWidget(QLabel("to"))
        time_layout.addWidget(self.time_max_box)
        layout.addLayout(time_layout)

        # Plot button
        plot_btn = QPushButton("Plot Fluorescence")
        plot_btn.clicked.connect(self.plot_fluorescence_signal)
        layout.addWidget(plot_btn)

        # Export button
        export_btn = QPushButton("Export DataFrame to CSV")
        export_btn.clicked.connect(self.export_dataframe)
        layout.addWidget(export_btn)

        self.setLayout(layout)

        # Listen for data loading to populate UI
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_experiment_loaded, "experiment_loaded")

    def on_image_data_loaded(self, image_data):
        # Populate positions and channels based on image_data shape
        shape = image_data.data.shape
        t_max, p_max, c_max = shape[0] - 1, shape[1] - 1, shape[2] - 1

        self.position_list.clear()
        for p in range(p_max + 1):
            item = QListWidgetItem(f"{p}")
            item.setSelected(True)
            self.position_list.addItem(item)

        self.channel_combo.clear()
        for c in range(1, c_max + 1):  # Fluorescence channels start from 1
            self.channel_combo.addItem(str(c))

        self.time_min_box.setRange(0, t_max)
        self.time_max_box.setRange(0, t_max)
        self.time_max_box.setValue(t_max)

    def on_experiment_loaded(self, experiment):
        self.experiment = experiment

    def plot_fluorescence_signal(self):
        # Get user selections
        selected_positions = [
            int(item.text()) for item in self.position_list.selectedItems()
        ]
        if not selected_positions:
            return

        channel = int(self.channel_combo.currentText())
        t_start = self.time_min_box.value()
        t_end = self.time_max_box.value()

        # Get the metrics DataFrame from the singleton
        df = self.metrics_service.df

        if df.is_empty():
            return
        
        # Test each filter condition separately
        pos_filter = df["position"].is_in(selected_positions)
        channel_filter = df["channel"] == channel
        time_start_filter = df["time"] >= t_start
        time_end_filter = df["time"] <= t_end
        
        print(f"Position filter matches: {df.filter(pos_filter).shape[0]} rows")
        # print(f"Channel filter matches: {df.filter(channel_filter).shape[0]} rows")
        print(f"Time start filter matches: {df.filter(time_start_filter).shape[0]} rows")
        print(f"Time end filter matches: {df.filter(time_end_filter).shape[0]} rows")


        # Filter for selected positions, channel, and time range
        mask = (
            df["position"].is_in(selected_positions)
            # TODO: include fluorescence for different channels in MetricsService
            # and reflect those here.
            # & (df["channel"] == channel)
            & (df["time"] >= t_start)
            & (df["time"] <= t_end)
        )
        subdf = df.filter(mask)

        if subdf.is_empty():
            return

        # Determine the metric to plot
        # metric_name = self.metric_combo.currentText()
        # if metric_name == "Mean Intensity":
        #     metric_col = "mean_intensity"
        # elif metric_name == "Integrated Intensity":
        #     metric_col = "integrated_intensity"
        # elif metric_name == "Normalized Intensity":
        #     metric_col = "normalized_intensity"
        # else:
        #     return  # Unknown metric
        # The metric we want is based on the selected channel
        metric_col = f"fluo_{channel}"
        metric_name = f"Fluorescence for channel {channel}"

        # Group by time, aggregate mean and std of the selected metric across cells and positions
        grouped = (
            subdf.group_by("time")
            .agg([
                pl.col(metric_col).mean().alias("mean_metric"),
                pl.col(metric_col).std().alias("std_metric"),
            ])
            .sort("time")
        )

        # Filter values smaller than epsilon, and multiply time by the experiment interval constant
        epsilon = 0.1
        valid = grouped.filter(pl.col("mean_metric") > epsilon)

        # Interval is in seconds, divide by 60 to get minutes
        factor = self.experiment.interval 
        result = valid.with_columns(
            (pl.col("time") * factor / 3600).alias("scaled_time")
        )

        times = result["scaled_time"].to_numpy()
        mean_metric = result["mean_metric"].to_numpy()
        std_metric = result["std_metric"].to_numpy()

        # Plot
        self.population_figure.clear()
        ax = self.population_figure.add_subplot(111)
        ax.plot(times, mean_metric, color="red", label=f"Mean {metric_name}")
        ax.fill_between(times, mean_metric - std_metric, mean_metric + std_metric, color="red", alpha=0.2, label="Std Dev")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(f"Mean Cell {metric_name}")
        ax.set_title(f"{metric_name} over Time (Positions: {selected_positions}, Channel: {channel})")
        ax.legend()
        self.population_canvas.draw()

    def export_dataframe(self):
        # Export the DataFrame to a CSV file
        df = self.metrics_service.df
        if not df.is_empty():
            df.write_csv("cell_metrics.csv")
            print("DataFrame exported to cell_metrics.csv")
        else:
            print("No data to export")

