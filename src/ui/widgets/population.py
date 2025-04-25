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

class PopulationWidget(QWidget):
    """
    Widget for plotting population-level fluorescence over time.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_service = MetricsService()  # Singleton instance
        self.init_ui()

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

        # Filter for selected positions, channel, and time range
        mask = (
            df["position"].is_in(selected_positions)
            & (df["channel"] == channel)
            & (df["time"] >= t_start)
            & (df["time"] <= t_end)
        )
        subdf = df.filter(mask)

        if subdf.is_empty():
            return

        # Determine the metric to plot
        metric_name = self.metric_combo.currentText()
        if metric_name == "Mean Intensity":
            metric_col = "mean_intensity"
        elif metric_name == "Integrated Intensity":
            metric_col = "integrated_intensity"
        elif metric_name == "Normalized Intensity":
            metric_col = "normalized_intensity"
        else:
            return  # Unknown metric

        # Group by time, aggregate mean and std of the selected metric across cells and positions
        grouped = (
            subdf.group_by("time")
            .agg([
                pl.col(metric_col).mean().alias("mean_metric"),
                pl.col(metric_col).std().alias("std_metric"),
            ])
            .sort("time")
        )

        times = grouped["time"].to_numpy()
        mean_metric = grouped["mean_metric"].to_numpy()
        std_metric = grouped["std_metric"].to_numpy()

        # Plot
        self.population_figure.clear()
        ax = self.population_figure.add_subplot(111)
        ax.plot(times, mean_metric, color="red", label=f"Mean {metric_name}")
        ax.fill_between(times, mean_metric - std_metric, mean_metric + std_metric, color="red", alpha=0.2, label="Std Dev")
        ax.set_xlabel("Time")
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

