import csv
import os
import pickle

import matplotlib.pyplot as plt
import polars as pl  # Import Polars
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
)
import seaborn as sns

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pubsub import pub

# Assume MetricsService is a singleton with a .df attribute (Polars DataFrame)
from nd2_analyzer.analysis.metrics_service import MetricsService
from nd2_analyzer.data.experiment import Experiment

from nd2_analyzer.analysis.population import FluoAnalysisConfig, filter_data, create_sample_data, calculate_population_statistics, generate_component_step_functions, component_intervals

class PopulationWidget(QWidget):
    """
    Widget for plotting population-level fluorescence over time.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_service = MetricsService()  # Singleton instance
        self.init_ui()
        self.experiment: Experiment = None

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
        mcherry_ch_layout = QHBoxLayout()
        mcherry_ch_layout.addWidget(QLabel("mCherry Channel:"))
        self.mcherry_channel_combo = QComboBox()
        mcherry_ch_layout.addWidget(self.mcherry_channel_combo)
        layout.addLayout(mcherry_ch_layout)

        yfp_ch_layout = QHBoxLayout()
        yfp_ch_layout.addWidget(QLabel("mCherry Channel:"))
        self.yfp_channel_combo = QComboBox()
        yfp_ch_layout.addWidget(self.yfp_channel_combo)
        layout.addLayout(yfp_ch_layout)

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

        # bottom buttons
        bottom_btn_layout = QHBoxLayout()

        # Plot button
        plot_btn = QPushButton("Plot Fluorescence")
        plot_btn.clicked.connect(self.on_plot_population_signal)
        bottom_btn_layout.addWidget(plot_btn)

        # Export DataFrame button
        export_btn = QPushButton("Export DataFrame to CSV")
        export_btn.clicked.connect(self.export_dataframe)
        bottom_btn_layout.addWidget(export_btn)

        # Calculate RPU button (new)
        rpu_btn = QPushButton("Calculate RPU Reference Values")
        # rpu_btn.clicked.connect(self.calculate_rpu_values)
        bottom_btn_layout.addWidget(rpu_btn)

        layout.addLayout(bottom_btn_layout)

        self.setLayout(layout)

        # Listen for data loading to populate UI
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_experiment_loaded, "experiment_loaded")

    def export_dataframe(self):
        # Export the DataFrame to a CSV file
        df = self.metrics_service.df
        if not df.is_empty():
            df.write_csv("cell_metrics.csv")
            print("DataFrame exported to cell_metrics.csv")
        else:
            print("No data to export")

    def get_selected_positions(self):
        selected_positions = [
            int(item.text()) for item in self.position_list.selectedItems()
        ]
        return selected_positions

    def get_selected_time(self):
        return (self.time_min_box.value(), self.time_max_box.value())

    def on_image_data_loaded(self, image_data):
        # Populate positions and channels based on image_data shape
        shape = image_data.data.shape
        t_max, p_max, c_max = shape[0] - 1, shape[1] - 1, shape[2] - 1

        self.position_list.clear()
        for p in range(p_max + 1):
            item = QListWidgetItem(f"{p}")
            self.position_list.addItem(item)
        for i in range(self.position_list.count()):
            self.position_list.item(i).setSelected(True)

        self.mcherry_channel_combo.clear()
        for c in range(1, c_max + 1):  # Fluorescence channels start from 1
            self.mcherry_channel_combo.addItem(str(c))

        self.yfp_channel_combo.clear()
        for c in range(1, c_max + 1):  # Fluorescence channels start from 1
            self.yfp_channel_combo.addItem(str(c))

        self.time_min_box.setRange(0, t_max)
        self.time_max_box.setRange(0, t_max)
        self.time_max_box.setValue(t_max)

        # Dummy plot for preview
        self.create_dummy_plot()

    def on_experiment_loaded(self, experiment):
        self.experiment = experiment

    def create_dummy_plot(self):
        self.population_figure.clear()

        # Configure based on selected values
        analysis_cgf = FluoAnalysisConfig()
        analysis_cgf.selected_positions = self.get_selected_positions()
        analysis_cgf.time_range = self.get_selected_time()

        df = create_sample_data(analysis_cgf)

        # Perform data processing
        df = filter_data(df, analysis_cgf)
        df = calculate_population_statistics(df, analysis_cgf)

        component_intervals = {
            'aTc': [(0, 19.25)],
            'IPTG': [(19.25, 30.42)],
            'M9': [(30.42, 100)]
        }

        # Normalize with RPU
        mcherry_channel = int(self.mcherry_channel_combo.currentText())
        yfp_channel = int(self.yfp_channel_combo.currentText())

        mcherry_subdf = df.filter(pl.col('fluorescence_channel') == mcherry_channel).to_pandas()
        yfp_subdf = df.filter(pl.col('fluorescence_channel') == yfp_channel).to_pandas()

        # mcherry_subdf['mean_intensity'] = mcherry_subdf['mean_intensity'] / mcherry_rpu
        # mcherry_subdf['std_intensity'] = mcherry_subdf['std_intensity'] / mcherry_rpu
        # yfp_subdf['mean_intensity'] = yfp_subdf['mean_intensity'] / yfp_no_roi_rpu
        # yfp_subdf['std_intensity'] = yfp_subdf['std_intensity'] / yfp_no_roi_rpu

        # Plot graph
        fig = self.population_figure
        axs = fig.subplots(5, 1, gridspec_kw={'height_ratios': [3.5, 3.5, 1, 1, 1]})

        # For channel 0 (mCherry)
        ch0 = mcherry_subdf
        sns.lineplot(data=ch0, x='time_hours', y='mean_intensity', ax=axs[0], color='red', label='mCherry')
        axs[0].fill_between(ch0['time_hours'],
                            ch0['mean_intensity'] - ch0['std_intensity'],
                            ch0['mean_intensity'] + ch0['std_intensity'],
                            color='red', alpha=0.3)
        axs[0].set_ylabel('mCherry')
        axs[0].set_xlabel('')
        axs[0].set_xticks(range(0, int(max(ch0['time_hours'])) + 1, 10))
        axs[0].legend().set_visible(False)  # Hide legend

        # For channel 1 (YFP)
        ch1 = yfp_subdf
        sns.lineplot(data=ch1, x='time_hours', y='mean_intensity', ax=axs[1], color='goldenrod', label='YFP')
        axs[1].fill_between(ch1['time_hours'],
                            ch1['mean_intensity'] - ch1['std_intensity'],
                            ch1['mean_intensity'] + ch1['std_intensity'],
                            color='goldenrod', alpha=0.3)
        axs[1].set_ylabel('YFP')
        axs[1].set_xlabel('')
        axs[1].set_xticks(range(0, int(max(ch1['time_hours'])) + 1, 10))
        axs[1].legend().set_visible(False)  # Hide legend

        plt.tight_layout()

        # Medium change steps
        t = df['time_hours']
        comp_steps = generate_component_step_functions(component_intervals, t)

        start_index = 2
        for idx, (comp, step) in enumerate(comp_steps.items()):
            axs[start_index + idx].step(t, step, label=f'{comp}', where='post', color='black')
            axs[start_index + idx].set_ylabel(comp)
            axs[start_index + idx].set_xticks(range(0, int(max(t)) + 1, 10))
            axs[start_index + idx].legend().set_visible(False)  # Hide legend

        axs[-1].set_xlabel('Time (h)')

        # plt.savefig('1_9_iptg_on_p_all.pdf')
        # plt.show()

    def on_plot_population_signal(self):
        self.population_figure.clear()

        # Configure based on selected values
        analysis_cgf = FluoAnalysisConfig()
        analysis_cgf.selected_positions = self.get_selected_positions()
        analysis_cgf.time_range = self.get_selected_time()

        # Current dataframe
        df = self.metrics_service.df

        # Perform data processing
        df = filter_data(df, analysis_cgf)
        df = calculate_population_statistics(df)

        component_intervals = {
            'aTc': [(0, 19.25)],
            'IPTG': [(19.25, 30.42)],
            'M9': [(30.42, 100)]
        }

        # Normalize with RPU
        mcherry_channel = int(self.mcherry_channel_combo.currentText())
        yfp_channel = int(self.yfp_channel_combo.currentText())

        mcherry_subdf = df.filter(pl.col('fluorescence_channel') == mcherry_channel).to_pandas()
        yfp_subdf = df.filter(pl.col('fluorescence_channel') == yfp_channel).to_pandas()

        # mcherry_subdf['mean_intensity'] = mcherry_subdf['mean_intensity'] / mcherry_rpu
        # mcherry_subdf['std_intensity'] = mcherry_subdf['std_intensity'] / mcherry_rpu
        # yfp_subdf['mean_intensity'] = yfp_subdf['mean_intensity'] / yfp_no_roi_rpu
        # yfp_subdf['std_intensity'] = yfp_subdf['std_intensity'] / yfp_no_roi_rpu

        # Plot graph
        fig = self.population_figure
        axs = fig.subplots(5, 1, gridspec_kw={'height_ratios': [3.5, 3.5, 1, 1, 1]}, figsize=(8, 10))

        # For channel 0 (mCherry)
        ch0 = mcherry_subdf
        sns.lineplot(data=ch0, x='time_hours', y='mean_intensity', ax=axs[0], color='red', label='mCherry')
        axs[0].fill_between(ch0['time_hours'],
                            ch0['mean_intensity'] - ch0['std_intensity'],
                            ch0['mean_intensity'] + ch0['std_intensity'],
                            color='red', alpha=0.3)
        axs[0].set_ylabel('mCherry')
        axs[0].set_xlabel('')
        axs[0].set_xticks(range(0, int(max(ch0['time_hours'])) + 1, 10))
        axs[0].legend().set_visible(False)  # Hide legend

        # For channel 1 (YFP)
        ch1 = yfp_subdf
        sns.lineplot(data=ch1, x='time_hours', y='mean_intensity', ax=axs[1], color='goldenrod', label='YFP')
        axs[1].fill_between(ch1['time_hours'],
                            ch1['mean_intensity'] - ch1['std_intensity'],
                            ch1['mean_intensity'] + ch1['std_intensity'],
                            color='goldenrod', alpha=0.3)
        axs[1].set_ylabel('YFP')
        axs[1].set_xlabel('')
        axs[1].set_xticks(range(0, int(max(ch1['time_hours'])) + 1, 10))
        axs[1].legend().set_visible(False)  # Hide legend

        plt.tight_layout()

        # Medium change steps
        t = df['time_hours']
        comp_steps = generate_component_step_functions(component_intervals, t)

        start_index = 2
        for idx, (comp, step) in enumerate(comp_steps.items()):
            axs[start_index + idx].step(t, step, label=f'{comp}', where='post', color='black')
            axs[start_index + idx].set_ylabel(comp)
            axs[start_index + idx].set_xticks(range(0, int(max(t)) + 1, 10))
            axs[start_index + idx].legend().set_visible(False)  # Hide legend

        axs[-1].set_xlabel('Time (h)')

        plt.savefig('1_9_iptg_on_p_all.pdf')
        plt.show()

    # def calculate_rpu_values(self):
    #     """
    #     Calculate RPU reference values from all segmented cells across all frames.
    #     Displays results in a dialog and offers to export to CSV.
    #     """
    #     from PySide6.QtWidgets import (
    #         QMessageBox,
    #         QDialog,
    #         QVBoxLayout,
    #         QLabel,
    #         QDialogButtonBox,
    #     )
    #
    #     # Get the metrics DataFrame from the singleton
    #     df = self.metrics_service.df
    #
    #     if df.is_empty():
    #         QMessageBox.warning(
    #             self,
    #             "No Data",
    #             "No metrics data available. Please run segmentation first.",
    #         )
    #         return
    #
    #     # Get available fluorescence channels
    #     fluo_columns = [col for col in df.columns if col.startswith("fluo_")]
    #
    #     if not fluo_columns:
    #         QMessageBox.warning(
    #             self,
    #             "No Data",
    #             "No fluorescence data found in metrics. Please run segmentation with fluorescence channels.",
    #         )
    #         return
    #
    #     # Calculate average values for each channel (ignoring zeros)
    #     rpu_values = {}
    #     for channel_col in fluo_columns:
    #         channel_num = int(channel_col.split("_")[1])  # Extract channel number
    #
    #         # Filter out zeros and calculate the average
    #         channel_data = df.filter(pl.col(channel_col) > 0.1)
    #
    #         if channel_data.height > 0:
    #             avg_value = channel_data[channel_col].mean()
    #             std_value = channel_data[channel_col].std()
    #             cell_count = channel_data.height
    #
    #             channel_name = f"Channel {channel_num}"
    #             if channel_num == 1:
    #                 channel_name = "mCherry"
    #             elif channel_num == 2:
    #                 channel_name = "YFP"
    #
    #             rpu_values[channel_num] = {
    #                 "name": channel_name,
    #                 "avg_value": avg_value,
    #                 "std_value": std_value,
    #                 "cell_count": cell_count,
    #             }
    #
    #     if not rpu_values:
    #         QMessageBox.warning(
    #             self,
    #             "No Data",
    #             "No valid fluorescence data found (all values are zero or missing).",
    #         )
    #         return
    #
    #     # Create a dialog to display the results
    #     dialog = QDialog(self)
    #     dialog.setWindowTitle("RPU Reference Values")
    #     dialog.setMinimumWidth(400)
    #
    #     layout = QVBoxLayout(dialog)
    #
    #     # Add title
    #     title_label = QLabel("<h3>RPU Reference Values</h3>")
    #     title_label.setAlignment(Qt.AlignCenter)
    #     layout.addWidget(title_label)
    #
    #     # Add description
    #     desc_label = QLabel(
    #         "The following reference values were calculated from single-cell analysis "
    #         "across all frames using segmentation model: UNet"
    #     )
    #     desc_label.setWordWrap(True)
    #     layout.addWidget(desc_label)
    #
    #     # Add the calculated values
    #     for channel_num, values in rpu_values.items():
    #         value_label = QLabel(
    #             f"<b>{values['name']} (Channel {channel_num}):</b> {values['avg_value']:.2f} ± {values['std_value']:.2f} "
    #             f"<i>(from {values['cell_count']} cells)</i>"
    #         )
    #         value_label.setTextFormat(Qt.RichText)
    #         layout.addWidget(value_label)
    #
    #     # Add note
    #     note_label = QLabel(
    #         "<i>Note: These values can be used as RPU reference values for normalizing "
    #         "fluorescence measurements in future experiments.</i>"
    #     )
    #     note_label.setWordWrap(True)
    #     layout.addWidget(note_label)
    #
    #     # Add buttons
    #     button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
    #     layout.addWidget(button_box)
    #
    #     # Connect button signals
    #     button_box.accepted.connect(lambda: self.export_rpu_values(rpu_values, dialog))
    #     button_box.rejected.connect(dialog.reject)
    #
    #     # Show the dialog
    #     dialog.exec_()
    #
    # def export_rpu_values(self, rpu_values, dialog):
    #     """Export the calculated RPU values to a CSV file"""
    #
    #     # Ask for save location
    #     file_path, _ = QFileDialog.getSaveFileName(
    #         self,
    #         "Save RPU Reference Values",
    #         "rpu_reference_values.csv",
    #         "CSV Files (*.csv)",
    #     )
    #
    #     if not file_path:
    #         return
    #
    #     try:
    #         with open(file_path, "w", newline="") as csvfile:
    #             writer = csv.writer(csvfile)
    #
    #             # Write header row
    #             writer.writerow(
    #                 [
    #                     "Channel",
    #                     "Channel Name",
    #                     "RPU Reference Value",
    #                     "Standard Deviation",
    #                     "Cell Count",
    #                 ]
    #             )
    #
    #             # Write data rows
    #             for channel_num, values in rpu_values.items():
    #                 writer.writerow(
    #                     [
    #                         channel_num,
    #                         values["name"],
    #                         f"{values['avg_value']:.6f}",
    #                         f"{values['std_value']:.6f}",
    #                         values["cell_count"],
    #                     ]
    #                 )
    #
    #             # Write metadata
    #             writer.writerow([])
    #             writer.writerow(["Segmentation Model", "UNet"])
    #
    #             # If experiment info is available, add it
    #             if self.experiment:
    #                 writer.writerow(
    #                     ["Experiment Name", getattr(self.experiment, "name", "Unknown")]
    #                 )
    #                 writer.writerow(
    #                     [
    #                         "ND2 Files",
    #                         ", ".join(
    #                             getattr(self.experiment, "nd2_files", ["Unknown"])
    #                         ),
    #                     ]
    #                 )
    #
    #         dialog.accept()
    #
    #         from PySide6.QtWidgets import QMessageBox
    #
    #         QMessageBox.information(
    #             self, "Export Complete", f"RPU reference values saved to:\n{file_path}"
    #         )
    #
    #     except Exception as e:
    #         from PySide6.QtWidgets import QMessageBox
    #
    #         QMessageBox.warning(
    #             self, "Export Error", f"Failed to export RPU values: {str(e)}"
    #         )
