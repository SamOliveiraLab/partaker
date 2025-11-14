"""
Cell View Dialog - Interactive Cell History Viewer

This dialog builds and visualizes cell histories, allowing you to:
1. Build cell-based organization from tracking data
2. Click on cells to see their complete time series
3. Validate the reorganization worked correctly
"""

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QSplitter,
    QWidget,
    QTextEdit,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from nd2_analyzer.analysis.cell_history import CellHistoryBuilder


class CellViewDialog(QDialog):
    """
    Dialog for viewing and validating cell histories.

    Shows:
    - Table of all cells with their stats
    - Click a cell to see its complete time series
    - Validation that data reorganization worked
    """

    def __init__(self, lineage_tracks, metrics_service, image_data=None, parent=None):
        super().__init__(parent)

        self.lineage_tracks = lineage_tracks
        self.metrics_service = metrics_service
        self.image_data = image_data
        self.builder = None
        self.cell_database = None
        self.current_cell_id = None

        # Set dialog properties
        self.setWindowTitle("Cell View - Cell History Viewer")
        self.setMinimumWidth(1400)
        self.setMinimumHeight(900)

        # Initialize UI
        self.init_ui()

        # Build cell histories automatically
        self.build_histories()

    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)

        # Status bar at top
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Building cell histories...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        status_layout.addWidget(self.status_label)

        self.rebuild_button = QPushButton("Rebuild")
        self.rebuild_button.clicked.connect(self.build_histories)
        status_layout.addWidget(self.rebuild_button)

        layout.addLayout(status_layout)

        # Create splitter for left (table) and right (plots)
        splitter = QSplitter(Qt.Horizontal)

        # LEFT SIDE: Cell list table
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(QLabel("All Cells (click to view):"))

        self.cell_table = QTableWidget()
        self.cell_table.setColumnCount(6)
        self.cell_table.setHorizontalHeaderLabels([
            "Cell ID", "Lifespan", "Fate", "Displacement", "Avg Velocity", "Coverage %"
        ])
        self.cell_table.cellClicked.connect(self.on_cell_clicked)
        left_layout.addWidget(self.cell_table)

        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))

        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "All Cells",
            "Long Cells (20+ frames)",
            "Divided Cells",
            "High Coverage (>90%)"
        ])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_combo)

        left_layout.addLayout(filter_layout)

        splitter.addWidget(left_widget)

        # RIGHT SIDE: Visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Cell info display
        self.cell_info_label = QLabel("Select a cell to view its history")
        self.cell_info_label.setStyleSheet("""
            background-color: #2b2b2b;
            color: #ffffff;
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
        """)
        right_layout.addWidget(self.cell_info_label)

        # Plots
        self.figure = plt.figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # Validation button
        validation_layout = QHBoxLayout()
        self.validate_button = QPushButton("🔍 Validate This Cell")
        self.validate_button.clicked.connect(self.validate_current_cell)
        self.validate_button.setEnabled(False)
        validation_layout.addWidget(self.validate_button)

        self.export_button = QPushButton("💾 Export Cell Histories CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.export_button.setEnabled(False)
        validation_layout.addWidget(self.export_button)

        right_layout.addLayout(validation_layout)

        splitter.addWidget(right_widget)

        # Set splitter sizes (30% table, 70% plots)
        splitter.setSizes([400, 1000])
        layout.addWidget(splitter)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def build_histories(self):
        """Build cell histories using CellHistoryBuilder"""
        print("\n" + "="*80)
        print("🔄 Building cell histories from GUI...")
        print("="*80)

        try:
            # Create builder
            self.builder = CellHistoryBuilder(self.lineage_tracks, self.metrics_service)

            # Build with minimum 5 frames
            self.cell_database = self.builder.build(min_track_length=5)

            if not self.cell_database:
                QMessageBox.warning(self, "Error", "No cells were processed!")
                return

            # Update status
            num_cells = len(self.cell_database)
            long_cells = len([c for c in self.cell_database.values() if c['lifespan'] >= 20])

            self.status_label.setText(
                f"✓ Built {num_cells} cell histories | {long_cells} long cells (20+ frames)"
            )
            self.status_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; padding: 10px; color: green;"
            )

            # Populate table
            self.populate_table()

            # Enable export
            self.export_button.setEnabled(True)

            print(f"✓ Cell histories built successfully: {num_cells} cells")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to build cell histories:\n{str(e)}")

    def populate_table(self, filter_func=None):
        """Populate the cell table with data"""
        self.cell_table.setRowCount(0)

        cells = self.cell_database.values()

        # Apply filter if provided
        if filter_func:
            cells = [c for c in cells if filter_func(c)]

        # Sort by lifespan (descending)
        cells = sorted(cells, key=lambda c: c['lifespan'], reverse=True)

        self.cell_table.setRowCount(len(cells))

        for row, cell in enumerate(cells):
            # Cell ID
            self.cell_table.setItem(row, 0, QTableWidgetItem(str(cell['cell_id'])))

            # Lifespan
            self.cell_table.setItem(row, 1, QTableWidgetItem(str(cell['lifespan'])))

            # Fate
            self.cell_table.setItem(row, 2, QTableWidgetItem(cell['fate']))

            # Displacement
            self.cell_table.setItem(row, 3, QTableWidgetItem(f"{cell['total_displacement']:.1f}"))

            # Avg Velocity
            self.cell_table.setItem(row, 4, QTableWidgetItem(f"{cell['avg_velocity']:.2f}"))

            # Coverage
            coverage_pct = cell['morphology_coverage'] * 100
            self.cell_table.setItem(row, 5, QTableWidgetItem(f"{coverage_pct:.1f}%"))

        self.cell_table.resizeColumnsToContents()

    def apply_filter(self, filter_name):
        """Apply filter to cell table"""
        if filter_name == "All Cells":
            self.populate_table()
        elif filter_name == "Long Cells (20+ frames)":
            self.populate_table(lambda c: c['lifespan'] >= 20)
        elif filter_name == "Divided Cells":
            self.populate_table(lambda c: c['fate'] == 'divided')
        elif filter_name == "High Coverage (>90%)":
            self.populate_table(lambda c: c['morphology_coverage'] > 0.9)

    def on_cell_clicked(self, row, column):
        """Handle cell selection from table"""
        cell_id_item = self.cell_table.item(row, 0)
        if not cell_id_item:
            return

        cell_id = int(cell_id_item.text())
        self.current_cell_id = cell_id

        # Get cell data
        cell = self.builder.get_cell(cell_id)
        if not cell:
            return

        # Update info display
        self.update_cell_info(cell)

        # Plot cell history
        self.plot_cell_history(cell)

        # Enable validation button
        self.validate_button.setEnabled(True)

    def update_cell_info(self, cell):
        """Update the cell info label"""
        info_text = f"""
<b>Cell ID:</b> {cell['cell_id']}<br>
<b>Lifespan:</b> {cell['lifespan']} frames<br>
<b>Fate:</b> {cell['fate']}<br>
<b>Time Range:</b> {cell['start_time']} to {cell['end_time']}<br>
<br>
<b>Movement:</b><br>
• Total Displacement: {cell['total_displacement']:.1f} pixels<br>
• Path Length: {cell['path_length']:.1f} pixels<br>
• Avg Velocity: {cell['avg_velocity']:.2f} pixels/frame<br>
• Directionality: {cell['directionality']:.3f}<br>
<br>
<b>Lineage:</b><br>
• Parent ID: {cell['parent_id'] if cell['parent_id'] else 'None'}<br>
• Children: {len(cell['children_ids'])} ({', '.join(map(str, cell['children_ids'])) if cell['children_ids'] else 'None'})<br>
<br>
<b>Data Quality:</b><br>
• Morphology Coverage: {cell['morphology_coverage']*100:.1f}%<br>
• Data Points Found: {cell['morphology_found']}/{cell['lifespan']}<br>
        """
        self.cell_info_label.setText(info_text)

    def plot_cell_history(self, cell):
        """Plot complete time series for a cell"""
        self.figure.clear()

        times = cell['times']

        # Create 2x2 subplot grid

        # Plot 1: Trajectory
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax1.plot(cell['x_positions'], cell['y_positions'], 'b-', alpha=0.6, linewidth=2)
        ax1.plot(cell['x_positions'][0], cell['y_positions'][0], 'go', markersize=12, label='Start', zorder=5)
        ax1.plot(cell['x_positions'][-1], cell['y_positions'][-1], 'ro', markersize=12, label='End', zorder=5)
        ax1.set_title(f"Cell {cell['cell_id']} - Trajectory", fontsize=12, fontweight='bold')
        ax1.set_xlabel("X Position (pixels)")
        ax1.set_ylabel("Y Position (pixels)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Area over time
        ax2 = self.figure.add_subplot(2, 2, 2)
        areas = cell['area']
        valid_idx = [i for i, a in enumerate(areas) if not np.isnan(a)]
        if valid_idx:
            valid_times = [times[i] for i in valid_idx]
            valid_areas = [areas[i] for i in valid_idx]
            ax2.plot(valid_times, valid_areas, 'b-o', markersize=4, linewidth=2)
            ax2.set_title("Area Over Time", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Time (frames)")
            ax2.set_ylabel("Area (pixels²)")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No area data', ha='center', va='center', transform=ax2.transAxes)

        # Plot 3: Length over time
        ax3 = self.figure.add_subplot(2, 2, 3)
        lengths = cell['length']
        valid_idx = [i for i, l in enumerate(lengths) if not np.isnan(l)]
        if valid_idx:
            valid_times = [times[i] for i in valid_idx]
            valid_lengths = [lengths[i] for i in valid_idx]
            ax3.plot(valid_times, valid_lengths, 'g-o', markersize=4, linewidth=2)
            ax3.set_title("Length Over Time", fontsize=12, fontweight='bold')
            ax3.set_xlabel("Time (frames)")
            ax3.set_ylabel("Length (pixels)")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No length data', ha='center', va='center', transform=ax3.transAxes)

        # Plot 4: Aspect Ratio over time
        ax4 = self.figure.add_subplot(2, 2, 4)
        aspect_ratios = cell['aspect_ratio']
        valid_idx = [i for i, ar in enumerate(aspect_ratios) if not np.isnan(ar)]
        if valid_idx:
            valid_times = [times[i] for i in valid_idx]
            valid_ratios = [aspect_ratios[i] for i in valid_idx]
            ax4.plot(valid_times, valid_ratios, 'r-o', markersize=4, linewidth=2)
            ax4.set_title("Aspect Ratio Over Time", fontsize=12, fontweight='bold')
            ax4.set_xlabel("Time (frames)")
            ax4.set_ylabel("Aspect Ratio")
            ax4.grid(True, alpha=0.3)

            # Add state thresholds
            ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Deforming threshold')
            ax4.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Deformed threshold')
            ax4.legend(fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No aspect ratio data', ha='center', va='center', transform=ax4.transAxes)

        self.figure.tight_layout()
        self.canvas.draw()

    def validate_current_cell(self):
        """Validate the currently selected cell by comparing to original data"""
        if not self.current_cell_id:
            return

        cell = self.builder.get_cell(self.current_cell_id)

        print("\n" + "="*80)
        print(f"🔍 VALIDATING CELL {self.current_cell_id}")
        print("="*80)

        # Pick 3 random timepoints to validate
        num_checks = min(3, len(cell['times']))
        if num_checks == 0:
            QMessageBox.warning(self, "Validation", "No timepoints to validate!")
            return

        check_indices = np.random.choice(len(cell['times']), num_checks, replace=False)

        validation_results = []
        all_valid = True

        for idx in check_indices:
            t = cell['times'][idx]
            x = cell['x_positions'][idx]
            y = cell['y_positions'][idx]
            area_from_db = cell['area'][idx]
            length_from_db = cell['length'][idx]

            print(f"\nChecking timepoint {t}:")
            print(f"  Cell database says:")
            print(f"    Position: ({x:.1f}, {y:.1f})")
            print(f"    Area: {area_from_db:.1f}")
            print(f"    Length: {length_from_db:.1f}")

            # Query original metrics_service
            metrics_df = self.metrics_service.query_optimized(time=t, cell_id=self.current_cell_id)

            if not metrics_df.is_empty():
                row = metrics_df.row(0, named=True)
                area_from_metrics = row['area']
                length_from_metrics = row['major_axis_length']

                print(f"  Metrics service says:")
                print(f"    Position: ({row['centroid_x']:.1f}, {row['centroid_y']:.1f})")
                print(f"    Area: {area_from_metrics:.1f}")
                print(f"    Length: {length_from_metrics:.1f}")

                # Check if they match
                area_match = abs(area_from_db - area_from_metrics) < 0.1 if not np.isnan(area_from_db) else False
                length_match = abs(length_from_db - length_from_metrics) < 0.1 if not np.isnan(length_from_db) else False

                if area_match and length_match:
                    print(f"  ✅ VERIFIED")
                    validation_results.append(f"✅ t={t}: Data matches")
                else:
                    print(f"  ⚠️ MISMATCH")
                    validation_results.append(f"⚠️ t={t}: Data mismatch!")
                    all_valid = False
            else:
                print(f"  ⚠️ No morphology data in metrics_service")
                if np.isnan(area_from_db):
                    validation_results.append(f"✅ t={t}: Correctly marked as NaN")
                else:
                    validation_results.append(f"⚠️ t={t}: Has data but shouldn't")
                    all_valid = False

        # Show results dialog
        result_text = f"Validation Results for Cell {self.current_cell_id}:\n\n"
        result_text += "\n".join(validation_results)
        result_text += f"\n\nOverall: {'✅ All checks passed!' if all_valid else '⚠️ Some mismatches detected'}"

        QMessageBox.information(self, "Validation Results", result_text)

    def export_csv(self):
        """Export cell database to CSV"""
        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Cell Histories",
            "cell_histories.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                self.builder.export_to_csv(file_path)
                QMessageBox.information(self, "Export Complete", f"Cell histories exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")
