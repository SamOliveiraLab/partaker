# motility_widget.py
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QTabWidget, QWidget, QMessageBox, QProgressDialog,
                             QFileDialog, QCheckBox, QDialogButtonBox)
from PySide6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage.measure import label, regionprops
import pandas as pd
from pubsub import pub

class MotilityDialog(QDialog):
    """
    Dialog for analyzing and visualizing cell motility.
    """
    
    def __init__(self, tracked_cells, lineage_tracks, parent=None):
        super().__init__(parent)
        
        self.tracked_cells = tracked_cells
        self.lineage_tracks = lineage_tracks
        self.motility_metrics = None
        
        # Set dialog properties
        self.setWindowTitle("Cell Motility Analysis")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)
        
        # Initialize UI
        self.init_ui()
        
        # Start analysis
        self.analyze_motility()
    
    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.combined_tab = QWidget()
        self.map_tab = QWidget()
        self.metrics_tab = QWidget()
        self.region_tab = QWidget()
        
        # Set up tab layouts
        self.combined_layout = QVBoxLayout(self.combined_tab)
        self.map_layout = QVBoxLayout(self.map_tab)
        self.metrics_layout = QVBoxLayout(self.metrics_tab)
        self.region_layout = QVBoxLayout(self.region_tab)
        
        # Add tabs
        self.tab_widget.addTab(self.combined_tab, "Motility by Region")
        self.tab_widget.addTab(self.map_tab, "Motility Map")
        self.tab_widget.addTab(self.metrics_tab, "Detailed Metrics")
        self.tab_widget.addTab(self.region_tab, "Regional Analysis")
        
        # Add summary label
        self.summary_label = QLabel()
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.summary_label)
        
        # Add tab widget
        layout.addWidget(self.tab_widget)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def analyze_motility(self):
        """Analyze cell motility"""
        # Delegate to a worker function
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Get chamber dimensions
            p = pub.sendMessage("get_current_p", default=0)
            c = pub.sendMessage("get_current_c", default=0)
            
            # Determine chamber dimensions from image data
            chamber_dimensions = (1392, 1040)  # default
            
            # Calculate motility metrics
            from tracking import enhanced_motility_index
            self.motility_metrics = enhanced_motility_index(
                self.tracked_cells, chamber_dimensions)
            
            # Create visualizations
            self.create_visualizations(chamber_dimensions)
            
            # Update summary
            self.update_summary()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to analyze motility: {str(e)}")
            
        finally:
            QApplication.restoreOverrideCursor()
    
    def create_visualizations(self, chamber_dimensions):
        """Create motility visualizations"""
        from tracking import (visualize_motility_with_chamber_regions, visualize_motility_map,
                            visualize_motility_metrics, analyze_motility_by_region)
        
        # Combined visualization
        combined_fig, _ = visualize_motility_with_chamber_regions(
            self.tracked_cells, [], chamber_dimensions, self.motility_metrics)
        combined_canvas = FigureCanvas(combined_fig)
        self.combined_layout.addWidget(combined_canvas)
        self.combined_fig = combined_fig
        
        # Motility map
        map_fig, _ = visualize_motility_map(
            self.tracked_cells, chamber_dimensions, self.motility_metrics)
        map_canvas = FigureCanvas(map_fig)
        self.map_layout.addWidget(map_canvas)
        self.map_fig = map_fig
        
        # Metrics visualization
        metrics_fig = visualize_motility_metrics(self.motility_metrics)
        metrics_canvas = FigureCanvas(metrics_fig)
        self.metrics_layout.addWidget(metrics_canvas)
        self.metrics_fig = metrics_fig
        
        # Regional analysis
        try:
            regional_analysis, region_fig = analyze_motility_by_region(
                self.tracked_cells, chamber_dimensions, self.motility_metrics)
            region_canvas = FigureCanvas(region_fig)
            self.region_layout.addWidget(region_canvas)
            self.region_fig = region_fig
            self.has_regional = True
        except Exception as e:
            print(f"Regional analysis failed: {e}")
            self.region_layout.addWidget(QLabel("Regional analysis not available"))
            self.has_regional = False
    
    def update_summary(self):
        """Update the summary label"""
        if not self.motility_metrics:
            return
            
        summary_text = (
            f"<h3>Motility Analysis Summary</h3>"
            f"<p><b>Population Average Motility Index:</b> {self.motility_metrics['population_avg_motility']:.1f}/100</p>"
            f"<p><b>Motility Heterogeneity:</b> {self.motility_metrics['population_heterogeneity']:.2f}</p>"
            f"<p><b>Sample Size:</b> {self.motility_metrics['sample_size']} cells</p>"
        )
        self.summary_label.setText(summary_text)
    
    def export_results(self):
        """Export analysis results"""
        export_dialog = QDialog(self)
        export_dialog.setWindowTitle("Export Options")
        export_layout = QVBoxLayout(export_dialog)
        
        export_label = QLabel("Select export options:")
        export_layout.addWidget(export_label)
        
        export_map = QCheckBox("Export Motility Map")
        export_map.setChecked(True)
        export_layout.addWidget(export_map)
        
        export_metrics = QCheckBox("Export Detailed Metrics Plot")
        export_metrics.setChecked(True)
        export_layout.addWidget(export_metrics)
        
        export_regional = QCheckBox("Export Regional Analysis")
        export_regional.setChecked(self.has_regional)
        export_regional.setEnabled(self.has_regional)
        export_layout.addWidget(export_regional)
        
        export_csv = QCheckBox("Export Metrics as CSV")
        export_csv.setChecked(True)
        export_layout.addWidget(export_csv)
        
        export_buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        export_buttons.accepted.connect(export_dialog.accept)
        export_buttons.rejected.connect(export_dialog.reject)
        export_layout.addWidget(export_buttons)
        
        if export_dialog.exec() == QDialog.Accepted:
            save_path, _ = QFileDialog.getSaveFileName(
                export_dialog, "Save Results", "", "All Files (*)")
                
            if save_path:
                base_path = save_path.replace(".png", "").replace(".csv", "")
                
                if export_map.isChecked():
                    self.map_fig.savefig(
                        f"{base_path}_motility_map.png", dpi=300, bbox_inches='tight')
                        
                if export_metrics.isChecked():
                    self.metrics_fig.savefig(
                        f"{base_path}_detailed_metrics.png", dpi=300, bbox_inches='tight')
                        
                if export_regional.isChecked() and self.has_regional:
                    self.region_fig.savefig(
                        f"{base_path}_regional_analysis.png", dpi=300, bbox_inches='tight')
                        
                if export_csv.isChecked():
                    metrics_df = pd.DataFrame(self.motility_metrics['individual_metrics'])
                    metrics_df.to_csv(f"{base_path}_motility_metrics.csv", index=False)
                    
                QMessageBox.information(
                    export_dialog, "Export Complete",
                    f"Results exported to {base_path}_*.png/csv")