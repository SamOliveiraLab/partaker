# motility_widget.py
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QTabWidget, QWidget, QMessageBox, QProgressDialog,
                             QFileDialog, QCheckBox, QDialogButtonBox, QTextEdit, QLineEdit,
                             QTableWidget, QTableWidgetItem, QComboBox)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage.measure import label, regionprops
import pandas as pd
from pubsub import pub

# Modify in motility_widget.py

class MotilityDialog(QDialog):
    """
    Dialog for analyzing and visualizing cell motility.
    """
    
    def __init__(self, tracked_cells, lineage_tracks, image_data=None, parent=None):
        super().__init__(parent)
        
        self.tracked_cells = tracked_cells
        self.lineage_tracks = lineage_tracks
        self.image_data = image_data  # Store image data for accessing segmentation cache
        self.motility_metrics = None
        
        # Set dialog properties
        self.setWindowTitle("Cell Motility Analysis")
        self.setMinimumWidth(1200)  # Increased width for the velocity tab
        self.setMinimumHeight(800)  # Increased height for the velocity tab
        
        # Initialize UI
        self.init_ui()
        
        # Start analysis
        self.analyze_motility()
    
    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget - make it fill the entire dialog
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
        
        self.track_selector_widget = QWidget()
        track_selector_layout = QHBoxLayout(self.track_selector_widget)

        track_label = QLabel("Select Track:")
        self.track_combo = QComboBox()
        self.track_combo.setEditable(True)
        self.track_combo.setInsertPolicy(QComboBox.NoInsert)
        self.track_combo.currentIndexChanged.connect(self.highlight_selected_track)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_track_highlight)

        track_selector_layout.addWidget(track_label)
        track_selector_layout.addWidget(self.track_combo, 1)
        track_selector_layout.addWidget(clear_button)

        # Add to combined layout at the top (before you add other widgets to combined_layout)
        self.combined_layout.insertWidget(0, self.track_selector_widget)
        
        # Add tabs
        self.tab_widget.addTab(self.combined_tab, "Motility by Region")
        self.tab_widget.addTab(self.map_tab, "Motility Map")
        
        # Add tab widget - it will now take up all the space
        layout.addWidget(self.tab_widget)
        
        # Create a status bar at the bottom for the summary, which will be shown only when needed
        self.status_bar = QWidget()
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(5, 0, 5, 0)  # Minimal margins
        
        # Summary label (now more compact)
        self.summary_label = QLabel()
        self.summary_label.setTextFormat(Qt.RichText)
        
        # Export and Close buttons
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        status_layout.addWidget(self.summary_label)
        status_layout.addStretch()
        status_layout.addWidget(self.export_button)
        status_layout.addWidget(self.close_button)
        
        layout.addWidget(self.status_bar)
        
        
        
        # Connect tab changed signal to update the summary visibility
        self.tab_widget.currentChanged.connect(self.update_summary_visibility)
    
    def analyze_motility(self):
        """Analyze cell motility"""
        # Delegate to a worker function
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Get current position and channel
            p = pub.sendMessage("get_current_p", default=0)
            if p is None:
                p = 0
                print(f"Current position was None, defaulting to position {p}")
            
            c = pub.sendMessage("get_current_c", default=0)
            if c is None:
                c = 0
                print(f"Current channel was None, defaulting to channel {c}")
            
            print(f"Using position={p}, channel={c}")
            
            # Determine chamber dimensions from image data
            chamber_dimensions = (1392, 1040)  # default
            if self.image_data and hasattr(self.image_data, "data"):
                if len(self.image_data.data.shape) >= 4:
                    height = self.image_data.data.shape[-2]
                    width = self.image_data.data.shape[-1]
                    chamber_dimensions = (width, height)
                    print(f"Using chamber dimensions from image data: {chamber_dimensions}")
            
            # Collect all cell positions
            all_cell_positions = self.collect_cell_positions(p, c)
            print(f"Collected {len(all_cell_positions)} cell positions for visualization")
            
            # Calculate motility metrics
            from tracking import enhanced_motility_index
            self.motility_metrics = enhanced_motility_index(
                self.tracked_cells, chamber_dimensions)
            
            # Create visualizations
            self.create_visualizations(chamber_dimensions, all_cell_positions)
            
            # Update summary
            self.update_summary()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to analyze motility: {str(e)}")
            
        finally:
            QApplication.restoreOverrideCursor()
    
    def collect_cell_positions(self, p, c):
        """Collect cell positions from segmentation cache or tracks"""
        all_cell_positions = []
        
        # Try to use segmentation cache if available
        if self.image_data and hasattr(self.image_data, 'segmentation_cache'):
            try:
                # Determine number of time frames
                t_max = 20
                if hasattr(self.image_data, 'data'):
                    t_max = min(20, self.image_data.data.shape[0])
                
                for t in range(t_max):
                    try:
                        binary_image = self.image_data.segmentation_cache[t, p, c]
                        if binary_image is not None:
                            labeled_image = label(binary_image)
                            regions = regionprops(labeled_image)
                            for region in regions:
                                y, x = region.centroid
                                all_cell_positions.append((x, y))
                        else:
                            print(f"Frame {t}: No binary image found")
                    except Exception as frame_error:
                        print(f"Error processing frame {t}: {str(frame_error)}")
                
            except Exception as e:
                print(f"Error collecting cell positions from segmentation: {str(e)}")
                all_cell_positions = []
        
        # Fall back to using tracks if needed
        if not all_cell_positions:
            print(f"Falling back to collecting positions from tracks")
            for track in self.lineage_tracks:
                if 'x' in track and 'y' in track:
                    all_cell_positions.extend(list(zip(track['x'], track['y'])))
            print(f"Collected {len(all_cell_positions)} cell positions from tracks")
        
        return all_cell_positions
    
    def create_visualizations(self, chamber_dimensions, all_cell_positions):
        """Create motility visualizations"""
        from tracking import (visualize_motility_with_chamber_regions, visualize_motility_map)
        
        # Combined visualization
        combined_fig, self.combined_ax = visualize_motility_with_chamber_regions(
            self.tracked_cells, all_cell_positions, chamber_dimensions, self.motility_metrics)
        combined_canvas = FigureCanvas(combined_fig)
        self.combined_layout.addWidget(combined_canvas)
        self.combined_fig = combined_fig
        self.combined_canvas = combined_canvas
        
        # Populate track selector
        self.populate_track_selector()
        
        # Motility map
        map_fig, _ = visualize_motility_map(
            self.tracked_cells, chamber_dimensions, self.motility_metrics)
        map_canvas = FigureCanvas(map_fig)
        self.map_layout.addWidget(map_canvas)
        self.map_fig = map_fig
        
       
        
        
    
    
    # Add these new methods to the MotilityDialog class

    def populate_track_selector(self):
        """Populate the track selector with ALL track IDs present in visualization"""
        self.track_combo.clear()
        self.track_combo.addItem("-- Select a track --", None)
        
        # Create a set of all track IDs
        track_ids = set()
        
        # First add all tracks from tracked_cells
        for track in self.tracked_cells:
            if 'ID' in track:
                track_ids.add(track['ID'])
        
        # Create a lookup for metrics (if available)
        metrics_lookup = {}
        if self.motility_metrics and 'individual_metrics' in self.motility_metrics:
            metrics_lookup = {m.get('track_id'): m for m in self.motility_metrics['individual_metrics']}
        
        # Sort track IDs numerically
        sorted_ids = sorted(track_ids)
        
        # Add each track to combo box
        for track_id in sorted_ids:
            # Check if we have metrics for this track
            if track_id in metrics_lookup:
                metric = metrics_lookup[track_id]
                motility = metric.get('motility_index', 0)
                self.track_combo.addItem(f"Track {track_id} (MI: {motility:.1f})", track_id)
            else:
                # Add track even without motility data
                self.track_combo.addItem(f"Track {track_id}", track_id)
                    

    def highlight_selected_track(self, index):
        """Highlight the selected track on the visualization"""
        if index <= 0:  # No selection or the default item
            self.clear_track_highlight()
            return
        
        # Get the track ID from the combo box
        track_id = self.track_combo.itemData(index)
        
        # Find the track
        selected_track = next((t for t in self.tracked_cells if t.get('ID', -1) == track_id), None)
        
        # Clear any existing highlight
        self.clear_track_highlight()
        
        if selected_track and 'x' in selected_track and 'y' in selected_track:
            # Highlight the track
            self.highlighted_line = self.combined_ax.plot(
                selected_track['x'], selected_track['y'], '-',
                linewidth=3, color='red', zorder=100)[0]
            
            # Add start/end markers
            self.highlighted_start = self.combined_ax.plot(
                selected_track['x'][0], selected_track['y'][0], 'o',
                markersize=8, color='red', zorder=100)[0]
            
            self.highlighted_end = self.combined_ax.plot(
                selected_track['x'][-1], selected_track['y'][-1], 's',
                markersize=8, color='red', zorder=100)[0]
            
            # Prepare info text with available data
            info_text = [f"Track ID: {track_id}"]
            
            # Add track length
            if 'x' in selected_track:
                info_text.append(f"Track Length: {len(selected_track['x'])} frames")
            
            # Calculate path length (if not available)
            if 'x' in selected_track and 'y' in selected_track and len(selected_track['x']) > 1:
                # Calculate path length
                path_length = 0
                for i in range(len(selected_track['x']) - 1):
                    dx = selected_track['x'][i+1] - selected_track['x'][i]
                    dy = selected_track['y'][i+1] - selected_track['y'][i]
                    path_length += np.sqrt(dx**2 + dy**2)
                info_text.append(f"Path Length: {path_length:.1f} px")
            
            # Look up motility metrics for this track (if available)
            if self.motility_metrics and 'individual_metrics' in self.motility_metrics:
                track_metric = next((m for m in self.motility_metrics['individual_metrics'] 
                                if m.get('track_id') == track_id), None)
                
                if track_metric:
                    # Add motility metrics
                    info_text.append(f"Motility Index: {track_metric.get('motility_index', 0):.1f}")
                    info_text.append(f"Avg Velocity: {track_metric.get('avg_velocity', 0):.2f} px/frame")
                    info_text.append(f"Confinement: {track_metric.get('confinement_ratio', 0):.2f}")
                    info_text.append(f"Persistence: {track_metric.get('directional_persistence', 0):.2f}")
            
            # Add text near track end
            self.highlighted_text = self.combined_ax.text(
                selected_track['x'][-1] + 10, selected_track['y'][-1] + 10,
                "\n".join(info_text), 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'),
                zorder=100)
            
            # Update the canvas
            self.combined_canvas.draw()
        
    
    def clear_track_highlight(self):
        """Clear the highlighted track"""
        # Remove any existing highlight elements
        if hasattr(self, 'highlighted_line') and self.highlighted_line:
            self.highlighted_line.remove()
            self.highlighted_line = None
        
        if hasattr(self, 'highlighted_start') and self.highlighted_start:
            self.highlighted_start.remove()
            self.highlighted_start = None
        
        if hasattr(self, 'highlighted_end') and self.highlighted_end:
            self.highlighted_end.remove()
            self.highlighted_end = None
        
        if hasattr(self, 'highlighted_text') and self.highlighted_text:
            self.highlighted_text.remove()
            self.highlighted_text = None
        
        # Update the canvas
        if hasattr(self, 'combined_canvas'):
            self.combined_canvas.draw()
    
    def update_summary_visibility(self, index):
        """Show or hide the summary based on the current tab"""
        # Hide summary for velocity analysis tab to maximize space
        tab_name = self.tab_widget.tabText(index)
        if tab_name == "Velocity Analysis":
            self.summary_label.setVisible(False)
        else:
            self.summary_label.setVisible(True)
            # Update the summary display if it exists
            if self.motility_metrics:
                self.update_summary()

    def update_summary(self):
        """Update the summary label - now more compact horizontal format"""
        if not self.motility_metrics:
            return
            
        summary_text = (
            f"<b>Population Avg MI:</b> {self.motility_metrics['population_avg_motility']:.1f}/100 | "
            f"<b>Heterogeneity:</b> {self.motility_metrics['population_heterogeneity']:.2f} | "
            f"<b>Sample:</b> {self.motility_metrics['sample_size']} cells"
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
                        
                if export_csv.isChecked():
                    metrics_df = pd.DataFrame(self.motility_metrics['individual_metrics'])
                    metrics_df.to_csv(f"{base_path}_motility_metrics.csv", index=False)
                    
                QMessageBox.information(
                    export_dialog, "Export Complete",
                    f"Results exported to {base_path}_*.png/csv")