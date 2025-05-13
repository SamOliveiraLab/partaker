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
        self.velocity_tab = QWidget()  # New velocity tab
        
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
        self.tab_widget.addTab(self.velocity_tab, "Velocity Analysis")  # Add velocity tab
        
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
        
        # Initialize the velocity tab
        self.velocity_analysis = VelocityAnalysisTab(self.tracked_cells, self.lineage_tracks)
        velocity_layout = QVBoxLayout(self.velocity_tab)
        velocity_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for maximum space
        velocity_layout.addWidget(self.velocity_analysis)
        
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
        from tracking import (visualize_motility_with_chamber_regions, visualize_motility_map,
                            visualize_motility_metrics, analyze_motility_by_region)
        
        # Combined visualization
        combined_fig, _ = visualize_motility_with_chamber_regions(
            self.tracked_cells, all_cell_positions, chamber_dimensions, self.motility_metrics)
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
                        
                if export_regional.isChecked() and self.has_regional:
                    self.region_fig.savefig(
                        f"{base_path}_regional_analysis.png", dpi=300, bbox_inches='tight')
                        
                if export_csv.isChecked():
                    metrics_df = pd.DataFrame(self.motility_metrics['individual_metrics'])
                    metrics_df.to_csv(f"{base_path}_motility_metrics.csv", index=False)
                    
                QMessageBox.information(
                    export_dialog, "Export Complete",
                    f"Results exported to {base_path}_*.png/csv")
                

# Add to motility_widget.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                              QWidget, QPushButton, QSpinBox, QDoubleSpinBox, 
                              QGroupBox, QScrollArea)
from PySide6.QtCore import Qt, Signal, Slot

class VelocityAnalysisTab(QWidget):
    """
    Tab for analyzing and visualizing cell velocity.
    """
    
    def __init__(self, tracked_cells, lineage_tracks, parent=None):
        super().__init__(parent)
        
        self.tracked_cells = tracked_cells
        self.lineage_tracks = lineage_tracks
        
        # Mock data for demonstration - same as the React version
        self.mock_tracks = [
            {
                "id": 15,
                "frames": [
                    {"x": 300, "y": 400, "frame": 1, "velocity": 2.7},
                    {"x": 320, "y": 380, "frame": 2, "velocity": 2.8},
                    {"x": 350, "y": 350, "frame": 3, "velocity": 2.9}
                ],
                "children": [28, 29]
            },
            {
                "id": 28,
                "frames": [
                    {"x": 350, "y": 350, "frame": 3, "velocity": 1.8},
                    {"x": 380, "y": 330, "frame": 4, "velocity": 2.0},
                    {"x": 410, "y": 310, "frame": 5, "velocity": 2.2},
                    {"x": 440, "y": 290, "frame": 6, "velocity": 2.1}
                ],
                "parent": 15,
                "children": [56, 57]
            },
            {
                "id": 29,
                "frames": [
                    {"x": 350, "y": 350, "frame": 3, "velocity": 2.1},
                    {"x": 370, "y": 320, "frame": 4, "velocity": 2.3},
                    {"x": 390, "y": 280, "frame": 5, "velocity": 2.4},
                    {"x": 410, "y": 240, "frame": 6, "velocity": 2.6}
                ],
                "parent": 15,
                "children": [58, 59]
            }
        ]
        
        # Mock vertical sampling data
        self.mock_vertical_samples = [
            {"position": 100, "avgVelocity": 1.2, "sampleCount": 18},
            {"position": 200, "avgVelocity": 1.8, "sampleCount": 25},
            {"position": 300, "avgVelocity": 2.5, "sampleCount": 42},
            {"position": 400, "avgVelocity": 3.1, "sampleCount": 38},
            {"position": 500, "avgVelocity": 2.8, "sampleCount": 31},
            {"position": 600, "avgVelocity": 2.4, "sampleCount": 27},
            {"position": 700, "avgVelocity": 2.0, "sampleCount": 22},
            {"position": 800, "avgVelocity": 1.7, "sampleCount": 19},
            {"position": 900, "avgVelocity": 1.5, "sampleCount": 15}
        ]
        
        # State variables
        self.selected_tab = "tracks"  # "tracks" or "profile"
        self.hovered_track = None
        self.selected_track = None
        self.current_frame = 3
        self.calibration = 0.5  # μm/pixel
        
        # Chamber dimensions
        self.chamber_width = 1000
        self.chamber_height = 600
        
        # Setup UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the tab UI with optimized layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for maximum space
        
        # Create a top control bar with minimal height
        top_controls = QWidget()
        top_controls.setMaximumHeight(50)  # Limit height of controls
        top_layout = QHBoxLayout(top_controls)
        top_layout.setContentsMargins(10, 0, 10, 0)  # Reduced margins
        
        # Calibration control (simplified)
        calibration_layout = QHBoxLayout()
        calibration_label = QLabel("μm/pixel:")
        self.calibration_spinbox = QDoubleSpinBox()
        self.calibration_spinbox.setRange(0.1, 2.0)
        self.calibration_spinbox.setSingleStep(0.1)
        self.calibration_spinbox.setValue(self.calibration)
        self.calibration_spinbox.valueChanged.connect(self.update_calibration)
        
        calibration_layout.addWidget(calibration_label)
        calibration_layout.addWidget(self.calibration_spinbox)
        
        # View switcher (as button group)
        view_layout = QHBoxLayout()
        
        self.tracks_button = QPushButton("Velocity Tracks")
        self.tracks_button.setCheckable(True)
        self.tracks_button.setChecked(True)
        self.tracks_button.clicked.connect(self.set_tracks_view)
        
        self.profile_button = QPushButton("Velocity Profile")
        self.profile_button.setCheckable(True)
        self.profile_button.clicked.connect(self.set_profile_view)
        
        view_layout.addWidget(self.tracks_button)
        view_layout.addWidget(self.profile_button)
        
        top_layout.addLayout(calibration_layout)
        top_layout.addStretch()
        top_layout.addLayout(view_layout)
        
        main_layout.addWidget(top_controls)
        
        # Main visualization - using matplotlib
        # Make figure fill the available space
        self.fig = Figure(dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        # Connect mouse events
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        main_layout.addWidget(self.canvas, 1)  # Give visualization maximum space
        
        # Bottom controls with minimal height
        bottom_controls = QWidget()
        bottom_controls.setMaximumHeight(50)  # Limit height
        bottom_layout = QHBoxLayout(bottom_controls)
        bottom_layout.setContentsMargins(10, 0, 10, 0)  # Reduced margins
        
        # Timeline controls
        timeline_layout = QHBoxLayout()
        timeline_label = QLabel("Frame:")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(6)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self.update_frame)
        
        self.frame_label = QLabel(f"{self.current_frame}/6")
        
        timeline_layout.addWidget(timeline_label)
        timeline_layout.addWidget(self.frame_slider)
        timeline_layout.addWidget(self.frame_label)
        
        # Status indicator
        self.status_label = QLabel("No cell selected")
        
        bottom_layout.addLayout(timeline_layout)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.status_label)
        
        main_layout.addWidget(bottom_controls)
        
        # Initialize the plot
        self.update_plot()
        
    def set_tracks_view(self):
        """Switch to tracks view"""
        self.selected_tab = "tracks"
        self.tracks_button.setChecked(True)
        self.profile_button.setChecked(False)
        self.update_plot()
        
    def set_profile_view(self):
        """Switch to velocity profile view"""
        self.selected_tab = "profile"
        self.tracks_button.setChecked(False)
        self.profile_button.setChecked(True)
        self.update_plot()
        
    def update_calibration(self, value):
        """Update calibration value"""
        self.calibration = value
        self.update_plot()
        
    def update_frame(self, value):
        """Update current frame"""
        self.current_frame = value
        self.frame_label.setText(f"{value}/6")
        self.update_plot()
        
    def on_mouse_move(self, event):
        """Handle mouse movement events"""
        if event.inaxes is None:
            return
            
        # Check if we're hovering over a track
        ax = self.fig.gca()
        artists = ax.get_children()
        
        old_hovered = self.hovered_track
        self.hovered_track = None
        
        # Check line segments and points
        for artist in artists:
            if hasattr(artist, 'track_id'):
                contains, _ = artist.contains(event)
                if contains:
                    self.hovered_track = artist.track_id
                    break
        
        # Only redraw if hover state changed
        if old_hovered != self.hovered_track:
            self.update_plot()
            
    def on_mouse_click(self, event):
        """Handle mouse click events"""
        if event.inaxes is None:
            return
            
        # Check if we're clicking on a track
        if self.hovered_track is not None:
            if self.selected_track == self.hovered_track:
                self.selected_track = None
            else:
                self.selected_track = self.hovered_track
            
            self.status_label.setText(f"Selected: Cell {self.selected_track}" if self.selected_track else "No cell selected")
            self.update_plot()
    
    def get_color_for_velocity(self, velocity):
        """Get color based on velocity value - same as the React version"""
        # Normalize between 1-4 μm/s
        normalized = min(max((velocity - 1) / 3, 0), 1)
        
        if normalized < 0.2:
            return '#3300ff'  # Deep blue
        elif normalized < 0.4:
            return '#0066ff'  # Blue
        elif normalized < 0.6:
            return '#00ccff'  # Light blue
        elif normalized < 0.8:
            return '#ffcc00'  # Orange
        else:
            return '#ffff00'  # Yellow
    
    def update_plot(self):
        """Update the plot based on current state"""
        self.fig.clear()
        
        # Create axis with correct dimensions
        ax = self.fig.add_subplot(111)
        ax.set_xlim(0, self.chamber_width + 100)  # Extra space for color scale
        
        # Track visualization includes the color scale,
        # Profile visualization needs more height for the chart below
        if self.selected_tab == "tracks":
            ax.set_ylim(0, self.chamber_height)
        else:
            ax.set_ylim(0, self.chamber_height + 250)  # Extra space for profile
        
        # Draw chamber regions
        self.draw_chamber_regions(ax)
        
        # Draw sampling lines
        self.draw_sampling_lines(ax)
        
        # Draw tracks
        self.draw_tracks(ax)
        
        # Draw color scale
        self.draw_color_scale(ax)
        
        # Draw velocity profile if selected
        if self.selected_tab == "profile":
            self.draw_velocity_profile(ax)
        
        # Draw tooltip if hovering over a track
        if self.hovered_track is not None:
            self.draw_tooltip(ax)
        
        # Draw lineage panel if a track is selected
        if self.selected_track is not None:
            self.draw_lineage_panel(ax)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Update the canvas
        self.canvas.draw()
        
    def draw_chamber_regions(self, ax):
        """Draw the chamber regions on the plot"""
        # Center region
        ax.add_patch(Rectangle((100, 50), self.chamber_width-200, self.chamber_height-100,
                              fill=True, color='white', alpha=0.05, 
                              linestyle='dashed', edgecolor='gray'))
        ax.text(self.chamber_width/2, self.chamber_height/2, "CENTER", 
               ha='center', va='center', color='gray', alpha=0.7)
        
        # Left inlet
        ax.add_patch(Rectangle((0, 50), 100, self.chamber_height-100,
                              fill=True, color='lightyellow', alpha=0.1,
                              linestyle='dashed', edgecolor='gray'))
        ax.text(50, self.chamber_height/2, "INLET", 
               ha='center', va='center', color='gray', alpha=0.7,
               rotation=90)
        
        # Right channel
        ax.add_patch(Rectangle((self.chamber_width-100, 50), 100, self.chamber_height-100,
                              fill=True, color='lightgreen', alpha=0.1,
                              linestyle='dashed', edgecolor='gray'))
        ax.text(self.chamber_width-50, self.chamber_height/2, "CHANNEL", 
               ha='center', va='center', color='gray', alpha=0.7,
               rotation=90)
        
        # Corners
        ax.add_patch(Rectangle((0, 0), 100, 50,
                              fill=True, color='mistyrose', alpha=0.1,
                              linestyle='dashed', edgecolor='gray'))
        ax.add_patch(Rectangle((0, self.chamber_height-50), 100, 50,
                              fill=True, color='mistyrose', alpha=0.1,
                              linestyle='dashed', edgecolor='gray'))
        ax.add_patch(Rectangle((self.chamber_width-100, 0), 100, 50,
                              fill=True, color='mistyrose', alpha=0.1,
                              linestyle='dashed', edgecolor='gray'))
        ax.add_patch(Rectangle((self.chamber_width-100, self.chamber_height-50), 100, 50,
                              fill=True, color='mistyrose', alpha=0.1,
                              linestyle='dashed', edgecolor='gray'))
                              
        ax.text(50, 25, "CORNER", ha='center', va='center', color='gray', alpha=0.7, fontsize=8)
        ax.text(self.chamber_width-50, self.chamber_height-25, "CORNER", 
               ha='center', va='center', color='gray', alpha=0.7, fontsize=8)
        
    def draw_sampling_lines(self, ax):
        """Draw vertical sampling lines"""
        for sample in self.mock_vertical_samples:
            ax.axvline(x=sample["position"], ymin=0, ymax=self.chamber_height, 
                      color='red', linestyle='dashed', alpha=0.5, linewidth=1)
    
    def draw_tracks(self, ax):
        """Draw cell tracks up to the current frame"""
        for track in self.mock_tracks:
            track_id = track["id"]
            track_frames = [f for f in track["frames"] if f["frame"] <= self.current_frame]
            
            if len(track_frames) < 2:
                continue
                
            # Draw line segments
            for i in range(len(track_frames) - 1):
                current = track_frames[i]
                next_frame = track_frames[i + 1]
                
                # Create line with track_id attribute for interaction
                line = Line2D([current["x"], next_frame["x"]], 
                             [current["y"], next_frame["y"]],
                             color=self.get_color_for_velocity(current["velocity"]),
                             linewidth=2 if (self.hovered_track == track_id or self.selected_track == track_id) else 0.8,
                             alpha=0.8,
                             picker=5)
                
                # Attach track_id to the line for identification in events
                line.track_id = track_id
                ax.add_line(line)
            
            # Add dots for each position
            for i, pos in enumerate(track_frames):
                point = ax.plot(pos["x"], pos["y"], 'o',
                               color=self.get_color_for_velocity(pos["velocity"]),
                               markersize=3,
                               alpha=0.8,
                               picker=5)[0]
                               
                # Attach track_id to point
                point.track_id = track_id
            
            # Add division marker if appropriate
            last_frame = track_frames[-1]
            is_dividing = (track.get("children") and len(track["children"]) > 0 and 
                          last_frame["frame"] == track["frames"][-1]["frame"])
            
            if is_dividing:
                # Draw triangle marker
                triangle = Polygon([[last_frame["x"]-5, last_frame["y"]], 
                                  [last_frame["x"]+5, last_frame["y"]], 
                                  [last_frame["x"], last_frame["y"]-8]],
                                 closed=True,
                                 color='red',
                                 alpha=0.8)
                                 
                # Attach track_id to triangle
                triangle.track_id = track_id
                ax.add_patch(triangle)
    
    def draw_color_scale(self, ax):
        """Draw the velocity color scale"""
        velocities = [1, 1.5, 2, 2.5, 3, 3.5, 4]
        scale_height = 200
        scale_width = 30
        start_y = 100
        
        # Add title
        ax.text(self.chamber_width + 20 + scale_width/2, start_y-20, 
               "Velocity (μm/s)", ha='center', va='center')
        
        # Draw color blocks
        for i, velocity in enumerate(velocities):
            y_pos = start_y + i * (scale_height / len(velocities))
            height = scale_height / len(velocities)
            
            ax.add_patch(Rectangle((self.chamber_width + 20, y_pos), 
                                scale_width, height,
                                color=self.get_color_for_velocity(velocity),
                                linewidth=0))
            
            # Add label
            ax.text(self.chamber_width + 20 + scale_width + 5, 
                  y_pos + height/2, 
                  f"{velocity:.1f}", 
                  va='center', fontsize=8)
    
    def draw_velocity_profile(self, ax):
        """Draw the velocity profile chart below the chamber"""
        max_velocity = max(sample["avgVelocity"] for sample in self.mock_vertical_samples)
        bar_width = 30
        profile_height = 150
        profile_y_offset = self.chamber_height + 50
        
        # Title
        ax.text(self.chamber_width/2, profile_y_offset + 20, 
               "Velocity Profile Across Chamber Width", 
               ha='center', va='center', fontweight='bold')
        
        # X axis
        ax.plot([0, self.chamber_width], 
               [profile_y_offset + profile_height, profile_y_offset + profile_height], 
               'k-', linewidth=1)
               
        ax.text(self.chamber_width/2, profile_y_offset + profile_height + 40, 
               "Chamber Position (x-coordinate)", 
               ha='center', va='center')
        
        # Y axis
        ax.plot([0, 0], 
               [profile_y_offset, profile_y_offset + profile_height], 
               'k-', linewidth=1)
               
        y_axis_label = ax.text(-40, profile_y_offset + profile_height/2, 
                             "Velocity (μm/s)", 
                             ha='center', va='center', rotation=90)
        
        # Draw bars
        for sample in self.mock_vertical_samples:
            bar_height = (sample["avgVelocity"] / max_velocity) * profile_height
            
            # Bar
            ax.add_patch(Rectangle((sample["position"] - bar_width/2, 
                                 profile_y_offset + profile_height - bar_height),
                               bar_width, bar_height,
                               color=self.get_color_for_velocity(sample["avgVelocity"]),
                               alpha=0.7))
                               
            # Sample count
            ax.text(sample["position"], profile_y_offset + profile_height - bar_height - 10,
                  str(sample["sampleCount"]),
                  ha='center', va='center', fontsize=8)
                  
            # Position label
            ax.text(sample["position"], profile_y_offset + profile_height + 15,
                  str(sample["position"]),
                  ha='center', va='center', fontsize=8)
        
        # Y-axis ticks
        for tick in range(5):
            tick_value = tick
            tick_y = profile_y_offset + profile_height - (tick/4) * profile_height
            
            # Tick mark
            ax.plot([-5, 5], [tick_y, tick_y], 'k-', linewidth=1)
            
            # Tick label
            ax.text(-10, tick_y, str(tick), ha='right', va='center', fontsize=8)
    
    def draw_tooltip(self, ax):
        """Draw tooltip for hovered track"""
        track_data = next((t for t in self.mock_tracks if t["id"] == self.hovered_track), None)
        
        if not track_data:
            return
            
        # Find frame closest to current frame
        relevant_frame = min(track_data["frames"], 
                            key=lambda f: abs(f["frame"] - self.current_frame))
        
        # Position tooltip near the current position
        x_pos = relevant_frame["x"]
        y_pos = relevant_frame["y"]
        
        # Create text for tooltip
        tooltip_text = (
            f"Cell ID: {self.hovered_track}\n"
            f"Frame: {relevant_frame['frame']}\n"
            f"Position: ({relevant_frame['x']}, {relevant_frame['y']})\n"
            f"Velocity: {relevant_frame['velocity']:.2f} μm/s"
        )
        
        if track_data.get("children") and len(track_data["children"]) > 0:
            tooltip_text += f"\nDivides into: {', '.join(map(str, track_data['children']))}"
        
        # Background rectangle for tooltip
        tooltip = ax.text(x_pos + 10, y_pos - 30, tooltip_text,
                       bbox=dict(facecolor='white', alpha=0.9, 
                               edgecolor='#cccccc', boxstyle='round'),
                       fontsize=8,
                       verticalalignment='top')
    
    def draw_lineage_panel(self, ax):
        """Draw lineage panel for selected track"""
        track_data = next((t for t in self.mock_tracks if t["id"] == self.selected_track), None)
        
        if not track_data:
            return
            
        # Find children and parent
        children = [t for t in self.mock_tracks if t.get("parent") == self.selected_track]
        parent = next((t for t in self.mock_tracks if t["id"] == track_data.get("parent")), None)
        
        # Calculate average velocity
        avg_velocity = sum(f["velocity"] for f in track_data["frames"]) / len(track_data["frames"])
        
        # Create panel content
        panel_text = f"Cell {self.selected_track} Lineage\n\n"
        
        if parent:
            panel_text += f"Parent: Cell {parent['id']}\n\n"
            
        panel_text += f"Current: Cell {self.selected_track}\n"
        panel_text += f"Average Velocity: {avg_velocity:.2f} μm/s\n"
        panel_text += f"Frames: {track_data['frames'][0]['frame']} - {track_data['frames'][-1]['frame']}\n\n"
        
        if children:
            panel_text += "Daughter Cells:\n"
            for child in children:
                panel_text += f"• Cell {child['id']} - Initial Velocity: {child['frames'][0]['velocity']:.1f} μm/s\n"
        
        # Draw panel in the upper right corner
        # Use a light rectangular background for the panel
        panel = ax.text(self.chamber_width + 20, self.chamber_height - 250, panel_text,
                      bbox=dict(facecolor='white', alpha=0.9, 
                              edgecolor='#cccccc', boxstyle='round'),
                      fontsize=10,
                      verticalalignment='top')