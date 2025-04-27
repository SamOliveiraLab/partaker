from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QComboBox, QTableWidget, QTableWidgetItem, 
                               QMessageBox, QFileDialog, QTabWidget)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pubsub import pub
import cv2

from morphology import (classify_morphology, extract_cells_and_metrics, 
                       annotate_binary_mask, extract_cell_morphologies)

class MorphologyWidget(QWidget):
    def __init__(self, parent=None, metrics_service=None):
        super().__init__(parent)
        self.cell_mapping = {}
        self.selected_cell_id = None
        self.tracked_cell_lineage = {}
        
        # Store reference to metrics service
        self.metrics_service = metrics_service
        
        # Set up morphology colors (same as original)
        self.morphology_colors = {
            "Artifact": (128, 128, 128),  # Gray
            "Divided": (255, 0, 0),       # Blue
            "Healthy": (0, 255, 0),       # Green
            "Elongated": (0, 255, 255),   # Yellow
            "Deformed": (255, 0, 255),    # Magenta
        }
        
        self.morphology_colors_rgb = {
            key: (color[2] / 255, color[1] / 255, color[0] / 255)
            for key, color in self.morphology_colors.items()
        }
        
        # Initialize UI components
        self.initUI()
        
        # Subscribe to relevant messages
        pub.subscribe(self.on_cell_mapping_updated, "cell_mapping_updated")
        pub.subscribe(self.on_segmentation_ready, "segmentation_ready")
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_current_view_changed, "current_view_changed")
        pub.subscribe(self.set_tracking_data, "tracking_data_available")
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Add the fetch metrics button at the top
        self.fetch_metrics_button = QPushButton("Get Metrics from Service")
        self.fetch_metrics_button.clicked.connect(self.fetch_metrics_from_service)
        self.fetch_metrics_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        layout.addWidget(self.fetch_metrics_button)
        
        # Create QTabWidget for inner tabs
        inner_tab_widget = QTabWidget()
        self.scatter_tab = QWidget()
        self.table_tab = QWidget()
        
        # Add tabs to the inner tab widget
        inner_tab_widget.addTab(self.scatter_tab, "PCA Plot")
        inner_tab_widget.addTab(self.table_tab, "Metrics Table")
        
        # Set up scatter plot tab (PCA)
        scatter_layout = QVBoxLayout(self.scatter_tab)
        
        # Annotated image display
        self.annotated_image_label = QLabel("Annotated image will be displayed here.")
        self.annotated_image_label.setFixedSize(300, 300)
        self.annotated_image_label.setAlignment(Qt.AlignCenter)
        self.annotated_image_label.setScaledContents(True)
        scatter_layout.addWidget(self.annotated_image_label)
        
        # PCA scatter plot display (no color by dropdown)
        self.figure_annot_scatter = plt.figure()
        self.canvas_annot_scatter = FigureCanvas(self.figure_annot_scatter)
        scatter_layout.addWidget(self.canvas_annot_scatter)
        
        # Metrics table tab
        table_layout = QVBoxLayout(self.table_tab)
        
        # Add Export button with tracking GIF capability
        self.button_layout = QHBoxLayout()
        
        # Export to CSV button
        self.export_button = QPushButton("Export to CSV")
        self.export_button.setStyleSheet("background-color: white; color: black; font-size: 14px;")
        self.export_button.clicked.connect(self.export_metrics_to_csv)
        self.button_layout.addWidget(self.export_button)
        
        # Save GIF button (initially hidden, will appear after cell selection)
        self.save_gif_button = None
        
        table_layout.addLayout(self.button_layout)
        
        # Create metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.itemClicked.connect(self.on_table_item_click)
        table_layout.addWidget(self.metrics_table)
        
        # Add tabs to main layout
        layout.addWidget(inner_tab_widget)
    
    def fetch_metrics_from_service(self):
        """Fetch cell metrics and classification from the metrics service dataframe"""
        if not self.metrics_service:
            QMessageBox.warning(self, "Error", "Metrics service not available.")
            return
        
        try:
            # Get current frame information to fetch relevant metrics
            # (These would be provided by pub/sub messages in real implementation)
            t = pub.sendMessage("get_current_t", default=0)
            p = pub.sendMessage("get_current_p", default=0)
            c = pub.sendMessage("get_current_c", default=0)
            
            print(f"Fetching metrics for T:{t}, P:{p}, C:{c}")
            
            # Request cell metrics from the service for current frame
            polars_df = self.metrics_service.query(time=t, position=p, channel=c)
            
            if polars_df.is_empty():
                print("No metrics found in polars dataframe")
                QMessageBox.warning(self, "No Data", "No metrics available for the current frame.")
                return
            
            print(f"Retrieved polars dataframe with shape: {polars_df.shape}")
            print(f"Columns: {polars_df.columns}")
            
            # Convert polars to pandas
            metrics_df = polars_df.to_pandas()
            print(f"Converted to pandas dataframe with shape: {metrics_df.shape}")
            print(f"Pandas columns: {metrics_df.columns}")
            
            # Convert the metrics dataframe to the cell_mapping format expected by existing code
            self.cell_mapping = {}
            
            for idx, row in metrics_df.iterrows():
                # Use 'cell_id' instead of 'ID'
                if 'cell_id' in row:
                    cell_id = int(row['cell_id'])
                else:
                    print(f"Warning: Row {idx} missing cell_id column")
                    continue
                
                # Extract bounding box - if not available, use defaults
                if all(col in row for col in ['y1', 'x1', 'y2', 'x2']):
                    bbox = (
                        int(row['y1']), 
                        int(row['x1']), 
                        int(row['y2']), 
                        int(row['x2'])
                    )
                else:
                    # Use centroid to create a small bounding box if available
                    if 'centroid_y' in row and 'centroid_x' in row:
                        cy, cx = row['centroid_y'], row['centroid_x']
                        bbox = (int(cy-5), int(cx-5), int(cy+5), int(cx+5))
                        print(f"Created bbox from centroid for cell {cell_id}: {bbox}")
                    else:
                        bbox = (0, 0, 10, 10)  # Default fallback
                        print(f"Using default bbox for cell {cell_id}")
                
                # Extract metrics dictionary - exclude certain columns
                exclude_cols = ['cell_id', 'y1', 'x1', 'y2', 'x2']
                metrics = {col: row[col] for col in row.index if col not in exclude_cols}
                
                # Create cell mapping entry
                self.cell_mapping[cell_id] = {
                    "bbox": bbox,
                    "metrics": metrics
                }
            
            print(f"Created cell_mapping with {len(self.cell_mapping)} entries")
            
            # Update the UI with the fetched metrics
            self.populate_metrics_table()
            self.update_annotation_scatter()
            
            # Provide user feedback
            QMessageBox.information(
                self, 
                "Metrics Loaded", 
                f"Successfully loaded metrics for {len(self.cell_mapping)} cells from metrics service."
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error fetching metrics: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to fetch metrics: {str(e)}")
    
    def on_table_item_click(self, item):
        """Handle clicks on the metrics table to select and track cells"""
        row = item.row()
        cell_id = self.metrics_table.item(row, 0).text()
        cell_id = int(cell_id)
        
        # Highlight the cell in the current frame
        self.highlight_cell_in_image(cell_id)
        
        # Set up tracking for this cell across frames
        print(f"Selected cell {cell_id} for tracking from table")
        self.select_cell_for_tracking(cell_id)
    
    def select_cell_for_tracking(self, cell_id):
        """Select a specific cell to track across frames."""
        print(f"Selected cell {cell_id} for tracking")
        self.selected_cell_id = cell_id
        
        # Initialize lineage_tracks if it doesn't exist
        if not hasattr(self, "lineage_tracks"):
            self.lineage_tracks = []
            print("Initializing empty lineage_tracks")
        
        # Check if tracking data is available
        if not self.lineage_tracks:
            reply = QMessageBox.question(
                self, "No Tracking Data",
                "No tracking data is available. Do you want to request tracking data?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                # Send message to generate tracking data
                print("Requesting tracking data generation")
                pub.sendMessage("track_cells_requested")
                QMessageBox.information(
                    self, "Tracking Requested", 
                    "Tracking data generation has been requested. Please try tracking again once it's complete."
                )
            else:
                print("User canceled tracking request")
            
            self.selected_cell_id = None
            return
            
        # If we have tracking data, proceed with finding the cell
        self.find_cell_in_tracking_data(cell_id)
        
    def set_tracking_data(self, lineage_tracks):
        """Set tracking data from external source"""
        print(f"Received tracking data with {len(lineage_tracks)} tracks")
        self.lineage_tracks = lineage_tracks
        
        # If we have a previously selected cell, try to track it now
        if hasattr(self, "selected_cell_id") and self.selected_cell_id:
            print(f"Re-attempting to track previously selected cell {self.selected_cell_id}")
            self.find_cell_in_tracking_data(self.selected_cell_id)
    
    def find_cell_in_tracking_data(self, cell_id):
        """Find a cell in the tracking data and prepare tracking information"""
        print(f"Finding cell {cell_id} in tracking data...")
        
        # Clear previous tracking
        self.tracked_cell_lineage = {}
        
        # Find the track for this cell
        selected_track = None
        for track in self.lineage_tracks:
            if track['ID'] == cell_id:
                selected_track = track
                break
                
        if not selected_track:
            print(f"Cell {cell_id} not found in tracking data")
            QMessageBox.warning(
                self, "Cell Not Found",
                f"Cell {cell_id} not found in tracking data."
            )
            return
            
        # Get all frames where this cell appears
        if 't' in selected_track:
            t_values = selected_track['t']
            print(f"Cell {cell_id} appears in frames: {t_values}")
            
            # Map each frame to this cell ID
            for t in t_values:
                if t not in self.tracked_cell_lineage:
                    self.tracked_cell_lineage[t] = []
                self.tracked_cell_lineage[t].append(cell_id)
                
            # Get children cells if any
            if 'children' in selected_track and selected_track['children']:
                print(f"Cell {cell_id} has children: {selected_track['children']}")
                self.add_children_to_tracking(selected_track['children'])
                
        QMessageBox.information(
            self, "Cell Tracking Prepared",
            f"Cell {cell_id} will be tracked across {len(self.tracked_cell_lineage)} frames.\n"
            f"Use the time slider to navigate frames."
        )
        
        # Notify that tracking data is ready
        pub.sendMessage("cell_tracking_ready", cell_id=cell_id, lineage_data=self.tracked_cell_lineage)
    
    def add_children_to_tracking(self, child_ids):
        """Add children cells to tracking data recursively"""
        for child_id in child_ids:
            # Find the child's track
            child_track = None
            for track in self.lineage_tracks:
                if track['ID'] == child_id:
                    child_track = track
                    break
                    
            if child_track and 't' in child_track:
                print(f"Adding child {child_id} to tracking")
                t_values = child_track['t']
                
                # Map each frame to this child ID
                for t in t_values:
                    if t not in self.tracked_cell_lineage:
                        self.tracked_cell_lineage[t] = []
                    self.tracked_cell_lineage[t].append(child_id)
                    
                # Recursively add this child's children
                if 'children' in child_track and child_track['children']:
                    print(f"Child {child_id} has children: {child_track['children']}")
                    self.add_children_to_tracking(child_track['children'])
    
    def save_tracked_cell_as_gif(self):
        """Save the movement of the tracked cell as an animated GIF."""
        if not hasattr(self, "tracked_cell_lineage") or not self.tracked_cell_lineage:
            QMessageBox.warning(self, "Error", "No cell is being tracked.")
            return
            
        # Ask user for save location
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Cell Movement as GIF", "", "GIF Files (*.gif)"
        )
        if not save_path:
            return
            
        # Request GIF creation via pub/sub
        pub.sendMessage("create_tracking_gif_requested", 
                       tracked_data=self.tracked_cell_lineage,
                       save_path=save_path)
    
    def highlight_cell_in_image(self, cell_id):
        """Highlight a specific cell in the current image view"""
        if not hasattr(self, "cell_mapping") or not self.cell_mapping:
            QMessageBox.warning(
                self, "Error", 
                "No cell mapping data available. Did you classify cells first?"
            )
            return
            
        if cell_id not in self.cell_mapping:
            QMessageBox.warning(
                self, "Error",
                f"Cell ID {cell_id} not found in cell mapping."
            )
            return
            
        # Send message to highlight the cell in main view
        pub.sendMessage("highlight_cell_requested", cell_id=cell_id)
    
    def update_annotation_scatter(self):
        """Update the PCA scatter plot with current cell morphology data"""
        try:
            if not hasattr(self, "cell_mapping") or not self.cell_mapping:
                return
                
            # Prepare DataFrame
            metrics_data = [
                {**{"ID": cell_id}, **data["metrics"], **
                    {"Class": data["metrics"]["morphology_class"]}}
                for cell_id, data in self.cell_mapping.items()
            ]
            morphology_df = pd.DataFrame(metrics_data)
            
            # Select numeric features for PCA
            numeric_features = [
                'area',
                'perimeter',
                'equivalent_diameter',
                'orientation',
                'aspect_ratio',
                'circularity',
                'solidity']
            X = morphology_df[numeric_features].values
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)
            
            # Store PCA results
            pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
            pca_df['Class'] = morphology_df['Class']
            pca_df['ID'] = morphology_df['ID']
            
            # Plot PCA scatter
            self.figure_annot_scatter.clear()
            ax = self.figure_annot_scatter.add_subplot(111)
            scatter = ax.scatter(
                pca_df['PC1'],
                pca_df['PC2'],
                c=[
                    self.morphology_colors_rgb[class_] for class_ in pca_df['Class']],
                s=50,
                edgecolor='w',
                picker=True)
                
            ax.set_title("PCA Scatter Plot")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            
            # Enable interactive annotations and highlighting
            self.annotate_scatter_points(ax, scatter, pca_df)
            
            self.canvas_annot_scatter.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
    
    def annotate_scatter_points(self, ax, scatter, pca_df):
        """
        Adds interactive hover annotations and click event to highlight a selected cell.
        """
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            ha="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)
        
        def update_annot(ind):
            """Update annotation text and position based on hovered point."""
            index = ind["ind"][0]
            pos = scatter.get_offsets()[index]
            annot.xy = pos
            selected_id = int(pca_df.iloc[index]["ID"])
            cell_class = pca_df.iloc[index]["Class"]
            annot.set_text(f"ID: {selected_id}\nClass: {cell_class}")
            annot.get_bbox_patch().set_alpha(0.8)
            
        def on_hover(event):
            """Handle hover events to show annotations."""
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    self.canvas_annot_scatter.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        self.canvas_annot_scatter.draw_idle()
                        
        def on_click(event):
            """Handle click events to highlight the selected cell in the segmented image."""
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    index = ind["ind"][0]
                    selected_id = int(pca_df.iloc[index]["ID"])
                    cell_class = pca_df.iloc[index]["Class"]
                    self.highlight_cell_in_image(selected_id)
                    
        self.canvas_annot_scatter.mpl_connect("motion_notify_event", on_hover)
        self.canvas_annot_scatter.mpl_connect("button_press_event", on_click)
        
        # Add legend using self.morphology_colors_rgb
        handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                color=color,
                label=key,
                markersize=8,
                linestyle='',
            )
            for key, color in self.morphology_colors_rgb.items()
        ]
        ax.legend(
            handles=handles,
            title="Morphology Class",
            loc='best',
        )
    
    
    def populate_metrics_table(self):
        """Populate the metrics table with cell data, showing only classification-relevant columns"""
        if not self.cell_mapping:
            return
            
        # Convert cell mapping to a DataFrame
        metrics_data = [
            {**{"ID": cell_id}, **data["metrics"]}
            for cell_id, data in self.cell_mapping.items()
        ]
        metrics_df = pd.DataFrame(metrics_data)
        
        print(f"Created metrics dataframe with columns: {metrics_df.columns}")
        
        # Select only classification-relevant columns
        # Include "ID" first, then morphology class, then relevant metrics
        columns_to_show = ["ID", "morphology_class", "area", "perimeter", 
                        "aspect_ratio", "circularity", "solidity", 
                        "equivalent_diameter"]
        
        # Only keep columns that exist in the dataframe
        available_columns = [col for col in columns_to_show if col in metrics_df.columns]
        
        # Filter dataframe to show only selected columns
        metrics_df = metrics_df[available_columns]
        
        # Round numeric columns to 2 decimal places (excluding ID and morphology_class)
        numeric_columns = [col for col in available_columns 
                        if col not in ["ID", "morphology_class"]]
        
        if numeric_columns:
            metrics_df[numeric_columns] = metrics_df[numeric_columns].round(2)
        
        # Update QTableWidget
        self.metrics_table.setRowCount(metrics_df.shape[0])
        self.metrics_table.setColumnCount(metrics_df.shape[1])
        self.metrics_table.setHorizontalHeaderLabels(metrics_df.columns)
        
        for row in range(metrics_df.shape[0]):
            for col, value in enumerate(metrics_df.iloc[row]):
                self.metrics_table.setItem(
                    row, col, QTableWidgetItem(str(value)))
                
                # Highlight the morphology class column with background color
                if metrics_df.columns[col] == "morphology_class":
                    morph_class = str(value)
                    if morph_class in self.morphology_colors:
                        item = self.metrics_table.item(row, col)
                        bgr_color = self.morphology_colors[morph_class]
                        # Convert BGR to RGB for Qt
                        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
                        item.setBackground(QColor(*rgb_color))
                        # Use contrasting text color
                        item.setForeground(QColor(255, 255, 255) if sum(rgb_color) < 384 else QColor(0, 0, 0))
                        
    
    def export_metrics_to_csv(self):
        """Exports the metrics table data to a CSV file."""
        try:
            if not self.cell_mapping:
                QMessageBox.warning(self, "Error", "No cell data available.")
                return
                
            metrics_data = [
                {**{"ID": cell_id}, **data["metrics"]}
                for cell_id, data in self.cell_mapping.items()
            ]
            metrics_df = pd.DataFrame(metrics_data)
            
            if metrics_df.empty:
                QMessageBox.warning(
                    self, "Error", "No data available to export.")
                return
                
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Metrics Data", "", "CSV Files (*.csv);;All Files (*)")
            if save_path:
                metrics_df.to_csv(save_path, index=False)
                QMessageBox.information(
                    self, "Success", f"Metrics data exported to {save_path}"
                )
            else:
                QMessageBox.warning(self, "Cancelled", "Export cancelled.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    # Handlers for pub/sub messages
    def on_cell_mapping_updated(self, cell_mapping):
        """Handler for updated cell mapping data"""
        self.cell_mapping = cell_mapping
        self.populate_metrics_table()
        self.update_annotation_scatter()
    
    def on_segmentation_ready(self, segmentation_data):
        """Handler for when segmentation is ready"""
        # Process segmentation results if needed
        pass
    
    def on_image_data_loaded(self, image_data):
        """Handler for when new image data is loaded"""
        # Reset state for new image data
        self.selected_cell_id = None
        self.tracked_cell_lineage = {}
        if self.save_gif_button:
            self.save_gif_button.setVisible(False)
    
    def on_current_view_changed(self, t, p, c):
        """Handler for when the current view changes (time/position/channel)"""
        # Update display if needed for new t/p/c values
        pass