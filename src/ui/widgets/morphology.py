from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QComboBox, QTableWidget, QTableWidgetItem, 
                               QMessageBox, QFileDialog, QTabWidget)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
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
    """
    Widget for morphology analysis of segmented cells.
    Uses PyPubSub for communication with other components.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize state variables
        self.cell_mapping = {}
        self.annotated_image = None
        self.current_time = 0
        self.current_position = 0
        self.current_channel = 0
        self.image_data = None
        
        # Define color mapping for morphology classes
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
        
        # Set up the UI
        self.init_ui()
        
        # Subscribe to relevant topics
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_current_view_changed, "current_view_changed")
        pub.subscribe(self.on_segmentation_ready, "segmentation_ready")
        pub.subscribe(self.on_image_ready, "image_ready")
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create QTabWidget for inner tabs
        self.inner_tab_widget = QTabWidget()
        self.scatter_tab = QWidget()
        self.table_tab = QWidget()
        
        # Add tabs to the inner tab widget
        self.inner_tab_widget.addTab(self.scatter_tab, "PCA Plot")
        self.inner_tab_widget.addTab(self.table_tab, "Metrics Table")
        
        # Classify Cells button
        classify_button = QPushButton("Classify Cells")
        classify_button.clicked.connect(self.classify_cells)
        layout.addWidget(classify_button)
        
        # ---- Scatter plot tab layout (PCA) ----
        scatter_layout = QVBoxLayout(self.scatter_tab)
        
        # Annotated image display
        self.annotated_image_label = QLabel("Annotated image will be displayed here.")
        self.annotated_image_label.setFixedSize(300, 300)
        self.annotated_image_label.setAlignment(Qt.AlignCenter)
        self.annotated_image_label.setScaledContents(True)
        scatter_layout.addWidget(self.annotated_image_label)
        
        # Dropdown for selecting metric to color PCA scatter plot
        self.color_dropdown = QComboBox()
        self.color_dropdown.addItems([
            "area", "perimeter", "aspect_ratio", "extent",
            "solidity", "equivalent_diameter", "orientation"
        ])
        
        dropdown_layout = QHBoxLayout()
        dropdown_layout.addWidget(QLabel("Color by:"))
        dropdown_layout.addWidget(self.color_dropdown)
        scatter_layout.addLayout(dropdown_layout)
        
        # PCA scatter plot display
        self.figure_scatter = plt.figure()
        self.canvas_scatter = FigureCanvas(self.figure_scatter)
        scatter_layout.addWidget(self.canvas_scatter)
        
        # Connect dropdown change to PCA plot update
        self.color_dropdown.currentTextChanged.connect(self.update_scatter_plot)
        
        # ---- Table tab layout (Metrics Table) ----
        table_layout = QVBoxLayout(self.table_tab)
        
        # Export to CSV button
        self.export_button = QPushButton("Export to CSV")
        self.export_button.setStyleSheet("background-color: white; color: black; font-size: 14px;")
        self.export_button.clicked.connect(self.export_metrics_to_csv)
        table_layout.addWidget(self.export_button)
        
        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.itemClicked.connect(self.on_table_item_click)
        table_layout.addWidget(self.metrics_table)
        
        # Add the inner tab widget to the main layout
        layout.addWidget(self.inner_tab_widget)

    def classify_cells(self):
        """
        Classify cells in the current image and update displays.
        """
        if not self.image_data:
            QMessageBox.warning(self, "Error", "No image data available.")
            return
            
        # Request current image
        pub.sendMessage("image_request", 
                       time=self.current_time, 
                       position=self.current_position, 
                       channel=self.current_channel)
        
        # Request segmentation
        pub.sendMessage("segmentation_request", 
                       time=self.current_time, 
                       position=self.current_position, 
                       channel=self.current_channel, 
                       model_name=None)
    
    def on_segmentation_ready(self, segmented_image, time, position, channel):
        """
        Handle segmented image data when ready.
        """
        # Check if this is the image we're expecting
        if (time != self.current_time or 
            position != self.current_position or 
            channel != self.current_channel):
            return
        
        if segmented_image is None:
            QMessageBox.warning(self, "Segmentation Error", "Segmentation failed.")
            return
        
        # Store segmentation for later use
        self.current_segmentation = segmented_image
        
        # Request original image data to extract metrics
        pub.sendMessage("image_request",
                      time=time,
                      position=position,
                      channel=channel)
    
    def on_image_ready(self, image, time, position, channel, mode):
        """
        Handle the raw image data for morphology analysis.
        """
        if mode != "normal":
            return
    
        # Check if this is the image we're expecting
        if (time != self.current_time or 
            position != self.current_position or 
            channel != self.current_channel):
            return
            
        # Make sure we have the segmentation
        if not hasattr(self, 'current_segmentation') or self.current_segmentation is None:
            # Request segmentation again
            pub.sendMessage("segmentation_request", 
                          time=time, 
                          position=position, 
                          channel=channel, 
                          model_name=None)
            return
            
        # Extract cells and metrics
        self.cell_mapping = extract_cells_and_metrics(image, self.current_segmentation)
        
        if not self.cell_mapping:
            QMessageBox.warning(self, "No Cells", "No cells detected in the current frame.")
            return
            
        # Annotate the binary segmented image
        self.annotated_image = annotate_binary_mask(self.current_segmentation, self.cell_mapping)
        
        # Display the annotated image
        height, width = self.annotated_image.shape[:2]
        qimage = QImage(
            self.annotated_image.data,
            width,
            height,
            self.annotated_image.strides[0],
            QImage.Format_RGB888
        )
        
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.annotated_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.annotated_image_label.setPixmap(pixmap)
        
        # Update the metrics table and PCA scatter plot
        self.populate_metrics_table()
        self.update_scatter_plot()
    
    def on_image_data_loaded(self, image_data):
        """
        Handle new image data loading notification.
        """
        # Store the image data
        self.image_data = image_data
        
        # Clear any existing data
        self.cell_mapping = {}
        self.annotated_image = None
        if hasattr(self, 'current_segmentation'):
            self.current_segmentation = None
        self.annotated_image_label.clear()
        self.metrics_table.setRowCount(0)
        self.figure_scatter.clear()
        self.canvas_scatter.draw()
    
    def on_current_view_changed(self, time, position, channel):
        """
        Handle notification that the currently displayed image has changed.
        """
        self.current_time = time
        self.current_position = position
        self.current_channel = channel
    
    def populate_metrics_table(self):
        """
        Populate the metrics table with cell data.
        """
        if not self.cell_mapping:
            return
            
        # Convert cell mapping to a DataFrame
        metrics_data = [
            {**{"ID": cell_id}, **data["metrics"]}
            for cell_id, data in self.cell_mapping.items()
        ]
        metrics_df = pd.DataFrame(metrics_data)
        
        # Round numerical columns to 2 decimal places
        numeric_columns = [
            'area', 'perimeter', 'equivalent_diameter', 'orientation',
            'aspect_ratio', 'circularity', 'solidity'
        ]
        
        if not metrics_df.empty:
            # Only round columns that exist in the DataFrame
            existing_numerics = [col for col in numeric_columns if col in metrics_df.columns]
            if existing_numerics:
                metrics_df[existing_numerics] = metrics_df[existing_numerics].round(2)
                
            # Update QTableWidget
            self.metrics_table.setRowCount(len(metrics_df))
            self.metrics_table.setColumnCount(len(metrics_df.columns))
            self.metrics_table.setHorizontalHeaderLabels(metrics_df.columns)
            
            for row in range(len(metrics_df)):
                for col, column_name in enumerate(metrics_df.columns):
                    value = metrics_df.iloc[row, col]
                    # Convert NaN to empty string
                    if pd.isna(value):
                        value = ""
                    self.metrics_table.setItem(row, col, QTableWidgetItem(str(value)))
    
    def update_scatter_plot(self):
        """
        Update the PCA scatter plot with current cell data.
        """
        try:
            if not self.cell_mapping:
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
                'area', 'perimeter', 'equivalent_diameter', 'orientation',
                'aspect_ratio', 'circularity', 'solidity'
            ]
            
            # Ensure all required columns are present
            available_features = [col for col in numeric_features if col in morphology_df.columns]
            
            if len(available_features) < 2:
                return  # Need at least 2 features for PCA
                
            X = morphology_df[available_features].values
            
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
            self.figure_scatter.clear()
            ax = self.figure_scatter.add_subplot(111)
            
            scatter = ax.scatter(
                pca_df['PC1'],
                pca_df['PC2'],
                c=[self.morphology_colors_rgb[class_] for class_ in pca_df['Class']],
                s=50,
                edgecolor='w',
                picker=True)
                
            ax.set_title("PCA Scatter Plot")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            
            # Enable interactive annotations and highlighting
            self.annotate_scatter_points(ax, scatter, pca_df)
            
            self.canvas_scatter.draw()
            
        except Exception as e:
            print(f"Error updating scatter plot: {e}")
    
    def annotate_scatter_points(self, ax, scatter, pca_df):
        """
        Adds interactive hover annotations and click event to highlight selected cells.
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
            index = ind["ind"][0]
            pos = scatter.get_offsets()[index]
            annot.xy = pos
            selected_id = int(pca_df.iloc[index]["ID"])
            cell_class = pca_df.iloc[index]["Class"]
            annot.set_text(f"ID: {selected_id}\nClass: {cell_class}")
            annot.get_bbox_patch().set_alpha(0.8)
            
        def on_hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    self.canvas_scatter.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        self.canvas_scatter.draw_idle()
                        
        def on_click(event):
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    index = ind["ind"][0]
                    selected_id = int(pca_df.iloc[index]["ID"])
                    self.highlight_cell_in_image(selected_id)
                    
        self.canvas_scatter.mpl_connect("motion_notify_event", on_hover)
        self.canvas_scatter.mpl_connect("button_press_event", on_click)
        
        # Add legend
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
    
    def on_table_item_click(self, item):
        """
        Handle clicks on the metrics table.
        """
        row = item.row()
        cell_id_item = self.metrics_table.item(row, 0)
        if cell_id_item:
            cell_id = cell_id_item.text()
            self.highlight_cell_in_image(cell_id)
    
    def highlight_cell_in_image(self, cell_id):
        """
        Highlight a selected cell in the annotated image.
        """
        # Ensure cell_id is an integer
        try:
            cell_id = int(cell_id)
        except ValueError:
            return
            
        if not hasattr(self, "cell_mapping") or not self.cell_mapping or cell_id not in self.cell_mapping:
            return
            
        # Get the bounding box coordinates
        y1, x1, y2, x2 = self.cell_mapping[cell_id]["bbox"]
        
        # Create a copy of the annotated image to highlight the selected cell
        if self.annotated_image is None:
            return
            
        highlighted_image = self.annotated_image.copy()
        
        # Draw a prominent border around the selected cell
        cv2.rectangle(highlighted_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Add text label for cell ID
        cv2.putText(
            highlighted_image,
            f"ID: {cell_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        # Convert to QImage and display
        height, width = highlighted_image.shape[:2]
        qimage = QImage(
            highlighted_image.data,
            width,
            height,
            highlighted_image.strides[0],
            QImage.Format_RGB888
        )
        
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.annotated_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.annotated_image_label.setPixmap(pixmap)
    
    def export_metrics_to_csv(self):
        """
        Export the metrics data to a CSV file.
        """
        if not self.cell_mapping:
            QMessageBox.warning(self, "Error", "No data available for export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Metrics Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Convert cell mapping to DataFrame
            metrics_data = [
                {**{"ID": cell_id}, **data["metrics"]}
                for cell_id, data in self.cell_mapping.items()
            ]
            metrics_df = pd.DataFrame(metrics_data)
            
            # Save to CSV
            metrics_df.to_csv(file_path, index=False)
            
            QMessageBox.information(
                self, "Export Successful", f"Metrics data exported to {file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"An error occurred during export: {str(e)}"
            )