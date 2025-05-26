from pubsub import pub
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPen, QPixmap, QImage, QPolygonF
from PySide6.QtWidgets import (
    QDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout
)
import numpy as np
import cv2

class ROISelectorDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('ROI Selector')

        # Main layout
        main_layout = QVBoxLayout(self)

        # Instructions label
        instructions = QLabel("Click to add points to the polygon. Double-click to complete.")
        main_layout.addWidget(instructions)

        # Graphics view for the image and ROI drawing
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        main_layout.addWidget(self.view)

        # Button layout
        button_layout = QHBoxLayout()

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_polygon)
        button_layout.addWidget(self.reset_button)

        # Complete button
        self.complete_button = QPushButton("Complete Polygon")
        self.complete_button.clicked.connect(self.complete_polygon)
        button_layout.addWidget(self.complete_button)

        # Accept button
        self.accept_button = QPushButton("Accept ROI")
        self.accept_button.clicked.connect(self.accept_roi)
        self.accept_button.setEnabled(False)
        button_layout.addWidget(self.accept_button)

        main_layout.addLayout(button_layout)

        # Initialize polygon drawing
        self.points = []
        self.polygon = None
        self.drawing = True
        self.mask = None

        # Connect mouse events
        self.view.mousePressEvent = self.mouse_press_event

        # Resize window
        self.resize(800, 600)

        # Subscribe to slider values response
        pub.subscribe(self._on_slider_vals_response, "view_area_slider_vals_response")

        # Subscribe to image response
        pub.subscribe(self._on_image_response, "image_ready")

        # Variables to hold slider values and image
        self.slider_vals = None
        self.image = None
        self.image_shape = None

        # Request slider values
        pub.sendMessage("view_area_slider_vals_request")

    def _on_slider_vals_response(self, slider_vals):
        # Receive slider values and request image from ImageData
        self.slider_vals = slider_vals
        # Assuming slider_vals contains time, position, channel
        t = slider_vals.get('time', 0)
        p = slider_vals.get('position', 0)
        c = slider_vals.get('channel', 0)
        pub.sendMessage("raw_image_request", time=t, position=p, channel=c)

    def _on_image_response(self, image, time, position, channel, mode):
        # Receive image from ImageData
        self.image = image
        if isinstance(image, np.ndarray):
            self.image_shape = image.shape[:2]
        else:
            self.image_shape = (image.height(), image.width())

        # Convert numpy array to QImage/QPixmap
        if isinstance(image, np.ndarray):
            if image.ndim == 2:  # Grayscale
                height, width = image.shape
                if image.dtype != np.uint8:
                    image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
                qimg = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
            else:  # RGB
                height, width, channels = image.shape
                qimg = QImage(image.data, width, height, width * channels, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
        else:
            pixmap = QPixmap(image)

        # Clear scene and add image
        self.scene.clear()
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Reset polygon drawing
        self.reset_polygon()

    def mouse_press_event(self, event):
        if not self.drawing:
            return

        scene_pos = self.view.mapToScene(event.pos())
        self.points.append((scene_pos.x(), scene_pos.y()))
        self.draw_polygon()

        if event.type() == QEvent.Type.MouseButtonDblClick:
            self.complete_polygon()

    def draw_polygon(self):
        if self.polygon is not None:
            self.scene.removeItem(self.polygon)

        if len(self.points) > 1:
            polygon = QPolygonF([QPointF(x, y) for x, y in self.points])
            self.polygon = self.scene.addPolygon(polygon, QPen(Qt.red, 2))

    def complete_polygon(self):
        if len(self.points) < 3:
            return

        self.drawing = False

        if self.points[0] != self.points[-1]:
            self.points.append(self.points[0])

        self.draw_polygon()

        if self.polygon is not None:
            self.polygon.setBrush(QColor(255, 0, 0, 50))

        self.accept_button.setEnabled(True)

    def reset_polygon(self):
        if self.polygon is not None:
            self.scene.removeItem(self.polygon)
            self.polygon = None

        self.points = []
        self.drawing = True
        self.accept_button.setEnabled(False)

    def create_mask(self):
        if self.image_shape is None:
            return

        height, width = self.image_shape
        self.mask = np.zeros((height, width), dtype=np.uint8)

        image_points = []
        for x, y in self.points:
            ix = int(x)
            iy = int(y)
            ix = max(0, min(ix, width - 1))
            iy = max(0, min(iy, height - 1))
            image_points.append((ix, iy))

        points_array = np.array(image_points, dtype=np.int32)
        cv2.fillPoly(self.mask, [points_array], 255)

    def accept_roi(self):
        self.create_mask()
        # Publish the ROI mask
        pub.sendMessage("set_segmentation_roi", binary_mask=self.mask)
        self.accept()

    def resizeEvent(self, event):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
