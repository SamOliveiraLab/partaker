"""
Singleton that stores the current application state.

Right now it consists of:
- Experiment details, files, etc.
- The segmentation state
- The collected metrics
- The currently selected segmentation model
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from pubsub import pub

from partaker.data.experiment import Experiment
from partaker.data.image_data import ImageData


class ApplicationState:
    _instance: Optional["ApplicationState"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.view_index: Optional[tuple[int, int, int]] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.experiment: Optional[Experiment] = None
        self.image_data: Optional[ImageData] = None
        self.selected_model: Optional[str] = None

        pub.subscribe(self.on_index_changed, "view_index_changed")
        pub.subscribe(self.on_roi_selected, "roi_selected")
        pub.subscribe(self.on_experiment_loaded, "experiment_loaded")
        pub.subscribe(self.on_image_data_loaded, "image_data_loaded")
        pub.subscribe(self.on_selected_model_changed, "selected_model_changed")

    def on_roi_selected(self, mask: np.ndarray, p) -> None:
        self.roi_mask = mask

    def on_index_changed(self, index: tuple[int, int, int]) -> None:
        self.view_index = index

    def on_experiment_loaded(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def on_image_data_loaded(self, image_data: ImageData) -> None:
        self.image_data = image_data

    def on_selected_model_changed(self, model: str) -> None:
        self.selected_model = model

    @classmethod
    def get_instance(cls) -> Optional["ApplicationState"]:
        """Return the current singleton instance, if it exists."""
        return cls._instance

    @classmethod
    def create_instance(cls, **kwargs) -> "ApplicationState":
        """Create and store a new singleton instance."""
        with cls._lock:
            instance = cls(**kwargs)
            cls._instance = instance
            return instance
