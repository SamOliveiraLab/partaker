'''
Singleton that stores the current application state.

Right now it consists of:
- Experiment details, files etc
- The segmentation state
- The collected metrics
'''
from typing import Optional, Tuple

import numpy as np
from pubsub import pub
import threading

class ApplicationState:
    _instance: Optional['ImageData'] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.view_index : Optional[Tuple[int, int, int]] = None
        self.roi_mask : Optional[np.ndarray] = None
        pub.subscribe(self.on_index_changed, "view_index_changed")
        pub.subscribe(self.on_roi_selected, "roi_selected")

    def on_roi_selected(self, mask: np.ndarray) -> None:
        self.roi_mask = mask

    def on_index_changed(self, index: Tuple[int, int, int]) -> None:
        self.view_index = index

    @classmethod
    def get_instance(cls) -> Optional['ApplicationState']:
        """Get the current singleton instance"""
        return cls._instance

    @classmethod
    def create_instance(cls, **kwargs) -> 'ApplicationState':
        """Create/replace the singleton instance"""
        with cls._lock:
            # Create new instance
            instance = cls(**kwargs)
            cls._instance = instance

            return instance