'''
Singleton that stores the current application state.

Right now it consists of:
- Experiment details, files etc
- The segmentation state
- The collected metrics
'''

from .experiment import Experiment
from .image_data import ImageData
from ..analysis.metrics_service import MetricsService


class ApplicationState:
    def __init__(self) -> None:
        self.experiment : Experiment = None
        #self.segmentation
        self.metricsService : MetricsService = MetricsService()

    def save(self, file_path: str) -> None:
        # Save experiment
        # Save segmentation
        # Save metrics

        pass

    def load(self, file_path: str) -> None:
        pass
