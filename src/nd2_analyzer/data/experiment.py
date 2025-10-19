import json
import os
from typing import List, Dict, Tuple, Optional

from nd2 import ND2File


class Experiment:
    """
    Represents a time-lapse microscopy experiment with analysis configuration.

    Attributes:
        name (str): Name of the experiment
        nd2_files (List[str]): List of paths to ND2 files
        phc_interval (float): Time step between phase contrast frames in seconds
        fluorescence_factor (float): Factor for fluorescence imaging frequency relative to PHC
        epsilon (float): Minimum fluorescence threshold for filtering
        selected_positions (List[int]): List of selected positions to analyze
        time_range (Tuple[int, int]): Time range for analysis (start, end) in frames
        channel_colors (Dict[str, str]): Mapping of channel names to color hex codes
        channel_names (Dict[str, str]): Mapping of channel identifiers to display names
        rpu_values (Dict[str, float]): Dictionary of RPU calibration values
        component_intervals (Dict[str, List[Tuple[float, float]]]): Component activation intervals in hours
        focus_loss_intervals (List[Tuple[float, float]]): Time intervals where focus was lost (in hours)
    """

    def __init__(
            self,
            name: str,
            nd2_files: List[str],
            interval: float,
            fluorescence_factor: float = 3.0,
            epsilon: float = 0.1,
            selected_positions: Optional[List[int]] = None,
            time_range: Optional[Tuple[int, int]] = None,
            channel_colors: Optional[Dict[str, str]] = None,
            channel_names: Optional[Dict[str, str]] = None,
            rpu_values: Optional[Dict[str, float]] = None,
            component_intervals: Optional[Dict[str, List[Tuple[float, float]]]] = None,
            focus_loss_intervals: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Initialize an experiment with analysis configuration.

        Args:
            name: Name of the experiment
            nd2_files: List of paths to ND2 files
            interval: Time step between PHC frames in seconds
            fluorescence_factor: Factor for fluorescence imaging frequency
            epsilon: Minimum fluorescence threshold
            selected_positions: List of positions to analyze (default: [0, 1, 2, 3])
            time_range: Time range for analysis (default: (0, 5000))
            channel_colors: Channel color mappings
            channel_names: Channel name mappings
            rpu_values: RPU calibration values
            component_intervals: Component activation time intervals
            focus_loss_intervals: Time intervals where autofocus failed (in hours)
        """
        self.name = name
        self.nd2_files = []
        self.phc_interval = interval
        self.fluorescence_factor = fluorescence_factor
        self.epsilon = epsilon

        # Set defaults for optional parameters
        self.selected_positions = selected_positions if selected_positions is not None else [0, 1, 2, 3]
        self.time_range = time_range if time_range is not None else (0, 5000)

        # Channel configuration defaults
        self.channel_colors = channel_colors if channel_colors is not None else {
            "mcherry": "#FF4444",  # Red
            "yfp": "#FFB347",  # Orange/Yellow
            "1": "#FF4444",
            "2": "#FFB347"
        }

        self.channel_names = channel_names if channel_names is not None else {
            "mcherry": "mCherry",
            "yfp": "YFP",
            "1": "mCherry",
            "2": "YFP"
        }

        self.rpu_values = rpu_values or {}
        self.component_intervals = component_intervals or {}
        self.focus_loss_intervals = focus_loss_intervals or []
        self.base_shape = ()

        # Add ND2 files
        for _file in nd2_files:
            self.add_nd2_file(_file)

    def add_nd2_file(self, file_path: str) -> None:
        """
        Add a new ND2 file to the experiment.

        Args:
            file_path: Path to the ND2 file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be opened as an ND2 file or if its shape is incompatible
        """
        if file_path in self.nd2_files:
            return

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        try:
            with ND2File(file_path) as reader:
                shape = reader.shape

                if len(self.base_shape) == 0:
                    self.base_shape = shape
                elif self.nd2_files:
                    if len(shape) != len(self.base_shape):
                        raise ValueError(
                            f"File {file_path} has different dimensions ({len(shape)}) "
                            f"than existing files ({len(self.base_shape)})."
                        )
                    if shape[1:] != self.base_shape[1:]:
                        raise ValueError(
                            f"File {file_path} shape {shape} is not compatible "
                            f"with existing files shape {self.base_shape}."
                        )

                self.nd2_files.append(file_path)

        except Exception as e:
            raise ValueError(f"Error opening ND2 file {file_path}: {str(e)}")

    def add_focus_loss_interval(self, start: float, end: float) -> None:
        """
        Add a time interval where focus was lost.

        Args:
            start: Start time in hours
            end: End time in hours

        Raises:
            ValueError: If start >= end
        """
        if start >= end:
            raise ValueError("Start time must be less than end time")
        self.focus_loss_intervals.append((start, end))
        # Keep intervals sorted by start time
        self.focus_loss_intervals.sort(key=lambda x: x[0])

    def is_in_focus(self, time_hours: float) -> bool:
        """
        Check if the given time point is in focus (not in any loss interval).

        Args:
            time_hours: Time point in hours

        Returns:
            True if in focus, False if in a focus loss interval
        """
        for start, end in self.focus_loss_intervals:
            if start <= time_hours < end:
                return False
        return True

    @property
    def time_interval_hours(self) -> float:
        """Calculate time interval in hours for analysis"""
        return (self.phc_interval * self.fluorescence_factor) / 3600

    def save(self, folder_path: str) -> None:
        """
        Save experiment configuration to a JSON file.

        Args:
            folder_path: Path to save the configuration
        """
        config = {
            "name": self.name,
            "nd2_files": self.nd2_files,
            "interval": self.phc_interval,
            "fluorescence_factor": self.fluorescence_factor,
            "epsilon": self.epsilon,
            "selected_positions": self.selected_positions,
            "time_range": self.time_range,
            "channel_colors": self.channel_colors,
            "channel_names": self.channel_names,
            "rpu_values": self.rpu_values,
            "component_intervals": self.component_intervals,
            "focus_loss_intervals": self.focus_loss_intervals,
        }

        file_path = os.path.join(folder_path, "experiment.json")
        with open(file_path, "w") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, folder_path: str) -> "Experiment":
        """
        Load experiment configuration from a JSON file.

        Args:
            folder_path: Path to the configuration file

        Returns:
            An Experiment instance
        """
        file_path = os.path.join(folder_path, "experiment.json")
        with open(file_path, "r") as f:
            config = json.load(f)

        return cls(
            name=config["name"],
            nd2_files=config["nd2_files"],
            interval=config["interval"],
            fluorescence_factor=config.get("fluorescence_factor", 3.0),
            epsilon=config.get("epsilon", 0.1),
            selected_positions=config.get("selected_positions"),
            time_range=config.get("time_range"),
            channel_colors=config.get("channel_colors"),
            channel_names=config.get("channel_names"),
            rpu_values=config.get("rpu_values"),
            component_intervals=config.get("component_intervals"),
            focus_loss_intervals=config.get("focus_loss_intervals", []),
        )

    def filter_track_frames(self, track):
        """
        Remove frames from track that fall within focus loss intervals.

        Parameters:
        -----------
        track : dict
            Track with 'x', 'y', 't' keys

        Returns:
        --------
        dict or None
            Filtered track, or None if all frames excluded
        """
        import numpy as np

        if not self.focus_loss_intervals:
            return track

        x = np.array(track["x"])
        y = np.array(track["y"])
        t = np.array(track.get("t", range(len(x))))

        # Find valid (in-focus) indices
        valid = []
        for i, frame in enumerate(t):
            time_h = frame * self.time_interval_hours
            if all(not (start <= time_h < end) for start, end in self.focus_loss_intervals):
                valid.append(i)

        if not valid:
            return None

        return {
            "ID": track["ID"],
            "x": x[valid].tolist(),
            "y": y[valid].tolist(),
            "t": t[valid].tolist(),
            "parent": track.get("parent"),
            "children": track.get("children", [])
        }
