import json
import os
from typing import List, Dict, Tuple, Optional

from nd2 import ND2File


# File extensions Partaker is willing to import as image data.
_TIFF_EXTENSIONS = (
    ".tif", ".tiff",
    ".ome.tif", ".ome.tiff",
    ".ome.btf", ".ome.tf2", ".ome.tf8",
)
_ND2_EXTENSIONS = (".nd2",)
_SUPPORTED_EXTENSIONS = _ND2_EXTENSIONS + _TIFF_EXTENSIONS


def _is_nd2(path: str) -> bool:
    return path.lower().endswith(_ND2_EXTENSIONS)


def _is_tiff(path: str) -> bool:
    return path.lower().endswith(_TIFF_EXTENSIONS)


class PositionsMismatchError(ValueError):
    """Raised when ND2 files differ only in the number of positions (P axis)."""

    def __init__(self, existing_p: int, new_p: int, file_path: str):
        super().__init__(
            f"ND2 position mismatch: existing P={existing_p}, new P={new_p} (file: {file_path})"
        )
        self.existing_p = existing_p
        self.new_p = new_p
        self.file_path = file_path


class Experiment:
    """
    Represents a time-lapse microscopy experiment with analysis configuration.

    Attributes:
        name (str): Name of the experiment
        image_files (List[str]): List of paths to image files (ND2 or TIFF)
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
        file_map (Dict): Optional mapping of (P, T, C) tuples to TIFF paths for directory imports
        import_mode (str): Selection mode for TIFF files (e.g. "batch_tiff", "stacked_tiff", "File", "Directory")
        truncate_positions (bool): If True, allow ND2 files with mismatched P to be truncated to smallest P
    """

    def __init__(
            self,
            name: str,
            image_files: Optional[List[str]] = None,
            interval: float = 60.0,
            fluorescence_factor: float = 3.0,
            epsilon: float = 0.1,
            selected_positions: Optional[List[int]] = None,
            time_range: Optional[Tuple[int, int]] = None,
            channel_colors: Optional[Dict[str, str]] = None,
            channel_names: Optional[Dict[str, str]] = None,
            rpu_values: Optional[Dict[str, float]] = None,
            component_intervals: Optional[Dict[str, List[Tuple[float, float]]]] = None,
            focus_loss_intervals: Optional[List[Tuple[float, float]]] = None,
            file_map: Optional[Dict] = None,
            import_mode: Optional[str] = None,
            truncate_positions: bool = False,
            # Back-compat: older callers (and saved configs) used "nd2_files".
            nd2_files: Optional[List[str]] = None,
    ):
        """
        Initialize an experiment with analysis configuration.

        Args:
            name: Name of the experiment
            image_files: List of paths to image files (ND2 or TIFF)
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
            file_map: Optional (P, T, C) -> TIFF path mapping for directory imports
            import_mode: Import mode keyword used by the loader
            truncate_positions: For ND2 with differing P, truncate to smallest instead of erroring
            nd2_files: Deprecated alias for `image_files`; kept for backward compatibility
        """
        # Accept the legacy `nd2_files` kwarg seamlessly.
        if image_files is None:
            image_files = nd2_files if nd2_files is not None else []

        self.name = name
        self.image_files: List[str] = []
        self.phc_interval = interval
        self.fluorescence_factor = fluorescence_factor
        self.epsilon = epsilon
        self.file_map = file_map or {}
        self.import_mode = import_mode

        self.selected_positions = selected_positions if selected_positions is not None else [0, 1, 2, 3]
        self.time_range = time_range if time_range is not None else (0, 5000)

        self.channel_colors = channel_colors if channel_colors is not None else {
            "mcherry": "#FF4444",  # Red
            "yfp": "#FFB347",  # Orange/Yellow
            "1": "#FF4444",
            "2": "#FFB347",
        }

        self.channel_names = channel_names if channel_names is not None else {
            "mcherry": "mCherry",
            "yfp": "YFP",
            "1": "mCherry",
            "2": "YFP",
        }

        self.rpu_values = rpu_values or {}
        self.component_intervals = component_intervals or {}
        self.focus_loss_intervals = focus_loss_intervals or []
        self.base_shape: Tuple[int, ...] = ()
        self.truncate_positions = truncate_positions

        for _file in image_files:
            self.add_image_file(_file)

    # ── Backward-compat property ─────────────────────────────────────
    @property
    def nd2_files(self) -> List[str]:
        """Deprecated alias for `image_files` kept for backward compatibility."""
        return self.image_files

    @nd2_files.setter
    def nd2_files(self, value: List[str]) -> None:
        self.image_files = value

    # ── File management ──────────────────────────────────────────────
    def add_image_file(self, file_path: str) -> None:
        """
        Add a new image file (ND2 or TIFF) to the experiment.

        For TIFF directory imports, file paths are appended without per-file
        shape validation because shapes are reconciled by the loader.

        Args:
            file_path: Path to the image file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be opened or its shape is incompatible
            PositionsMismatchError: For ND2 files where P differs and `truncate_positions` is False
        """
        # TIFF directory imports skip per-file validation; the loader handles it.
        if self.import_mode in ("batch_tiff", "stacked_tiff", "Directory"):
            self.image_files.append(file_path)
            return

        if file_path in self.image_files:
            return

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        try:
            if _is_nd2(file_path):
                with ND2File(file_path) as reader:
                    shape = reader.shape
                    self._check_nd2_shape(shape, file_path)
                    self.image_files.append(file_path)
            elif _is_tiff(file_path):
                import tifffile
                with tifffile.TiffFile(file_path) as tif:
                    shape = tif.series[0].shape
                    self._check_tiff_shape(shape, file_path)
                    self.image_files.append(file_path)
            else:
                raise ValueError(
                    f"Unsupported image format for {file_path}. "
                    f"Expected one of: {_SUPPORTED_EXTENSIONS}"
                )
        except PositionsMismatchError:
            # Let the caller handle the "different positions" UX.
            raise
        except Exception as e:
            raise ValueError(f"Error opening image file {file_path}: {str(e)}")

    # Back-compat: older callers used add_nd2_file.
    def add_nd2_file(self, file_path: str) -> None:
        """Deprecated alias for `add_image_file`."""
        self.add_image_file(file_path)

    def _check_nd2_shape(self, shape, file_path: str) -> None:
        """Validate an incoming ND2 shape against the existing base shape.

        Allows P (position) to differ when `truncate_positions=True`; still
        requires C/Y/X to match.
        """
        if len(self.base_shape) == 0:
            self.base_shape = shape
            return

        if not self.image_files:
            return

        if len(shape) != len(self.base_shape):
            raise ValueError(
                f"File {file_path} has different dimensions ({len(shape)}) "
                f"than existing files ({len(self.base_shape)})."
            )

        # ND2 shape convention: (T, P, C, Y, X). We allow P to differ when
        # `truncate_positions=True`, but still require C/Y/X to match.
        if shape[2:] != self.base_shape[2:]:
            raise ValueError(
                f"File {file_path} shape {shape} is not compatible "
                f"with existing files shape {self.base_shape}."
            )

        existing_p = self.base_shape[1]
        new_p = shape[1]
        if new_p != existing_p and not self.truncate_positions:
            raise PositionsMismatchError(
                existing_p=existing_p, new_p=new_p, file_path=file_path
            )

    def _check_tiff_shape(self, shape, file_path: str) -> None:
        """Validate an incoming TIFF series shape.

        TIFFs may legitimately differ in dimensionality between files (e.g. a
        single-channel stack vs. a multi-channel one). We only enforce that
        Y/X (the trailing two axes) match across the dataset.
        """
        if len(self.base_shape) == 0:
            self.base_shape = shape
            return

        if not self.image_files:
            return

        if shape[-2:] != self.base_shape[-2:]:
            raise ValueError(
                f"File {file_path} shape {shape} is not compatible "
                f"with existing files shape {self.base_shape}."
            )

    # ── Focus loss / time helpers ────────────────────────────────────
    def add_focus_loss_interval(self, start: float, end: float) -> None:
        """Add a time interval (hours) where focus was lost."""
        if start >= end:
            raise ValueError("Start time must be less than end time")
        self.focus_loss_intervals.append((start, end))
        self.focus_loss_intervals.sort(key=lambda x: x[0])

    def is_in_focus(self, time_hours: float) -> bool:
        """Whether the given time point falls outside every focus-loss interval."""
        for start, end in self.focus_loss_intervals:
            if start <= time_hours < end:
                return False
        return True

    @property
    def time_interval_hours(self) -> float:
        """Time interval per analysis frame, in hours."""
        return (self.phc_interval * self.fluorescence_factor) / 3600

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, folder_path: str) -> None:
        """Save experiment configuration to a JSON file inside `folder_path`."""
        config = {
            "name": self.name,
            "image_files": self.image_files,
            # Keep "nd2_files" written for back-compat readers.
            "nd2_files": self.image_files,
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
            "import_mode": self.import_mode,
        }

        file_path = os.path.join(folder_path, "experiment.json")
        with open(file_path, "w") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, folder_path: str) -> "Experiment":
        """Load experiment configuration from a JSON file in `folder_path`."""
        file_path = os.path.join(folder_path, "experiment.json")
        with open(file_path, "r") as f:
            config = json.load(f)

        # Prefer the new key, fall back to the legacy one.
        image_files = config.get("image_files")
        if image_files is None:
            image_files = config.get("nd2_files", [])

        return cls(
            name=config["name"],
            image_files=image_files,
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
            import_mode=config.get("import_mode"),
        )
