from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class TLFrame:
    index: Tuple[int, int]  # Time, Position
    labeled_phc: np.ndarray  # TODO: add typing
    mcherry: Optional[np.ndarray]
    yfp: Optional[np.ndarray]
    # Bright-field / phase (channel 0); used as the intensity source when no
    # fluorescence channels exist (e.g. single-channel TIFF imports).
    phase: Optional[np.ndarray] = None
