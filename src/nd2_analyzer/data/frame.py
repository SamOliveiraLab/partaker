from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TLFrame:
    index: Tuple[int, int]  # Time, Position
    labeled_phc: np.ndarray  # TODO: add typing
    mcherry: np.ndarray
    yfp: np.ndarray
