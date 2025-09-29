from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

@dataclass
class TLFrame:
    index: Tuple[int, int]  # Time, Position
    labeled_phc: np.ndarray  # TODO: add typing
    mcherry: Optional[np.ndarray]
    yfp: Optional[np.ndarray]
