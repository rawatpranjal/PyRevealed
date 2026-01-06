"""Type aliases for PyRevealed."""

from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

# Matrix types
FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

# Cycle representation (tuple of observation indices)
Cycle: TypeAlias = tuple[int, ...]
