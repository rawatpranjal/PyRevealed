"""Deprecated: moved to pyrevealed.contrib.intertemporal."""
import warnings as _warnings
_warnings.warn(
    "pyrevealed.algorithms.intertemporal has moved to pyrevealed.contrib.intertemporal. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
import pyrevealed.contrib.intertemporal as _mod
import sys as _sys
_self = _sys.modules[__name__]
for _name in dir(_mod):
    setattr(_self, _name, getattr(_mod, _name))
