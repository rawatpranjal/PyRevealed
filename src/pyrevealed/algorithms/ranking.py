"""Deprecated: moved to pyrevealed.contrib.ranking."""
import warnings as _warnings
_warnings.warn(
    "pyrevealed.algorithms.ranking has moved to pyrevealed.contrib.ranking. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
import pyrevealed.contrib.ranking as _mod
import sys as _sys
_self = _sys.modules[__name__]
for _name in dir(_mod):
    setattr(_self, _name, getattr(_mod, _name))
