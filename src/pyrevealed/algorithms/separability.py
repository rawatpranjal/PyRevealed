"""Deprecated: moved to pyrevealed.contrib.separability."""
import warnings as _warnings
_warnings.warn(
    "pyrevealed.algorithms.separability has moved to pyrevealed.contrib.separability. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)
import pyrevealed.contrib.separability as _mod
import sys as _sys
_self = _sys.modules[__name__]
for _name in dir(_mod):
    setattr(_self, _name, getattr(_mod, _name))
