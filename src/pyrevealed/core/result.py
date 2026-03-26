"""Backward compatibility - imports from results/ subpackage.

All result dataclasses have been moved to pyrevealed.core.results/ submodules.
This shim re-exports everything so existing imports continue to work:
    from pyrevealed.core.result import GARPResult  # still works
"""
from pyrevealed.core.results import *  # noqa: F401,F403
