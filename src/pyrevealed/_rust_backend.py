"""Conditional Rust backend import.

If the Rust extension (_rust_core) is installed, use it for batch analysis.
Otherwise, fall back to the pure-Python implementation.
"""

try:
    from pyrevealed._rust_core import analyze_batch as _rust_analyze_batch
    from pyrevealed._rust_core import analyze_menu_batch as _rust_analyze_menu_batch
    from pyrevealed._rust_core import build_preference_graph as _rust_build_preference_graph
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    _rust_analyze_batch = None
    _rust_analyze_menu_batch = None
    _rust_build_preference_graph = None
