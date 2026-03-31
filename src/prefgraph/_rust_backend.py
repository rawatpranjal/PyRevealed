"""Conditional Rust backend import.

If the Rust extension (_rust_core) is installed, use it for batch analysis.
Otherwise, fall back to the pure-Python implementation.
"""

try:
    from prefgraph._rust_core import analyze_batch as _rust_analyze_batch
    from prefgraph._rust_core import analyze_menu_batch as _rust_analyze_menu_batch
    from prefgraph._rust_core import build_preference_graph as _rust_build_preference_graph
    from prefgraph._rust_core import rum_consistency_batch as _rust_rum_batch
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    _rust_analyze_batch = None
    _rust_analyze_menu_batch = None
    _rust_build_preference_graph = None
    _rust_rum_batch = None

# Parquet support (compiled with --features parquet)
try:
    from prefgraph._rust_core import analyze_parquet_file as _rust_analyze_parquet_file
    HAS_PARQUET_RUST = True
except ImportError:
    HAS_PARQUET_RUST = False
    _rust_analyze_parquet_file = None
