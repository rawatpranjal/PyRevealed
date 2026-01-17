"""pytest configuration and fixtures."""

# Ignore source files from test collection
# (functions like test_warp_la, test_ram_consistency are API functions, not tests)
collect_ignore_glob = [
    "**/src/**/*.py",
    "src/**/*.py",
]

# Additionally prevent pytest from collecting functions that match test_* from imports
# by filtering them out based on their module
def pytest_collection_modifyitems(config, items):
    """Filter out test functions that are actually API functions from source code."""
    # Filter items that come from pyrevealed source modules
    filtered = []
    for item in items:
        # Skip items that come from pyrevealed.* modules (these are API functions, not tests)
        if hasattr(item, 'module') and item.module:
            module_name = item.module.__name__ if hasattr(item.module, '__name__') else ''
            if module_name.startswith('pyrevealed.'):
                continue
        filtered.append(item)
    items[:] = filtered
