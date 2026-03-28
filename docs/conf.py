# Configuration file for the Sphinx documentation builder.

import os
import sys
import warnings

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# Suppress deprecation warnings from contrib module shims during autodoc
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Mock the compiled Rust extension so autodoc works without Rust toolchain (RTD)
autodoc_mock_imports = ["prefgraph._rust_core"]

import sys
from unittest.mock import MagicMock

class MockNumba(MagicMock):
    @classmethod
    def njit(cls, *args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
        
    @classmethod
    def prange(cls, *args):
        return range(*args)

sys.modules['numba'] = MockNumba()

# -- Project information -----------------------------------------------------

project = "PrefGraph"
copyright = "2024, PrefGraph Contributors"
author = "PrefGraph Contributors"
release = "0.5.13"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "archive/**",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

# Add the meta tag needed for Algolia Crawler domain verification
html_meta = {
    "algolia-site-verification": "72A7D416CC979088"
}

html_theme_options = {
    "github_url": "https://github.com/rawatpranjal/PrefGraph",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": False,
    "secondary_sidebar_items": [],  # no right sidebar anywhere
}

# All pages get left sidebar with nav tree + in-page section TOC
# Enforce left sidebar (nav tree + in-page TOC) on ALL pages, including index
html_sidebars = {
    "**": ["sidebar-nav-bs", "page-toc"],
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_ivar = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
