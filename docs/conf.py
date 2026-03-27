# Configuration file for the Sphinx documentation builder.

import os
import sys
import warnings

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# Suppress deprecation warnings from contrib module shims during autodoc
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Mock the compiled Rust extension so autodoc works without Rust toolchain (RTD)
autodoc_mock_imports = ["pyrevealed._rust_core", "numba"]

# -- Project information -----------------------------------------------------

project = "PyRevealed"
copyright = "2024, PyRevealed Contributors"
author = "PyRevealed Contributors"
release = "0.5.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

html_theme_options = {
    "github_url": "https://github.com/rawatpranjal/PyRevealed",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": False,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

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

