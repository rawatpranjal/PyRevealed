#!/usr/bin/env python3
"""Compatibility wrapper for legacy benchmarks entrypoint.

This script now delegates to docs/perf to keep all RTD-related
generation code in one place.
"""

from __future__ import annotations

import os
import sys

# Ensure local package is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_PERF = os.path.join(REPO_ROOT, "docs", "perf")
if DOCS_PERF not in sys.path:
    sys.path.insert(0, DOCS_PERF)

from docs.perf.generate import generate_all  # type: ignore
from docs.perf.utils import full_config  # type: ignore


def main() -> int:
    print("[benchmarks] Using docs.perf module to generate scaling plots...")
    cfg = full_config()
    paths = generate_all(cfg)
    for p in paths:
        print(f"  wrote: {p}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
