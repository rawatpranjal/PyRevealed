#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from .generate import generate_all
from .utils import full_config, light_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate performance plots for RTD.")
    parser.add_argument(
        "--mode",
        choices=["light", "full"],
        default="light",
        help="Light runs quickly (CI/RTD-friendly). Full matches the paper/RTD figures.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory (defaults to docs/_static).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Limit compute threads (sets RAYON_NUM_THREADS, OMP/MKL/BLAS to this value).",
    )
    args = parser.parse_args()

    cfg = light_config() if args.mode == "light" else full_config()
    if args.out_dir:
        cfg.out_dir = os.path.abspath(args.out_dir)

    # Optionally pin parallelism to avoid high CPU usage
    if args.threads is not None and args.threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

    print("=" * 60)
    extra = f", threads={args.threads}" if args.threads else ""
    print(f" Generating performance plots (mode={args.mode}{extra})")
    print("=" * 60)
    paths = generate_all(cfg)
    for p in paths:
        print(f"  wrote: {p}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
