#!/usr/bin/env python3
"""
Speed up GIF animations in docs slightly, without touching site text/CSS transitions.

What this script does
---------------------
- Finds source GIFs under docs/_static and docs/archive/scripts/_static
- Reads per-frame durations (ms) and reduces them by a uniform factor (default: 1.25x faster)
- Writes back the GIFs in-place with updated durations, preserving frame order and loop count

Notes
-----
- We intentionally do not modify any CSS/JS; only the embedded GIF file timings change.
- Pillow (PIL) reports frame durations in milliseconds; we scale by 1/speed and round.
- We set a small floor on durations to avoid edge cases where browsers clamp very small delays.

Usage
-----
python tools/speed_up_gifs.py                # default 1.25x faster on default dirs
python tools/speed_up_gifs.py --factor 1.2   # custom factor
python tools/speed_up_gifs.py docs/_static   # custom path(s)

Implementation choices
----------------------
- Per-frame durations are preserved proportionally, so pauses remain but are slightly shorter.
- We save with save_all=True and preserve loop=0 (infinite) if present; disposal is left to Pillow.
- We avoid optimizing palette to minimize visual changes; optimization can change appearance.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageSequence


def iter_gifs(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from (x for x in p.rglob("*.gif") if "_build" not in x.parts)
        elif p.is_file() and p.suffix.lower() == ".gif":
            if "_build" not in p.parts:
                yield p


def read_frames_and_durations(img_path: Path) -> Tuple[List[Image.Image], List[int], dict]:
    im = Image.open(img_path)
    frames: List[Image.Image] = []
    durations_ms: List[int] = []
    # Preserve basic info if present
    base_info = {
        key: im.info.get(key)
        for key in ("loop", "background", "transparency", "version")
        if key in im.info
    }

    for i, frame in enumerate(ImageSequence.Iterator(im)):
        # PIL updates im.info on seek; frame.info may not carry duration reliably.
        # Use the container's info after seeking to each frame.
        try:
            im.seek(i)
        except Exception:
            pass
        duration = int(im.info.get("duration", 100))  # ms
        # Copy the visual content of each frame so saving does not depend on lazy seeks.
        frames.append(frame.copy())
        durations_ms.append(duration)

    return frames, durations_ms, base_info


def scale_durations(durations_ms: List[int], factor: float, min_ms: int = 40) -> List[int]:
    scaled: List[int] = []
    for d in durations_ms:
        # Speed up by reducing duration per frame: new = old / factor
        new_d = max(min_ms, int(round(d / factor)))
        # GIF encodes durations in 10ms units; round to nearest 10ms to avoid bloat.
        new_d = int(round(new_d / 10.0)) * 10
        scaled.append(max(10, new_d))
    return scaled


def summarize(label: str, durations: List[int]) -> str:
    if not durations:
        return f"{label}: 0 frames"
    return (
        f"{label}: {len(durations)} frames | "
        f"mean={statistics.mean(durations):.1f}ms, "
        f"median={statistics.median(durations):.1f}ms, "
        f"min={min(durations)}ms, max={max(durations)}ms"
    )


def process_gif(path: Path, factor: float, dry_run: bool = False) -> None:
    frames, orig_durations, base_info = read_frames_and_durations(path)

    if not frames:
        print(f"SKIP (no frames): {path}")
        return

    new_durations = scale_durations(orig_durations, factor)

    print(f"\n{path}")
    print("  " + summarize("original", orig_durations))
    print("  " + summarize("new     ", new_durations))

    if dry_run:
        return

    # Save back in-place with updated durations.
    # Keep palette/mode of first frame as-is to minimize visual diffs.
    first = frames[0]
    rest = frames[1:]
    save_kwargs = dict(
        save_all=True,
        append_images=rest,
        duration=new_durations,
        loop=base_info.get("loop", 0) if base_info.get("loop") is not None else 0,
        disposal=2,  # restore to background between frames helps avoid trails
        optimize=False,
    )
    # Attempt to carry transparency if present
    if base_info.get("transparency") is not None:
        save_kwargs["transparency"] = base_info["transparency"]

    tmp_path = path.with_suffix(".tmp.gif")
    first.save(tmp_path, format="GIF", **save_kwargs)
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Slightly speed up GIFs by reducing frame durations.")
    parser.add_argument("paths", nargs="*", type=Path, help="Directories/files to process")
    parser.add_argument("--factor", type=float, default=1.25, help="Speed-up factor (>1 = faster)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing GIFs")
    args = parser.parse_args()

    default_roots = [Path("docs/_static"), Path("docs/archive/scripts/_static")]
    targets = args.paths if args.paths else default_roots

    gifs = sorted(set(iter_gifs(targets)))
    if not gifs:
        print("No GIFs found in:")
        for t in targets:
            print(f" - {t}")
        return

    print(f"Processing {len(gifs)} GIF(s) with factor={args.factor} (1.0=unchanged, 1.25=25% faster)")
    for g in gifs:
        process_gif(g, factor=args.factor, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

