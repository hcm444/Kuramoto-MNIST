#!/usr/bin/env python3
"""Stitch digits/progress_10x10.png from digits/progress_manifest.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from mnist_bench.digits import (
    DEFAULT_PROGRESS_MANIFEST,
    DEFAULT_PROGRESS_OUTPUT,
    stitch_progress_grid,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROGRESS_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROGRESS_OUTPUT)
    parser.add_argument("--cell-scale", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = stitch_progress_grid(
        manifest_path=args.manifest,
        output_image=args.output,
        cell_scale=args.cell_scale,
    )
    print(f"Wrote {manifest.output_image} ({len(manifest.rows)} rows)")


if __name__ == "__main__":
    main()
