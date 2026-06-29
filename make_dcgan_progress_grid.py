#!/usr/bin/env python3
"""Train DCGAN with progress rows and build a 10×10 training-progress grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mnist_bench.digits import progress_epoch_step, stitch_progress_grid, training_subprocess_env
from un0.common import resolve_device

DEFAULT_OUTPUT = Path("digits/dcgan/progress_10x10.png")
DEFAULT_ROWS_DIR = Path("digits/dcgan/progress_rows")
DEFAULT_MANIFEST = Path("digits/dcgan/progress_manifest.json")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/dcgan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train DCGAN, save 10 training progress rows, and compose a 10×10 grid "
            "(rows = training time, columns = digits 0-9)."
        ),
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Build the grid from an existing progress manifest only.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from an existing checkpoint (e.g. checkpoints/dcgan/final.pt).",
    )
    parser.add_argument("--snapshots", type=int, default=10, help="Number of progress rows in the grid.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-dir", type=Path, default=DEFAULT_ROWS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Save a progress row every N epochs (overrides --snapshots spacing).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument(
        "--cell-scale",
        type=int,
        default=4,
        help="Upscale each digit cell for readability (4 → 128×128 cells).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(str(args.device))
    epochs = int(args.epochs)
    if args.progress_every is not None:
        progress_every = int(args.progress_every)
    else:
        progress_every = progress_epoch_step(epochs, num_rows=int(args.snapshots))

    if not args.skip_train:
        print(
            f"Training DCGAN on {device}: {epochs} epochs, "
            f"progress row every {progress_every}…",
        )
        cmd = [
            sys.executable,
            "train_dcgan.py",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(int(args.batch_size)),
            "--device",
            str(device),
            "--seed",
            str(int(args.seed)),
            "--sample-every",
            "0",
            "--progress-every",
            str(progress_every),
            "--progress-candidates",
            str(int(args.candidates)),
            "--progress-rows-dir",
            str(args.rows_dir),
            "--progress-manifest",
            str(args.manifest),
            "--progress-output",
            str(args.output),
            "--checkpoint-dir",
            str(args.checkpoint_dir),
        ]
        if args.resume is not None:
            print(f"  Resuming from {args.resume}")
            cmd.extend(["--resume", str(args.resume)])
        subprocess.run(cmd, check=True, env=training_subprocess_env())
        print(f"Training finished; grid at {args.output}")
        return

    print("Stitching progress grid from manifest…")
    manifest = stitch_progress_grid(
        manifest_path=args.manifest,
        output_image=args.output,
        cell_scale=int(args.cell_scale),
    )
    print(f"Wrote {args.output}")
    print(f"  Row grids: {args.rows_dir}/epoch_XXXX.png")
    print(f"  Manifest:  {args.manifest}")
    for row in manifest.rows:
        print(f"  epoch {int(row['epoch']):>4}: {Path(str(row['image'])).name}")


if __name__ == "__main__":
    main()
