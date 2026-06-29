#!/usr/bin/env python3
"""Train Kuramoto with snapshots and build a 10×10 training-progress grid."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from mnist_bench.digits import (
    MAC_FAST_TRAIN_KWARGS,
    MAC_LITE_TRAIN_KWARGS,
    kuramoto_train_command,
    progress_epoch_step,
    stitch_progress_grid,
    train_kwargs_for_device,
    training_subprocess_env,
)
from un0.common import resolve_device

DEFAULT_SNAPSHOT_DIR = Path("checkpoints/kuramoto/snapshots")
DEFAULT_OUTPUT = Path("digits/progress_10x10.png")


def build_parser() -> argparse.ArgumentParser:
    defaults = train_kwargs_for_device("auto")
    parser = argparse.ArgumentParser(
        description=(
            "Train Kuramoto, save 10 training snapshots, and compose a 10×10 grid "
            "(rows = training time, columns = digits 0-9)."
        ),
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Build the grid from existing snapshots only.",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Mac preview only (~20 min). Blobs, not digits — use --fast for quality.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Mac-optimized: 6k samples, 20 epochs (~5–15 min, uses more RAM).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from an existing checkpoint (e.g. checkpoints/kuramoto/final.pt).",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=10,
        help="Number of training rows in the progress grid.",
    )
    parser.add_argument("--batch-size", type=int, default=int(defaults["batch_size"]))
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-dir", type=Path, default=Path("digits/progress_rows"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Progress manifest path (default: parent of --rows-dir / progress_manifest.json).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Save a progress row every N epochs (overrides --snapshots spacing).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=16)
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

    if args.lite:
        kwargs = dict(MAC_LITE_TRAIN_KWARGS)
        epochs = int(args.epochs) if args.epochs is not None else int(kwargs["epochs"])
        snapshots = int(args.snapshots)
        batch_size = int(kwargs["batch_size"])
        mode = "LITE"
    elif args.fast:
        kwargs = dict(MAC_FAST_TRAIN_KWARGS)
        epochs = int(args.epochs) if args.epochs is not None else int(kwargs["epochs"])
        snapshots = int(args.snapshots)
        batch_size = int(kwargs["batch_size"])
        mode = "FAST"
    else:
        kwargs = train_kwargs_for_device(str(device))
        epochs = int(args.epochs) if args.epochs is not None else int(kwargs["epochs"])
        snapshots = int(args.snapshots)
        batch_size = int(args.batch_size)
        mode = "standard"

    kwargs["epochs"] = epochs
    kwargs["batch_size"] = batch_size
    if args.progress_every is not None:
        progress_every = int(args.progress_every)
    else:
        progress_every = progress_epoch_step(epochs, num_rows=snapshots)
    kwargs.setdefault("sample_every", 0)
    kwargs["progress_every"] = progress_every
    kwargs.setdefault("progress_candidates", int(args.candidates))
    manifest_path = args.manifest if args.manifest is not None else args.rows_dir.parent / "progress_manifest.json"
    kwargs["progress_rows_dir"] = str(args.rows_dir)
    kwargs["progress_manifest"] = str(manifest_path)
    kwargs["progress_output"] = str(args.output)
    kwargs["device"] = str(device)
    checkpoint_dir = args.snapshot_dir.parent

    if not args.skip_train:
        steps = (int(kwargs.get("max_samples", 0)) or 60000) // batch_size
        print(
            f"[{mode}] Training on {device}: {epochs} epochs, ~{steps} steps/epoch, "
            f"progress row every {progress_every}…",
        )
        if args.lite:
            print("  (preview quality — use --fast for digit-like results)")
        elif args.fast:
            print("  (6000 samples, 256 oscillators, progress every", progress_every, "epochs)")
        elif device.type == "cuda":
            preset = "6GB" if int(kwargs.get("n_oscillators", 1024)) <= 512 else "cloud"
            print(
                f"  (full 60k MNIST, CUDA {preset} preset — "
                f"dino={kwargs['dino_weight']}, pixel={kwargs['pixel_weight']}, "
                f"collapse={kwargs.get('collapse_weight', 0.0)})",
            )
        if args.resume is not None:
            print(f"  Resuming from {args.resume}")
        cmd = kuramoto_train_command(
            checkpoint_dir=checkpoint_dir,
            snapshot_every=0,
            train_kwargs=kwargs,
            resume=args.resume,
        )
        subprocess.run(cmd, check=True, env=training_subprocess_env())
        print(f"Training finished; grid at {args.output}")
        return

    manifest_path = args.manifest if args.manifest is not None else args.rows_dir.parent / "progress_manifest.json"
    print("Stitching progress grid from manifest…")
    manifest = stitch_progress_grid(
        manifest_path=manifest_path,
        output_image=args.output,
        cell_scale=int(args.cell_scale),
    )
    print(f"Wrote {args.output}")
    print(f"  Row grids: {args.rows_dir}/epoch_XXXX.png")
    print(f"  Manifest:  {args.output.with_suffix('.json')}")
    for row in manifest.rows:
        print(f"  epoch {int(row['epoch']):>4}: {Path(str(row['image'])).name}")


if __name__ == "__main__":
    main()
