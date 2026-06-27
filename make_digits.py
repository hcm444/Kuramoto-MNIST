#!/usr/bin/env python3
"""Train Kuramoto on MNIST and export ten grayscale digits (0-9)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from mnist_bench.digits import (
    export_ten_digits,
    kuramoto_train_command,
    train_kwargs_for_device,
    training_subprocess_env,
)
from un0.common import resolve_device

DEFAULT_CHECKPOINT = Path("checkpoints/kuramoto/final.pt")
DEFAULT_OUTPUT = Path("digits")


def build_parser() -> argparse.ArgumentParser:
    defaults = train_kwargs_for_device("auto")
    parser = argparse.ArgumentParser(
        description="Train Kuramoto on MNIST and write digits/0.png … digits/9.png",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only generate digits from an existing checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidates",
        type=int,
        default=8,
        help="Samples per digit; keep the best-looking one.",
    )
    parser.add_argument("--epochs", type=int, default=int(defaults["epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(defaults["batch_size"]))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(str(args.device))

    if not args.skip_train:
        kwargs = train_kwargs_for_device(str(device))
        kwargs["epochs"] = int(args.epochs)
        kwargs["batch_size"] = int(args.batch_size)
        print(f"Training on {device} (batch {args.batch_size}, {args.epochs} epochs)…")
        cmd = kuramoto_train_command(
            checkpoint_dir=args.checkpoint.parent,
            train_kwargs=kwargs,
        )
        subprocess.run(cmd, check=True, env=training_subprocess_env())
    elif not args.checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}. Run without --skip-train first.")

    print(f"Generating 10 digits ({args.candidates} candidates each)…")
    manifest = export_ten_digits(
        checkpoint=args.checkpoint,
        output_dir=args.output,
        device=str(args.device),
        seed=int(args.seed),
        candidates_per_digit=int(args.candidates),
    )
    print(f"Wrote digits to {args.output}/")
    for digit in range(10):
        print(f"  {digit}.png")
    print("  grid.png")
    print(f"  manifest.json  (checkpoint: {manifest.checkpoint})")


if __name__ == "__main__":
    main()
