#!/usr/bin/env python3
"""Pretrain the MNIST digit feature encoder used by train_kuramoto.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from mnist_bench.digit_features import DEFAULT_ENCODER_PATH, pretrain_digit_encoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pretrain_digit_encoder(
        output_path=args.output,
        data_root=str(args.data_root),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
