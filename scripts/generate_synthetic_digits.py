"""Generate a labeled synthetic MNIST digit dataset from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from mnist_bench.factory import export_synthetic_digits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("kuramoto", "dcgan"), required=True)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--num-samples",
        type=int,
        help="Total images to generate (balanced across digits 0-9).",
    )
    group.add_argument(
        "--per-class",
        type=int,
        help="Images per digit class (0-9). Total = 10 * per_class.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--layout",
        choices=("flat", "by_class"),
        default="flat",
        help="flat: images/000000.png + labels.csv; by_class: images/3/00042.png",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Keep 3-channel color output (default: save MNIST-style grayscale).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = export_synthetic_digits(
        checkpoint=args.checkpoint,
        output=args.output,
        model_kind=args.model,
        num_samples=args.num_samples,
        per_class=args.per_class,
        batch_size=int(args.batch_size),
        layout=args.layout,
        device=str(args.device),
        seed=int(args.seed),
        grayscale=not args.rgb,
    )
    print(f"Wrote {manifest.num_images} images to {args.output}")
    print(f"  labels:  {args.output / 'labels.csv'}")
    print(f"  manifest: {args.output / 'manifest.json'}")
    print("  per class:", ", ".join(f"{k}:{v}" for k, v in manifest.per_class_counts.items()))


if __name__ == "__main__":
    main()
