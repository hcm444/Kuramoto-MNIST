"""Dump padded MNIST train images as PNGs for clean-FID reference statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm.auto import tqdm

from mnist_bench.data import build_mnist_dataloader, flat_to_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/mnist_reals"))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-images", type=int, default=0, help="0 = full train set (60k).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loader = build_mnist_dataloader(
        root=str(args.data_root),
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    idx = 0
    limit = int(args.max_images) if int(args.max_images) > 0 else None
    for batch in tqdm(loader, desc="dump reals"):
        for img in flat_to_images(batch["data"]):
            save_image(img, out / f"real_{idx:06d}.png")
            idx += 1
            if limit is not None and idx >= limit:
                print(f"Wrote {idx} images to {out}")
                return
    print(f"Wrote {idx} images to {out}")


if __name__ == "__main__":
    main()
