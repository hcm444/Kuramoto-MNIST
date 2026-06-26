"""Benchmark inference throughput for Kuramoto or DCGAN on MNIST checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mnist_bench.dcgan import Generator
from un0.apple import (
    autotune_inference_batch_size,
    configure_apple_runtime,
    generate_samples,
    prepare_model_for_inference,
    synchronize_device,
)
from un0.common import resolve_device, seed_everything
from un0.model import build_cifar10_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("kuramoto", "dcgan"), required=True)
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _time_kuramoto(model, class_ids: torch.Tensor, device: torch.device, *, batch_size: int) -> float:
    prepare_model_for_inference(model)
    effective = batch_size or autotune_inference_batch_size(model, device)
    synchronize_device(device)
    start = time.perf_counter()
    generate_samples(model, class_ids, device, batch_size=effective, warmup=True)
    synchronize_device(device)
    return time.perf_counter() - start


def _time_dcgan(generator, class_ids: torch.Tensor, device: torch.device, *, latent_dim: int, batch_size: int) -> float:
    effective = batch_size or 128
    synchronize_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for start_idx in range(0, class_ids.shape[0], effective):
            batch_ids = class_ids[start_idx : start_idx + effective]
            noise = torch.randn(batch_ids.shape[0], latent_dim, device=device)
            generator(noise, batch_ids)
    synchronize_device(device)
    return time.perf_counter() - start


def main() -> None:
    args = build_parser().parse_args()
    configure_apple_runtime()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state.get("config") or {}
    num_images = int(args.num_images)
    class_ids = torch.arange(num_images, device=device, dtype=torch.long) % 10

    if args.model == "kuramoto":
        model = build_cifar10_model(
            n_oscillators=int(config.get("n_oscillators", 1024)),
            n_conditional_oscillators=int(config.get("n_conditional_oscillators", 8)),
            class_dropout_prob=float(config.get("class_dropout_prob", 0.1)),
            num_steps=int(config.get("num_steps", 10)),
            parameterization="standard",
            relativization="mean_relative",
            encoding="sin_cos",
            solver=str(config.get("solver", "euler")),
        ).to(device)
        model.load_state_dict(state["model"])
        elapsed = _time_kuramoto(model, class_ids, device, batch_size=int(args.batch_size))
        params = sum(p.numel() for p in model.parameters())
    else:
        latent_dim = int(config.get("latent_dim", 128))
        generator = Generator(latent_dim=latent_dim).to(device)
        generator.load_state_dict(state["generator"])
        generator.eval()
        elapsed = _time_dcgan(
            generator,
            class_ids,
            device,
            latent_dim=latent_dim,
            batch_size=int(args.batch_size),
        )
        params = sum(p.numel() for p in generator.parameters())

    ms_per_image = (elapsed / num_images) * 1000.0
    result = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "num_images": num_images,
        "elapsed_s": elapsed,
        "ms_per_image": ms_per_image,
        "params": params,
    }
    print(f"{args.model}: {num_images} images in {elapsed * 1000:.0f} ms ({ms_per_image:.2f} ms/image)")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
