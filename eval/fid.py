"""Compute clean-FID for a trained Kuramoto or DCGAN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mnist_bench.metrics import compute_dcgan_fid, compute_kuramoto_fid
from mnist_bench.dcgan import Generator
from un0.common import disable_torchscript_gpu_fuser_on_blackwell, resolve_device, seed_everything
from un0.model import build_cifar10_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("kuramoto", "dcgan"), required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--real-dir", type=Path, default=Path("data/mnist_reals"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    disable_torchscript_gpu_fuser_on_blackwell()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    if not args.real_dir.is_dir():
        raise SystemExit(f"Missing real image dir {args.real_dir}. Run scripts/dump_mnist_reals.py first.")

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state.get("config") or {}

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
        model.eval()
        fid = compute_kuramoto_fid(
            model,
            num_samples=int(args.num_samples),
            batch_size=int(args.batch_size),
            device=device,
            real_image_dir=args.real_dir,
        )
        params = sum(p.numel() for p in model.parameters())
    else:
        latent_dim = int(config.get("latent_dim", 128))
        generator = Generator(latent_dim=latent_dim).to(device)
        generator.load_state_dict(state["generator"])
        generator.eval()
        fid = compute_dcgan_fid(
            generator,
            num_samples=int(args.num_samples),
            batch_size=int(args.batch_size),
            device=device,
            real_image_dir=args.real_dir,
            latent_dim=latent_dim,
        )
        params = sum(p.numel() for p in generator.parameters())

    result = {
        "fid": fid,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "num_samples": int(args.num_samples),
        "params": params,
    }
    print(f"FID: {fid:.4f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
