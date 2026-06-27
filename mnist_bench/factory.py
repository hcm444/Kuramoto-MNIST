"""Export labeled synthetic MNIST digits from a trained generator."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.utils import save_image

from mnist_bench.constants import NUM_CLASSES
from mnist_bench.data import flat_to_images, images_to_grayscale
from mnist_bench.dcgan import Generator
from un0.common import disable_torchscript_gpu_fuser_on_blackwell, resolve_device, seed_everything
from un0.model import build_cifar10_model

ModelKind = Literal["kuramoto", "dcgan"]
LayoutKind = Literal["flat", "by_class"]


@dataclass
class FactoryManifest:
    model: str
    checkpoint: str
    num_images: int
    layout: str
    seed: int
    device: str
    per_class_counts: dict[str, int]


def build_class_ids(
    *,
    num_samples: int | None,
    per_class: int | None,
    device: torch.device,
) -> Tensor:
    if (num_samples is None) == (per_class is None):
        msg = "Specify exactly one of num_samples or per_class."
        raise ValueError(msg)
    if per_class is not None:
        if per_class <= 0:
            raise ValueError("per_class must be positive.")
        ids = torch.arange(NUM_CLASSES, device=device).repeat_interleave(int(per_class))
        return ids[torch.randperm(ids.numel(), device=device)]
    assert num_samples is not None
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    per = num_samples // NUM_CLASSES
    ids = torch.arange(NUM_CLASSES, device=device).repeat_interleave(per)
    remainder = num_samples - ids.numel()
    if remainder:
        ids = torch.cat([ids, torch.randint(NUM_CLASSES, (remainder,), device=device)])
    return ids[torch.randperm(ids.numel(), device=device)]


def load_kuramoto(checkpoint: Path, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state.get("config") or {}
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
    return model


def load_dcgan(checkpoint: Path, device: torch.device) -> tuple[Generator, int]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state.get("config") or {}
    latent_dim = int(config.get("latent_dim", 128))
    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(state["generator"])
    generator.eval()
    return generator, latent_dim


def _generate_batch(
    model: nn.Module,
    class_ids: Tensor,
    *,
    model_kind: ModelKind,
    latent_dim: int | None,
) -> Tensor:
    if model_kind == "kuramoto":
        return model.sample(class_ids)
    assert latent_dim is not None
    noise = torch.randn(class_ids.shape[0], latent_dim, device=class_ids.device)
    return model(noise, class_ids)


def export_synthetic_digits(
    *,
    checkpoint: Path,
    output: Path,
    model_kind: ModelKind,
    num_samples: int | None = None,
    per_class: int | None = None,
    batch_size: int = 64,
    layout: LayoutKind = "flat",
    device: str = "auto",
    seed: int = 42,
    grayscale: bool = True,
) -> FactoryManifest:
    """Generate PNGs plus labels.csv and manifest.json."""
    disable_torchscript_gpu_fuser_on_blackwell()
    seed_everything(int(seed))
    torch_device = resolve_device(device)

    latent_dim: int | None = None
    if model_kind == "kuramoto":
        model = load_kuramoto(checkpoint, torch_device)
    else:
        model, latent_dim = load_dcgan(checkpoint, torch_device)

    class_ids = build_class_ids(
        num_samples=num_samples,
        per_class=per_class,
        device=torch_device,
    )
    output = Path(output)
    images_root = output / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int]] = []
    per_class_counts = {str(d): 0 for d in range(NUM_CLASSES)}
    global_idx = 0

    with torch.no_grad():
        for start in range(0, class_ids.numel(), batch_size):
            batch_ids = class_ids[start : start + batch_size]
            flat = _generate_batch(
                model,
                batch_ids,
                model_kind=model_kind,
                latent_dim=latent_dim,
            )
            for img, label in zip(flat_to_images(flat), batch_ids.tolist(), strict=True):
                if grayscale:
                    img = images_to_grayscale(img.unsqueeze(0)).squeeze(0)
                if layout == "by_class":
                    class_dir = images_root / str(label)
                    class_dir.mkdir(parents=True, exist_ok=True)
                    local_idx = per_class_counts[str(label)]
                    rel_path = Path("images") / str(label) / f"{local_idx:05d}.png"
                    save_image(img, output / rel_path)
                else:
                    rel_path = Path("images") / f"{global_idx:06d}.png"
                    save_image(img, output / rel_path)
                rows.append((str(rel_path).replace("\\", "/"), int(label)))
                per_class_counts[str(label)] += 1
                global_idx += 1

    labels_path = output / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    manifest = FactoryManifest(
        model=model_kind,
        checkpoint=str(checkpoint.resolve()),
        num_images=len(rows),
        layout=layout,
        seed=int(seed),
        device=str(torch_device),
        per_class_counts=per_class_counts,
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")

    return manifest
