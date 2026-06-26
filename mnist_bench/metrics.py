"""FID helpers for MNIST (custom clean-FID reference statistics)."""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn
from torchvision.utils import save_image

from mnist_bench.constants import IMAGE_SIZE, MNIST_STATS_NAME, NUM_CLASSES
from mnist_bench.data import flat_to_images


def _class_balanced_ids(num_samples: int, device: torch.device) -> Tensor:
    per_class = num_samples // NUM_CLASSES
    ids = torch.arange(NUM_CLASSES, device=device).repeat_interleave(per_class)
    remainder = num_samples - ids.numel()
    if remainder:
        ids = torch.cat([ids, torch.randint(NUM_CLASSES, (remainder,), device=device)])
    return ids[torch.randperm(ids.numel(), device=device)]


def _dump_kuramoto_samples(
    model: nn.Module,
    class_ids: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    image_dir: Path,
) -> None:
    model.eval()
    idx = 0
    with torch.no_grad():
        for start in range(0, class_ids.numel(), batch_size):
            batch_ids = class_ids[start : start + batch_size]
            flat = model.sample(batch_ids)
            for img in flat_to_images(flat):
                save_image(img, image_dir / f"gen_{idx:06d}.png")
                idx += 1


def _dump_dcgan_samples(
    generator: nn.Module,
    class_ids: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    image_dir: Path,
    latent_dim: int,
) -> None:
    generator.eval()
    idx = 0
    with torch.no_grad():
        for start in range(0, class_ids.numel(), batch_size):
            batch_ids = class_ids[start : start + batch_size]
            noise = torch.randn(batch_ids.shape[0], latent_dim, device=device)
            flat = generator(noise, batch_ids)
            for img in flat_to_images(flat):
                save_image(img, image_dir / f"gen_{idx:06d}.png")
                idx += 1


def ensure_mnist_reference_stats(
    real_image_dir: str | Path,
    *,
    num_real_samples: int | None = None,
) -> None:
    from cleanfid import fid as cleanfid

    if not cleanfid.test_stats_exists(MNIST_STATS_NAME, "clean"):
        cleanfid.make_custom_stats(
            MNIST_STATS_NAME,
            str(real_image_dir),
            num=num_real_samples,
            mode="clean",
        )


def score_directory(gen_dir: str | Path, *, real_image_dir: str | Path) -> float:
    from cleanfid import fid as cleanfid

    ensure_mnist_reference_stats(real_image_dir)
    return float(
        cleanfid.compute_fid(
            str(gen_dir),
            dataset_name=MNIST_STATS_NAME,
            dataset_split="custom",
            mode="clean",
        )
    )


def compute_kuramoto_fid(
    model: nn.Module,
    *,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    real_image_dir: str | Path,
    image_dir: str | Path | None = None,
) -> float:
    class_ids = _class_balanced_ids(num_samples, device)

    def _run(path: Path) -> float:
        _dump_kuramoto_samples(model, class_ids, batch_size=batch_size, device=device, image_dir=path)
        return score_directory(path, real_image_dir=real_image_dir)

    if image_dir is not None:
        out = Path(image_dir)
        out.mkdir(parents=True, exist_ok=True)
        return _run(out)
    with tempfile.TemporaryDirectory() as td:
        return _run(Path(td))


def compute_dcgan_fid(
    generator: nn.Module,
    *,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    real_image_dir: str | Path,
    latent_dim: int,
    image_dir: str | Path | None = None,
) -> float:
    class_ids = _class_balanced_ids(num_samples, device)

    def _run(path: Path) -> float:
        _dump_dcgan_samples(
            generator,
            class_ids,
            batch_size=batch_size,
            device=device,
            image_dir=path,
            latent_dim=latent_dim,
        )
        return score_directory(path, real_image_dir=real_image_dir)

    if image_dir is not None:
        out = Path(image_dir)
        out.mkdir(parents=True, exist_ok=True)
        return _run(out)
    with tempfile.TemporaryDirectory() as td:
        return _run(Path(td))
