"""FID helpers for MNIST (custom clean-FID reference statistics)."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.utils import save_image

from mnist_bench.constants import IMAGE_SIZE, MNIST_STATS_NAME, NUM_CLASSES
from mnist_bench.data import flat_to_images


def _fid_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cleanfid_kwargs() -> dict:
    # num_workers=0 avoids macOS spawn pickle errors in clean-fid's DataLoader.
    return {"device": _fid_device(), "num_workers": 0}


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Fréchet distance; scipy>=1.14 removed sqrtm(..., disp=False)."""
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


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
            **_cleanfid_kwargs(),
        )


def score_directory(gen_dir: str | Path, *, real_image_dir: str | Path) -> float:
    from cleanfid.features import build_feature_extractor, get_reference_statistics
    from cleanfid.fid import get_folder_features

    ensure_mnist_reference_stats(real_image_dir)
    fid_kwargs = _cleanfid_kwargs()
    ref_mu, ref_sigma = get_reference_statistics(
        MNIST_STATS_NAME,
        "na",
        mode="clean",
        split="custom",
    )
    feat_model = build_feature_extractor("clean", fid_kwargs["device"])
    np_feats = get_folder_features(
        str(gen_dir),
        feat_model,
        mode="clean",
        description=f"FID {Path(gen_dir).name} : ",
        **fid_kwargs,
    )
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    return _frechet_distance(mu, sigma, ref_mu, ref_sigma)


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
