"""Train and export ten MNIST-style Kuramoto digits (0-9)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image

from mnist_bench.constants import CHANNELS, IMAGE_SIZE, NUM_CLASSES
from mnist_bench.data import flat_to_images, images_to_grayscale
from mnist_bench.factory import load_kuramoto
from un0.common import disable_torchscript_gpu_fuser_on_blackwell, resolve_device, seed_everything

# MNIST loss mix: trust pixel matching over DINO (simple strokes, not CIFAR semantics).
_MNIST_LOSS_KWARGS: dict[str, float] = {
    "pixel_weight": 0.06,
    "dino_weight": 0.2,
    "channel_weight": 0.1,
    "collapse_weight": 0.01,
}

# Apple Silicon (MPS) — faster iteration, MNIST-focused loss mix.
MAC_TRAIN_KWARGS: dict[str, str | int | float] = {
    **_MNIST_LOSS_KWARGS,
    "batch_size": 64,
    "epochs": 40,
    "lr": 0.001,
    "num_pos": 8,
    "precision": "bf16",
    "n_oscillators": 512,
    "n_conditional_oscillators": 8,
    "num_steps": 8,
    "feature_batch_size": 32,
    "num_workers": 0,
    "queue_size": 256,
    "sample_every": 5,
    "save_every": 5,
}

# Mac progress-grid demo (~5–15 min): subset + light DINO for stable digits.
MAC_FAST_TRAIN_KWARGS: dict[str, str | int | float] = {
    "pixel_weight": 0.08,
    "dino_weight": 0.15,
    "channel_weight": 0.1,
    "collapse_weight": 0.01,
    "batch_size": 64,
    "epochs": 20,
    "lr": 0.0008,
    "num_pos": 8,
    "precision": "bf16",
    "n_oscillators": 256,
    "n_conditional_oscillators": 4,
    "num_steps": 8,
    "feature_batch_size": 16,
    "num_workers": 0,
    "queue_size": 128,
    "max_samples": 6000,
    "sample_every": 2,
    "save_every": 2,
}

# Mac-safe preview: low memory, shorter run. Use --fast for digit-like quality.
MAC_LITE_TRAIN_KWARGS: dict[str, str | int | float] = {
    "batch_size": 32,
    "epochs": 12,
    "lr": 0.0008,
    "pixel_weight": 0.08,
    "dino_weight": 0.15,
    "channel_weight": 0.1,
    "collapse_weight": 0.0,
    "num_pos": 8,
    "precision": "fp32",
    "n_oscillators": 256,
    "n_conditional_oscillators": 4,
    "num_steps": 8,
    "feature_batch_size": 8,
    "num_workers": 0,
    "queue_size": 128,
    "max_samples": 6000,
    "sample_every": 1,
    "save_every": 1,
}
MNIST_TRAIN_KWARGS = MAC_TRAIN_KWARGS

# Cloud / Vast.ai (≥8 GB VRAM): Un-0-scale epoch budget, MNIST-tuned losses.
CLOUD_TRAIN_KWARGS: dict[str, str | int | float] = {
    **_MNIST_LOSS_KWARGS,
    "batch_size": 512,
    "epochs": 1200,
    "lr": 0.001,
    "num_pos": 64,
    "precision": "bf16",
    "n_oscillators": 1024,
    "n_conditional_oscillators": 8,
    "num_steps": 10,
    "feature_batch_size": 64,
    "num_workers": 4,
    "queue_size": 1024,
    "sample_every": 120,
    "save_every": 10,
}

# CUDA on ≥8 GB VRAM — shorter default for quick progress grids.
CUDA_TRAIN_KWARGS: dict[str, str | int | float] = {
    **_MNIST_LOSS_KWARGS,
    "batch_size": 512,
    "epochs": 100,
    "lr": 0.001,
    "num_pos": 64,
    "precision": "bf16",
    "n_oscillators": 1024,
    "n_conditional_oscillators": 8,
    "num_steps": 10,
    "feature_batch_size": 64,
    "num_workers": 4,
    "queue_size": 1024,
    "sample_every": 10,
    "save_every": 10,
}

# CUDA on ≤6 GB laptop GPUs — ~9 hr for 60 epochs on RTX A1000; resume-friendly.
CUDA_6GB_TRAIN_KWARGS: dict[str, str | int | float] = {
    **_MNIST_LOSS_KWARGS,
    "batch_size": 128,
    "epochs": 60,
    "lr": 0.001,
    "num_pos": 32,
    "precision": "bf16",
    "n_oscillators": 512,
    "n_conditional_oscillators": 8,
    "num_steps": 10,
    "feature_batch_size": 16,
    "num_workers": 2,
    "queue_size": 512,
    "sample_every": 6,
    "save_every": 6,
}


def _cuda_vram_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.get_device_properties(device).total_memory) / (1024**3)


def train_kwargs_for_device(device: str = "auto") -> dict[str, str | int | float]:
    preset = os.environ.get("KURAMOTO_PRESET", "").strip().lower()
    if preset == "cloud":
        return dict(CLOUD_TRAIN_KWARGS)
    if preset == "6gb":
        return dict(CUDA_6GB_TRAIN_KWARGS)
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        vram_gb = _cuda_vram_gb(resolved)
        if vram_gb is not None and vram_gb < 8.0:
            return dict(CUDA_6GB_TRAIN_KWARGS)
        return dict(CUDA_TRAIN_KWARGS)
    if resolved.type == "mps":
        return dict(MAC_TRAIN_KWARGS)
    return dict(MAC_TRAIN_KWARGS)


def training_subprocess_env() -> dict[str, str]:
    """Environment for subprocess training (MPS tweaks on macOS only)."""
    import os
    import sys

    env = os.environ.copy()
    if sys.platform == "darwin":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # Strip MPS watermark overrides — invalid values crash with
        # "RuntimeError: invalid low watermark ratio".
        for key in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"):
            env.pop(key, None)
    return env


def mac_training_env(*, lite: bool = False) -> dict[str, str]:
    """Backward-compatible alias for training_subprocess_env()."""
    return training_subprocess_env()


def kuramoto_train_command(
    *,
    checkpoint_dir: Path,
    snapshot_every: int = 0,
    train_kwargs: dict[str, str | int | float] | None = None,
    train_script: Path | None = None,
    resume: Path | None = None,
) -> list[str]:
    """Build argv for train_kuramoto.py from device-tuned kwargs."""
    import sys

    kw = dict(train_kwargs or train_kwargs_for_device("auto"))
    script = train_script or Path(__file__).resolve().parents[1] / "train_kuramoto.py"
    cmd = [
        sys.executable,
        str(script),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--epochs",
        str(kw["epochs"]),
        "--batch-size",
        str(kw["batch_size"]),
        "--lr",
        str(kw["lr"]),
        "--pixel-weight",
        str(kw["pixel_weight"]),
        "--dino-weight",
        str(kw["dino_weight"]),
        "--channel-weight",
        str(kw["channel_weight"]),
        "--collapse-weight",
        str(kw.get("collapse_weight", 0.0)),
        "--num-pos",
        str(kw["num_pos"]),
        "--precision",
        str(kw["precision"]),
        "--n-oscillators",
        str(kw["n_oscillators"]),
        "--n-conditional-oscillators",
        str(kw["n_conditional_oscillators"]),
        "--num-steps",
        str(kw["num_steps"]),
        "--feature-batch-size",
        str(kw["feature_batch_size"]),
        "--num-workers",
        str(kw["num_workers"]),
        "--queue-size",
        str(kw["queue_size"]),
        "--sample-every",
        str(kw["sample_every"]),
        "--save-every",
        str(kw["save_every"]),
    ]
    if int(kw.get("max_samples", 0)) > 0:
        cmd.extend(["--max-samples", str(int(kw["max_samples"]))])
    if snapshot_every > 0:
        cmd.extend(["--snapshot-every", str(snapshot_every)])
    if resume is not None:
        cmd.extend(["--resume", str(resume)])
    return cmd


@dataclass
class ProgressGridManifest:
    output_image: str
    rows: list[dict[str, str | int]]
    seed: int
    device: str
    candidates_per_digit: int
    cell_scale: int


def generate_digit_row(
    model: nn.Module,
    device: torch.device,
    *,
    candidates: int,
) -> list[Tensor]:
    """Return ten grayscale (1, H, W) tensors for digits 0-9."""
    row: list[Tensor] = []
    for digit in range(NUM_CLASSES):
        flat = _pick_best_flat(model, digit, candidates=candidates, device=device)
        row.append(_to_grayscale_batch(flat).squeeze(0))
    return row


def build_progress_grid(
    *,
    checkpoints: list[tuple[int, Path]],
    output_image: Path,
    device: str = "auto",
    seed: int = 42,
    candidates_per_digit: int = 4,
    cell_scale: int = 4,
    rows_dir: Path | None = None,
) -> ProgressGridManifest:
    """Compose a rows×10 grid: each row is one training snapshot, columns are digits 0-9."""
    disable_torchscript_gpu_fuser_on_blackwell()
    seed_everything(int(seed))
    torch_device = resolve_device(device)

    cells: list[Tensor] = []
    row_meta: list[dict[str, str | int]] = []

    for epoch, checkpoint in checkpoints:
        model = load_kuramoto(checkpoint, torch_device)
        row = generate_digit_row(model, torch_device, candidates=candidates_per_digit)
        if rows_dir is not None:
            rows_dir.mkdir(parents=True, exist_ok=True)
            row_grid = make_grid(row, nrow=NUM_CLASSES, padding=2)
            save_image(row_grid, rows_dir / f"epoch_{epoch:04d}.png")
        cells.extend(row)
        row_meta.append({"epoch": epoch, "checkpoint": str(checkpoint.resolve())})

    batch = torch.stack(cells)
    if cell_scale > 1:
        batch = F.interpolate(batch, scale_factor=cell_scale, mode="nearest")
    grid = make_grid(batch, nrow=NUM_CLASSES, padding=2)

    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_image)

    manifest = ProgressGridManifest(
        output_image=str(output_image.resolve()),
        rows=row_meta,
        seed=int(seed),
        device=str(torch_device),
        candidates_per_digit=int(candidates_per_digit),
        cell_scale=int(cell_scale),
    )
    manifest_path = output_image.with_suffix(".json")
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def list_snapshot_checkpoints(snapshot_dir: Path, *, limit: int = 10) -> list[tuple[int, Path]]:
    """Return up to ``limit`` checkpoints sorted by epoch, evenly spaced if more exist."""
    paths = sorted(snapshot_dir.glob("epoch_*.pt"))
    if not paths:
        msg = f"No snapshots in {snapshot_dir}. Train with --snapshot-every N first."
        raise FileNotFoundError(msg)

    labeled = [(int(path.stem.split("_", 1)[1]), path) for path in paths]
    if len(labeled) <= limit:
        return labeled

    indices = [round(i * (len(labeled) - 1) / (limit - 1)) for i in range(limit)]
    return [labeled[i] for i in indices]


@dataclass
class DigitManifest:
    checkpoint: str
    output_dir: str
    seed: int
    device: str
    candidates_per_digit: int
    digits: dict[str, str]


def anti_collapse_loss(x_flat: Tensor) -> Tensor:
    """Discourage flat black/white outputs during training."""
    per_sample = x_flat.reshape(x_flat.shape[0], CHANNELS, -1).std(dim=(1, 2))
    return -per_sample.mean()


def grayscale_consistency_loss(x_flat: Tensor) -> Tensor:
    """Penalize RGB channel mismatch so outputs stay MNIST-like."""
    pixels = x_flat.reshape(x_flat.shape[0], CHANNELS, -1)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    return ((r - g).square() + (g - b).square() + (r - b).square()).mean()


def _to_grayscale_batch(flat: Tensor) -> Tensor:
    return images_to_grayscale(flat_to_images(flat))


def score_digit(image: Tensor) -> float:
    """Higher is better: MNIST-like ink, contrast, and smooth grayscale."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    gray = images_to_grayscale(image)
    ink_fraction = float((gray < 0.5).float().mean())
    contrast = float(gray.std())
    if contrast < 0.02:
        return -1.0
    # Reject harsh binary blobs — real MNIST exports are mostly mid-tones.
    saturated = float(((gray < 0.05) | (gray > 0.95)).float().mean())
    ink_target = 0.18
    contrast_target = 0.28
    ink_penalty = abs(ink_fraction - ink_target)
    contrast_penalty = abs(contrast - contrast_target) * 0.5
    return contrast - ink_penalty - contrast_penalty - saturated * 2.0


def _pick_best_flat(model: nn.Module, digit: int, *, candidates: int, device: torch.device) -> Tensor:
    class_id = torch.tensor([digit], device=device, dtype=torch.long)
    best_flat: Tensor | None = None
    best_score = float("-inf")
    with torch.no_grad():
        for _ in range(candidates):
            flat = model.sample(class_id)
            score = score_digit(_to_grayscale_batch(flat))
            if score > best_score:
                best_score = score
                best_flat = flat
    assert best_flat is not None
    return best_flat


def export_ten_digits(
    *,
    checkpoint: Path,
    output_dir: Path,
    device: str = "auto",
    seed: int = 42,
    candidates_per_digit: int = 8,
) -> DigitManifest:
    """Write digits/0.png … digits/9.png plus digits/grid.png and manifest.json."""
    disable_torchscript_gpu_fuser_on_blackwell()
    seed_everything(int(seed))
    torch_device = resolve_device(device)
    model = load_kuramoto(checkpoint, torch_device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    digit_paths: dict[str, str] = {}
    tiles: list[Tensor] = []

    for digit in range(NUM_CLASSES):
        flat = _pick_best_flat(
            model,
            digit,
            candidates=candidates_per_digit,
            device=torch_device,
        )
        gray = _to_grayscale_batch(flat).squeeze(0)
        out_path = output_dir / f"{digit}.png"
        save_image(gray, out_path)
        digit_paths[str(digit)] = out_path.name
        tiles.append(gray)

    grid = make_grid(tiles, nrow=NUM_CLASSES, padding=2)
    save_image(grid, output_dir / "grid.png")

    manifest = DigitManifest(
        checkpoint=str(checkpoint.resolve()),
        output_dir=str(output_dir.resolve()),
        seed=int(seed),
        device=str(torch_device),
        candidates_per_digit=int(candidates_per_digit),
        digits=digit_paths,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
