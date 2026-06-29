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

# MNIST loss mix with digit CNN features (default backbone in train_kuramoto.py).
_MNIST_LOSS_KWARGS: dict[str, str | float] = {
    "feature_backbone": "digit",
    "pixel_weight": 0.06,
    "dino_weight": 0.35,
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

# CUDA on ≤6 GB laptop GPUs — resume-friendly; ~1 hr per 100 epochs on RTX A1000.
CUDA_6GB_TRAIN_KWARGS: dict[str, str | int | float] = {
    **_MNIST_LOSS_KWARGS,
    "batch_size": 128,
    "epochs": 200,
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

# Local quality experiment (6 GB): sharper pixel loss, less feature smear, 400 epochs.
QUALITY_6GB_TRAIN_KWARGS: dict[str, str | int | float] = {
    **CUDA_6GB_TRAIN_KWARGS,
    "pixel_weight": 0.10,
    "dino_weight": 0.20,
    "epochs": 400,
    "sample_every": 0,
    "save_every": 25,
}


def _cuda_vram_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.get_device_properties(device).total_memory) / (1024**3)


def train_kwargs_for_device(device: str = "auto") -> dict[str, str | int | float]:
    preset = os.environ.get("KURAMOTO_PRESET", "").strip().lower()
    if preset == "cloud":
        return dict(CLOUD_TRAIN_KWARGS)
    if preset in ("quality", "quality6gb", "quality_6gb"):
        return dict(QUALITY_6GB_TRAIN_KWARGS)
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
    # Keep torch inductor caches off home disk (avoids EDQUOT on small quotas).
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/torch_inductor_kuramoto")
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
        "--feature-backbone",
        str(kw.get("feature_backbone", "digit")),
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
    progress_every = int(kw.get("progress_every", 0))
    if progress_every > 0:
        cmd.extend(
            [
                "--progress-every",
                str(progress_every),
                "--progress-candidates",
                str(int(kw.get("progress_candidates", 32))),
            ],
        )
        if "progress_rows_dir" in kw:
            cmd.extend(["--progress-rows-dir", str(kw["progress_rows_dir"])])
        if "progress_manifest" in kw:
            cmd.extend(["--progress-manifest", str(kw["progress_manifest"])])
        if "progress_output" in kw:
            cmd.extend(["--progress-output", str(kw["progress_output"])])
        if "device" in kw:
            cmd.extend(["--device", str(kw["device"])])
    if resume is not None:
        cmd.extend(["--resume", str(resume)])
    return cmd


DEFAULT_PROGRESS_ROWS_DIR = Path("digits/progress_rows")
DEFAULT_PROGRESS_OUTPUT = Path("digits/progress_10x10.png")
DEFAULT_PROGRESS_MANIFEST = Path("digits/progress_manifest.json")


@dataclass
class ProgressGridManifest:
    output_image: str
    rows: list[dict[str, str | int]]
    seed: int
    device: str
    candidates_per_digit: int
    cell_scale: int


def progress_epoch_step(total_epochs: int, *, num_rows: int = 10) -> int:
    """Epoch interval for ``num_rows`` evenly spaced progress snapshots."""
    return max(1, int(total_epochs) // int(num_rows))


def init_progress_manifest(
    *,
    manifest_path: Path = DEFAULT_PROGRESS_MANIFEST,
    candidates_per_digit: int = 32,
    cell_scale: int = 4,
    seed: int = 42,
) -> None:
    """Reset the progress manifest (call before a fresh training run)."""
    payload = {
        "rows": [],
        "candidates_per_digit": int(candidates_per_digit),
        "cell_scale": int(cell_scale),
        "seed": int(seed),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_progress_row(
    model: nn.Module,
    epoch: int,
    *,
    rows_dir: Path = DEFAULT_PROGRESS_ROWS_DIR,
    manifest_path: Path = DEFAULT_PROGRESS_MANIFEST,
    device: torch.device,
    candidates: int = 32,
    seed: int = 42,
) -> Path:
    """Best-of-N digits 0–9 at ``epoch``; append to manifest (no checkpoint .pt needed)."""
    was_training = model.training
    model.eval()
    seed_everything(int(seed) + int(epoch))
    row = generate_digit_row(model, device, candidates=candidates)
    if was_training:
        model.train()

    rows_dir.mkdir(parents=True, exist_ok=True)
    row_path = rows_dir / f"epoch_{int(epoch):04d}.png"
    row_cpu = [cell.detach().cpu() for cell in row]
    save_image(make_grid(row_cpu, nrow=NUM_CLASSES, padding=0), row_path)

    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {"rows": [], "candidates_per_digit": int(candidates), "cell_scale": 4, "seed": int(seed)}
    rows: list[dict[str, str | int]] = [r for r in data.get("rows", []) if int(r["epoch"]) != int(epoch)]
    rows.append({"epoch": int(epoch), "image": str(row_path.resolve())})
    rows.sort(key=lambda r: int(r["epoch"]))
    data["rows"] = rows
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return row_path


def stitch_progress_grid(
    *,
    manifest_path: Path = DEFAULT_PROGRESS_MANIFEST,
    output_image: Path = DEFAULT_PROGRESS_OUTPUT,
    cell_scale: int | None = None,
) -> ProgressGridManifest:
    """Compose progress_10x10.png from manifest-listed row images only."""
    from torchvision.io import read_image

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_meta: list[dict[str, str | int]] = data.get("rows", [])
    if not rows_meta:
        msg = f"No progress rows listed in {manifest_path}"
        raise FileNotFoundError(msg)

    scale = int(cell_scale if cell_scale is not None else data.get("cell_scale", 4))
    seed = int(data.get("seed", 42))
    candidates = int(data.get("candidates_per_digit", 32))

    cells: list[Tensor] = []
    for row in rows_meta:
        img = read_image(str(row["image"])).float() / 255.0
        height, width = int(img.shape[1]), int(img.shape[2])
        cell_w = width // NUM_CLASSES
        cell_h = height
        if cell_w * NUM_CLASSES != width:
            msg = f"Row image width not divisible by {NUM_CLASSES}: {row['image']}"
            raise ValueError(msg)
        for digit in range(NUM_CLASSES):
            x0 = digit * cell_w
            patch = img[:, :cell_h, x0 : x0 + cell_w]
            gray = patch.mean(dim=0, keepdim=True) if patch.shape[0] == 3 else patch.unsqueeze(0)
            cells.append(gray)

    batch = torch.stack(cells)
    if scale > 1:
        batch = F.interpolate(batch, scale_factor=scale, mode="nearest")
    grid = make_grid(batch, nrow=NUM_CLASSES, padding=2)

    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_image)

    manifest = ProgressGridManifest(
        output_image=str(output_image.resolve()),
        rows=rows_meta,
        seed=seed,
        device="cpu",
        candidates_per_digit=candidates,
        cell_scale=scale,
    )
    manifest_path_out = output_image.with_suffix(".json")
    manifest_path_out.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


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


def _pick_best_dcgan_flat(
    generator: nn.Module,
    digit: int,
    *,
    latent_dim: int,
    candidates: int,
    device: torch.device,
) -> Tensor:
    class_id = torch.tensor([digit], device=device, dtype=torch.long)
    best_flat: Tensor | None = None
    best_score = float("-inf")
    with torch.no_grad():
        for _ in range(candidates):
            noise = torch.randn(1, int(latent_dim), device=device)
            flat = generator(noise, class_id)
            score = score_digit(_to_grayscale_batch(flat))
            if score > best_score:
                best_score = score
                best_flat = flat
    assert best_flat is not None
    return best_flat


def generate_dcgan_digit_row(
    generator: nn.Module,
    device: torch.device,
    *,
    latent_dim: int,
    candidates: int,
) -> list[Tensor]:
    """Return ten grayscale (1, H, W) tensors for digits 0-9 from a DCGAN generator."""
    row: list[Tensor] = []
    for digit in range(NUM_CLASSES):
        flat = _pick_best_dcgan_flat(
            generator,
            digit,
            latent_dim=int(latent_dim),
            candidates=candidates,
            device=device,
        )
        row.append(_to_grayscale_batch(flat).squeeze(0))
    return row


def save_dcgan_progress_row(
    generator: nn.Module,
    epoch: int,
    *,
    latent_dim: int,
    rows_dir: Path,
    manifest_path: Path,
    device: torch.device,
    candidates: int = 32,
    seed: int = 42,
) -> Path:
    """Best-of-N DCGAN digits 0–9 at ``epoch``; append to manifest."""
    was_training = generator.training
    generator.eval()
    seed_everything(int(seed) + int(epoch))
    row = generate_dcgan_digit_row(
        generator,
        device,
        latent_dim=int(latent_dim),
        candidates=candidates,
    )
    if was_training:
        generator.train()

    rows_dir.mkdir(parents=True, exist_ok=True)
    row_path = rows_dir / f"epoch_{int(epoch):04d}.png"
    row_cpu = [cell.detach().cpu() for cell in row]
    save_image(make_grid(row_cpu, nrow=NUM_CLASSES, padding=0), row_path)

    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {"rows": [], "candidates_per_digit": int(candidates), "cell_scale": 4, "seed": int(seed)}
    rows: list[dict[str, str | int]] = [r for r in data.get("rows", []) if int(r["epoch"]) != int(epoch)]
    rows.append({"epoch": int(epoch), "image": str(row_path.resolve())})
    rows.sort(key=lambda r: int(r["epoch"]))
    data["rows"] = rows
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return row_path


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
