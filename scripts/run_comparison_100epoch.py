#!/usr/bin/env python3
"""Run 100-epoch Kuramoto vs DCGAN comparison and write research findings."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from mnist_bench.digits import train_kwargs_for_device
from un0.common import resolve_device

REPO = Path(__file__).resolve().parents[1]
EPOCHS = 100
SNAPSHOTS = 10
CANDIDATES = 32

KURAMOTO_ROWS = Path("digits/kuramoto/progress_rows")
KURAMOTO_MANIFEST = Path("digits/kuramoto/progress_manifest.json")
KURAMOTO_GRID = Path("digits/kuramoto/progress_10x10.png")
KURAMOTO_CKPT = Path("checkpoints/kuramoto")

DCGAN_ROWS = Path("digits/dcgan/progress_rows")
DCGAN_MANIFEST = Path("digits/dcgan/progress_manifest.json")
DCGAN_GRID = Path("digits/dcgan/progress_10x10.png")
DCGAN_CKPT = Path("checkpoints/dcgan")

FINDINGS = Path("research/comparison_100epoch.md")


def clean_comparison_artifacts() -> None:
    for path in (
        KURAMOTO_ROWS,
        DCGAN_ROWS,
        KURAMOTO_MANIFEST,
        DCGAN_MANIFEST,
        KURAMOTO_GRID,
        DCGAN_GRID,
    ):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    for ckpt_dir in (KURAMOTO_CKPT, DCGAN_CKPT):
        for name in ("final.pt", "latest.pt", "smoke.pt"):
            p = ckpt_dir / name
            if p.is_file():
                p.unlink()
        samples = ckpt_dir / "samples"
        if samples.is_dir():
            shutil.rmtree(samples)
    print("Cleaned Kuramoto and DCGAN comparison artifacts.")


def run_kuramoto(device: str) -> None:
    kwargs = train_kwargs_for_device(device)
    kwargs["epochs"] = EPOCHS
    cmd = [
        sys.executable,
        "make_progress_grid.py",
        "--device",
        device,
        "--epochs",
        str(EPOCHS),
        "--snapshots",
        str(SNAPSHOTS),
        "--candidates",
        str(CANDIDATES),
        "--output",
        str(KURAMOTO_GRID),
        "--rows-dir",
        str(KURAMOTO_ROWS),
        "--manifest",
        str(KURAMOTO_MANIFEST),
        "--batch-size",
        str(int(kwargs["batch_size"])),
    ]
    subprocess.run(cmd, check=True, cwd=REPO, env=dict(**__import__("os").environ))


def run_dcgan(device: str) -> None:
    cmd = [
        sys.executable,
        "make_dcgan_progress_grid.py",
        "--device",
        device,
        "--epochs",
        str(EPOCHS),
        "--snapshots",
        str(SNAPSHOTS),
        "--candidates",
        str(CANDIDATES),
        "--output",
        str(DCGAN_GRID),
        "--rows-dir",
        str(DCGAN_ROWS),
        "--manifest",
        str(DCGAN_MANIFEST),
        "--checkpoint-dir",
        str(DCGAN_CKPT),
    ]
    subprocess.run(cmd, check=True, cwd=REPO, env=dict(**__import__("os").environ))


def _manifest_epochs(manifest_path: Path) -> list[int]:
    if not manifest_path.is_file():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [int(r["epoch"]) for r in data.get("rows", [])]


def write_findings(device: str) -> None:
    kwargs = train_kwargs_for_device(device)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    kuramoto_epochs = _manifest_epochs(KURAMOTO_MANIFEST)
    dcgan_epochs = _manifest_epochs(DCGAN_MANIFEST)

    FINDINGS.parent.mkdir(parents=True, exist_ok=True)
    body = f"""# Kuramoto vs DCGAN — 100-epoch MNIST comparison

Generated: {now}

## Goal

Train both generators for **{EPOCHS} epochs** on full MNIST and compare qualitative progress via **10×10 grids** (10 time snapshots × 10 digit classes).

## Grids

| Model | Progress grid | Row snapshots |
|-------|---------------|---------------|
| Kuramoto (Un-0 + digit encoder) | ![Kuramoto 10×10]({KURAMOTO_GRID.as_posix()}) | `{KURAMOTO_ROWS.as_posix()}/` |
| DCGAN (class-conditional) | ![DCGAN 10×10]({DCGAN_GRID.as_posix()}) | `{DCGAN_ROWS.as_posix()}/` |

## Configuration

| Setting | Kuramoto | DCGAN |
|---------|----------|-------|
| Epochs | {EPOCHS} | {EPOCHS} |
| Progress snapshots | {SNAPSHOTS} (every {EPOCHS // SNAPSHOTS} epochs) | {SNAPSHOTS} (every {EPOCHS // SNAPSHOTS} epochs) |
| Candidates per digit | {CANDIDATES} | {CANDIDATES} |
| Batch size | {kwargs['batch_size']} | 128 |
| Device | {device} | {device} |
| Checkpoint | `{KURAMOTO_CKPT.as_posix()}/final.pt` | `{DCGAN_CKPT.as_posix()}/final.pt` |

### Kuramoto-specific

| Setting | Value |
|---------|-------|
| Oscillators | {kwargs.get('n_oscillators', '—')} |
| Feature backbone | digit encoder |
| Pixel weight | {kwargs.get('pixel_weight', '—')} |
| DINO/drift weight | {kwargs.get('dino_weight', '—')} |
| Collapse weight | {kwargs.get('collapse_weight', '—')} |

### DCGAN-specific

| Setting | Value |
|---------|-------|
| Latent dim | 128 |
| Learning rate | 2e-4 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |

## Progress epochs captured

- **Kuramoto:** {', '.join(str(e) for e in kuramoto_epochs) or '—'}
- **DCGAN:** {', '.join(str(e) for e in dcgan_epochs) or '—'}

## Qualitative notes

Fill in after visual inspection:

1. **Early training (epochs 10–30):** Kuramoto vs DCGAN — noise level, structure emergence.
2. **Mid training (epochs 40–70):** Class separation, stroke-like features.
3. **Late training (epochs 80–100):** Mode collapse, digit readability, per-class diversity.
4. **Winner (subjective):** Which model produces more MNIST-like digits at epoch 100?

## Reproduce

```bash
python scripts/run_comparison_100epoch.py --device cuda
```

Kuramoto only:

```bash
python make_progress_grid.py --device cuda --epochs 100 --candidates 32 \\
  --output digits/kuramoto/progress_10x10.png \\
  --rows-dir digits/kuramoto/progress_rows \\
  --manifest digits/kuramoto/progress_manifest.json
```

DCGAN only:

```bash
python make_dcgan_progress_grid.py --device cuda --epochs 100 --candidates 32
```
"""
    FINDINGS.write_text(body, encoding="utf-8")
    print(f"Wrote {FINDINGS}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--kuramoto-only", action="store_true")
    parser.add_argument("--dcgan-only", action="store_true")
    parser.add_argument("--findings-only", action="store_true", help="Rewrite findings from existing grids.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = str(resolve_device(str(args.device)))

    if args.findings_only:
        write_findings(device)
        return

    if not args.skip_clean:
        clean_comparison_artifacts()

    run_both = not args.kuramoto_only and not args.dcgan_only
    if run_both or args.kuramoto_only:
        run_kuramoto(device)
    if run_both or args.dcgan_only:
        run_dcgan(device)

    write_findings(device)


if __name__ == "__main__":
    main()
