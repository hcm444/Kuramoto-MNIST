#!/usr/bin/env bash
# Fresh local quality run: pixel=0.10, dino=0.20, 400 epochs, 6 GB preset.
# Progress grid: epochs 40, 80, …, 400 (~4 hr on RTX A1000 6 GB).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate

export KURAMOTO_PRESET=quality6gb

EPOCHS="${EPOCHS:-400}"
SNAPSHOTS="${SNAPSHOTS:-10}"
CANDIDATES="${CANDIDATES:-32}"

echo "==> Local quality experiment (fresh train)"
echo "    Preset: quality6gb  pixel=0.10  dino=0.20"
echo "    Epochs: $EPOCHS  Snapshots: $SNAPSHOTS  Candidates: $CANDIDATES"
echo ""

# Fresh run — drop stale checkpoint and progress rows.
rm -f checkpoints/kuramoto/final.pt checkpoints/kuramoto/latest.pt checkpoints/kuramoto/smoke.pt
rm -rf checkpoints/kuramoto/samples/*
rm -f digits/progress_10x10.png digits/progress_10x10.json digits/progress_manifest.json
rm -f digits/progress_rows/epoch_*.png

python make_progress_grid.py \
  --device cuda \
  --epochs "$EPOCHS" \
  --snapshots "$SNAPSHOTS" \
  --candidates "$CANDIDATES" \
  --output digits/progress_10x10.png \
  --rows-dir digits/progress_rows \
  --manifest digits/progress_manifest.json

echo ""
echo "Done."
echo "  Grid:       digits/progress_10x10.png"
echo "  Checkpoint: checkpoints/kuramoto/final.pt"
echo ""
echo "Export digits:"
echo "  python make_digits.py --skip-train --device cuda --candidates 64"
