#!/usr/bin/env bash
# Fair 400-epoch Kuramoto vs DCGAN on full MNIST (60k).
# Progress row every 10 epochs → 40 snapshots per model.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate

EPOCHS="${EPOCHS:-400}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
CANDIDATES="${CANDIDATES:-32}"
DEVICE="${DEVICE:-cuda}"
LOG="${LOG:-fair_comparison_400.log}"

KURAMOTO_ROWS="digits/kuramoto/progress_rows"
KURAMOTO_MANIFEST="digits/kuramoto/progress_manifest.json"
KURAMOTO_GRID="digits/kuramoto/progress_40x10.png"
KURAMOTO_CKPT="checkpoints/kuramoto"

DCGAN_ROWS="digits/dcgan/progress_rows"
DCGAN_MANIFEST="digits/dcgan/progress_manifest.json"
DCGAN_GRID="digits/dcgan/progress_40x10.png"
DCGAN_CKPT="checkpoints/dcgan"

FINDINGS="research/fair_comparison_400epoch.md"

clean_model_artifacts() {
  local ckpt_dir="$1"
  local rows_dir="$2"
  local manifest="$3"
  local grid="$4"
  rm -f "$ckpt_dir/final.pt" "$ckpt_dir/latest.pt" "$ckpt_dir/smoke.pt"
  rm -rf "$ckpt_dir/samples"/*
  mkdir -p "$rows_dir"
  find "$rows_dir" -maxdepth 1 -name 'epoch_*.png' -delete 2>/dev/null || true
  rm -f "$manifest" "$grid" "${grid%.png}.json"
}

write_findings_header() {
  mkdir -p research
  cat >"$FINDINGS" <<EOF
# Fair comparison — 400 epochs, full MNIST

Protocol: matched epoch budget, batch 128, full 60k train set, progress every ${PROGRESS_EVERY} epochs (40 rows × 10 digits).

| | Kuramoto | DCGAN |
|---|----------|-------|
| Epochs | ${EPOCHS} | ${EPOCHS} |
| Batch | 128 | 128 |
| Progress | every ${PROGRESS_EVERY} | every ${PROGRESS_EVERY} |
| Candidates/digit | ${CANDIDATES} | ${CANDIDATES} |
| Kuramoto preset | \`6gb\` default losses | stable DCGAN + \`dcgan=True\` data |
| Data note | MNIST mean/std scaling | [-1, 1] tanh range |

## Grids (filled when training completes)

| Model | Full progress | Rows |
|-------|---------------|------|
| Kuramoto | ![Kuramoto](${KURAMOTO_GRID}) | \`${KURAMOTO_ROWS}/\` |
| DCGAN | ![DCGAN](${DCGAN_GRID}) | \`${DCGAN_ROWS}/\` |

## Results

Training in progress — see \`${LOG}\`.
EOF
}

echo "==> Fair comparison: ${EPOCHS} epochs, progress every ${PROGRESS_EVERY}"
write_findings_header

echo "==> Cleaning prior artifacts…"
clean_model_artifacts "$KURAMOTO_CKPT" "$KURAMOTO_ROWS" "$KURAMOTO_MANIFEST" "$KURAMOTO_GRID"
clean_model_artifacts "$DCGAN_CKPT" "$DCGAN_ROWS" "$DCGAN_MANIFEST" "$DCGAN_GRID"

echo "==> [1/2] Kuramoto (${EPOCHS} epochs, preset=6gb)…"
export KURAMOTO_PRESET=6gb
python make_progress_grid.py \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --progress-every "$PROGRESS_EVERY" \
  --candidates "$CANDIDATES" \
  --output "$KURAMOTO_GRID" \
  --rows-dir "$KURAMOTO_ROWS" \
  --manifest "$KURAMOTO_MANIFEST"

echo "==> [2/2] DCGAN (${EPOCHS} epochs, fixed training)…"
python make_dcgan_progress_grid.py \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --progress-every "$PROGRESS_EVERY" \
  --candidates "$CANDIDATES" \
  --output "$DCGAN_GRID" \
  --rows-dir "$DCGAN_ROWS" \
  --manifest "$DCGAN_MANIFEST" \
  --checkpoint-dir "$DCGAN_CKPT"

python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path

findings = Path("research/fair_comparison_400epoch.md")
text = findings.read_text(encoding="utf-8")
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
text = text.replace(
    "Training in progress — see `fair_comparison_400.log`.",
    f"Completed: {now}\n\nCompare side-by-side grids above. Epoch rows: 10, 20, …, 400.",
)
findings.write_text(text, encoding="utf-8")
print(f"Updated {findings}")
PY

echo ""
echo "Done."
echo "  Kuramoto: $KURAMOTO_GRID"
echo "  DCGAN:    $DCGAN_GRID"
echo "  Findings: $FINDINGS"
