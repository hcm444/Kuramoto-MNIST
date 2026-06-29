#!/usr/bin/env bash
# Resume fair Kuramoto vs DCGAN comparison from final.pt to a higher epoch budget.
# Keeps existing progress rows/manifest; appends epochs 410, 420, … up to EPOCHS.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate

EPOCHS="${EPOCHS:-800}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
CANDIDATES="${CANDIDATES:-32}"
DEVICE="${DEVICE:-cuda}"
LOG="${LOG:-fair_comparison_800.log}"

KURAMOTO_ROWS="digits/kuramoto/progress_rows"
KURAMOTO_MANIFEST="digits/kuramoto/progress_manifest.json"
KURAMOTO_GRID="digits/kuramoto/progress_80x10.png"
KURAMOTO_CKPT="checkpoints/kuramoto/final.pt"

DCGAN_ROWS="digits/dcgan/progress_rows"
DCGAN_MANIFEST="digits/dcgan/progress_manifest.json"
DCGAN_GRID="digits/dcgan/progress_80x10.png"
DCGAN_CKPT="checkpoints/dcgan/final.pt"

FINDINGS="research/fair_comparison_800epoch.md"

if [[ ! -f "$KURAMOTO_CKPT" ]]; then
  echo "Missing $KURAMOTO_CKPT — run scripts/run_fair_comparison_400.sh first." >&2
  exit 1
fi
if [[ ! -f "$DCGAN_CKPT" ]]; then
  echo "Missing $DCGAN_CKPT — run scripts/run_fair_comparison_400.sh first." >&2
  exit 1
fi

mkdir -p research
cat >"$FINDINGS" <<EOF
# Fair comparison — resume to ${EPOCHS} epochs

Continued from 400-epoch checkpoints. Progress every ${PROGRESS_EVERY} epochs (rows 10…400 from first run, then 410…${EPOCHS}).

| | Kuramoto | DCGAN |
|---|----------|-------|
| Target epochs | ${EPOCHS} | ${EPOCHS} |
| Resume from | \`checkpoints/kuramoto/final.pt\` | \`checkpoints/dcgan/final.pt\` |
| Progress | every ${PROGRESS_EVERY} | every ${PROGRESS_EVERY} |

## Grids

| Model | Full progress | Rows |
|-------|---------------|------|
| Kuramoto | ![Kuramoto](${KURAMOTO_GRID}) | \`${KURAMOTO_ROWS}/\` |
| DCGAN | ![DCGAN](${DCGAN_GRID}) | \`${DCGAN_ROWS}/\` |

## Results

Training in progress — see \`${LOG}\`.
EOF

echo "==> Resuming fair comparison to ${EPOCHS} epochs (progress every ${PROGRESS_EVERY})"

echo "==> [1/2] Kuramoto (resume → ${EPOCHS} epochs, preset=6gb)…"
export KURAMOTO_PRESET=6gb
python make_progress_grid.py \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --progress-every "$PROGRESS_EVERY" \
  --candidates "$CANDIDATES" \
  --resume "$KURAMOTO_CKPT" \
  --output "$KURAMOTO_GRID" \
  --rows-dir "$KURAMOTO_ROWS" \
  --manifest "$KURAMOTO_MANIFEST"

echo "==> [2/2] DCGAN (resume → ${EPOCHS} epochs)…"
python make_dcgan_progress_grid.py \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --progress-every "$PROGRESS_EVERY" \
  --candidates "$CANDIDATES" \
  --resume "$DCGAN_CKPT" \
  --output "$DCGAN_GRID" \
  --rows-dir "$DCGAN_ROWS" \
  --manifest "$DCGAN_MANIFEST" \
  --checkpoint-dir "$(dirname "$DCGAN_CKPT")"

python - <<PY
from datetime import datetime, timezone
from pathlib import Path

findings = Path("$FINDINGS")
text = findings.read_text(encoding="utf-8")
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
text = text.replace(
    "Training in progress — see \`$LOG\`.",
    f"Completed: {now}\n\nFull grid: epochs 10, 20, …, $EPOCHS.",
)
findings.write_text(text, encoding="utf-8")
print(f"Updated {findings}")
PY

echo ""
echo "Done."
echo "  Kuramoto: $KURAMOTO_GRID"
echo "  DCGAN:    $DCGAN_GRID"
echo "  Findings: $FINDINGS"
