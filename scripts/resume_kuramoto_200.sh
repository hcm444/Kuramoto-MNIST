#!/usr/bin/env bash
# Resume Kuramoto from checkpoints/kuramoto/final.pt and train to 200 epochs.
# Builds a 10×10 progress grid (epochs 20, 40, …, 200).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate

ROWS_DIR="digits/progress_rows"
MANIFEST="digits/progress_manifest.json"
CHECKPOINT="checkpoints/kuramoto/final.pt"
EPOCHS="${EPOCHS:-200}"
SNAPSHOTS="${SNAPSHOTS:-10}"
CANDIDATES="${CANDIDATES:-32}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing $CHECKPOINT — run a 100-epoch train first." >&2
  exit 1
fi

mkdir -p "$ROWS_DIR"

# Seed manifest with every-20-epoch rows from the completed 100-epoch run.
python - <<'PY'
import json
import shutil
from pathlib import Path

repo = Path(".")
src = repo / "digits/kuramoto/progress_rows"
dst = repo / "digits/progress_rows"
manifest_path = repo / "digits/progress_manifest.json"

rows = []
for epoch in (20, 40, 60, 80, 100):
    name = f"epoch_{epoch:04d}.png"
    src_path = src / name
    if not src_path.is_file():
        src_path = dst / name
    if not src_path.is_file():
        raise SystemExit(f"Missing progress row for epoch {epoch}: {src_path}")
    if src_path.parent != dst:
        shutil.copy2(src_path, dst / name)
    rows.append({"epoch": epoch, "image": str((dst / name).resolve())})

manifest_path.write_text(
    json.dumps(
        {
            "rows": rows,
            "candidates_per_digit": 32,
            "cell_scale": 4,
            "seed": 42,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Seeded {manifest_path} with epochs: {[r['epoch'] for r in rows]}")
PY

echo "==> Resuming Kuramoto to ${EPOCHS} epochs (progress every $((EPOCHS / SNAPSHOTS)))"
python make_progress_grid.py \
  --device cuda \
  --epochs "$EPOCHS" \
  --snapshots "$SNAPSHOTS" \
  --candidates "$CANDIDATES" \
  --resume "$CHECKPOINT" \
  --output digits/progress_10x10.png \
  --rows-dir "$ROWS_DIR" \
  --manifest "$MANIFEST"

echo ""
echo "Done."
echo "  Grid:       digits/progress_10x10.png"
echo "  Checkpoint: checkpoints/kuramoto/final.pt"
