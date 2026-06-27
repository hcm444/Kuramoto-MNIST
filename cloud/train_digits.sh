#!/usr/bin/env bash
# Train Kuramoto on CUDA and export digits/0.png … digits/9.png
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CHECKPOINT="${CHECKPOINT:-checkpoints/kuramoto/final.pt}"

python train_kuramoto.py \
  --checkpoint-dir checkpoints/kuramoto \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --pixel-weight 0.02 \
  --dino-weight 0.5 \
  --channel-weight 0.05 \
  --num-pos 64 \
  --precision bf16 \
  --sample-every 10 \
  --save-every 10

python make_digits.py \
  --skip-train \
  --checkpoint "$CHECKPOINT" \
  --output digits \
  --device cuda \
  --candidates 16

echo ""
echo "Training complete. Download to your Mac:"
echo "  ./cloud/fetch_results.sh root@YOUR_DROPLET_IP"
