#!/usr/bin/env bash
# Run on your Mac to pull checkpoints and generated digits from the droplet.
set -euo pipefail

REMOTE="${1:?usage: ./cloud/fetch_results.sh user@droplet-ip}"
REMOTE_DIR="${2:-un0-mnist-bench}"

echo "==> Fetching checkpoints/"
rsync -avz --progress "$REMOTE:$REMOTE_DIR/checkpoints/" ./checkpoints/

echo "==> Fetching digits/"
rsync -avz --progress "$REMOTE:$REMOTE_DIR/digits/" ./digits/

echo ""
echo "Done. Generate locally on MPS if needed:"
echo "  python make_digits.py --skip-train --device mps"
