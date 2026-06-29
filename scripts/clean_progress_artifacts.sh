#!/usr/bin/env bash
# Remove stale training images so progress stitching only sees the current run.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

rm -rf checkpoints/kuramoto/samples/*
rm -rf checkpoints/kuramoto/snapshots/*
rm -rf checkpoints/dcgan/samples/*
rm -f checkpoints/kuramoto/smoke.pt checkpoints/dcgan/smoke.pt
rm -f digits/progress_10x10.png digits/progress_10x10.json
rm -f digits/progress_manifest.json
rm -f digits/manifest.json digits/grid.png
rm -f digits/[0-9].png
find digits/progress_rows -maxdepth 1 -name 'epoch_*.png' -delete 2>/dev/null || true
rm -rf digits/kuramoto/progress_rows digits/dcgan/progress_rows
rm -f digits/kuramoto/progress_manifest.json digits/dcgan/progress_manifest.json
rm -f digits/kuramoto/progress_10x10.png digits/dcgan/progress_10x10.png

echo "Cleaned progress artifacts under digits/ and checkpoints/{kuramoto,dcgan}/"
