# Kuramoto-MNIST

Train [Un-0](https://github.com/hcm444/Un-0) Kuramoto dynamics on MNIST and **generate ten grayscale digits** — one per class `0`–`9`.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Train and export all ten digits:

```bash
python make_digits.py --device mps
```

Output:

```
digits/
  0.png … 9.png    # one digit per class
  grid.png         # all ten in a row
  manifest.json
```

On Apple Silicon, training scripts set `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically.

## Training on Mac (Apple Silicon)

### Recommended: full dataset

Best quality on a local Mac — all **60,000** training images, 40 epochs (~2–4 hours):

```bash
source .venv/bin/activate
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO PYTORCH_MPS_LOW_WATERMARK_RATIO

# Optional smoke test (~2 min) — confirms MPS works before a long run
python train_kuramoto.py --device mps --epochs 1 --max-steps 50 --batch-size 64 \
  --dino-weight 0.2 --pixel-weight 0.06 --n-oscillators 512 \
  --feature-batch-size 32 --num-workers 0 --precision bf16

# Train + 10×10 progress grid
rm -rf checkpoints/kuramoto/snapshots digits/progress_rows digits/progress_10x10.png
python make_progress_grid.py --device mps --epochs 40 --candidates 16

# Or train + export final ten digits only
python make_digits.py --device mps --epochs 40 --candidates 16
```

Check training progress in `checkpoints/kuramoto/samples/epoch_XXXX.png`. Top rows of the progress grid start noisy; bottom rows should show recognizable digit strokes.

If you hit OOM, retry with `--batch-size 32`. Close other heavy apps while training.

### Faster presets (subset training)

| Flag | Data | Epochs | Time | Quality |
|------|------|--------|------|---------|
| *(none)* | 60k | 40 | ~2–4 hr | Best on Mac |
| `--fast` | 6k | 20 | ~45–90 min | Good |
| `--lite` | 6k | 12 | ~20 min | Preview only (blobs, not digits) |

```bash
python make_progress_grid.py --fast --device mps
python make_progress_grid.py --lite --device mps   # smoke test only
```

### Regenerate without retraining

```bash
python make_progress_grid.py --skip-train --device mps --candidates 16
python make_digits.py --skip-train --device mps --candidates 16
```

## Options

```bash
# Train longer for sharper digits
python make_digits.py --device mps --epochs 60

# Regenerate from an existing checkpoint
python make_digits.py --skip-train --checkpoint checkpoints/kuramoto/final.pt

# Try more samples per digit and keep the best
python make_digits.py --skip-train --candidates 32
```

## Training progress grid (10×10)

Show improvement over training: **10 rows** (snapshots) × **10 columns** (digits 0–9).

```bash
python make_progress_grid.py --device mps
```

Writes `digits/progress_10x10.png` (top row = early training, bottom = latest). Also saves each row as `digits/progress_rows/epoch_XXXX.png`.

```bash
python make_progress_grid.py --epochs 40 --snapshots 10 --skip-train --device mps
```

## Cloud training (GPU)

Mac training is slow. Use a cloud GPU for sharper digits in ~1 hour — see **[cloud/README.md](cloud/README.md)**.

DigitalOcean GPUs are often sold out. **Google Colab (free T4)** or **RunPod / Vast.ai** work well.

```bash
# Any GPU server — sync, train, fetch
rsync -avz --exclude .venv --exclude checkpoints ./ root@GPU_HOST:~/un0-mnist-bench/
ssh root@GPU_HOST 'cd ~/un0-mnist-bench && ./cloud/bootstrap.sh && ./cloud/train_progress.sh'
./cloud/fetch_results.sh root@GPU_HOST
```

## How it works

1. **Train** — drift loss + DINO features, with MNIST-tuned settings:
   - higher pixel loss, lower DINO weight
   - grayscale consistency penalty (RGB channels should match)
   - MSE warmup loss while the per-class queue fills
   - `bf16` + batch 64 on MPS

2. **Generate** — for each digit `0`–`9`, sample several candidates and keep the best by MNIST-like contrast, ink coverage, and smooth grayscale.

MNIST is padded 28→32 and repeated to 3 channels so the CIFAR-10 Un-0 architecture applies unchanged. Exports are converted to true grayscale PNGs.

## Legacy benchmark tools

The original DCGAN comparison, FID eval, and bulk synthetic export scripts are still available:

- `train_dcgan.py`, `eval/`, `scripts/generate_synthetic_digits.py`

## References

- [unconv-ai/Un-0](https://github.com/unconv-ai/Un-0)
- [hcm444/Un-0](https://github.com/hcm444/Un-0) — fork used as a dependency

## License

MIT
