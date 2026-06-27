# Kuramoto-MNIST

Train [Un-0](https://github.com/hcm444/Un-0) Kuramoto dynamics on MNIST and **generate ten grayscale digits** — one per class `0`–`9`.

## Ubuntu GPU (recommended for quality)

Use an **Ubuntu machine with an NVIDIA GPU** for the best results — full 60k dataset, ~100 epochs, clean digit shapes.

**Full guide:** **[cloud/UBUNTU.md](cloud/UBUNTU.md)**

```bash
git clone https://github.com/hcm444/Kuramoto-MNIST.git un0-mnist-bench
cd un0-mnist-bench
chmod +x cloud/*.sh
./cloud/bootstrap.sh
tmux new -s train
./cloud/train_progress.sh      # → digits/progress_10x10.png
```

Download results to your laptop: `./cloud/fetch_results.sh user@your-server`

---

## Mac (Apple Silicon)

Slower than Ubuntu GPU. Activate the venv first:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO PYTORCH_MPS_LOW_WATERMARK_RATIO
```

**Full dataset, best local quality** (~2–4 hours):

```bash
python make_progress_grid.py --device mps --epochs 40 --candidates 16
```

**Faster subset** (~45–90 min, good but not as sharp):

```bash
python make_progress_grid.py --fast --device mps --candidates 16
```

Do **not** use `--lite` if you want digits — it's preview-only.

---

## Outputs

```
digits/
  progress_10x10.png     # 10 training snapshots × 10 digits (0–9)
  progress_rows/         # one row PNG per snapshot
  0.png … 9.png          # final digits (from make_digits.py)
  grid.png
checkpoints/kuramoto/
  final.pt               # trained model
  snapshots/             # epoch checkpoints for the progress grid
```

---

## Options

```bash
# Regenerate from checkpoint (no retraining)
python make_progress_grid.py --skip-train --device cuda --candidates 32
python make_digits.py --skip-train --device cuda --candidates 32

# Train longer
EPOCHS=150 ./cloud/train_progress.sh          # Ubuntu
python make_digits.py --device mps --epochs 60  # Mac
```

---

## How it works

1. **Train** — drift loss + DINO features on padded 32×32 RGB MNIST
2. **Generate** — for each digit `0`–`9`, sample many candidates and keep the best

Ubuntu uses `CUDA_TRAIN_KWARGS` (batch 512, DINO 0.5, 1024 oscillators). Mac uses smaller presets to avoid OOM.

---

## Legacy benchmark tools

- `train_dcgan.py`, `eval/`, `scripts/generate_synthetic_digits.py`

## References

- [unconv-ai/Un-0](https://github.com/unconv-ai/Un-0)
- [hcm444/Un-0](https://github.com/hcm444/Un-0) — fork used as a dependency

## License

MIT
