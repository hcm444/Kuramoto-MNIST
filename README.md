# Kuramoto-MNIST

Train [Un-0](https://github.com/unconv-ai/Un-0) Kuramoto dynamics on MNIST and generate **ten grayscale digits** (0–9). Includes a **class-conditional DCGAN baseline** and a **fair comparison** on full MNIST.

## Fair comparison (Kuramoto vs DCGAN)

**800 epochs each** on 60k MNIST — matched batch size, progress cadence, and best-of-32 selection.

| | Kuramoto | DCGAN |
|---|----------|-------|
| Result @ 800 ep | All classes; thick, jagged digits | Clean MNIST-like digits |
| Verdict | Learns structure; plateaus ~100 ep | **Stronger generator** on this benchmark |

| Progress grid | Path |
|---------------|------|
| Kuramoto (80×10) | [`digits/kuramoto/progress_80x10.png`](digits/kuramoto/progress_80x10.png) |
| DCGAN (80×10) | [`digits/dcgan/progress_80x10.png`](digits/dcgan/progress_80x10.png) |

**Full write-up:** [`research/fair_comparison.md`](research/fair_comparison.md) · [`research/README.md`](research/README.md)

```bash
# Reproduce: 400 ep fresh + 400 ep resume → 800 total
./scripts/run_fair_comparison_400.sh
./scripts/resume_fair_comparison.sh
```

---

## Quick start (local CUDA)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Pretrain digit encoder (once)
python scripts/pretrain_digit_encoder.py --epochs 5 --device cuda

# Kuramoto progress grid (100 epochs)
./cloud/train_progress.sh

# Kuramoto quality preset (400 epochs, sharper loss mix)
./scripts/train_quality_local.sh

# DCGAN baseline (100 epochs)
python make_dcgan_progress_grid.py --device cuda --epochs 100
```

| Preset | GPU | Epochs | ~time |
|--------|-----|--------|-------|
| `quality6gb` | 6 GB | 400 | ~4 hr |
| `6gb` (auto) | 6 GB | 200 | ~2 hr |
| `cloud` | ≥8 GB | 1200 | 2–5 days |
| Fair comparison | 6 GB | 800×2 models | ~8 hr |

---

## Vast.ai (1200 epochs)

**Guide:** **[cloud/vast/README.md](cloud/vast/README.md)**

```bash
curl -fsSL https://raw.githubusercontent.com/hcm444/Kuramoto-MNIST/main/cloud/vast/onstart.sh | bash
```

Resume from a laptop checkpoint:

```bash
RESUME=checkpoints/kuramoto/final.pt EPOCHS=1200 ./cloud/train_long.sh
```

---

## Mac (Apple Silicon)

```bash
python make_progress_grid.py --device mps --epochs 40 --candidates 16
```

---

## Outputs

```
digits/
  kuramoto/              # fair-comparison Kuramoto grids + rows
  dcgan/                 # fair-comparison DCGAN grids + rows
  progress_10x10.png     # legacy Kuramoto grid
  0.png … 9.png          # final digits (make_digits.py)
checkpoints/kuramoto/    # trained model (gitignored)
checkpoints/dcgan/       # DCGAN checkpoint (gitignored)
checkpoints/mnist_digit_encoder.pt
research/                # experiment write-ups
```

---

## How it works

| | **Kuramoto (Un-0)** | **DCGAN** |
|---|---|---|
| Objective | Drift loss (digit encoder + pixel) | Adversarial (G vs D) |
| Entry point | `train_kuramoto.py` | `train_dcgan.py` |
| Progress grids | `make_progress_grid.py` | `make_dcgan_progress_grid.py` |

Kuramoto uses a frozen **MNIST digit CNN** as the feature backbone (`pixel=0.06`, `dino=0.35`, `collapse=0.01` on the `6gb` preset).

Training notes: **[FINDINGS.md](FINDINGS.md)** · **[research/](research/)**

---

## Commands

```bash
# Re-stitch progress grid from manifest
python make_progress_grid.py --skip-train --device cuda \
  --manifest digits/kuramoto/progress_manifest.json \
  --output digits/kuramoto/progress_80x10.png

# Export final digits 0–9
python make_digits.py --skip-train --device cuda --candidates 32

# FID + throughput (requires checkpoints)
python eval/compare.py \
  --kuramoto checkpoints/kuramoto/final.pt \
  --dcgan checkpoints/dcgan/final.pt
```

---

## References

- [unconv-ai/Un-0](https://github.com/unconv-ai/Un-0) — oscillator image generator  
- [Un-0 blog](https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/)

MIT
