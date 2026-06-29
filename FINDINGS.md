# CUDA training findings (June 2026)

## Summary

| Run | Preset | Result |
|-----|--------|--------|
| Early CUDA | `dino=0.5`, 100 epochs | Mode collapse — identical blobs |
| MNIST-tuned | digit encoder, 512 osc | Readable 0–9 at 100 epochs |
| **Fair comparison** | 800 ep × 2 models | DCGAN wins on polish; Kuramoto learns all classes — see [`research/fair_comparison.md`](research/fair_comparison.md) |
| Quality experiment | `quality6gb`, 400 ep | Sharper than `6gb`; still jagged vs DCGAN |
| Cloud target | `KURAMOTO_PRESET=cloud`, 1200 ep | Vast.ai / `train_long.sh` |

Early runs used a CIFAR-style loss mix on MNIST. **MNIST-tuned presets** (`mnist_bench/digits.py`) fix mode collapse with the digit encoder backbone.

---

## Fair comparison (800 epochs)

Matched protocol on full MNIST: batch 128, progress every 10 epochs, 32 candidates per digit.

| Model | @ 800 epochs | Notes |
|-------|--------------|-------|
| **DCGAN** | Clean MNIST-like digits | Stable training after v3 data fix (`dcgan=True`) |
| **Kuramoto** | Readable but thick/noisy | Loss plateaus ~epoch 100; more epochs help only slightly |

Grids: `digits/kuramoto/progress_80x10.png`, `digits/dcgan/progress_80x10.png`

Reproduce:

```bash
./scripts/run_fair_comparison_400.sh
./scripts/resume_fair_comparison.sh
```

---

## Kuramoto vs DCGAN

| | **Kuramoto (Un-0)** | **DCGAN** |
|---|---|---|
| Objective | Drift loss (digit encoder + pixel) | Adversarial (G vs D) |
| Discriminator | None | ConvNet critic |
| Entry point | `train_kuramoto.py` | `train_dcgan.py` |

Kuramoto is **not** a GAN. With the digit encoder and tuned weights it learns class-conditional digits without a discriminator, but on this benchmark DCGAN produces cleaner samples.

Pilot 100-epoch notes (incl. DCGAN v1/v2/v3 history): [`research/comparison_100epoch.md`](research/comparison_100epoch.md)

---

## Legacy early CUDA run (superseded)

| Setting | Value |
|---------|-------|
| DINO weight | 0.5 (too high for MNIST) |
| Result | Mode collapse — identical blobs at epoch 100 |

![Legacy 10×10 grid](digits/progress_10x10.png)

Re-export digits from a trained checkpoint:

```bash
python make_digits.py --skip-train --device cuda --candidates 32
```

Compare models:

```bash
python eval/compare.py \
  --kuramoto checkpoints/kuramoto/final.pt \
  --dcgan checkpoints/dcgan/final.pt
```
