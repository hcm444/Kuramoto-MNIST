# Kuramoto vs DCGAN — 100-epoch MNIST comparison

> **Superseded by the fair comparison:** see [`fair_comparison.md`](fair_comparison.md) (400 + 800 epochs, matched protocol). This doc retains the early pilot history (DCGAN v1/v2 debugging).

Generated: 2026-06-28 12:27 UTC

## Goal

Train both generators for **100 epochs** on full MNIST and compare qualitative progress via **10×10 grids** (10 time snapshots × 10 digit classes).

## Grids

| Model | Progress grid | Row snapshots |
|-------|---------------|---------------|
| Kuramoto (Un-0 + digit encoder) | ![Kuramoto 10×10](digits/kuramoto/progress_10x10.png) | `digits/kuramoto/progress_rows/` |
| DCGAN (class-conditional) | ![DCGAN 10×10](digits/dcgan/progress_10x10.png) | `digits/dcgan/progress_rows/` |

## Configuration

| Setting | Kuramoto | DCGAN |
|---------|----------|-------|
| Epochs | 100 | 100 |
| Progress snapshots | 10 (every 10 epochs) | 10 (every 10 epochs) |
| Candidates per digit | 32 | 32 |
| Batch size | 128 | 128 |
| Device | cuda | cuda |
| Checkpoint | `checkpoints/kuramoto/final.pt` | `checkpoints/dcgan/final.pt` |

### Kuramoto-specific

| Setting | Value |
|---------|-------|
| Oscillators | 512 |
| Feature backbone | digit encoder |
| Pixel weight | 0.06 |
| DINO/drift weight | 0.35 |
| Collapse weight | 0.01 |

### DCGAN-specific

| Setting | Value |
|---------|-------|
| Latent dim | 128 |
| Learning rate | 2e-4 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |

## Progress epochs captured

- **Kuramoto:** 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
- **DCGAN:** 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

## Summary

| Model | Best result @ 100 epochs | Verdict |
|-------|--------------------------|---------|
| **Kuramoto** (Un-0 + digit encoder) | Recognizable digits 0–9; noisy edges | **Success** |
| **DCGAN v1** (broken training) | Static/noise | **Failed** — D collapsed G by epoch 2 |
| **DCGAN v2** (fixed training) | Noisy blobs / faint structure; not readable digits | **Partial** — stable training, poor sample quality |

**Winner on digit quality: Kuramoto.** Fixed DCGAN trains stably but still does not produce clean MNIST digits at 100 epochs on this architecture.

### DCGAN fixes applied (v2)

- Removed **BatchNorm from discriminator** (DCGAN convention)
- **Label smoothing** on real targets (0.9)
- **Non-saturating G loss** (`softplus(-logits)`)
- **Lower D LR** (1e-4 vs G 2e-4)
- **2× G steps** per D step

---

## Findings

### 1. Kuramoto learns digit structure without a discriminator

With the **MNIST digit encoder** (replacing DINO) and tuned loss weights (`pixel=0.06`, `dino_weight=0.35`, `collapse=0.01`), the Kuramoto oscillator model learned class-conditional digit generation in **100 epochs on full MNIST** (60k, batch 128, 512 oscillators, RTX A1000 6 GB).

This is a major improvement over the earlier **DINO-backed 100-epoch run** documented in `FINDINGS.md`, which collapsed to identical blobs across all classes. Switching the feature backbone to a pretrained digit CNN fixed mode collapse for this epoch budget.

**Learning trajectory (visual, progress rows every 10 epochs):**

| Phase | Epochs | Kuramoto |
|-------|--------|----------|
| Early | 10–30 | Thick blobby strokes; faint class hints; heavy pixel noise |
| Mid | 40–70 | Distinct per-class shapes (loops, vertical 1s, 7 diagonals); jagged edges |
| Late | 80–100 | All ten digits readable; some columns sharper than others; no collapse to one blob |

**Epoch 100 row:** 0, 8, 3, 2, 4/9, 0, 1, 7, 5/9, 0 — class diversity preserved.

**Remaining issues:** High-frequency noise and blocky boundaries; best-of-32 selection helps but outputs are not clean MNIST exports. Likely needs more epochs (200–400) or cloud-scale training (1200) for polish, not a different loss mix at 100.

### 2. DCGAN v1 failed; v2 trains but samples stay noisy

**v1 (broken):** salt-and-pepper noise at all epochs. `d_loss → 0`, `g_loss → 26.8` by epoch 100.

**v2 (fixed, `train_dcgan_100.log`):**

| Epoch | d_loss | g_loss | Samples |
|-------|--------|--------|---------|
| 10 | ~0.33 | ~8 | Horizontal stripe blobs |
| 50 | ~0.33 | ~17 | Noisy blobs, no digits |
| 100 | ~0.33 | ~17.5 | Blobs + vertical artifacts; not readable 0–9 |

Training is **stable** (D no longer hits zero), but G loss stays high and **best-of-32 selection cannot fix unstructured noise**.

### 3. Fair comparison requires a fixed protocol

For research reproducibility we used:

- Same dataset (full MNIST train)
- Same epoch budget (**100**)
- Same progress cadence (**every 10 epochs**, 10 rows)
- Same selection (**32 candidates per digit**, `score_digit` heuristic)
- Same batch size (**128**)
- Same hardware (**CUDA**, 6 GB preset for Kuramoto)

**Conclusion:** Kuramoto produces readable digits at 100+ epochs. Fixed DCGAN trains stably but still blobs at 100 — may need 200+ epochs or further tuning.

### 4. Recommended epoch budgets (from this experiment)

| Goal | Kuramoto | DCGAN | Notes |
|------|----------|-------|-------|
| Research snapshot (10×10 grid) | **100** | **100** (after fix) | ~1 hr + ~20 min on A1000 6 GB |
| Cleaner Kuramoto digits | **200–400** | — | Diminishing returns; optional follow-up |
| Un-0 scale / publication | **1200** (cloud, ≥8 GB) | — | See `CLOUD_TRAIN_KWARGS` |

### 5. Updated comparison (Kuramoto 400 ep quality vs DCGAN 100 ep fixed)

| | Kuramoto (400 ep, quality preset) | DCGAN v2 (100 ep, fixed) |
|---|-----------------------------------|---------------------------|
| Grid | `digits/progress_10x10.png` | `digits/dcgan/progress_10x10.png` |
| Epoch 100/400 row | Readable 0–9, still jagged | Noisy blobs |
| Train time | ~4 hr | ~25 min |
| Inference | 4.3 ms/image | 1.0 ms/image |

**Kuramoto wins on quality.** DCGAN wins on speed but has not matched digit readability in this repo.

### 6. Open questions

1. DCGAN at 200 epochs with v2 training — does G loss drop and digits emerge?
2. Kuramoto at cloud scale (1200 ep, 1024 osc) — how clean vs DCGAN at same compute budget?
3. FID / classifier accuracy — not yet measured.

---

## Qualitative notes

1. **Early training (epochs 10–30):** Kuramoto shows thick strokes and emerging topology; DCGAN is already pure static with no digit structure.
2. **Mid training (epochs 40–70):** Kuramoto develops class-separated strokes; DCGAN unchanged (noise).
3. **Late training (epochs 80–100):** Kuramoto — readable 0–9, noisy but diverse; DCGAN — still noise, no mode collapse because G never modes.
4. **Winner:** **Kuramoto** — readable digits at 100+ epochs; DCGAN v2 trains but blobs at 100.

## Reproduce

Fixed DCGAN (100 epochs):

```bash
python make_dcgan_progress_grid.py --device cuda --epochs 100 --candidates 32
```

```bash
python make_progress_grid.py --device cuda --epochs 100 --candidates 32 \
  --output digits/kuramoto/progress_10x10.png \
  --rows-dir digits/kuramoto/progress_rows \
  --manifest digits/kuramoto/progress_manifest.json
```

DCGAN only:

```bash
python make_dcgan_progress_grid.py --device cuda --epochs 100 --candidates 32
```
