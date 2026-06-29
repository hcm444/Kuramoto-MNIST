# Fair comparison — Kuramoto vs DCGAN on MNIST

**Final protocol:** 800 epochs each, batch 128, full 60k MNIST, progress row every 10 epochs (80 snapshots × 10 digits), 32 candidates per digit, seed 42, CUDA (RTX A1000 6 GB).

Training ran in two legs:

1. **400 epochs** — fresh run (`scripts/run_fair_comparison_400.sh`, completed 2026-06-29)
2. **400→800** — resume from `final.pt` (`scripts/resume_fair_comparison.sh`, completed 2026-06-29)

## Progress grids

| Model | 400 ep grid | 800 ep grid (full) |
|-------|-------------|-------------------|
| Kuramoto | ![K400](../digits/kuramoto/progress_40x10.png) | ![K800](../digits/kuramoto/progress_80x10.png) |
| DCGAN | ![D400](../digits/dcgan/progress_40x10.png) | ![D800](../digits/dcgan/progress_80x10.png) |

Row snapshots: `digits/kuramoto/progress_rows/`, `digits/dcgan/progress_rows/`.

## Matched settings

| Setting | Kuramoto | DCGAN |
|---------|----------|-------|
| Epoch budget | 800 | 800 |
| Batch size | 128 | 128 |
| Train set | 60,000 | 60,000 |
| Progress cadence | every 10 epochs | every 10 epochs |
| Selection | best-of-32 per digit | best-of-32 per digit |
| Seed | 42 | 42 |

## Model-specific settings (by design)

| | Kuramoto | DCGAN |
|---|----------|-------|
| Architecture | Un-0, 512 oscillators | Class-cond DCGAN (~10× params) |
| Objective | Pixel + digit-encoder drift + collapse | Non-saturating GAN |
| Loss weights | `pixel=0.06`, `dino=0.35`, `collapse=0.01` | `lr_g=2e-4`, `lr_d=1e-4`, 2× G steps |
| Data scaling | MNIST mean/std → padded RGB | `[0,1]` → `[-1,1]` for tanh G |
| Time / 400 ep | ~3.5 hr | ~1.7 hr |

The data-scaling difference is intentional: Kuramoto uses the Un-0 MNIST pipeline; DCGAN uses the standard `[-1,1]` real-image range with `dcgan=True` in the dataloader (critical fix — see [comparison_100epoch.md](comparison_100epoch.md)).

## Qualitative results

### DCGAN

- **~epoch 50–100:** readable digits emerge; thin strokes, natural variation
- **epoch 400:** clean 0–9, minor class quirks (e.g. fuzzy 0)
- **epoch 800:** stable; small refinements, no major visual jump after 400

Final-row quality: MNIST-like handwriting, low background noise, good class diversity.

### Kuramoto

- **~epoch 30–50:** class-separated thick blobs
- **~epoch 100–150:** recognizable 0–9; loss and `grad_norm` plateau
- **epoch 400–800:** slow visual polish — strokes stay **thick, blocky, high-contrast** with salt-and-pepper edge noise

Extra training (400→800) did **not** close the gap with DCGAN. The model learns all ten classes without collapse but appears stuck in a local optimum of the drift objective.

## Training signals (epoch 800)

| Metric | Kuramoto | DCGAN |
|--------|----------|-------|
| Primary loss | `total ≈ 0.023`, flat since ~100 | `d_loss ≈ 0.38`, `g_loss ≈ 4.4` |
| Grad norm | `≈ 0.017` (near zero updates) | G still improving slowly |
| Mode collapse | None — all classes distinct | None |

## Verdict

| Criterion | Winner |
|-----------|--------|
| Digit readability @ 800 ep | **DCGAN** |
| Class diversity | Tie |
| Training stability (after fixes) | Tie |
| Compute efficiency | **DCGAN** (~2× faster per epoch) |
| Novelty / oscillator dynamics | Kuramoto (research interest) |

**Bottom line:** On this fair MNIST benchmark, the tuned DCGAN baseline is the stronger generator. Kuramoto reliably learns class-conditional structure but needs architectural or loss changes (e.g. `quality6gb` preset, higher pixel weight, cloud-scale 1200 ep) to match DCGAN polish — more epochs alone are insufficient.

## Follow-ups

1. **Kuramoto `quality6gb`** — `pixel=0.10`, `dino=0.20`; rerun fair protocol
2. **FID / throughput** — `python eval/compare.py --kuramoto … --dcgan …`
3. **Cloud scale** — Kuramoto 1200 ep (`KURAMOTO_PRESET=cloud`) vs DCGAN at matched wall-clock

## Reproduce

```bash
# Leg 1: fresh 400 epochs
./scripts/run_fair_comparison_400.sh

# Leg 2: resume to 800 (or EPOCHS=1200)
./scripts/resume_fair_comparison.sh

# Monitor
tail -f fair_comparison_800.log
```

Checkpoints: `checkpoints/kuramoto/final.pt`, `checkpoints/dcgan/final.pt` (local only, gitignored).
