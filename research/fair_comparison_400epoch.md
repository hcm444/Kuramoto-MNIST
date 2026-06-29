# Fair comparison — 400 epochs, full MNIST

Protocol: matched epoch budget, batch 128, full 60k train set, progress every 10 epochs (40 rows × 10 digits).

| | Kuramoto | DCGAN |
|---|----------|-------|
| Epochs | 400 | 400 |
| Batch | 128 | 128 |
| Progress | every 10 | every 10 |
| Candidates/digit | 32 | 32 |
| Kuramoto preset | `6gb` default losses | stable DCGAN + `dcgan=True` data |
| Data note | MNIST mean/std scaling | [-1, 1] tanh range |

## Grids (filled when training completes)

| Model | Full progress | Rows |
|-------|---------------|------|
| Kuramoto | ![Kuramoto](digits/kuramoto/progress_40x10.png) | `digits/kuramoto/progress_rows/` |
| DCGAN | ![DCGAN](digits/dcgan/progress_40x10.png) | `digits/dcgan/progress_rows/` |

## Results

Completed: 2026-06-29 13:55 UTC

**Leg 1 of the fair comparison.** See the full write-up in [fair_comparison.md](fair_comparison.md).

| Model | @ 400 epochs |
|-------|----------------|
| **DCGAN** | Clean, MNIST-like digits — already strong |
| **Kuramoto** | All classes readable; thick, jagged strokes; loss plateau ~100–150 |

Continued to 800 epochs via [`scripts/resume_fair_comparison.sh`](../scripts/resume_fair_comparison.sh) → [fair_comparison_800epoch.md](fair_comparison_800epoch.md).
