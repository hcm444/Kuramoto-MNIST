# Fair comparison — resume to 800 epochs

Continued from 400-epoch checkpoints. Progress every 10 epochs (rows 10…400 from first run, then 410…800).

| | Kuramoto | DCGAN |
|---|----------|-------|
| Target epochs | 800 | 800 |
| Resume from | `checkpoints/kuramoto/final.pt` | `checkpoints/dcgan/final.pt` |
| Progress | every 10 | every 10 |

## Grids

| Model | Full progress | Rows |
|-------|---------------|------|
| Kuramoto | ![Kuramoto](digits/kuramoto/progress_80x10.png) | `digits/kuramoto/progress_rows/` |
| DCGAN | ![DCGAN](digits/dcgan/progress_80x10.png) | `digits/dcgan/progress_rows/` |

## Results

Completed: 2026-06-29 22:17 UTC

**Leg 2 of the fair comparison** (resume from 400-epoch checkpoints). Full analysis: [fair_comparison.md](fair_comparison.md).

| Model | 400→800 change |
|-------|----------------|
| **DCGAN** | Marginal refinement; already good at 400 |
| **Kuramoto** | Slight smoothing; still thick/noisy vs DCGAN |

Combined grid: epochs 10, 20, …, 800 (80 rows × 10 digits).
