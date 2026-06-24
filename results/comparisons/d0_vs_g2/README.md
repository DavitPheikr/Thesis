# D0 (LiDAR) vs G2 (LiDAR + RGB + Lovász) — focused comparison

Full-coverage held-out **test** set (seed 42). D0→G2 is the thesis spine
(geometry-only vs best geometry+appearance). NOTE: not a clean RGB ablation
(G2 also adds Lovász loss, more features, more epochs) — system-level comparison.

## Headline
| metric | D0 | G2 | G2 − D0 |
| --- | ---: | ---: | ---: |
| marking IoU | 0.4147 | 0.5177 | +0.1030 |
| precision | 0.4832 | 0.6811 | +0.1979 |
| recall | 0.7452 | 0.6833 | -0.0620 |
| F1 | 0.5863 | 0.6822 | +0.0959 |
| pred/true | 1.5423 | 1.0031 | -0.5392 |
| road IoU | 0.9274 | 0.9489 | +0.0215 |
| other IoU | 0.9774 | 0.9850 | +0.0076 |

**Read:** the gain is precision/calibration — G2 lifts precision (0.483→0.681) and fixes over-prediction (pred/true 1.54→1.00), at a small recall cost (0.745→0.683).

## Files

- `metrics_d0_vs_g2.csv` — full headline table.
- `per_sequence_d0_vs_g2.csv` / `fig_per_sequence.png` — per scene (065 = night).
- `distance_d0_vs_g2.csv` / `fig_distance.png` — by range.
- `fig_metrics_bars.png`, `fig_overprediction.png`, `fig_error_breakdown.png`.
- `fig_confusion_D0.png`, `fig_confusion_G2.png` — row-normalized confusion.
- `fig_val_vs_test.png` — matched full-coverage validation vs test (generalization).
