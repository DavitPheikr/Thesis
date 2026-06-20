# H0 Conclusions

This compares H0 (candidate) against baseline(s) G2, F0 on the held-out test set.
Single-seed (seed 42) controlled comparison; see `logs/milestone_g/notes/single_seed_limitation.md` before reading IoU deltas near the ~0.008 noise floor.

## Headline

| run | epoch | marking IoU | F1 | precision | recall | mIoU | pred/true |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 best | 68 | 0.550444 | 0.710047 | 0.635617 | 0.804219 | 0.827825 | 1.265 |
| F0 best | 13 | 0.482770 | 0.651173 | 0.549249 | 0.799543 | 0.798682 | 1.456 |
| LiDAR + RGB + Lovász + Jitter best | 37 | 0.550671 | 0.710236 | 0.658916 | 0.770225 | 0.826728 | 1.169 |
| LiDAR + RGB + Lovász + Jitter final | 100 | 0.544748 | 0.705291 | 0.626238 | 0.807185 | 0.826938 |  |

## Interpretation

- vs **G2** best: marking IoU `+0.000227`, precision `+0.023299`, recall `-0.033995`.
- vs **F0** best: marking IoU `+0.067901`, precision `+0.109667`, recall `-0.029319`.
- H0 final-minus-best IoU drift is `-0.005923`.

## Sampled Best-Checkpoint Check

- sampled checkpoint: `/home/coder/project/logs/milestone_h/runs/H0_rgb_jitter/checkpoints/ckpt_epoch_00037.pth`
- sampled all marking IoU: `0.540719`
- sampled all pred/true: `1.175`
- RGB-valid active point share: `0.817`
- RGB-valid marking IoU / pred-true: `0.540583` / `1.183`
- RGB-invalid active point share: `0.183`
- RGB-invalid marking IoU / pred-true: `0.541221` / `1.145`

## Source Files

- `summary.csv`
- `best_vs_final.csv`
- `comparison_epoch_metrics.csv`
- `pred_true_ratio_by_epoch.csv`
- `loss_component_summary.csv`
- `sampled_error_analysis_epoch*/`
