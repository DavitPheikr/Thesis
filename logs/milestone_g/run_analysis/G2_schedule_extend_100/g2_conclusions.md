# G2 Conclusions

This compares G2 (candidate) against baseline(s) F0, D0 on the held-out test set.
Single-seed (seed 42) controlled comparison; see `logs/milestone_g/notes/single_seed_limitation.md` before reading IoU deltas near the ~0.008 noise floor.

## Headline

| run | epoch | marking IoU | F1 | precision | recall | mIoU | pred/true |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F0 best | 13 | 0.482770 | 0.651173 | 0.549249 | 0.799543 | 0.798682 | 1.456 |
| D0 best | 18 | 0.440294 | 0.611394 | 0.514788 | 0.752636 | 0.780829 | 1.462 |
| G2 best | 68 | 0.550444 | 0.710047 | 0.635617 | 0.804219 | 0.827825 | 1.265 |
| G2 final | 100 | 0.546559 | 0.706806 | 0.630213 | 0.804593 | 0.826876 |  |

## Interpretation

- vs **F0** best: marking IoU `+0.067674`, precision `+0.086368`, recall `+0.004676`.
- vs **D0** best: marking IoU `+0.110150`, precision `+0.120830`, recall `+0.051583`.
- G2 final-minus-best IoU drift is `-0.003885`.

## Sampled Best-Checkpoint Check

- sampled checkpoint: `/home/coder/project/logs/milestone_g/runs/G2_schedule_extend_100/checkpoints/ckpt_epoch_00068.pth`
- sampled all marking IoU: `0.540046`
- sampled all pred/true: `1.271`
- RGB-valid active point share: `0.817`
- RGB-valid marking IoU / pred-true: `0.540187` / `1.289`
- RGB-invalid active point share: `0.183`
- RGB-invalid marking IoU / pred-true: `0.539515` / `1.205`

## Source Files

- `summary.csv`
- `best_vs_final.csv`
- `comparison_epoch_metrics.csv`
- `pred_true_ratio_by_epoch.csv`
- `loss_component_summary.csv`
- `sampled_error_analysis_epoch*/`
