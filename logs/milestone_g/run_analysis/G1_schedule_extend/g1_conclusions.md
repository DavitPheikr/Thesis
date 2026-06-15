# G1 Conclusions

G1 tests one question: did G0 stop before its unchanged ReduceLROnPlateau scheduler reached a useful lower-LR refinement phase?

## Headline

| run | epoch | marking IoU | F1 | precision | recall | mIoU | pred/true |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 best | 18 | 0.523048 | 0.686844 | 0.629949 | 0.755036 | 0.817407 | 1.199 |
| G1 best | 27 | 0.542659 | 0.703538 | 0.635133 | 0.788456 | 0.823239 | 1.241 |
| G1 final | 35 | 0.532956 | 0.695331 | 0.619992 | 0.791513 | 0.820118 | |

## Interpretation

- G1 best marking IoU changed by `+0.019611` versus G0 best.
- G1 best precision changed by `+0.005184`.
- G1 best recall changed by `+0.033419`.
- G1 final-minus-best IoU drift is `-0.009703`.

## Sampled Best-Checkpoint Check

- sampled checkpoint: `/home/coder/project/logs/milestone_g/runs/G1_schedule_extend/checkpoints/ckpt_epoch_00027.pth`
- sampled all marking IoU: `0.534884`
- sampled all pred/true: `1.232`
- RGB-valid active point share: `0.817`
- RGB-valid marking IoU / pred-true: `0.535114` / `1.277`
- RGB-invalid active point share: `0.183`
- RGB-invalid marking IoU / pred-true: `0.533968` / `1.068`

## Source Files

- `summary.csv`
- `best_vs_final.csv`
- `g0_vs_g1_epoch_metrics.csv`
- `pred_true_ratio_by_epoch.csv`
- `loss_component_summary.csv`
- `sampled_error_analysis_epoch*/`
