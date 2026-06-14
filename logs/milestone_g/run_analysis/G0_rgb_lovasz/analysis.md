# G0 RGB + Lovasz Run-Level Analysis

This report uses saved training artifacts only (`eval_history.csv`,
`loss_components.csv`, `confusion_epoch_*.npy`). It does not run a fresh
sampled inference pass.

Milestone G = Milestone F0 with the loss changed to
`weighted_CE + 0.5 * Lovasz-Softmax`. Everything
else (RGB front features, softened marking weight effective 15.0,
`dim_features=16`) is unchanged. The config-snapshot check below is the
formal proof of that.

## Provenance Checks

| run | eval rows | eval jsons | confusions | checkpoints | loss_components |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 | 25 | 25 | 25 | 25 | n/a |
| E0 | 25 | 25 | 25 | 25 | n/a |
| F0 | 25 | 25 | 25 | 25 | n/a |
| G0 | 25 | 25 | 25 | 25 | 25 |

G0 loss-component consistency (all within tolerance, else this report
would have failed):

- lovasz_lambda: `0.5`
- max |eval val_loss - val_total_loss|: `0.00e+00`
- max |val_total - (val_ce + lambda*val_lovasz)|: `8.71e-10`
- max |train_total - (train_ce + lambda*train_lovasz)|: `6.84e-10`

G0 config snapshot check:

- feature mode: `intensity_rgb_front`
- class weights: `[119562394.0, 14344000.0, 173473484.0]`
- dim_features: `16`
- model input channels: `8`
- loss: `weighted_ce_lovasz`, lambda `0.5`, classes `present`

## Headline

Best checkpoint for every run is selected by maximum raw marking IoU
(`lane_iou`), the same rule across D0/E0/F0/G0.

| metric | D0 ep18 | E0 ep14 | F0 ep13 | G0 ep18 | G0-F0 | G0-D0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| marking_iou | 0.440294 | 0.438439 | 0.482770 | 0.523048 | +0.040279 | +0.082754 |
| marking_f1 | 0.611394 | 0.609604 | 0.651173 | 0.686844 | +0.035671 | +0.075450 |
| marking_precision | 0.514788 | 0.470298 | 0.549249 | 0.629949 | +0.080700 | +0.115161 |
| marking_recall | 0.752636 | 0.866170 | 0.799543 | 0.755036 | -0.044507 | +0.002400 |
| miou | 0.780829 | 0.780658 | 0.798682 | 0.817407 | +0.018726 | +0.036578 |

## CE Loss Comparison (fair: pure CE only)

F0 `val_loss` is pure weighted CE. G0 `val_ce` is the CE component of
the combined loss, computed identically. These are directly comparable.
G0 `val_loss` is the TOTAL (CE + Lovasz) and must NOT be compared to F0.

| run | epoch | val CE | val total | val Lovasz (raw) |
| --- | ---: | ---: | ---: | ---: |
| F0 best | 13 | 0.118228 | 0.118228 | 0.000000 |
| G0 best | 18 | 0.112872 | 0.209156 | 0.192567 |

## Best vs Final Epoch (drift)

The F0 hypothesis G tests: does adding the IoU-surrogate Lovasz term
reduce the best-to-final drift seen in F0 (val loss kept improving while
marking IoU degraded after the best epoch)?

| run | metric | best | final | delta final-best |
| --- | --- | ---: | ---: | ---: |
| F0 | marking IoU | 0.482770 | 0.449982 | -0.032787 |
| G0 | marking IoU | 0.523048 | 0.478593 | -0.044455 |
| F0 | precision | 0.549249 | 0.484524 | -0.064725 |
| G0 | precision | 0.629949 | 0.550910 | -0.079039 |
| F0 | recall | 0.799543 | 0.863239 | +0.063696 |
| G0 | recall | 0.755036 | 0.784758 | +0.029722 |

## Marking Overprediction Check

| run | label | marking pred/true ratio |
| --- | --- | ---: |
| D0 | official | 1.462032 |
| E0 | best | 1.841747 |
| F0 | best | 1.455702 |
| G0 | best | 1.198568 |
| F0 | final | 1.781625 |
| G0 | final | 1.424476 |

G0 marking pred/true ratio at best is `1.199` vs F0 `1.456`. Closer to 1.0 means better calibrated.

## Key Confusion Changes (best checkpoints)

| error | D0 | E0 | F0 | G0 | G0-F0 | G0-D0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| road->marking | 249846 | 363006 | 243315 | 168907 | -74408 | -80939 |
| marking->road | 95638 | 51945 | 78971 | 92659 | +13688 | -2979 |
| other->marking | 28267 | 22130 | 17837 | 4976 | -12861 | -23291 |

## What Still Needs Sampled / Component Analysis

The run-level metrics show whether G beat F0, but not whether the Lovasz
term is healthy (scale, stability, IoU-alignment) or whether residual
errors moved. Those are answered by:

- `g0_loss_component_analysis.py` (CE vs Lovasz vs total, alignment)
- `g0_sampled_error_analysis.py` (RGB-valid, sequence, distance, subtype)

## Outputs

- `summary.csv`
- `confusion_breakdown.csv`
- `pred_true_ratio.csv`
- `artifact_counts.csv`
- `analysis.md`
