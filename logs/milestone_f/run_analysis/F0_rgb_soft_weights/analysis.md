# F0 RGB Soft-Weights Run-Level Analysis

This report uses saved training artifacts only: `eval_history.csv` and
`confusion_epoch_*.npy`. It does not run a fresh sampled inference pass.

## Provenance Checks

| run | eval rows | eval jsons | confusions | checkpoints |
| --- | ---: | ---: | ---: | ---: |
| D0 | 25 | 25 | 25 | 25 |
| E0 | 25 | 25 | 25 | 25 |
| F0 | 25 | 25 | 25 | 25 |

F0 config snapshot check:

- feature mode: `intensity_rgb_front`
- camera: `front_camera`
- camera lookup: `nearest_timestamp`
- color sampling: `bilinear`
- RGB normalization: `divide_by_255`
- class weights: `[119562394.0, 14344000.0, 173473484.0]`
- model input channels: `8`
- dim_features: `16`

## Headline

F0 is the first run in this series that clearly improves over the D0
LiDAR-only baseline and over the E0 RGB-front run with the original
marking weight.

| metric | D0 epoch 18 | E0 epoch 14 | F0 epoch 13 | F0-D0 | F0-E0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| marking_iou | 0.440294 | 0.438439 | 0.482770 | +0.042476 | +0.044330 |
| marking_f1 | 0.611394 | 0.609604 | 0.651173 | +0.039778 | +0.041568 |
| marking_precision | 0.514788 | 0.470298 | 0.549249 | +0.034461 | +0.078951 |
| marking_recall | 0.752636 | 0.866170 | 0.799543 | +0.046907 | -0.066627 |
| miou | 0.780829 | 0.780658 | 0.798682 | +0.017852 | +0.018023 |

Interpretation: E0 showed that RGB could increase marking recall, but it
overpredicted marking. F0 softened the marking class pressure and improved
precision while keeping recall above D0.

## F0 Best vs Final Epoch

| metric | F0 best epoch 13 | F0 final epoch 25 | delta final-best |
| --- | ---: | ---: | ---: |
| marking IoU | 0.482770 | 0.449982 | -0.032787 |
| precision | 0.549249 | 0.484524 | -0.064725 |
| recall | 0.799543 | 0.863239 | +0.063696 |
| val loss | 0.118228 | 0.106725 | -0.011504 |

F0 still drifts late: validation loss improves after the best marking IoU,
but marking precision and IoU degrade. The official F0 checkpoint is epoch
`13`, not epoch `25`.

## Marking Overprediction Check

| run | marking pred/true ratio |
| --- | ---: |
| D0 official | 1.462032 |
| E0 best | 1.841747 |
| F0 best | 1.455702 |

F0 marking pred/true ratio is `1.456` vs E0 `1.842`.
This is the first check that softened weighting reduced the E0
overprediction pattern.

## Key Confusion Changes

| error | D0 count | E0 count | F0 count | F0-D0 | F0-E0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| road->marking | 249846 | 363006 | 243315 | -6531 | -119691 |
| marking->road | 95638 | 51945 | 78971 | -16667 | +27026 |
| other->marking | 28267 | 22130 | 17837 | -10430 | -4293 |

## What Still Needs Sampled Analysis

The saved confusion matrices show F0 improved the run-level metric, but they
do not answer whether remaining errors are RGB-valid, sequence-specific,
distance-dependent, or raw-subtype-specific. The next stage is the
epoch-13 sampled inference analysis.

## Outputs

- `summary.csv`
- `confusion_breakdown.csv`
- `pred_true_ratio.csv`
- `artifact_counts.csv`
- `analysis.md`
