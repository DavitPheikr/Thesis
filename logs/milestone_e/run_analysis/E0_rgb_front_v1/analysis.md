# E0 RGB Front Run-Level Analysis

This report uses saved training artifacts only: `eval_history.csv` and
`confusion_epoch_*.npy`. It does not run a fresh inference pass, so it
cannot yet answer per-point `rgb_valid` prediction questions.

## Provenance Checks

| run | eval rows | eval jsons | confusions | checkpoints |
| --- | ---: | ---: | ---: | ---: |
| D0 | 25 | 25 | 25 | 25 |
| E0 | 25 | 25 | 25 | 25 |

E0 config snapshot check:

- feature mode: `intensity_rgb_front`
- camera: `front_camera`
- color sampling: `bilinear`
- RGB normalization: `divide_by_255`
- model input channels: `8`

## Headline

E0's best marking-IoU checkpoint is epoch `14`. D0's
official LiDAR-only checkpoint is epoch `18`.

| metric | D0 epoch 18 | E0 best | delta |
| --- | ---: | ---: | ---: |
| marking IoU | 0.440294 | 0.438439 | -0.001854 |
| marking F1 | 0.611394 | 0.609604 | -0.001790 |
| marking precision | 0.514788 | 0.470298 | -0.044489 |
| marking recall | 0.752636 | 0.866170 | +0.113534 |
| mIoU | 0.780829 | 0.780658 | -0.000171 |

E0 did not beat D0 on the main balanced marking metrics. It strongly
increased recall, but precision fell enough that IoU and F1 stayed slightly
below D0.

## E0 Best vs Final Epoch

| metric | E0 best | E0 final | delta final-best |
| --- | ---: | ---: | ---: |
| marking IoU | 0.438439 | 0.399308 | -0.039131 |
| precision | 0.470298 | 0.415527 | -0.054771 |
| recall | 0.866170 | 0.910953 | +0.044783 |
| val loss | 0.138934 | 0.129703 | -0.009231 |

The final epoch has lower validation loss and higher recall but worse marking
IoU. This repeats D0's pattern: loss keeps improving while positive-class
calibration drifts toward overprediction.

## Overprediction Check

| class | D0 pred/true | E0 pred/true | delta |
| --- | ---: | ---: | ---: |
| road | 1.001824 | 0.985442 | -0.016382 |
| marking | 1.462032 | 1.841747 | +0.379714 |
| other | 0.985293 | 0.985778 | +0.000485 |

E0 marking pred/true ratio is `1.842` vs D0 `1.462`.
This confirms that E0 increased marking overprediction.

## Key Confusion Changes

| error | D0 count | E0 count | delta |
| --- | ---: | ---: | ---: |
| road->marking | 249846 | 363006 | +113160 |
| marking->road | 95638 | 51945 | -43693 |
| other->marking | 28267 | 22130 | -6137 |

Interpretation: E0 likely reduced missed markings, but it also produced many
more false-positive markings from road/other. The next diagnostic must be a
fresh sampled inference pass that stratifies errors by `rgb_valid`.

## What This Does Not Yet Prove

This report does not prove that RGB itself is harmful. It only proves that
the E0 training setup with RGB plus the D0 weighted-CE objective did not
improve the balanced marking metric. The likely causes remain:

1. RGB signal is useful but amplified by the aggressive marking class weight.
2. Invalid-RGB fallback behavior hurts some frames/sequences.
3. RGB projection/color quality introduces false-positive cues.

The next required analysis is sampled checkpoint inference at E0 epoch 14
with per-point `rgb_valid`, sequence, distance, raw subtype, intensity, and
RGB summaries.

## Outputs

- `summary.csv`
- `confusion_breakdown.csv`
- `pred_true_ratio.csv`
- `analysis.md`
