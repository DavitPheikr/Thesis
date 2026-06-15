# G1 Schedule Extension Analysis

G1 is a copied continuation of G0. It resumes from G0 epoch 25 and trains to epoch 35 with the original scheduler unchanged.

## Validation

- feature mode: `intensity_rgb_front`
- loss: `weighted_ce_lovasz` with lambda `0.5`
- scheduler patience: `6`
- loss composition residual, val: `8.71e-10`

## Headline

| run | epoch | marking IoU | F1 | precision | recall | mIoU | pred/true marking | LR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 best | 18 | 0.523048 | 0.686844 | 0.629949 | 0.755036 | 0.817407 | 1.199 | 0.001400 |
| G1 best | 27 | 0.542659 | 0.703538 | 0.635133 | 0.788456 | 0.823239 | 1.241 | 0.000700 |
| G1 final | 35 | 0.532956 | 0.695331 | 0.619992 | 0.791513 | 0.820118 | | 0.000700 |

## LR Events

| epoch | LR before | LR after | factor |
| ---: | ---: | ---: | ---: |
| 27 | 0.001400 | 0.000700 | 0.500 |

## Best vs Final Drift

| run | metric | best epoch | final epoch | best | final | final-best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G0 | marking_iou | 18 | 25 | 0.523048 | 0.478593 | -0.044455 |
| G0 | marking_f1 | 18 | 25 | 0.686844 | 0.647363 | -0.039481 |
| G0 | marking_precision | 18 | 25 | 0.629949 | 0.550910 | -0.079039 |
| G0 | marking_recall | 18 | 25 | 0.755036 | 0.784758 | +0.029722 |
| G0 | miou | 18 | 25 | 0.817407 | 0.795974 | -0.021433 |
| G1 | marking_iou | 27 | 35 | 0.542659 | 0.532956 | -0.009703 |
| G1 | marking_f1 | 27 | 35 | 0.703538 | 0.695331 | -0.008206 |
| G1 | marking_precision | 27 | 35 | 0.635133 | 0.619992 | -0.015141 |
| G1 | marking_recall | 27 | 35 | 0.788456 | 0.791513 | +0.003057 |
| G1 | miou | 27 | 35 | 0.823239 | 0.820118 | -0.003121 |

## Output Map

- Core plots: `plots/`
- Sampled best-checkpoint analysis: `sampled_error_analysis_epochXX/`
- Visual inspection notes: `visual_inspection_notes.md`
