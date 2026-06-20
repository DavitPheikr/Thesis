# G2 Run Analysis

Candidate run **G2** compared against baseline **F0**. All runs are single-seed; see `single_seed_limitation.md`.

## Validation

- feature mode: `intensity_rgb_front`
- loss: `weighted_ce_lovasz` with lambda `0.5`
- scheduler patience: `6`
- loss composition residual, val: `9.41e-10`
- config differences vs G1 reference: `{}`

## Headline

| run | epoch | marking IoU | F1 | precision | recall | mIoU | pred/true marking | LR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F0 best | 13 | 0.482770 | 0.651173 | 0.549249 | 0.799543 | 0.798682 | 1.456 | 0.001400 |
| G2 best | 68 | 0.550444 | 0.710047 | 0.635617 | 0.804219 | 0.827825 | 1.265 | 0.000044 |
| G2 final | 100 | 0.546559 | 0.706806 | 0.630213 | 0.804593 | 0.826876 | | 0.000003 |

## LR Events

| epoch | LR before | LR after | factor |
| ---: | ---: | ---: | ---: |
| 27 | 0.001400 | 0.000700 | 0.500 |
| 36 | 0.000700 | 0.000350 | 0.500 |
| 45 | 0.000350 | 0.000175 | 0.500 |
| 53 | 0.000175 | 0.000087 | 0.500 |
| 61 | 0.000087 | 0.000044 | 0.500 |
| 69 | 0.000044 | 0.000022 | 0.500 |
| 77 | 0.000022 | 0.000011 | 0.500 |
| 85 | 0.000011 | 0.000005 | 0.500 |
| 93 | 0.000005 | 0.000003 | 0.500 |

## Best vs Final Drift

| run | metric | best epoch | final epoch | best | final | final-best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F0 | marking_iou | 13 | 25 | 0.482770 | 0.449982 | -0.032787 |
| F0 | marking_f1 | 13 | 25 | 0.651173 | 0.620673 | -0.030500 |
| F0 | marking_precision | 13 | 25 | 0.549249 | 0.484524 | -0.064725 |
| F0 | marking_recall | 13 | 25 | 0.799543 | 0.863239 | +0.063696 |
| F0 | miou | 13 | 25 | 0.798682 | 0.787873 | -0.010809 |
| D0 | marking_iou | 18 | 25 | 0.440294 | 0.410337 | -0.029956 |
| D0 | marking_f1 | 18 | 25 | 0.611394 | 0.581900 | -0.029495 |
| D0 | marking_precision | 18 | 25 | 0.514788 | 0.451201 | -0.063587 |
| D0 | marking_recall | 18 | 25 | 0.752636 | 0.819195 | +0.066559 |
| D0 | miou | 18 | 25 | 0.780829 | 0.771793 | -0.009037 |
| G2 | marking_iou | 68 | 100 | 0.550444 | 0.546559 | -0.003885 |
| G2 | marking_f1 | 68 | 100 | 0.710047 | 0.706806 | -0.003240 |
| G2 | marking_precision | 68 | 100 | 0.635617 | 0.630213 | -0.005404 |
| G2 | marking_recall | 68 | 100 | 0.804219 | 0.804593 | +0.000374 |
| G2 | miou | 68 | 100 | 0.827825 | 0.826876 | -0.000949 |

## Output Map

- Core plots: `plots/`
- Sampled best-checkpoint analysis: `sampled_error_analysis_epochXX/`
- Visual inspection notes: `visual_inspection_notes.md`
