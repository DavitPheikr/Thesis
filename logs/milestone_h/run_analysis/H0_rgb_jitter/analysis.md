# H0 Run Analysis

Candidate run **H0** compared against baseline **G2**. All runs are single-seed; see `single_seed_limitation.md`.

## Validation

- feature mode: `intensity_rgb_front`
- loss: `weighted_ce_lovasz` with lambda `0.5`
- scheduler patience: `6`
- loss composition residual, val: `2.03e-09`
- config differences vs G1 reference: `{}`

## Headline

| run | epoch | marking IoU | F1 | precision | recall | mIoU | pred/true marking | LR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 best | 68 | 0.550444 | 0.710047 | 0.635617 | 0.804219 | 0.827825 | 1.265 | 0.000044 |
| H0 best | 37 | 0.550671 | 0.710236 | 0.658916 | 0.770225 | 0.826728 | 1.169 | 0.000350 |
| H0 final | 100 | 0.544748 | 0.705291 | 0.626238 | 0.807185 | 0.826938 | | 0.000001 |

## LR Events

| epoch | LR before | LR after | factor |
| ---: | ---: | ---: | ---: |
| 23 | 0.001400 | 0.000700 | 0.500 |
| 36 | 0.000700 | 0.000350 | 0.500 |
| 44 | 0.000350 | 0.000175 | 0.500 |
| 52 | 0.000175 | 0.000087 | 0.500 |
| 60 | 0.000087 | 0.000044 | 0.500 |
| 68 | 0.000044 | 0.000022 | 0.500 |
| 76 | 0.000022 | 0.000011 | 0.500 |
| 84 | 0.000011 | 0.000005 | 0.500 |
| 92 | 0.000005 | 0.000003 | 0.500 |
| 100 | 0.000003 | 0.000001 | 0.500 |

## Best vs Final Drift

| run | metric | best epoch | final epoch | best | final | final-best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G2 | marking_iou | 68 | 100 | 0.550444 | 0.546559 | -0.003885 |
| G2 | marking_f1 | 68 | 100 | 0.710047 | 0.706806 | -0.003240 |
| G2 | marking_precision | 68 | 100 | 0.635617 | 0.630213 | -0.005404 |
| G2 | marking_recall | 68 | 100 | 0.804219 | 0.804593 | +0.000374 |
| G2 | miou | 68 | 100 | 0.827825 | 0.826876 | -0.000949 |
| F0 | marking_iou | 13 | 25 | 0.482770 | 0.449982 | -0.032787 |
| F0 | marking_f1 | 13 | 25 | 0.651173 | 0.620673 | -0.030500 |
| F0 | marking_precision | 13 | 25 | 0.549249 | 0.484524 | -0.064725 |
| F0 | marking_recall | 13 | 25 | 0.799543 | 0.863239 | +0.063696 |
| F0 | miou | 13 | 25 | 0.798682 | 0.787873 | -0.010809 |
| H0 | marking_iou | 37 | 100 | 0.550671 | 0.544748 | -0.005923 |
| H0 | marking_f1 | 37 | 100 | 0.710236 | 0.705291 | -0.004945 |
| H0 | marking_precision | 37 | 100 | 0.658916 | 0.626238 | -0.032678 |
| H0 | marking_recall | 37 | 100 | 0.770225 | 0.807185 | +0.036960 |
| H0 | miou | 37 | 100 | 0.826728 | 0.826938 | +0.000210 |

## Output Map

- Core plots: `plots/`
- Sampled best-checkpoint analysis: `sampled_error_analysis_epochXX/`
- Visual inspection notes: `visual_inspection_notes.md`
