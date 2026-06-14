# F0 Epoch-13 Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time
epoch-13 validation sample.

- checkpoint: `/home/coder/project/logs/milestone_f/runs/F0_rgb_soft_weights/checkpoints/ckpt_epoch_00013.pth`
- steps: `2160`
- seed: `42`
- device: `cuda`

## Main Metrics

| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.480235 | 0.546206 | 0.799040 | 1.463 | 710399 | 228774 |
| rgb_valid | 0.471318 | 0.527015 | 0.816840 | 1.550 | 613824 | 163526 |
| rgb_invalid | 0.519858 | 0.640107 | 0.734558 | 1.148 | 96575 | 65248 |

Interpretation rule:

- If `rgb_valid` still has most road->marking errors, remaining false
  positives are still tied to valid front-camera RGB regions.
- If `rgb_invalid` is disproportionately bad, invalid-RGB fallback behavior
  is a major issue.
