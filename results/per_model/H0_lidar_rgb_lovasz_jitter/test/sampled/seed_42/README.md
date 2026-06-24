# G0 Best-Epoch Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time
best-epoch validation sample.

- checkpoint: `/home/coder/project/logs/milestone_h/runs/H0_rgb_jitter/checkpoints/ckpt_epoch_00037.pth`
- steps: `2160`
- seed: `42`
- device: `cuda`

## Main Metrics

| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.525287 | 0.717911 | 0.661905 | 0.922 | 312265 | 416381 |
| rgb_valid | 0.534458 | 0.722481 | 0.672524 | 0.931 | 285688 | 370389 |
| rgb_invalid | 0.425695 | 0.660914 | 0.544649 | 0.824 | 26577 | 45992 |

Interpretation rule (compare to F0 sampled epoch-13):

- If `rgb_valid` pred/true fell below F0's 1.550, Lovasz reduced the
  RGB-valid overprediction fingerprint.
- If `rgb_valid` still holds most road->marking errors, residual false
  positives remain tied to valid front-camera RGB regions.
