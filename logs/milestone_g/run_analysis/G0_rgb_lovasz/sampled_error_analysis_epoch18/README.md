# G0 Best-Epoch Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time
best-epoch validation sample.

- checkpoint: `/home/coder/project/logs/milestone_g/runs/G0_rgb_lovasz/checkpoints/ckpt_epoch_00018.pth`
- steps: `2160`
- seed: `42`
- device: `cuda`

## Main Metrics

| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.522980 | 0.632140 | 0.751772 | 1.189 | 489746 | 275549 |
| rgb_valid | 0.525595 | 0.622253 | 0.771878 | 1.240 | 412083 | 198163 |
| rgb_invalid | 0.512477 | 0.676404 | 0.678933 | 1.004 | 77663 | 77386 |

Interpretation rule (compare to F0 sampled epoch-13):

- If `rgb_valid` pred/true fell below F0's 1.550, Lovasz reduced the
  RGB-valid overprediction fingerprint.
- If `rgb_valid` still holds most road->marking errors, residual false
  positives remain tied to valid front-camera RGB regions.
