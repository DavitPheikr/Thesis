# G0 Best-Epoch Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time
best-epoch validation sample.

- checkpoint: `/home/coder/project/logs/milestone_g/runs/G2_schedule_extend_100/checkpoints/ckpt_epoch_00068.pth`
- steps: `2160`
- seed: `42`
- device: `cuda`

## Main Metrics

| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.536834 | 0.627290 | 0.788261 | 1.257 | 1095781 | 496466 |
| rgb_valid | 0.533278 | 0.621330 | 0.790049 | 1.272 | 890855 | 390770 |
| rgb_invalid | 0.550995 | 0.651373 | 0.781446 | 1.200 | 204926 | 105696 |

Interpretation rule (compare to F0 sampled epoch-13):

- If `rgb_valid` pred/true fell below F0's 1.550, Lovasz reduced the
  RGB-valid overprediction fingerprint.
- If `rgb_valid` still holds most road->marking errors, residual false
  positives remain tied to valid front-camera RGB regions.
