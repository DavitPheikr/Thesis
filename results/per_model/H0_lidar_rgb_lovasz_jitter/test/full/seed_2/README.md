# G0 Best-Epoch Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time
best-epoch validation sample.

- checkpoint: `/home/coder/project/logs/milestone_h/runs/H0_rgb_jitter/checkpoints/ckpt_epoch_00037.pth`
- steps: `2160`
- seed: `2`
- device: `cuda`

## Main Metrics

| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.508771 | 0.698475 | 0.651963 | 0.933 | 679716 | 876133 |
| rgb_valid | 0.515571 | 0.701818 | 0.660185 | 0.941 | 630013 | 795321 |
| rgb_invalid | 0.422225 | 0.650321 | 0.546239 | 0.840 | 49703 | 80812 |

Interpretation rule (compare to F0 sampled epoch-13):

- If `rgb_valid` pred/true fell below F0's 1.550, Lovasz reduced the
  RGB-valid overprediction fingerprint.
- If `rgb_valid` still holds most road->marking errors, residual false
  positives remain tied to valid front-camera RGB regions.
