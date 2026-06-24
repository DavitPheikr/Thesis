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
| all | 0.526043 | 0.714686 | 0.665882 | 0.932 | 316132 | 407316 |
| rgb_valid | 0.535543 | 0.719486 | 0.676874 | 0.941 | 289285 | 362067 |
| rgb_invalid | 0.421963 | 0.654019 | 0.543222 | 0.831 | 26847 | 45249 |

Interpretation rule (compare to F0 sampled epoch-13):

- If `rgb_valid` pred/true fell below F0's 1.550, Lovasz reduced the
  RGB-valid overprediction fingerprint.
- If `rgb_valid` still holds most road->marking errors, residual false
  positives remain tied to valid front-camera RGB regions.
