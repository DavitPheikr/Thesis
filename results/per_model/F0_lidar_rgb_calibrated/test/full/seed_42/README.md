# G0 Best-Epoch Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time
best-epoch validation sample.

- checkpoint: `/home/coder/project/logs/milestone_f/runs/F0_rgb_soft_weights/checkpoints/ckpt_epoch_00013.pth`
- steps: `2160`
- seed: `42`
- device: `cuda`

## Main Metrics

| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.449537 | 0.544637 | 0.720241 | 1.322 | 1358722 | 711423 |
| rgb_valid | 0.455906 | 0.545356 | 0.735419 | 1.349 | 1290847 | 625262 |
| rgb_invalid | 0.359254 | 0.532016 | 0.525237 | 0.987 | 67875 | 86161 |

Interpretation rule (compare to F0 sampled epoch-13):

- If `rgb_valid` pred/true fell below F0's 1.550, Lovasz reduced the
  RGB-valid overprediction fingerprint.
- If `rgb_valid` still holds most road->marking errors, residual false
  positives remain tied to valid front-camera RGB regions.
