# G0 Best-Epoch Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time
best-epoch validation sample.

- checkpoint: `/home/coder/project/logs/milestone_e/runs/E0_rgb_front_v1/checkpoints/ckpt_epoch_00014.pth`
- steps: `2160`
- seed: `42`
- device: `cuda`

## Main Metrics

| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.399581 | 0.443177 | 0.802448 | 1.811 | 2392864 | 497062 |
| rgb_valid | 0.401894 | 0.442204 | 0.815117 | 1.843 | 2272724 | 432703 |
| rgb_invalid | 0.365172 | 0.459731 | 0.639691 | 1.391 | 120140 | 64359 |

Interpretation rule (compare to F0 sampled epoch-13):

- If `rgb_valid` pred/true fell below F0's 1.550, Lovasz reduced the
  RGB-valid overprediction fingerprint.
- If `rgb_valid` still holds most road->marking errors, residual false
  positives remain tied to valid front-camera RGB regions.
