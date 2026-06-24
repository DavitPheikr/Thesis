# G0 Best-Epoch Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time
best-epoch validation sample.

- checkpoint: `/home/coder/project/logs/milestone_g/runs/G2_schedule_extend_100/checkpoints/ckpt_epoch_00068.pth`
- steps: `2160`
- seed: `2`
- device: `cuda`

## Main Metrics

| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.530317 | 0.693901 | 0.692264 | 0.998 | 366140 | 374722 |
| rgb_valid | 0.539929 | 0.699240 | 0.703248 | 1.006 | 333962 | 332067 |
| rgb_invalid | 0.425886 | 0.627865 | 0.569688 | 0.907 | 32178 | 42655 |

Interpretation rule (compare to F0 sampled epoch-13):

- If `rgb_valid` pred/true fell below F0's 1.550, Lovasz reduced the
  RGB-valid overprediction fingerprint.
- If `rgb_valid` still holds most road->marking errors, residual false
  positives remain tied to valid front-camera RGB regions.
