# E0 Epoch-14 Sampled Error Analysis

Fresh sampled validation inference pass. This is not the exact training-time
epoch-14 validation sample.

- checkpoint: `/home/coder/project/logs/milestone_e/runs/E0_rgb_front_v1/checkpoints/ckpt_epoch_00014.pth`
- steps: `2160`
- seed: `42`
- device: `cuda`

## Main Metrics

| stratum | marking IoU | precision | recall | pred/true | road->marking | marking->road |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.436871 | 0.469718 | 0.862018 | 1.835 | 1055189 | 155521 |
| rgb_valid | 0.421745 | 0.449251 | 0.873229 | 1.944 | 909871 | 111732 |
| rgb_invalid | 0.506883 | 0.569667 | 0.821402 | 1.442 | 145318 | 43789 |

Interpretation rule:

- If `rgb_valid` has most road->marking errors, RGB is likely being
  over-amplified by the current objective.
- If `rgb_invalid` is disproportionately bad, invalid-RGB fallback behavior
  is a major issue.
