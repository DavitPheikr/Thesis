# G0 Marking-Logit Bias Sweep

- checkpoint: `/home/coder/project/logs/milestone_g/runs/G0_rgb_lovasz/checkpoints/ckpt_epoch_00018.pth`
- split/steps/seed: validation / 2160 / 42
- raw-remap mismatch rate: 0.00e+00
- decision gate: IoU gain over b=0 must be >= 0.008

**This is a diagnostic only.** A bias-tuned IoU must NOT be reported as the
model's result; it only reveals the direction of calibration headroom.

| bias | IoU | precision | recall | F1 | pred/true | rgb_valid p/t | rgb_invalid p/t |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.50 | 0.522109 | 0.6835 | 0.6885 | 0.6860 | 1.007 | 1.060 | 0.815 |
| -0.40 | 0.523468 | 0.6740 | 0.7010 | 0.6872 | 1.040 | 1.093 | 0.848 |
| -0.30 | 0.524409 | 0.6642 | 0.7136 | 0.6880 | 1.074 | 1.127 | 0.883 |
| -0.20 | 0.524622 | 0.6539 | 0.7262 | 0.6882 | 1.111 | 1.163 | 0.921 |  <-- best IoU
| -0.10 | 0.524195 | 0.6433 | 0.7390 | 0.6878 | 1.149 | 1.200 | 0.961 |
| +0.00 | 0.522980 | 0.6321 | 0.7518 | 0.6868 | 1.189 | 1.240 | 1.004 |  <-- b=0
| +0.10 | 0.520954 | 0.6203 | 0.7649 | 0.6850 | 1.233 | 1.284 | 1.048 |
| +0.20 | 0.517784 | 0.6075 | 0.7781 | 0.6823 | 1.281 | 1.332 | 1.097 |
| +0.30 | 0.513066 | 0.5931 | 0.7917 | 0.6782 | 1.335 | 1.385 | 1.152 |
| +0.40 | 0.506838 | 0.5774 | 0.8058 | 0.6727 | 1.396 | 1.446 | 1.212 |
| +0.50 | 0.498380 | 0.5590 | 0.8212 | 0.6652 | 1.469 | 1.521 | 1.282 |

- IoU at b=0: `0.522980`
- best IoU at b=`-0.20`: `0.524622` (gain `+0.0016`)

## Verdict

b=0 is best (or gain below the single-seed gate) -> G0 is well-calibrated. Do NOT lower lambda. Stop, or run only a schedule continuation as polish.

**Route:** CASE B: keep G0; optional schedule-extend run.

Long-range pred/true columns (`pred_true_*`) are in the CSV: check whether a
positive bias pulls the >30 m buckets back toward 1.0 (G0 under-predicts there).
