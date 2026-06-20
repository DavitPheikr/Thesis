# G2_schedule_extend_100 — Marking-Logit Bias Sweep (operating-point diagnostic)

- checkpoint: `/home/coder/project/logs/milestone_g/runs/G2_schedule_extend_100/checkpoints/ckpt_epoch_00068.pth`
- split/steps/seed: validation / 2160 / 42
- raw-remap mismatch rate: 0.00e+00

**Diagnostic only.** The official evaluation is plain argmax (b = 0); a bias-tuned
number is NOT reported as the model's result. Treat IoU gains below the single-seed
noise floor (~0.008) as not real.

| bias | IoU | precision | recall | F1 | pred/true | road->mk FP | mk->road FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.50 | 0.545457 | 0.6681 | 0.7482 | 0.7059 | 1.120 | 419230 | 284285 |  <-- best IoU
| -0.40 | 0.545401 | 0.6606 | 0.7577 | 0.7058 | 1.147 | 438679 | 273429 |
| -0.30 | 0.544968 | 0.6529 | 0.7673 | 0.7055 | 1.175 | 459299 | 262521 |
| -0.20 | 0.543896 | 0.6447 | 0.7767 | 0.7046 | 1.205 | 481419 | 251810 |
| -0.10 | 0.542405 | 0.6360 | 0.7866 | 0.7033 | 1.237 | 505708 | 240606 |
| +0.00 | 0.540046 | 0.6267 | 0.7962 | 0.7013 | 1.271 | 532367 | 229626 |  <-- b=0
| +0.10 | 0.537227 | 0.6170 | 0.8060 | 0.6990 | 1.306 | 561096 | 218439 |
| +0.20 | 0.533585 | 0.6066 | 0.8160 | 0.6959 | 1.345 | 592960 | 207116 |
| +0.30 | 0.529231 | 0.5955 | 0.8262 | 0.6922 | 1.387 | 628066 | 195488 |
| +0.40 | 0.523772 | 0.5836 | 0.8363 | 0.6875 | 1.433 | 667197 | 183942 |
| +0.50 | 0.517948 | 0.5717 | 0.8464 | 0.6824 | 1.481 | 708193 | 172422 |

- IoU at b=0 (official argmax): `0.540046`
- best IoU at b=`-0.50`: `0.545457` (gain `+0.0054`)

## Verdict

Argmax (b=0) is at/near the best operating point (best bias -0.50, IoU gain +0.0054 < 0.008 noise floor). The model is well-calibrated; the precision/recall trade is not freely improvable by thresholding.

Per-distance pred/true columns (`pred_true_*`) are in the CSV: a positive bias pulling the far buckets back toward 1.0 indicates under-prediction at range.
