# G1 RGB Brightness Sensitivity Recommendation

Eval-only probe. No training, no dataset/cache/config mutation, no writes to the G1 run directory.

## Provenance

- generated_at: `2026-06-17T02:24:59`
- script: `logs/milestone_h/run_analysis/analysis_code/g1_rgb_brightness_sensitivity.py`
- config: `/home/coder/project/logs/milestone_g/configs/g1_schedule_extend.yml`
- checkpoint: `/home/coder/project/logs/milestone_g/runs/G1_schedule_extend/checkpoints/ckpt_epoch_00027.pth`
- split: `validation`
- steps: `2160`
- seed: `42`
- device: `cuda`
- rgb_valid threshold: `0.5`
- scales: `0.80, 0.85, 1.00, 1.15, 1.20`

## Answers

1. **Does darkening RGB reduce road->marking false positives?** Yes. 0.85: 454798 (-52314, -10.32%); 0.80: 435009 (-72103, -14.22%).
2. **Does brightening RGB increase road->marking false positives?** Yes. 1.15: 556667 (+49555, +9.77%); 1.20: 574396 (+67284, +13.27%).
3. **Is the effect monotonic across scales?** Yes for road->marking FP count.
4. **Is 0.85-1.15 enough?** Yes (max mild FP change 10.32%; max strong FP change 14.22%).
5. **Should we train H0_rgb_jitter?** GO: H0_rgb_jitter is justified by eval-time brightness sensitivity.
6. **Recommended H0 brightness range:** `0.85-1.15`.

## Scale Table

| scale | road->marking | precision | recall | IoU | F1 | pred/true | mIoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 435009 | 0.657808 | 0.746809 | 0.537857 | 0.699489 | 1.135 | 0.822288 |
| 0.85 | 454798 | 0.650720 | 0.757287 | 0.538426 | 0.699970 | 1.164 | 0.822440 |
| 1.00 | 507112 | 0.631443 | 0.777689 | 0.534893 | 0.696977 | 1.232 | 0.820587 |
| 1.15 | 556667 | 0.612559 | 0.787846 | 0.525824 | 0.689233 | 1.286 | 0.816201 |
| 1.20 | 574396 | 0.605724 | 0.789684 | 0.521581 | 0.685578 | 1.304 | 0.814181 |

## Limitation

This probe tests sensitivity to absolute/global RGB brightness scaling only.
It does not test all possible relative, contextual, hue, saturation, camera-exposure,
or geometry/RGB interaction shortcuts.

## Thesis-safe interpretation

The frozen G1 model is measurably sensitive to global RGB brightness:
road->marking false positives move when RGB is darkened/brightened.
This supports running a controlled train-only RGB jitter experiment to test
whether mild brightness robustness can reduce bright-road false positives.