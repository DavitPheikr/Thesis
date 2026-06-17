# RGB Brightness/Jitter Context

This note records why Milestone H moved from z-cropping and marking-aware
sampling toward a controlled RGB brightness/contrast jitter experiment.

The current baseline entering this decision is:

```text
run:        G1_schedule_extend
checkpoint: epoch 27
input:      x, y, z, intensity, r, g, b, rgb_valid
loss:       weighted CE + 0.5 * Lovasz-Softmax
marking IoU: 0.542659
F1:          0.703538
precision:   0.635133
recall:      0.788456
mIoU:        0.823239
```

Source: `logs/milestone_g/run_analysis/G1_schedule_extend/summary.csv`.

## Why RGB Jitter Became The Next Candidate

Milestone H first audited two supervisor-suggested data changes:

1. **Z-crop**: rejected.
   A wide z crop such as `[-7.0, 5.0]` is geometrically safe, but it does not
   touch the dominant G1 error. The sampled G1 false positives are road-level
   bright-road points, not high/low `other` points. The best-case marking-IoU
   gain from the crop was estimated as only `+0.0071`, below the single-seed
   decision margin.

   Sources:
   - `logs/milestone_h/dataset_analysis/z_crop_audit/z_crop_recommendation.md`
   - `logs/milestone_h/dataset_analysis/z_crop_audit/z_crop_error_effect.md`

2. **Marking-aware / hard-negative sampling**: rejected for H0.
   The patch-center audit showed low leverage. Under the real RandLA-Net
   32768-neighbor patch geometry, changing patch centers barely changes what
   the model sees:

   ```text
   marking per patch: uniform 589.9 -> marking-centered 738.1  (x1.251)
   hardneg per patch: uniform 1431.6 -> hardneg-centered 1630.9 (x1.139)
   strongest candidate seen-marking frequency: x1.083 vs uniform
   ```

   This is not enough to justify a sampler change, especially because G1's main
   issue is precision/false positives, while naive marking-centered sampling
   usually pushes recall and can revive E0/F0-style overprediction.

   Source:
   - `logs/milestone_h/dataset_analysis/marking_aware_sampling_audit/sampling_recommendation.md`

Those audits point to the same practical conclusion: the remaining bottleneck is
not missing marking exposure or irrelevant vertical `other`; it is the RGB-based
road-to-marking false-positive mechanism.

## The Suspected RGB Shortcut

G1 uses front-camera RGB as point features:

```text
[x, y, z, intensity, r, g, b, rgb_valid]
```

RGB improved the project overall, but the residual G1 errors suggest the model
can still treat bright, low-saturation, road-like regions as markings. This is
not proof that the model literally uses a single brightness rule, but the
evidence is consistent with a brightness/luminance shortcut:

```text
bright / near-neutral road-looking RGB -> predicted marking
```

The key sampled G1 error budget was:

```text
road->marking false positives: 507,459
other->marking false positives: 15,317
marking->road false negatives: 250,859
marking->other false negatives: 4,836
```

So nearly all marking false positives are road points, and the precision problem
is larger than the missed-marking problem. Source:
`logs/milestone_g/run_analysis/G1_schedule_extend/sampled_error_analysis_epoch27/`
and summarized in `logs/milestone_h/notes/milestone_h_context.md`.

## Eval-Only Brightness Sensitivity Probe

Before implementing any training change, Milestone H ran an eval-only probe on
the frozen G1 epoch-27 checkpoint.

Source outputs:

```text
logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/
  brightness_sensitivity_summary.csv
  brightness_sensitivity_confusions.json
  brightness_sensitivity_recommendation.md
  plots/road_to_marking_fp_vs_scale.png
  plots/marking_precision_recall_iou_vs_scale.png
  plots/pred_true_vs_scale.png
```

Probe details:

```text
checkpoint: G1_schedule_extend epoch 27
split: validation
steps: 2160
seed: 42
device: cuda
scales: 0.80, 0.85, 1.00, 1.15, 1.20
operation: RGB *= scale only where rgb_valid >= 0.5
RGB columns: features[:, :, 4:7]
rgb_valid column: features[:, :, 7]
```

The probe changed only the batch tensor inside the analysis script before the
model forward pass. It did not train, rebuild cache, mutate the dataset, change
the config, or write into the G1 run directory.

Source script:
`logs/milestone_h/run_analysis/analysis_code/g1_rgb_brightness_sensitivity.py`.

## What The Probe Found

The frozen G1 model is clearly sensitive to absolute RGB brightness.

| RGB scale | road->marking FP | precision | recall | marking IoU | F1 | pred/true | mIoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 435,009 | 0.657808 | 0.746809 | 0.537857 | 0.699489 | 1.135 | 0.822288 |
| 0.85 | 454,798 | 0.650720 | 0.757287 | 0.538426 | 0.699970 | 1.164 | 0.822440 |
| 1.00 | 507,112 | 0.631443 | 0.777689 | 0.534893 | 0.696977 | 1.232 | 0.820587 |
| 1.15 | 556,667 | 0.612559 | 0.787846 | 0.525824 | 0.689233 | 1.286 | 0.816201 |
| 1.20 | 574,396 | 0.605724 | 0.789684 | 0.521581 | 0.685578 | 1.304 | 0.814181 |

Main facts:

- Darkening RGB reduces road-to-marking false positives.
  - `0.85`: `454,798`, down `52,314` points (`-10.32%`) from scale `1.00`.
  - `0.80`: `435,009`, down `72,103` points (`-14.22%`) from scale `1.00`.
- Brightening RGB increases road-to-marking false positives.
  - `1.15`: `556,667`, up `49,555` points (`+9.77%`) from scale `1.00`.
  - `1.20`: `574,396`, up `67,284` points (`+13.27%`) from scale `1.00`.
- The road-to-marking false-positive count is monotonic across the tested
  scales.
- Darkening improves precision but reduces recall. For example:
  - precision improves from `0.631443` at scale `1.00` to `0.650720` at scale
    `0.85`;
  - recall drops from `0.777689` to `0.757287`.
- Brightening does the opposite: recall rises slightly but precision, IoU, F1,
  mIoU, and predicted/true calibration all worsen.

This means fixed darkening is not the final solution. The probe is diagnostic:
it shows that brightness controls the precision/recall operating point.

## Why Train-Only Jitter Is The Logical Follow-Up

The goal is not to make validation images darker. The goal is to make the model
less dependent on the exact absolute brightness seen during training.

The proposed H0 experiment should therefore use **train-only** RGB
brightness/contrast jitter:

```text
training:   RGB is mildly jittered
validation: original RGB, unchanged
test:       original RGB, unchanged
```

Validation and test must stay unchanged because they measure real performance on
the actual dataset and keep H0 comparable with G1. The brightness probe modified
validation only as a diagnostic stress test, not as an official evaluation
protocol.

Recommended H0 jitter:

```text
brightness factor: 0.85-1.15
contrast factor:   0.90-1.10
scope:             train only
granularity:       one factor per sampled patch
apply to:          RGB-valid points only, rgb_valid >= 0.5
clip RGB:          [0, 1]
leave unchanged:   x, y, z, intensity, labels, rgb_valid
```

Reasons for the mild range:

- `0.85-1.15` already produced about a 10% false-positive swing in the probe.
- `0.80-1.20` is stronger, but the dark end already reduces recall and the bright
  end worsens overprediction.
- Mild jitter tests robustness while preserving brightness as a useful cue for
  real markings.

## What H0 Would Test

Hypothesis:

```text
If G1 relies too strongly on absolute RGB brightness, then train-only mild
brightness/contrast jitter should reduce road->marking false positives on
unchanged validation data while preserving most of G1's recall.
```

This is a clean experiment if everything except train-time RGB jitter remains
G1-identical:

```text
same input dimensions
same cache
same model architecture
same loss
same class weights
same sampler
same optimizer/scheduler
same split
same checkpoint selection by marking IoU
```

The run should be trained from scratch, not resumed from G1, because the
regularization must affect learning from epoch 1.

## Success And Failure Criteria

Compare H0 against official G1 epoch 27 and the G1 sampled analysis.

Strong success:

```text
marking IoU >= 0.551   # roughly +0.008 over G1 official IoU
precision >= 0.635 or close with a clear IoU/F1 gain
recall does not collapse, ideally >= 0.76
predicted/true marking ratio <= 1.30
road->marking false positives decrease vs G1 sampled analysis
mIoU does not regress
```

Neutral / weak result:

```text
IoU gain < 0.008, or precision/FPs improve only by trading away too much recall.
```

Failure:

```text
road->marking false positives increase,
predicted/true drifts upward toward E0/F0 behavior,
or recall/IoU collapses because brightness was regularized too strongly.
```

If H0 fails, G1 remains the final model. The thesis-safe interpretation would be
that G1 is brightness-sensitive, but simple train-time brightness/contrast
jitter did not produce a defensible improvement; richer RGB representations or
illumination-aware augmentation become future work.

If H0 improves, the thesis-safe interpretation is:

```text
The G1 model showed measurable sensitivity to absolute RGB brightness. A
controlled train-only RGB brightness/contrast jitter reduced reliance on this
brightness shortcut and improved road-marking segmentation on unchanged
validation data.
```

## Limitation

The probe tests only absolute/global RGB brightness sensitivity. It does not
test all possible RGB shortcuts:

- local contrast patterns,
- hue or saturation cues,
- camera exposure variation,
- geometry/RGB interaction,
- annotation ambiguity,
- sequence-specific visual context.

Therefore H0 should be framed as a controlled test of brightness robustness, not
as a complete fix for all RGB-related failure modes.

## Current Decision

Milestone H has evidence to justify H0_rgb_jitter.

Do not re-open z-crop or sampler changes unless visual inspection or a later
analysis gives new evidence. The next training candidate is:

```text
H0_rgb_jitter:
  G1 setup + train-only RGB brightness/contrast jitter
  brightness 0.85-1.15
  contrast   0.90-1.10
  validation/test unchanged
```

