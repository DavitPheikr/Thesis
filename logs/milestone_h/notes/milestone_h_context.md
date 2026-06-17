# Milestone H Context Document

Working context for Milestone H. Updated as each H analysis lands. Current state:
the **z-crop direction is resolved (do not crop)**, the **marking-aware sampling
direction is resolved (do not change sampler for H0)**, and the current
recommended H0 lever is **train-only RGB brightness/contrast jitter**.

---

## 1. Starting Point

The current best model before H is:

```text
run:        G1_schedule_extend
checkpoint: epoch 27
input:      xyz + intensity + front RGB + rgb_valid
loss:       weighted CE + 0.5 * Lovasz-Softmax
scheduler:  ReduceLROnPlateau on smoothed marking IoU
```

G1 best metrics:

| metric | value |
| --- | ---: |
| marking IoU | 0.542659 |
| F1 | 0.703538 |
| precision | 0.635133 |
| recall | 0.788456 |
| mIoU | 0.823239 |

Sources: `logs/milestone_g/run_analysis/G1_schedule_extend/summary.csv`,
`logs/milestone_g/notes/milestone_g_context.md`.

H must be compared against G1, not against older D/F/G0 runs.

---

## 2. Why H Exists

The supervisor proposed making the training problem more road-marking-focused:

1. **Z-axis crop** — remove high/low points (irrelevant vertical `other`).
2. **Marking-aware sampling** — sample neighborhoods containing markings more
   often.
3. **Longer training** (~100 epochs) once the data/sampling changes are proven
   safe.

Item 1 has now been audited and **closed (do not crop)** — see Section 3. Item 2
has also been audited and **closed for H0 (keep uniform sampling)** — see Section
5. The current remaining H0 lever is the RGB brightness shortcut — see Section 6.

---

## 3. Z-Crop Audit — RESOLVED: do not crop

Full audit in `logs/milestone_h/dataset_analysis/z_crop_audit/` (train+val, full
resolution, 344 M points; `z_crop_recommendation.md` is the authoritative verdict).
The investigation ran in three stages:

1. **Coordinate frame.** `z` is the per-frame LiDAR ego-pose height (the exact z
   the model sees; `recenter` touches x,y only), verified empirically (road
   median z = -0.253 m). It is **not gravity-aligned** and moves with the sensor.
   (`z_coordinate_frame.md`.)

2. **Tight crops -> rejected.** Because the frame tilts on grades, markings smear
   vertically on sloped/curved sequences (per-sequence marking z-spread ~8-10 m on
   hills vs ~0.7 m on flat scenes). Every tight band clipped markings on the
   sloped sequences; worst was **validation 034** (downhill, lanes only at the
   bottom) losing 22-82% of its markings. All tight candidates failed the
   guardrails.

3. **Hill-aware wide crops -> a dataset-safe band exists.** `[-7.0, 5.0]` keeps
   **99.98% of markings** (every busy sequence >= 99%, all raw 8/9/10 >= 99.97%)
   and still removes **~16% of `other`**, passing the geometric safety + 10%
   usefulness gate. The lower bound is forced to -7 by the downhills (040, 044),
   the upper bound by the uphills (043, 039).

4. **Error-effect check -> safe but not worth it.** Against G1's *actual* errors
   (`z_crop_error_effect.md`, computed from G1's confusion + the audit's removal),
   `[-7.0, 5.0]` touches at most **~4.2% of marking false positives, realistically
   ~0%**: the dominant road->marking FPs are **bright road at road level (z ~ 0),
   inside the band**. Best-case marking-IoU gain is **+0.0071, below the
   single-seed noise floor (~0.008)**. The 32 M `other` points it removes are
   ~99.96% correctly-classified `other` that never confused with marking.

**Decision: do not crop.** A height-based crop cannot reach errors that live at
road level. This also retires the gravity-aligned / ground-relative crop idea for
*accuracy* (a better frame would let you crop more `other`, but the errors are
still at road level); cropping remains only a possible efficiency optimization,
not pursued now.

If a crop were ever adopted it would have to be applied identically to
train/val/test with the cache re-baked, and `G1` class weights kept (cropped
weights are informational only).

---

## 4. The Real Bottleneck (cross-cutting finding)

Three independent analyses (F, G, and this audit) converge on the same point:
**G1's bottleneck is precision, not recall and not height.**

```text
G1 epoch-27 sampled error budget (70.6 M points):
  road->marking  FP : 507,459   (97% of all marking FPs; bright road at road level)
  other->marking FP :  15,317   (3%)
  marking->road  FN : 250,859
  marking->other FN :   4,836
  total marking FP  : 522,776   vs   total marking FN : 255,695   (~2:1)
```

The dominant error is the **RGB luminance shortcut**: bright, near-neutral,
road-like-intensity road points read as marking. This is a feature/RGB problem at
road level, which is why neither a z-crop nor longer training can touch it.

---

## 5. Marking-Aware Sampling Audit — RESOLVED: keep uniform sampling for H0

Full audit in
`logs/milestone_h/dataset_analysis/marking_aware_sampling_audit/`
(`sampling_recommendation.md` is the authoritative verdict). The audit used the
train split only and the real patch geometry:

```text
grid_size: 0.04
patch size: 32768 nearest neighbors
sampler family: uniform, marking-centered, bright-road-hard-negative-centered
```

Key center-leverage results:

```text
marking per patch: uniform 589.9 -> marking-centered 738.1  (x1.251)
hardneg per patch: uniform 1431.6 -> hardneg-centered 1630.9 (x1.139)
uniform zero-marking patch rate: 0.1533
strongest candidate seen-marking frequency: x1.083 vs uniform
```

These gains are too small to justify a sampler change. The patches already cover
a large part of each frame, so choosing a different center rarely changes the
visible class balance enough to matter. More importantly, G1's dominant failure
is **precision / road->marking false positives**, while naive marking-aware
sampling mainly pushes recall and risks reviving E0/F0-style overprediction.

**Decision: do not change the sampler for H0.** Keep uniform sampling and pivot
to the RGB feature/brightness mechanism.

## 6. RGB Brightness Sensitivity Probe — H0 candidate justified

Full probe in:

```text
logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/
```

The probe used the frozen G1 epoch-27 checkpoint and changed only RGB values in
the analysis batch before the forward pass. It did not train, rebuild cache,
modify the dataset, alter validation/test config, or write into the G1 run
directory.

Probe facts:

```text
split: validation
steps: 2160
seed: 42
scales: 0.80, 0.85, 1.00, 1.15, 1.20
RGB columns: features[..., 4:7]
rgb_valid column: features[..., 7]
apply scale only where rgb_valid >= 0.5
```

Result:

| RGB scale | road->marking FP | precision | recall | marking IoU | pred/true |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 435,009 | 0.657808 | 0.746809 | 0.537857 | 1.135 |
| 0.85 | 454,798 | 0.650720 | 0.757287 | 0.538426 | 1.164 |
| 1.00 | 507,112 | 0.631443 | 0.777689 | 0.534893 | 1.232 |
| 1.15 | 556,667 | 0.612559 | 0.787846 | 0.525824 | 1.286 |
| 1.20 | 574,396 | 0.605724 | 0.789684 | 0.521581 | 1.304 |

Interpretation:

- Darkening RGB reduces road->marking false positives and improves precision,
  but lowers recall.
- Brightening RGB increases road->marking false positives and recall, but lowers
  precision, IoU, F1, mIoU, and calibration.
- The effect is monotonic for road->marking false positives.
- `0.85-1.15` is already strong enough to expose the sensitivity; `0.80-1.20`
  is not needed for training.

**Decision: H0_rgb_jitter is justified.** The experiment should train from
scratch with train-only RGB brightness/contrast jitter while validation/test stay
unchanged.

Recommended H0:

```text
brightness: 0.85-1.15
contrast:   0.90-1.10
apply only to RGB-valid training points, rgb_valid >= 0.5
leave x/y/z/intensity/labels/rgb_valid unchanged
everything else G1-identical
```

Detailed rationale:
`logs/milestone_h/notes/rgb_jitter_context.md`.

## 7. Implications For The Remaining H Levers

- **Z-crop:** rejected for accuracy; can only be revisited as an efficiency idea.
- **Marking-aware sampling:** rejected for H0; low patch-center leverage.
- **RGB brightness robustness:** current best H0 lever; directly targets the
  measured road-level false-positive mechanism.
- **Longer training alone:** low expected value; G1 already stabilized at the
  tail, and the bottleneck is a feature shortcut rather than lack of epochs.

---

## 8. Risks To Control

### 8.1 Z-crop risk — CLOSED
Resolved by the audit (Section 3): do not crop. No residual risk to manage.

### 8.2 Marking-aware sampling risk — CLOSED for H0
Marking-aware sampling can recover recall but can reintroduce E0/F0-style
overprediction if the model sees an artificially marking-heavy distribution. The
sampler must preserve enough road/other context so the model still learns when
**not** to predict marking. Given Section 4, prioritise precision-preserving
designs. The H audit found low center leverage, so H0 should not change the
sampler.

### 8.3 RGB jitter risk
Brightness is also a real marking cue. If jitter is too strong, the model may
become conservative and lose true markings. This is why the recommended range is
mild (`0.85-1.15` brightness, `0.90-1.10` contrast) and train-only.

### 8.4 Longer-training risk
100 epochs can overfit/drift if monitoring is weak. Save every checkpoint and keep
per-epoch analysis so the best checkpoint can be selected post-run by marking IoU.

---

## 9. Decisions: settled vs open

Settled:

- **z-crop: do not crop** (Section 3).
- **sampler: do not change for H0** (Section 5).
- **RGB brightness sensitivity exists** and justifies H0_rgb_jitter (Section 6).
- H baseline to beat = **G1 epoch 27**.
- Checkpoint selection = max marking IoU; G1 class weights kept by default.

Still open (answer before full H0 training):

- exact implementation placement for train-only RGB jitter.
- smoke-test results proving validation/test are unchanged.
- whether H0 should use 35 epochs initially or a longer budget after smoke tests.

Do not create the final H0 config until these are answered by local analysis.

---

## 10. Proposed H Work Sequence

1. ~~Dataset z-crop audit~~ — **done; do not crop** (Section 3).
2. ~~Marking-aware sampling audit~~ — **done; keep uniform sampling** (Section 5).
3. ~~G1 RGB brightness sensitivity probe~~ — **done; H0_rgb_jitter justified**
   (Section 6).
4. Implement train-only RGB brightness/contrast jitter.
5. Small data-loader / batch-feature smoke test.
6. Tiny training smoke test.
7. Full H0 run (only if smoke tests are clean), judged against G1 epoch 27 on
   marking IoU / precision / recall / pred-true ratio / road->marking FPs /
   visual sanity.
8. H0 analysis package mirroring G1, with explicit RGB-jitter comparison.

---

## 11. Source Map

Z-crop audit (resolved):

- `logs/milestone_h/dataset_analysis/z_crop_audit/z_crop_recommendation.md`
- `logs/milestone_h/dataset_analysis/z_crop_audit/z_crop_error_effect.md`
- `logs/milestone_h/dataset_analysis/z_crop_audit/z_coordinate_frame.md`
- `logs/milestone_h/dataset_analysis/z_crop_audit/` (CSVs + plots + analysis_code)

G1 baseline + residuals:

- `logs/milestone_g/run_analysis/G1_schedule_extend/summary.csv`
- `logs/milestone_g/run_analysis/G1_schedule_extend/g1_conclusions.md`
- `logs/milestone_g/run_analysis/G1_schedule_extend/sampled_error_analysis_epoch27/`
- `logs/milestone_g/notes/milestone_g_context.md`,
  `logs/milestone_g/notes/milestone_g_deep_verdict.md`

H documents:

- `logs/milestone_h/README.md`
- `logs/milestone_h/notes/milestone_h_design_questions.md`
- `logs/milestone_h/notes/rgb_jitter_context.md`

Sampler audit:

- `logs/milestone_h/dataset_analysis/marking_aware_sampling_audit/sampling_recommendation.md`
- `logs/milestone_h/dataset_analysis/marking_aware_sampling_audit/` (CSVs + plots + analysis_code)

RGB brightness probe:

- `logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/brightness_sensitivity_recommendation.md`
- `logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/brightness_sensitivity_summary.csv`
- `logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/brightness_sensitivity_confusions.json`
- `logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/plots/`
- `logs/milestone_h/run_analysis/analysis_code/g1_rgb_brightness_sensitivity.py`
