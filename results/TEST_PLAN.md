# Test-Set Evaluation — End-to-End Plan & Decision Record

Held-out **test-set** evaluation of the thesis (PandaSet road-marking
segmentation, RandLA-Net, LiDAR ± front-camera RGB). This is the complete,
self-contained specification: the decision, the reasoning behind every choice,
the code, the run sequence, the outputs, the presentation, and the caveats.
Everything lives under `results/`; committed milestone code is not modified.

**Status:** protocol **RESOLVED** (Option B+, §1). A sampler bug found during the
smokes (§3) invalidated the earlier test numbers — they are discarded. Remaining
work is implementation + verification + the run (§9), not further deliberation.

---

## 1. The decision (Option B+) and why

### 1.1 The decision

```
SELECTION (already done, frozen):
  Epochs and models were selected during development on the SAMPLED validation
  metric. We do NOT re-select on full-coverage validation or on test.

FINAL REPORTING:
  Re-evaluate the FROZEN selected checkpoints with FULL SPATIAL COVERAGE on BOTH
  validation and the held-out test set. Headline = full-coverage TEST (micro/
  pooled marking IoU). Supplement = macro per-sequence.

ROBUSTNESS:
  - sampled-vs-full cross-check on the close pair (G2, H0) on test;
  - coverage-count distribution (per true class);
  - prediction-agreement rate for multiply-covered points (per true class);
  - (if clean) a per-point voted confusion matrix as direct no-voting confirmation.

SEEDS:
  D0/E0/F0: 1 full-coverage seed (val + test).
  G2/H0:    3 full-coverage TEST seeds (mean ± std on the close call); 1 full val.

NOT DOING:
  - per-point logit voting as the headline metric (disclosed limitation);
  - any threshold/bias sweep or tuning on test.
```

### 1.2 Why full coverage instead of sampling (the core choice)
- RandLA-Net consumes fixed 32,768-point **patches**, never a whole frame. Every
  metric is built by drawing patches → predicting → accumulating a 3×3 confusion
  matrix. **How patches are drawn on the test set is the methodological choice.**
- Measured fact (from the smoke): fully covering a frame takes **~8.2 patches**,
  so a complete pass over the 720 test frames is **~5,900 patches**. Our sampled
  protocol uses 2,160 → it evaluates only **~40 % of the test set's points per
  seed**. That is fine as a *development* signal but wrong for a *final, held-out*
  number.
- A held-out test result should answer *"how well does the model do on the test
  set?"*, not *"on a 40 % random subset of patches from it."*
- **Marking is the rare class the whole thesis rests on.** A partial sample is
  exactly where rare-class results are most exposed to sampling luck (which
  marking fragments were hit). Full coverage removes that variance and the
  "you sampled your test set" objection.
- Cost is one overnight run + a small code switch — the right place to spend
  compute for final thesis evidence.

### 1.3 Why selection stays on the *sampled* validation metric
Model/epoch selection happened during development on sampled validation, before
test was ever in play. Re-selecting on full-coverage val or on test would be
tuning on the final metric (selection bias). The clean, standard workflow is:
**select on the development metric, report on the rigorous final metric, never
re-select.**

### 1.4 Why validation is re-measured with full coverage
To make the val→test comparison a *single identical protocol* so the gap is pure
generalization, not protocol difference. This yields **two validation numbers**,
labelled explicitly and never mixed:
- **Selection-val** (sampled) — used for development/selection.
- **Full-val** (full coverage) — post-selection, only a protocol-matched
  reference for test.

If they differ, that is not a contradiction (different protocols, different
roles). If full-coverage **test** is lower than sampled validation, that is *more
honest*, not "worse": the development metric was mildly optimistic; full-coverage
test is the unbiased final estimate.

### 1.5 Why no per-point voting as the headline
The textbook gold standard is full coverage **with** per-point voting (collapse
each unique point to one prediction — ideally by averaging logits — and count it
once). Our engine instead accumulates predictions **per patch**, so an
overlapping point is counted multiple times. The two differ in two ways:
(a) over-covered points get extra weight; (b) we take hard argmax per patch
rather than averaging logits before argmax. We do **not** make voting the headline
because:
- under *full* coverage the sampler equalises per-point coverage, so (a) is small
  (far smaller than under random sampling);
- implementing voting as the headline metric is a risky late change to the number
  the whole thesis rests on;
- we instead **disclose it honestly** ("full spatial-coverage patch evaluation,
  not per-point logit voting") and **back it empirically** with the
  agreement-rate diagnostic (and, if clean, a voted confusion matrix showing the
  two agree). Voting-as-headline is named as a limitation / future refinement.

### 1.6 Why this is right for *this* thesis
It fixes a real bug caught before reporting invalid numbers, upgrades the *final*
evaluation to the most complete form affordable, and shores up the exact thing
the thesis stands on — the rare marking class — while staying honest about
selection and about what the metric is and isn't. Rigorous, not over-engineered.

### 1.7 Honest alternative (for the record)
**Option A** (random-sampled test, matched to sampled val) would also pass for a
bachelor thesis and is simpler / internally consistent with every existing
sampled figure. We choose **B+** because it is *more* defensible, the corrected
coverage numbers show A is a genuine ~40 % sample, and the rare-class core
warrants it.

---

## 2. How a metric is produced here (background)
- Input = a 32,768-point patch (KNN-gathered around a centre point). A forward-
  facing PandaSet frame has more points than that, so each frame needs several
  patches to be fully seen.
- We loop over patches, take `argmax` of model scores per point, and **add each
  patch's (true, pred) pairs into one running 3×3 confusion matrix** (rows = true,
  cols = pred, order `road(0) / marking(1) / other(2)`). All headline metrics are
  derived from that matrix.
- **Per-patch accumulation, not per-point voting** (see §1.5). Identical on val
  and test, so the comparison is fair regardless.
- **Test split = 720 frames** = 9 sequences × 80
  (001, 002, 015, 065, 090, 101, 102, 103, 117); all contain markings.
- **"Point"** throughout means a point of the model's **grid-subsampled
  evaluation cloud** (the pipeline subsamples before patching). Metrics are
  computed over that cloud; predictions are **not** projected back to the raw
  PandaSet cloud. This is identical on val and test, so the comparison is
  unaffected; "full coverage" therefore means every *evaluation-cloud* point is
  covered ≥ once.

---

## 3. The test-sampling bug (why earlier test numbers are discarded)
Open3D-ML provides two samplers and **forces** a different one for the test split
— the spatially-regular full-coverage sampler, which is appropriate in itself.
The bug was on our side: our engine **capped** it with `--steps` (below).

| split | sampler | behaviour |
| --- | --- | --- |
| train / validation | `SemSegRandomSampler` | `permutation(length)` → `length` **random** patches across all frames (length = `steps_per_epoch`, e.g. 2160) |
| **test** | `SemSegSpatiallyRegularSampler` (**hard-coded**, ignores config) | `gen_test` walks frames **strictly in order**, fully covering each before moving on; stops at `self.length` |

```python
# open3d/_ml3d/datasets/base_dataset.py — BaseDatasetSplit.__init__
if split in ['test']:
    sampler_cls = get_module('sampler', 'SemSegSpatiallyRegularSampler')   # forced
else:
    sampler_cls = get_module('sampler', self.cfg['sampler']['name'])       # our choice
```
```python
# SemSegSpatiallyRegularSampler.gen_test — walks frames 0..length IN ORDER
while curr_could_id < self.length:
    if self.min_possibilities[curr_could_id] > 0.5:   # this frame covered -> next
        curr_could_id += 1; continue
    self.cloud_id = curr_could_id; yield self.cloud_id
```

**Consequence:** our engine capped runs at `--steps`, which for this sampler caps
*how many frames in order it ever reaches*:
- `--steps 50` → covered only the **first ~50 frames** (sequence 001 + start of
  002), ~409 patches. The reported **G2 = 0.2456 / D0 = 0.149** were on **2
  sequences in alphabetical order, not the test set** → **invalid, discarded.**
- `--steps 2160` → walked toward frame 2160 while only 720 exist → indexed
  `min_possibilities[720]` → **`IndexError`**.

**Fixes (both needed):**
- *Sampled path* (commit `0f2ef05`): force `SemSegRandomSampler` on the test split
  so the sampled protocol matches validation exactly (used for the cross-check).
- *Full-coverage path* (to implement, §8): use `SemSegSpatiallyRegularSampler`
  **correctly** — `length = frame count (720)`, never capped — so it covers
  **all** frames.

---

## 4. Models under test
Run codes map to incremental descriptive names (no alphabet codes in figures).

| code | thesis name | run dir | selected epoch | feature_mode / in_ch | has_rgb |
| --- | --- | --- | ---: | --- | --- |
| D0 | LiDAR | `logs/milestone_d/runs/D0_weighted_ce_25ep` | 18 | intensity / 4 | no |
| E0 | LiDAR + RGB | `logs/milestone_e/runs/E0_rgb_front_v1` | 14 | intensity_rgb_front / 8 | yes |
| F0 | LiDAR + RGB (calibrated) | `logs/milestone_f/runs/F0_rgb_soft_weights` | 13 | intensity_rgb_front / 8 | yes |
| G2 | LiDAR + RGB + Lovász | `logs/milestone_g/runs/G2_schedule_extend_100` | 68 | intensity_rgb_front / 8 | yes |
| H0 | LiDAR + RGB + Lovász + Jitter | `logs/milestone_h/runs/H0_rgb_jitter` | 37 | intensity_rgb_front / 8 | yes |

- Config = each run's committed `config_snapshot.yml`; checkpoint =
  `checkpoints/ckpt_epoch_<NNNNN>.pth` at the epoch above.
- Selected epochs were chosen on **validation** (max marking IoU); not re-selected
  on test.
- **Selection-val reference** (sampled best-epoch marking IoU, for read-back
  sanity): D0 `0.440294`, E0 `0.438439`, F0 `0.482770`, G2 `0.550444`,
  H0 `0.550671`.

### Critical attribution caveat
**D0 → G2/H0 is NOT a clean RGB on/off ablation.** D0 differs from G2 in four
ways: +RGB, +Lovász, +`dim_features` (8→16), +epochs (25→100). Therefore:
- the **clean "RGB adds X"** claim uses **D0 → E0** (RGB added, loss/epochs ~fixed);
- **D0 → G2/H0** is a **system-level** comparison (LiDAR baseline vs full RGB
  system). This is why all five models are tested.

---

## 5. Dataset / splits / labels
- Frozen sequence split (Milestone B): **58 train / 9 val / 9 test**.
- **Test = 001, 002, 015, 065, 090, 101, 102, 103, 117** — all lane-bearing;
  **065 is night** (`configs/splits/test.txt`, `logs/milestone_b_split_report.txt`).
- Label mode `road_marking3`: active classes after ignore-filtering are
  `0=road, 1=marking, 2=other`; **marking = raw 8 (lane line) + 9 (stop line) +
  10 (other road marking)**; ignored raw `{1,2,3,4}`.
- **Naming alias:** in all CSV/JSON, `lane_*` ≡ `marking_*`.
- **Do not compare to Milestone C** (strict lane-line labels) without stating the
  label-definition change.

---

## 6. Evaluation methodology (every decision)
1. **Full spatial coverage** for the final numbers (§1.2): spatially-regular
   sampler, `length = 720`, uncapped — every point of the (grid-subsampled)
   evaluation cloud is covered ≥ once, for test and full-val alike (see §2 on
   what "point" means here).
2. **Averaging.** Headline = **micro / pooled** marking IoU (one confusion matrix
   over all points; matches val + segmentation convention). Supplement =
   **macro / per-sequence** marking IoU (rare-class consistency across the 9
   scenes). If micro and macro diverge, report both and frame them as different
   questions ("overall point-level segmentation" vs "cross-scene consistency") —
   never pick the better-looking one.
3. **Seeds.** Full coverage is near-deterministic (the sampler's `rand()*1e-3`
   possibility init is a tiny tie-breaker), so **1 full-coverage seed** for
   D0/E0/F0 and for all full-val. The headline call is the close **G2-vs-H0**, so
   **3 full-coverage TEST seeds** for G2/H0 → mean ± std. This is *evaluation*
   variance only; the dominant uncertainty remains the **single training seed**.
4. **Selection vs reporting** (§1.3): selected on sampled val; test is evaluated
   **only after model selection is frozen**, and **no checkpoint/model choice is
   made from test results** (the G2/H0 multi-seed runs are reporting, not
   selection). G2 and H0 reported **both** as a precision/recall trade-off.
5. **Sampled-vs-full cross-check** (G2/H0, test): run the *sampled* protocol
   (3 seeds, mean ± std) and show its band agrees with the full-coverage number —
   demonstrates the development metric was representative and bounds the
   sampling-vs-coverage effect.
6. **No-voting defense (diagnostics on the full-coverage runs):**
   - **coverage-count distribution** per true class (road/marking/other): median,
     p10, p90, max number of patches covering a point. Similar marking vs
     road/other counts ⇒ double-counting is near-uniform ⇒ small bias.
   - **prediction-agreement rate** = fraction of multiply-covered points whose
     predicted class is identical across all covering patches, overall and **by
     true class**. High *marking* agreement ⇒ voting would barely change the
     marking result.
   - **supplementary, optional:** if a safe per-point mapping is already
     available, a **voted confusion matrix** (each unique point counted once,
     prediction = vote/logit consensus) may be reported beside the
     patch-accumulated one as direct confirmation they agree. Otherwise the
     coverage-count and agreement-rate diagnostics stand on their own and voting
     remains future work. It is **never a second headline metric** and never a
     blocker.
7. **No bias/threshold sweep on test** — operating-point analysis stays on
   validation; the G2-vs-H0 contrast already gives two real operating points.

### Metric definitions (re-derived from the 3×3 confusion matrix)
`cm` rows = true, cols = pred, order road(0)/marking(1)/other(2):
- marking IoU = `cm11 / (cm11 + (cm01+cm21) + (cm10+cm12))`
- precision = `cm11 / (cm11+cm01+cm21)`; recall = `cm11 / (cm11+cm10+cm12)`;
  F1 = harmonic mean
- per-class IoU analogously; **mIoU** = mean of the three class IoUs
- **pred/true marking ratio** = `cm[:,1].sum() / cm[1,:].sum()` (1.0 = calibrated;
  >1 = over-prediction)
- road→marking false positives = `cm[0,1]`; marking→road misses = `cm[1,0]`

---

## 7. Metric re-derivation is exact (verified)
`build_comparison.metrics_from_cm` re-derives all metrics from
`confusion_matrix.npy`. On G2's known sampled validation matrix it reproduced
marking IoU `0.540046`, precision `0.626677`, recall `0.796193`, pred/true
`1.270500`, road→marking `532367` — so every downstream number is recomputable
from the saved matrices alone (no GPU/pandaset needed for aggregation).

---

## 8. Code (all under `results/test_suite/`)
Isolated, self-contained copy of the polished reproducible suite (thesis run
names, log-scale LR plot, no-overlap legends, opaque overlays). Committed
milestone suites are not modified.

### 8.1 Already done
- **D0 generalisation** of `_sampled_error_engine.py` and `04_prediction_images.py`
  (accept `intensity` 4-ch as well as `intensity_rgb_front` 8-ch; `has_rgb` gates
  the RGB-only outputs; `summary.json` gains `feature_mode`/`has_rgb`). RGB path
  byte-identical for E0/F0/G2/H0.
- Engine is invoked **directly** (not via the G1-specific `02` wrapper, whose
  `enrich_rgb_valid_metrics` needs the RGB strata CSV and crashes on D0). It is
  standalone (own repo-root finder, `__main__` guard, explicit
  `--config/--checkpoint/--split/--steps/--seed/--device/--out-dir`).
- **Sampler fix** (`0f2ef05`): force `SemSegRandomSampler` on the test split so the
  sampled protocol matches validation.

### 8.2 To implement for Option B+
- **`--coverage {sampled,full}` switch** in `_sampled_error_engine.py`:
  - `sampled` (current): force `SemSegRandomSampler`, `length = --steps` (2160).
  - `full`: force `SemSegSpatiallyRegularSampler`, `length = len(split)` (frame
    count), do **not** cap with `steps_per_epoch`; works for any split (so
    full-val uses it too). Verify it completes each frame before advancing.
- **Diagnostics** accumulated during a `full` run (per frame, keyed by subsampled
  point index): coverage count per point, per-class agreement, and (optional)
  voted CM. Emit `coverage_count_by_class.csv`, `agreement_rate.csv`, and (if
  done) `confusion_matrix_voted.npy` + a `voted_marking_iou` field in
  `summary.json`.
- **Drivers:** `run_test_all.py` gains `--coverage` and a `--val` mode so it can
  produce full-test (all 5), full-val (all 5), and sampled-test (G2/H0) with the
  per-model seed counts of §6.3. `build_comparison.py` gains the two-validation-
  column layout (§10) and confusion-matrix plots; `run_qualitative.py` unchanged
  in logic (uses `--passes-per-frame` for near-full-frame overlays).

---

## 9. Run sequence (server) and compute
```bash
cd ~/project

# 0) VERIFY the new full-coverage path on G2 first (the one real remaining risk):
#    must iterate ~5-6k patches, cover all 720 frames, emit diagnostics, not crash.
python results/test_suite/_sampled_error_engine.py \
  --config logs/milestone_g/runs/G2_schedule_extend_100/config_snapshot.yml \
  --checkpoint logs/milestone_g/runs/G2_schedule_extend_100/checkpoints/ckpt_epoch_00068.pth \
  --split test --coverage full --seed 42 --device cuda --out-dir results/_smoke_g2_full
#    check: ~720 frames touched, active_points ~ full set, coverage/agreement CSVs present.

# 1) full run (detached, overnight) — full-coverage test + val + sampled cross-check
python results/test_suite/run_test_all.py --device cuda   # honours §6.3 seed plan

# 2) assemble package (CPU; can run locally)
python results/test_suite/build_comparison.py

# 3) qualitative overlays (slow; edit HAND_PICKS/065 frames first)
python results/test_suite/run_qualitative.py --device cuda
```

**Compute (≈70 min/full run, ≈27 min/sampled run):**

| set | runs | ≈ time |
| --- | ---: | ---: |
| full-coverage **test** — D0/E0/F0 ×1 + G2/H0 ×3 | 9 | ~10.5 h |
| full-coverage **val** — 5 ×1 | 5 | ~5.8 h |
| sampled **test** cross-check — G2/H0 ×3 | 6 | ~2.7 h |
| **total (recommended)** | **20** | **~19 h** |

**Leaner fallback** (if compute is tight): 1 full-coverage seed everywhere
(10 full runs ≈ 12 h) + sampled G2/H0 cross-check (6 runs ≈ 2.7 h) ≈ **15 h**;
the close-pair stability then rests on the sampled cross-check band rather than
3 full-coverage seeds.

---

## 10. Outputs (`results/`) and presentation
```
results/
  TEST_PLAN.md
  test_master_table.csv          # headline; columns below
  README.md                      # generated provenance + master table
  per_model/<folder>/<split>/<coverage>/seed_<S>/
      summary.json               # has_rgb, feature_mode, coverage, sampled/full metrics, voted_marking_iou?
      confusion_matrix.npy       # 3x3 (source of truth)
      confusion_matrix_voted.npy # optional (no-voting confirmation)
      per_sequence_metrics.csv   # macro / per-scene
      distance_bucket_metrics.csv
      coverage_count_by_class.csv   # full runs: median/p10/p90/max by true class
      agreement_rate.csv            # full runs: agreement overall + by true class
      frame_error_summary.csv       # drives qualitative auto-pick
      raw_subtype_rgb_stratified_metrics.csv / group_feature_summary.csv
      rgb_valid_stratified_metrics.csv   # RGB models only (D0 omits)
  comparisons/
      rgb_effect__D0_vs_E0/        # clean "add RGB" effect
      system__D0_vs_G2_vs_H0/      # LiDAR baseline vs full systems
      shortcut__G2_vs_H0/          # rgb_valid over-prediction gap + brightness fingerprint
      plots/                       # progression, val-vs-test, per-class IoU, distance, shortcut, confusion
  qualitative/<category>/<seq>_f<frame>/<folder>/<mode>/   # gt/pred/error PNGs (opaque)
```

**Master table (two clearly-named validation columns — never mixed):**

```
Model | Selected epoch | Selection-val IoU | Full-val IoU | Full-test IoU (±std for G2/H0) |
      Test precision | Test recall | Test F1 | Test pred/true | Macro per-seq IoU
```
- **Selection-val** = sampled validation used for checkpoint selection.
- **Full-val / Full-test** = post-selection full-coverage metrics, for reporting
  and the generalization gap only.

**Caption template:** "*Selection validation* denotes the sampled validation
metric used for checkpoint selection during development. *Full-coverage
validation* and *full-coverage test* denote post-selection evaluations under one
identical spatially-regular coverage protocol; they are used for final reporting
and the generalization gap, not for selection."

**Thesis methods paragraph (drop-in):** "During development, checkpoints were
selected using the sampled validation protocol used consistently in training,
where metrics were computed from randomly sampled 32,768-point patches. After the
selected checkpoints were fixed, final evaluation was performed with a spatially
regular full-coverage patch protocol on both validation and the held-out test
split. The full-coverage validation results are reported only as a
protocol-matched reference for the test results and were not used for checkpoint
or model selection. Metrics are computed by accumulating a confusion matrix over
evaluated patch points; this is full spatial-coverage patch evaluation, not
per-point logit voting. To assess that choice, the evaluation records per-class
point-coverage counts and a prediction-agreement rate across overlapping patches;
per-point voting is left as a limitation and possible future refinement."

If full-coverage test < sampled validation, frame it as honesty: "the sampled
development metric was optimistic; the full-coverage test score is the final
unbiased held-out estimate."

---

## 11. What the comparisons produce
- **rgb_effect__D0_vs_E0** — the only clean RGB on/off step ("RGB adds X").
- **system__D0_vs_G2_vs_H0** — LiDAR baseline vs full systems (system-level).
- **shortcut__G2_vs_H0** — `rgb_valid_gap.csv` (rgb_valid vs rgb_invalid precision
  + pred/true, over-prediction gap) and `brightness_fingerprint.csv`
  (road→marking FP RGB/intensity profile vs road-TP/marking-TP): the brightness
  shortcut before/after jitter.

Expected (from validation, to re-check on full-coverage test): jitter (H0)
reduces over-prediction and road→marking FPs and raises precision **at a recall
cost**, leaving marking IoU tied with G2 within noise — a precision/calibration
result, not an IoU win; shortcut reduced, not eliminated.

---

## 12. Caveats / limitations (state in the thesis)
1. **Single training seed (42)**; ~**0.008** marking-IoU noise floor → differences
   within it are ties (why G2 ≈ H0). The full-coverage eval seed fixes
   *evaluation* variance, **not** training variance.
2. **Patch-accumulated confusion matrix, not per-point logit voting** — defended
   by the coverage-count and agreement-rate diagnostics (§6.6); voting is future
   work.
3. **Select-on-sampled-val, report-on-full-coverage-test** (clean workflow §1.3),
   with two named validation columns.
4. **D0 → G2/H0 is not a clean RGB ablation** (§4); the clean RGB effect is D0→E0.
5. **H0 vs G2 also differ in training regime**: H0 = 100 epochs from scratch;
   G2 = resume chain (G0→G1→G2). Same data/seed/schedule, noted.
6. **RGB-shortcut framing is correlational** (group-level fingerprint; no causal
   channel ablation). Train-with-jitter is stronger than the earlier
   frozen-checkpoint probe, but not causal proof.
7. **Night sequence 065 is a single case study** — see §13.

---

## 13. Night sequence (065) framing
Do **not** claim broadly "RGB helps by day but hurts at night" from one sequence.
Safe wording: "Sequence 065 is a night-time stress case. If the RGB models lose
some advantage over the LiDAR-only baseline here, it suggests low-light camera
conditions as a potential failure mode. As only one night sequence is present, we
treat this as a case-specific observation, not a general conclusion." Render
GT/pred/error overlays for D0/G2/H0 on the same 065 frames for qualitative
support.

---

## 14. Verification status
**Done:** scripts syntax-clean; engine standalone-importable; D0 generalisation
structurally verified; `build_comparison` ran end-to-end on synthetic per-model
data (master table + comparisons + plots) PASS; metric re-derivation exact (§7);
`val_metrics` reads real `eval_history` numbers; sampler root-cause traced in
Open3D source and the sampled fix committed (`0f2ef05`).

**Remaining (the only real risk):** runtime-verify the **`--coverage full`** path
end-to-end on G2 (§9 step 0) — all 720 frames covered, no boundary crash,
diagnostics emitted — before the ~19 h run. Then implement the diagnostics +
two-column table, run, assemble.

---

## 15. Source map
| Thing | Path |
| --- | --- |
| This plan | `results/TEST_PLAN.md` |
| Engine (+ `--coverage`, diagnostics) | `results/test_suite/_sampled_error_engine.py` |
| Overlay viewer | `results/test_suite/04_prediction_images.py` |
| Test driver | `results/test_suite/run_test_all.py` |
| Comparison/aggregation | `results/test_suite/build_comparison.py` |
| Qualitative driver | `results/test_suite/run_qualitative.py` |
| Plot style + thesis names | `results/test_suite/_suite_style.py` (`DISPLAY_NAMES`) |
| Test split | `configs/splits/test.txt`; `logs/milestone_b_split_report.txt` |
| Milestone context docs | `docs/milestone_{a..h}/…`; `docs/thesis_context/MILESTONE_CONTEXT_INDEX.md` |
| Per-model configs/checkpoints | `logs/milestone_{d,e,f,g,h}/runs/<run>/` (§4) |
