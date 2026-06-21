# Test-Set Evaluation — End-to-End Plan (for review)

This document specifies the held-out **test-set** evaluation phase of the thesis
project (PandaSet road-marking segmentation with RandLA-Net, LiDAR ± front-camera
RGB). It is a complete, self-contained handoff: goals, decisions, code, run
sequence, outputs, what was verified, and caveats. Everything described lives
under `results/` and was built without modifying any committed milestone code.

Status: harness built + structurally verified; D0 + G2 runtime smokes ran. **A
sampling-protocol problem was found during the smokes (see §0). No valid test
numbers exist yet — the earlier smoke numbers were measured on a biased subset
and must be discarded.** The open decision in §0 must be resolved before the
full run.

---

## 0. OPEN METHODOLOGICAL DECISION — how to sample the test set (READ FIRST)

> This section is a self-contained brief for an external reviewer. It explains
> how metrics are computed in this project, the sampler bug we hit, why the
> choice it forces is genuinely non-obvious, and the options. **The question we
> need help deciding is in §0.7.**

### 0.1 One-paragraph summary
RandLA-Net cannot ingest a whole LiDAR frame at once; it consumes fixed-size
**32,768-point patches**. So *every* metric in this project is computed by
drawing patches, running the model, and accumulating a 3×3 confusion matrix
(true × predicted over `road / marking / other`), then deriving IoU /
precision / recall from it. **How you draw those patches on the test set is the
open question.** Validation (the baseline we compare against) was measured with
**random** patch sampling. The deep-learning library (Open3D-ML) silently forces
a **different**, order-based sampler on the test split, and that difference (a)
made our first test numbers meaningless and (b) is now a real methodological
choice for the final thesis number.

### 0.2 How a "metric" is produced here (important background)
- The model input is a patch of 32,768 points centred on some point, gathered by
  nearest-neighbour search. A single PandaSet forward-facing frame has more
  points than that, so each frame needs **several patches** to be fully seen.
- For evaluation we loop over patches, take `argmax` of the model scores per
  point, and **add each patch's (true, pred) pairs into one running 3×3 confusion
  matrix**. All headline metrics are derived from that matrix.
- **Crucial nuance:** this engine does **per-patch accumulation, not per-point
  voting.** If a point is covered by several overlapping patches, it contributes
  several times. (Textbook RandLA-Net test inference averages logits per point
  and emits one prediction per point. We do not — and neither did our validation.
  Whatever we choose, val and test use the *same* per-patch accumulation, so the
  comparison stays internally fair.)
- **Test split = 720 frames** (9 sequences × 80 frames: 001, 002, 015, 065, 090,
  101, 102, 103, 117). All 9 contain markings.

### 0.3 The two samplers (verbatim Open3D-ML behaviour)
**(a) `SemSegRandomSampler`** — used for **training and validation**:
```python
def gen():
    ids = np.random.permutation(self.length)   # self.length = steps_per_epoch (= 2160)
    for i in ids:
        yield i                                 # frame index; dataloader does index % num_frames
```
→ draws `length` **random** patches spread across **all** frames (frames revisited
via wraparound). This is how the validation marking-IoU numbers (G2 = 0.55, etc.)
were produced.

**(b) `SemSegSpatiallyRegularSampler`** — Open3D **forces** this for the test split:
```python
def gen_test():
    curr_could_id = 0
    while curr_could_id < self.length:                  # self.length controls how far it walks
        if self.min_possibilities[curr_could_id] > 0.5: # this frame is "covered" -> next frame
            curr_could_id += 1
            continue
        self.cloud_id = curr_could_id
        yield self.cloud_id                             # keep patching THIS frame until covered
```
→ walks frames **strictly in order** (0, 1, 2, …), fully covering each before
moving on. It is designed for **complete test-time coverage** (predict every
point). It is forced by a hard-coded rule in the library — it **ignores our
config**:
```python
if split in ['test']:
    sampler_cls = get_module('sampler', 'SemSegSpatiallyRegularSampler')   # forced
else:
    sampler_cls = get_module('sampler', self.cfg['sampler']['name'])       # our choice (random)
```

### 0.4 The bug we hit, and why the smoke numbers are invalid
Our engine caps a run at `--steps` patches. For the **random** sampler that
simply sets the patch count. For the **spatially-regular** sampler, `--steps`
caps `self.length` — i.e. **how many frames (in order) it ever reaches**:
- Smoke with `--steps 50` → it covered only the **first ~50 frames in order**
  (sequence 001 + the start of 002), taking ~409 patches. The reported
  **G2 = 0.2456** and **D0 = 0.149** marking IoU were therefore measured on **2
  sequences in alphabetical order, not the 9-sequence test set.** That is why
  they looked alarmingly low. **These numbers are an artifact and are discarded.**
- Re-running with `--steps 2160` made it walk toward frame 2160 while only 720
  frames exist → it indexed `min_possibilities[720]` → **`IndexError`** (crash).

So: we currently have **no valid test number**, and val↔test were never measured
the same way.

### 0.5 Why this is a genuine choice, not an obvious fix
- **The validation baseline is itself random-sampled.** The numbers we compare
  test against (val marking IoU per model) come from training-time validation,
  which used `SemSegRandomSampler`. "Match how val was measured" ⇒ random.
- **Dense random sampling ≈ full coverage in aggregate.** 2160 patches ×
  32,768 pts ≈ **70.8M point-evaluations** spread over 720 frames (~98k per
  frame) — every frame is heavily oversampled, so the aggregate IoU is expected
  to be ~equal to full coverage. (In the smoke the spatially-regular sampler
  needed ~8 patches to "cover" a frame, so a true full-coverage pass over 720
  frames is ≈ **5–6k patches**.)
- **Marking is a rare class.** Coverage uniformity matters more than usual: a
  reviewer could argue random sampling under/over-represents sparse marking
  points, whereas full coverage guarantees every marking point is evaluated.
- **Neither method is per-point voting** (see §0.2), so "full coverage" here is
  "every point covered ≥ once with overlap-weighting," not the textbook
  single-vote-per-point. This narrows the practical gap between the two options.

### 0.6 The options
**Option A — Random-sampled test (currently implemented).**
Force `SemSegRandomSampler` on the test split too; 2160 random patches, identical
to validation. *Claim:* "test measured with the identical protocol used for
validation." *Pros:* directly comparable to the existing val numbers; supports
the 3-eval-seed variance band; simple; ~equal to full coverage numerically.
*Cons:* it is a (dense) *sample* of the test set, not every point — a small
"why sample your final test set?" vulnerability, sharpest for the rare class.
*Cost:* ~27 min/run. *Code:* done (commit `0f2ef05`).

**Option B — Full-coverage test + matched full-coverage val (recommended).**
Use the spatially-regular sampler **correctly** (set its length to the frame
count = 720 so it covers *all* frames, never capped at `--steps`), AND re-run
**validation** the same way for the selected checkpoints, so both sides of the
val↔test comparison are full coverage. *Claim:* "every point of val and test was
evaluated, under one identical protocol." *Pros:* canonical test-set evaluation;
removes the sampling objection; strongest for the rare class. *Cons:* the val
numbers in existing figures/docs are the sampled training-time ones, so we'd
carry **two val numbers** (sampled = selection metric; full-coverage = comparison
metric) and must explain both; more compute. *Cost:* ~40–60 min/run × (5 models ×
val+test) ≈ overnight. *Code:* add a `--coverage {sampled,full}` switch.

**Option C — Both (A as primary, one full-coverage G2 cross-check).**
Report random-sampled as primary (matches val) but run full coverage on G2 once
to show the two agree within the seed band — a robustness sentence
("sampled and full-coverage agree to within 0.0X"). *Cost:* A + ~40 min.

**Independent sub-decisions (orthogonal to A/B/C):**
- *Selection integrity (settled):* epoch/model selection was done on **sampled
  training-time validation** — standard, with **no test leakage**. Any
  full-coverage val in Option B is a *post-hoc, comparison-only* re-measurement.
- *Eval seeds:* random sampling is stochastic → 3 seeds give a variance band
  (planned for G2/H0). Full coverage is near-deterministic → 1 seed suffices.

### 0.7 The question we want help deciding
> For a **bachelor thesis** reporting held-out test-set performance of a
> rare-class (road-marking) segmentation model, where (i) all development/
> validation metrics were produced by **dense random patch sampling**, (ii) the
> evaluation accumulates a **per-patch confusion matrix (no per-point voting)**,
> and (iii) dense random sampling numerically approximates full coverage — is it
> **more correct and defensible** to report the test number with the **same
> random-sampled protocol as validation (Option A)**, or to switch the final
> test (and a matched re-run of validation) to **full spatial coverage
> (Option B)**? What would a methods examiner most likely expect, and what is the
> cleanest way to frame "we selected on sampled validation but report on
> full-coverage test" without it looking inconsistent?

Our current lean: **Option B** (full coverage is the more defensible *final*
number, especially for a rare class), with the selection-integrity framing in
§0.6. The counter-argument for **A** is internal consistency with every existing
sampled-validation figure and the simplicity of one protocol everywhere.

---

## 1. Goal and narrative

All reported metrics so far are **validation** numbers — the split used to make
every development decision (epoch selection, run-to-run comparison, loss/weight
choices). They are therefore mildly optimistic. The test split (frozen in
Milestone B, never touched during development) gives the **unbiased** estimate of
generalization. This phase evaluates the already-selected checkpoints on test,
**once**, and assembles the thesis results package.

Thesis story the test must support:
1. **Without RGB → with RGB**: LiDAR-only baseline (D0) vs the RGB systems.
2. **Brightness shortcut**: G2 vs H0 — does train-time RGB jitter reduce the
   bright-road→marking false-positive shortcut.

---

## 2. Models under test

Label scheme used in figures (no alphabet codes in the thesis): the run codes map
to descriptive names that build up incrementally.

| code | thesis name | run dir | selected epoch | feature_mode / in_ch | has_rgb |
| --- | --- | --- | ---: | --- | --- |
| D0 | LiDAR | `logs/milestone_d/runs/D0_weighted_ce_25ep` | 18 | intensity / 4 | no |
| E0 | LiDAR + RGB | `logs/milestone_e/runs/E0_rgb_front_v1` | 14 | intensity_rgb_front / 8 | yes |
| F0 | LiDAR + RGB (calibrated) | `logs/milestone_f/runs/F0_rgb_soft_weights` | 13 | intensity_rgb_front / 8 | yes |
| G2 | LiDAR + RGB + Lovász | `logs/milestone_g/runs/G2_schedule_extend_100` | 68 | intensity_rgb_front / 8 | yes |
| H0 | LiDAR + RGB + Lovász + Jitter | `logs/milestone_h/runs/H0_rgb_jitter` | 37 | intensity_rgb_front / 8 | yes |

- Each model's config is its committed `config_snapshot.yml`; checkpoint is
  `checkpoints/ckpt_epoch_<NNNNN>.pth` at the epoch above.
- Selected epochs were chosen on **validation** (max marking IoU). They are NOT
  re-selected on test.
- Validation reference (best-epoch marking IoU, for sanity-checking the test
  read-back): D0 `0.440294`, E0 `0.438439`, F0 `0.482770`, G2 `0.550444`,
  H0 `0.550671`.

### Critical attribution caveat
**D0 → G2/H0 is NOT a clean RGB on/off ablation.** D0 differs from G2 in four
ways: +RGB, +Lovász loss, +`dim_features` (8→16), +epochs (25→100). So:
- the **clean "RGB adds X"** claim must use **D0 → E0** (RGB added, loss/epochs ~fixed);
- **D0 → G2/H0** is a **system-level** comparison (LiDAR baseline vs full RGB system).
This is why all five models are tested, not only D0/G2/H0.

---

## 3. Dataset / splits / labels

- Frozen sequence-level split (Milestone B): **58 train / 9 val / 9 test**.
- **Test sequences: 001, 002, 015, 065, 090, 101, 102, 103, 117** — all 9 are
  lane-bearing (every test sequence has marking content). Source:
  `logs/milestone_b_split_report.txt`, `configs/splits/test.txt`.
- Label mode `road_marking3`: active classes after ignore-filtering are
  `0=road, 1=marking, 2=other`; **marking = raw 8 (lane line) + raw 9 (stop line)
  + raw 10 (other road marking)**. Ignored: raw {1,2,3,4}.
- **Metric naming alias**: in all CSV/JSON the columns `lane_*` mean `marking_*`.
- **Do not compare to Milestone C** (strict lane-line labels) without stating the
  label-definition change.

---

## 4. Evaluation methodology (decisions)

1. **Sampled, not exhaustive.** Every metric in the project (train/val) was
   computed on random 32,768-point patches via `SemSegRandomSampler`, not
   exhaustive per-point inference. The test pass uses the **same** sampled
   protocol so val and test are measured identically. Disclose: "metrics are
   sampled-patch, not exhaustive." **(Partly superseded — see §0: Open3D forces a
   different sampler on the test split; whether the final test number should be
   random-sampled (Option A) or full-coverage (Option B) is the open decision.)**
2. **Steps = 2160** patches per pass (matches the validation diagnostics).
3. **Eval seeds:** D0/E0/F0 → 1 seed (42). **G2/H0 → 3 seeds (42, 1, 2)**,
   reported as mean ± std. This captures **evaluation-sampling** variance (which
   patches are drawn), **not training-seed** variance (which would require
   retraining). It puts error bars on the close G2-vs-H0 call.
4. **Selection vs reporting:** checkpoints/models were selected on validation;
   test is evaluated **once** and **not tuned on**. No model is chosen by its
   test score (that would be selection bias). G2 and H0 are **both** reported as
   a precision/recall trade-off; if one headline is needed, pick on validation
   reasoning (G2 = best IoU/recall) before seeing test.
5. **Single-seed (training) limitation** stands: each model was trained once
   (seed 42). ~**0.008 marking-IoU noise floor** — treat smaller differences as
   ties (this is exactly why G2≈H0).
6. **No bias sweep on test** (it is a what-if operating-point diagnostic and
   risks looking like test-tuning; the G2-vs-H0 contrast already gives two real
   operating points).

### Metric definitions (re-derived from the 3×3 confusion matrix)
`cm` rows = true, cols = pred, order road(0)/marking(1)/other(2):
- marking IoU = `cm11 / (cm11 + (cm01+cm21) + (cm10+cm12))`
- marking precision = `cm11 / (cm11+cm01+cm21)`; recall = `cm11 / (cm11+cm10+cm12)`;
  F1 = harmonic mean
- per-class IoU for road/other analogously; **mIoU** = mean of the three class IoUs
- **pred/true marking ratio** = `cm[:,1].sum() / cm[1,:].sum()` (1.0 = calibrated; >1 = over-prediction)
- road→marking false positives = `cm[0,1]`; marking→road misses = `cm[1,0]`

---

## 5. Code (all under `results/test_suite/`)

The suite is an **isolated, self-contained copy** of the polished reproducible
testing suite (the one used for the G2/H0 validation analysis: thesis run names,
log-scale LR plot, no-overlap legends, opaque overlays). Committed milestone
suites were **not** modified.

### 5.1 Generalised for the LiDAR baseline (D0)
The original engine/viewer were RGB-only (hard-rejected `feature_mode != intensity_rgb_front`
and assumed 8 channels). Two files were generalised to also run D0 (4-channel,
intensity-only); the RGB path is byte-identical for E0/F0/G2/H0.
- `results/test_suite/_sampled_error_engine.py` — metrics/diagnostics engine.
  - config guard now accepts `intensity` **and** `intensity_rgb_front`;
  - `_load_sample` builds 8-ch (RGB) or 1-feature (D0) accordingly;
  - transform extracts RGB only when present (dummy zeros otherwise);
  - **output gating** via `has_rgb`: for D0 it writes core metrics + confusion +
    distance + raw-subtype("all"), and **skips** the RGB-only outputs
    (`rgb_valid_stratified_metrics.csv`, brightness fingerprint, RGB subtype strata);
  - `summary.json` gains `feature_mode` and `has_rgb`.
- `results/test_suite/04_prediction_images.py` — front-camera overlays, same D0
  generalisation (overlays color by predicted/true/error class; `rgb_valid` is
  metadata only, zeroed for D0). Default `--alpha 1.0` (opaque).

### 5.2 Why the engine is called directly (not via `02`)
`results/test_suite/02_sampled_error_analysis.py` is a G1-specific **wrapper**
whose post-processing (`enrich_rgb_valid_metrics`) requires the RGB strata CSV and
**would crash on D0**. The test driver therefore calls
`_sampled_error_engine.py` **directly** (it is standalone: own repo-root finder,
all globals defined, `__main__` guard, accepts explicit
`--config/--checkpoint/--split/--steps/--seed/--device/--out-dir`).

### 5.3 New drivers (built for this phase)
- `results/test_suite/run_test_all.py` — runs all 5 models on `--split test` into
  `results/per_model/<folder>/test/seed_<S>/`. 3 eval-seeds for G2/H0, 1 for
  others. Pre-flight verifies every config + checkpoint exists. `--only D0,G2`
  to subset; `--steps`, `--device` flags. Runs on the **server** (GPU + checkpoints).
- `results/test_suite/build_comparison.py` — **CPU/file-only, no GPU/pandaset
  import** (runs anywhere). Re-derives every metric from each seed's
  `confusion_matrix.npy`, aggregates mean ± std, reads validation numbers from
  each run's `eval_history.csv` at the selected epoch, and writes the master
  table, comparison folders, plots (thesis-styled via `_suite_style.py`), and
  `results/README.md`.
- `results/test_suite/run_qualitative.py` — auto-picks interesting test frames
  from the reference model's `frame_error_summary.csv` (best / worst / over-
  prediction) plus `HAND_PICKS` (currently `065` = night), and renders GT +
  prediction + error overlays (opaque, `--passes-per-frame 8` for near-full
  coverage) for D0/G2/H0 on the **same** frames via `04`. Knobs at top of file
  (`RENDER_MODELS`, `MODES`, `K_PER_CATEGORY`, `MIN_TRUE_MARKING`, `HAND_PICKS`).

---

## 6. Run sequence (server)

```bash
cd ~/project

# 0) one-time D0 smoke (~2-4 min) — proves the 4-channel generalisation runs
python results/test_suite/_sampled_error_engine.py \
  --config logs/milestone_d/runs/D0_weighted_ce_25ep/config_snapshot.yml \
  --checkpoint logs/milestone_d/runs/D0_weighted_ce_25ep/checkpoints/ckpt_epoch_00018.pth \
  --split test --steps 50 --seed 42 --device cuda --out-dir results/_smoke_d0
# expect: summary.json with split=test, has_rgb=false, feature_mode=intensity,
# a sampled_metrics.marking_iou; NO rgb_valid_stratified_metrics.csv. Then: rm -rf results/_smoke_d0

# 1) full test pass, all 5 models (~5-6 h; run detached with nohup)
python results/test_suite/run_test_all.py --device cuda

# 2) assemble the results package (seconds; can also run locally)
python results/test_suite/build_comparison.py

# 3) qualitative overlays (slow; edit HAND_PICKS/K first if desired)
python results/test_suite/run_qualitative.py --device cuda
```

---

## 7. Output structure (`results/`)

```
results/
  TEST_PLAN.md                         # this document
  test_master_table.csv                # headline: all models, val + test, mean±std
  README.md                            # generated: provenance + master table
  per_model/<model_folder>/test/seed_<S>/
      summary.json                     # has_rgb, feature_mode, sampled_metrics, confusion
      confusion_matrix.npy             # 3x3 (source of truth for all re-derived metrics)
      distance_bucket_metrics.csv      # marking metrics by range bucket
      per_sequence_metrics.csv         # per test sequence
      raw_subtype_rgb_stratified_metrics.csv   # recall by raw 8/9/10 (strata: all [+rgb for RGB models])
      group_feature_summary.csv        # per-outcome intensity (+RGB for RGB models)
      frame_error_summary.csv          # per-frame metrics (drives qualitative auto-pick)
      top_frames_by_road_to_marking.csv / top_frames_by_marking_to_road.csv
      rgb_valid_stratified_metrics.csv # RGB models ONLY (D0 omits)
  comparisons/
      rgb_effect__D0_vs_E0/            # clean "add RGB" effect (test_metrics.csv + deltas)
      system__D0_vs_G2_vs_H0/          # LiDAR baseline vs full RGB systems
      shortcut__G2_vs_H0/              # rgb_valid over-prediction gap + brightness fingerprint
      plots/                           # progression, val-vs-test, marking metrics,
                                       #   per-class IoU, distance, shortcut over-prediction
  qualitative/<category>/<seq>_f<frame>/<model_folder>/<mode>/   # gt/pred/error PNGs
```

Folder names carry both code and description (`G2_lidar_rgb_lovasz`) for
traceability; figure text uses the thesis names from `_suite_style.DISPLAY_NAMES`.

---

## 8. What the comparisons produce

- **rgb_effect__D0_vs_E0** — the only clean RGB on/off step; supports "RGB adds X".
- **system__D0_vs_G2_vs_H0** — LiDAR baseline vs full systems (bundled; system-level claim).
- **shortcut__G2_vs_H0** — `rgb_valid_gap.csv` (rgb_valid vs rgb_invalid precision +
  pred/true, and the over-prediction gap) and `brightness_fingerprint.csv`
  (road→marking FP RGB/intensity profile vs road-TP/marking-TP). This is the
  brightness-shortcut before/after.

Expected (from validation, to be re-checked on test): jitter (H0) reduces
over-prediction and road→marking FPs and raises precision, **at a recall cost**,
leaving marking IoU tied with G2 within noise — i.e. a precision/calibration
result, not an IoU win; shortcut reduced, not eliminated.

---

## 9. Verification already done (bulletproofing)

- All scripts syntax-clean; engine is standalone-importable.
- Engine/`04` D0 generalisation: structurally verified (config compat, no-RGB
  branches, `has_rgb` gating). Runtime confirmation = the step-0 smoke.
- `build_comparison.py` ran **end-to-end on synthetic per-model data** → master
  table + comparison folders + 6 plots, PASS; synthetic data cleaned up.
- **Metric re-derivation is exact**: `metrics_from_cm` on G2's known validation
  sampled `confusion_matrix.npy` reproduced marking IoU `0.540046`, precision
  `0.626677`, recall `0.796193`, pred/true `1.270500`, road→marking `532367`.
- `val_metrics` reads the real validation numbers from `eval_history.csv`
  correctly (D0 0.4403 … H0 0.5507).

---

## 10. Caveats / limitations (state in the thesis)

1. **Sampled**, not exhaustive per-point evaluation (consistent across train/val/test).
2. **Single training seed (42)**; ~0.008 IoU noise floor; G2-vs-H0 differences
   within that are ties. The 3 eval-seeds for G2/H0 are sampling variance, not
   training variance.
3. **Validation vs test method nuance**: validation numbers in the master table
   come from the per-epoch training validation (720-step) at the selected epoch;
   test is a fresh 2160-step sampled pass. Both sampled; the `val−test` column is
   the generalization gap (expect test slightly lower). A fully method-matched
   gap would require also re-running validation through the engine at 2160 steps
   (optional).
4. **D0→G2/H0 is not a clean RGB ablation** (see §2); the clean RGB effect is D0→E0.
5. **H0 vs G2 also differs in training regime**: H0 was trained 100 epochs from
   scratch; G2 is a resume chain (G0→G1→G2). Same data/seed/schedule, but noted.
6. **RGB-shortcut framing is correlational** (group-level fingerprint; no causal
   channel ablation). Train-with-jitter is stronger evidence than the earlier
   frozen-checkpoint brightness probe, but still not a causal proof.

---

## 11. Things to scrutinise (for the reviewer)

- Is "sampled, single eval-seed for D0/E0/F0" acceptable, or should all five get
  3 eval-seeds? (Recommendation: only the close pair G2/H0 needs it.)
- Should validation be re-run through the engine at 2160 steps for a fully
  method-matched val→test gap? (Recommendation: optional; the eval_history val
  number is the canonical development metric.)
- Headline model choice: report G2 and H0 both (trade-off) vs designate one.
- Qualitative frame selection: auto best/worst/over-prediction + 065 night — any
  additional scenes worth pinning (uphill/downhill/weather)?

---

## 12. Source map

| Thing | Path |
| --- | --- |
| This plan | `results/TEST_PLAN.md` |
| Generalised engine | `results/test_suite/_sampled_error_engine.py` |
| Generalised overlay viewer | `results/test_suite/04_prediction_images.py` |
| Test driver | `results/test_suite/run_test_all.py` |
| Comparison/aggregation | `results/test_suite/build_comparison.py` |
| Qualitative driver | `results/test_suite/run_qualitative.py` |
| Plot style + thesis names | `results/test_suite/_suite_style.py` (`DISPLAY_NAMES`) |
| Test split | `configs/splits/test.txt`; `logs/milestone_b_split_report.txt` |
| Milestone context docs | `docs/milestone_{a..h}/…`; `docs/thesis_context/MILESTONE_CONTEXT_INDEX.md` |
| Per-model configs/checkpoints | `logs/milestone_{d,e,f,g,h}/runs/<run>/` (see §2) |
