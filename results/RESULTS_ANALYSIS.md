# Final Results & Test-Set Evaluation — Analysis Document

**Purpose.** This is the end-to-end record of the **final evaluation on the
held-out test split** of the thesis (PandaSet forward-facing road-marking
semantic segmentation, RandLA-Net, LiDAR ± front-camera RGB). It documents what
was run, where every number/file/plot lives, what the findings are and what they
*mean*, and where to look to go deeper. It is the test-phase companion to
[`results/TEST_PLAN.md`](TEST_PLAN.md) (which holds the protocol decision and its
justification). It is **not** a milestone-development record — it covers only the
final results discovered *now*, on the test set.

**Status:** evaluation COMPLETE. All numbers below are **full spatial coverage**
unless explicitly marked "sampled". Chosen final model = **G2**.

---

## 0. TL;DR — the findings in six lines
1. **G2 is the best model**: full-coverage **test marking IoU 0.517 ± 0.0005**, a **+0.103** system-level gain over the LiDAR-only baseline D0 (0.415).
2. **The D0→G2 gain is precision/calibration, not coverage**: precision 0.48→0.68, recall 0.75→0.68, over-prediction `pred/true` 1.54→1.00. RGB+Lovász roughly **halves false markings** (road→marking 1.75M→0.79M points).
3. **Raw RGB alone (E0) *hurts*** (IoU −0.015 vs D0) by amplifying over-prediction (`pred/true` 1.54→1.81) — the brightness shortcut. Calibration (F0) and the Lovász loss (G2) are what convert RGB into a gain.
4. **Jitter (H0) ≈ G2 on IoU** (0.510 vs 0.517, within the single-seed noise floor); it buys higher precision (0.70) and less over-prediction at a recall cost — a calibration result, not an IoU win.
5. **It generalizes** (val→test IoU gap ~0.015–0.027), **but recall drops** from ~0.80 (validation) to 0.68 (test): validation was optimistic on recall; on test the model is better-calibrated (`pred/true` 1.26→1.00).
6. **RGB helps specifically where the camera sees** (G2 IoU 0.52 in camera-visible regions vs 0.43 where it doesn't), and **markings degrade with distance** (IoU 0.59 near → 0.35 at 60m+).

---

## 1. What was run (the experiment matrix)

Five frozen, already-selected checkpoints were evaluated. Epoch was chosen during
development on the (sampled) training-time validation; **not** re-selected on test.

| code | thesis name | run dir | epoch | feat / in_ch | rgb |
| --- | --- | --- | ---: | --- | --- |
| D0 | LiDAR | `logs/milestone_d/runs/D0_weighted_ce_25ep` | 18 | intensity / 4 | no |
| E0 | LiDAR + RGB | `logs/milestone_e/runs/E0_rgb_front_v1` | 14 | intensity_rgb_front / 8 | yes |
| F0 | LiDAR + RGB (calibrated) | `logs/milestone_f/runs/F0_rgb_soft_weights` | 13 | intensity_rgb_front / 8 | yes |
| G2 | LiDAR + RGB + Lovász | `logs/milestone_g/runs/G2_schedule_extend_100` | 68 | intensity_rgb_front / 8 | yes |
| H0 | LiDAR + RGB + Lovász + Jitter | `logs/milestone_h/runs/H0_rgb_jitter` | 37 | intensity_rgb_front / 8 | yes |

Runs performed (coverage is a **folder level**: `…/full/…` vs `…/sampled/…`):

| Model | test FULL | test SAMPLED (cross-check) | validation FULL |
| --- | --- | --- | --- |
| D0 | ✅ seed 42 | — | ✅ seed 42 |
| E0 | ✅ seed 42 | — | — |
| F0 | ✅ seed 42 | — | — |
| G2 | ✅ seeds 42,1,2 | ✅ seeds 42,1,2 | ✅ seed 42 |
| H0 | ✅ seeds 42,1,2 | ✅ seeds 42,1,2 | ✅ seed 42 |

- **Test split** = 9 sequences (001, 002, 015, 065, 090, 101, 102, 103, 117) × 80 frames = **720 frames**; 065 is night.
- **Full coverage** = spatially-regular sampler over every frame (~5,275 patches; each point hit ~4–5× via overlap) → confusion matrices total ~172.6M point-predictions. **This is the protocol for all reported numbers.**
- **Sampled** (G2/H0 only) = 2,160 random patches; exists *only* to verify "sampled ≈ full" (cross-check). Not a result.
- Labels `road_marking3`: active classes `0=road, 1=marking, 2=other`; **marking = raw 8 (lane line) + 9 (stop line) + 10 (other road marking)**; in CSVs `lane_*` ≡ `marking_*`.

Protocol decision, justification, and the sampler bug that was found+fixed: see
[`TEST_PLAN.md`](TEST_PLAN.md) §1–§3. Confusion matrices are **patch-accumulated**
(not per-point logit voting); the voted variant is reported as a check (§6).

---

## 2. Headline results (master table)

Source: [`results/test_master_table.csv`](test_master_table.csv). All **full
coverage** except the two columns flagged sampled.

| code | sel-val* | full-val | **full-test** | ±std | voted | gap | prec | recall | F1 | mIoU | road IoU | other IoU | pred/true |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | 0.440* | 0.430 | **0.4147** | 0.000 | 0.435 | 0.015 | 0.483 | 0.745 | 0.586 | 0.773 | 0.927 | 0.977 | 1.542 |
| E0 | 0.438* | — | **0.3996** | 0.000 | 0.430 | 0.039 | 0.443 | 0.802 | 0.571 | 0.768 | 0.924 | 0.981 | 1.811 |
| F0 | 0.483* | — | **0.4495** | 0.000 | 0.470 | 0.033 | 0.545 | 0.720 | 0.620 | 0.790 | 0.939 | 0.982 | 1.322 |
| **G2** | 0.550* | 0.537 | **0.5170** | 0.0005 | 0.528 | 0.020 | 0.681 | 0.683 | 0.682 | 0.817 | 0.949 | 0.985 | 1.003 |
| H0 | 0.551* | 0.537 | **0.5096** | 0.0008 | 0.519 | 0.027 | 0.699 | 0.653 | 0.675 | 0.815 | 0.949 | 0.985 | 0.934 |

`*sel-val` = **sampled** training-time validation used only to *select* the
checkpoint (not a performance result). `full-val` / `full-test` = the reported
full-coverage metrics. `gap` = full-val − full-test (uses full-val where present).
`±std` = over 3 evaluation seeds (G2/H0). road/other IoU are ~0.93–0.99 for every
model — **marking is the only discriminating class.**

---

## 3. Findings and what they mean

### 3.1 D0 → G2: the system improvement is precision/calibration
Source: [`comparisons/d0_vs_g2/metrics_d0_vs_g2.csv`](comparisons/d0_vs_g2/metrics_d0_vs_g2.csv).

- IoU **+0.103** (0.415→0.518), precision **+0.198** (0.483→0.681), recall **−0.062** (0.745→0.683), `pred/true` **1.542→1.003**.
- Road→marking false positives **1.75M → 0.79M** (−55%); other→marking **0.30M → 0.04M** (−88%).
- **Meaning:** the full RGB system does **not** find *more* markings than LiDAR-only — it stops painting *false* ones on ambiguous road. The whole gain is fewer false positives + perfect calibration. This is the central, honest narrative for the thesis. (Caveat: D0→G2 is *system-level*, not a clean RGB ablation — G2 also adds Lovász, more features, more epochs; the clean RGB step is D0→E0 below.)

### 3.2 Raw RGB alone hurts — the brightness shortcut appears
Source: [`comparisons/rgb_effect__D0_vs_E0/deltas_vs_baseline.csv`](comparisons/rgb_effect__D0_vs_E0/deltas_vs_baseline.csv).

- D0→E0 (the clean RGB on/off step): IoU **−0.015**, precision **−0.040**, recall **+0.057**, `pred/true` **+0.268** (1.54→1.81).
- **Meaning:** naively adding camera RGB makes the model paint *more* marking (higher recall) but with worse precision — it leans on bright appearance to guess "marking", i.e. the brightness shortcut. RGB only becomes a net gain once over-prediction is controlled by calibration (F0, IoU 0.450) and the Lovász loss (G2, IoU 0.517). **The progression `pred/true` 1.54→1.81→1.32→1.00→0.93 (D0→E0→F0→G2→H0) is the shortcut being progressively tamed.**

### 3.3 G2 vs H0: jitter is a calibration move, not an IoU win
- IoU 0.517 vs 0.510 → **0.007 apart, below the ~0.008 single-training-seed noise floor → a tie.**
- H0 has higher precision (0.699 vs 0.681), lower recall (0.653 vs 0.683), `pred/true` 0.934 (under-predicts) vs 1.003.
- **Meaning:** the night-jitter augmentation trades recall for precision and pushes the model from perfectly-calibrated to slightly conservative. It does not improve the headline IoU. This is why H0 is a *secondary* discussion point, not the final model.

### 3.4 Generalization is healthy, but recall is lower on test than validation
Sources: master table (`full_val_iou`, `full_test_iou`); recall reconciliation in
§3.4 below was computed from the confusion matrices in
`per_model/G2.../{validation,test}/full/` and `…/test/sampled/`.

- val→test IoU gaps are small and uniform (D0 0.015, G2 0.020, H0 0.027).
- **But marking recall drops: validation ~0.80 → test 0.68.** Verified to be a *data* effect, not a protocol artifact:
  - validation recall ≈ 0.79–0.80 under **both** sampled (eval_history 0.804) and full coverage (0.788);
  - test recall ≈ 0.68 under **both** full (0.683) and sampled (0.689).
- IoU is nevertheless preserved because **precision rises** (0.63→0.68): on validation G2 over-predicts marking (`pred/true` 1.26); on the held-out test set it is **calibrated** (`pred/true` 1.00), so it catches fewer markings but makes fewer false positives.
- **Meaning:** the sampled validation was **optimistic on recall**. On unseen data the same model is better calibrated and more conservative. Report honestly: *"on test the model is well-calibrated (pred/true ≈ 1.0) but recalls fewer markings than validation suggested (0.68 vs 0.80), trading sensitivity for precision at near-unchanged IoU."*

### 3.5 G2 stratified analysis (where it works / fails)
All from `per_model/G2_lidar_rgb_lovasz/test/full/seed_42/`.

- **Per class** (`confusion_matrix.npy`): road IoU 0.949, **marking 0.518**, other 0.985. Marking is the hard class; geometry alone nearly solves road/other.
- **By distance** (`distance_bucket_metrics.csv`): marking IoU 0.59 (0–10m) → 0.52 → 0.51 → 0.54 (to 40m) → **0.44 (40–60m) → 0.35 (60m+)**. 92% of markings lie within 40m; clear degradation beyond. Recall falls to 0.55 at 40–60m; precision collapses to 0.45 at 60m+ (sparse, ~1.4% of markings — noisy).
- **By sequence** (`per_sequence_metrics.csv`): best **117 (0.648)**, **090 (0.580), 102 (0.578)**; worst **001 (0.302), 002 (0.358)** (complex urban scenes — these two are exactly what the discarded sampler-bug smoke accidentally measured); **night 065 = 0.531** (above average — RGB does *not* collapse at night here). Macro per-sequence mean ≈ 0.511 ≈ micro 0.517 → the headline isn't carried by a few scenes.
- **By marking subtype** (`raw_subtype_rgb_stratified_metrics.csv`, the "subtype/all" stratum is recall): **lane lines 0.728** (best), **stop lines 0.690**, **other road markings 0.591** (worst). The model is strongest on the most frequent/regular markings.
- **By RGB-validity** (`rgb_valid_stratified_metrics.csv`): **camera-visible (rgb_valid) IoU 0.524, recall 0.692** vs **not-visible (rgb_invalid) IoU 0.434, recall 0.575**. **Meaning:** G2 performs measurably better where the front camera actually sees the road — direct evidence that the RGB channel contributes (a partial answer to the "is the gain really from RGB?" question, since D0→G2 isn't a clean ablation).

### 3.6 The brightness shortcut, after Lovász (G2 vs H0)
Source: [`comparisons/shortcut__G2_vs_H0/`](comparisons/shortcut__G2_vs_H0/).

- `rgb_valid_gap.csv`: in **camera-visible** regions both models over-predict marking more than where the camera can't see — G2 `pred/true` 1.011 (valid) vs 0.900 (invalid), an **over-prediction gap of 0.111**; H0 reduces it to **0.096**. So jitter (H0) trims the camera-driven over-prediction by ~14%, but does not eliminate it. **Shortcut reduced, not removed** — consistent with the validation-era story.
- `brightness_fingerprint.csv`: G2's residual road→marking **false positives** have RGB brightness **0.389** — *lower* than correctly-classified road (0.404) and far below true markings (0.438) — i.e. the residual false markings are **not** the bright pixels. But they have **elevated LiDAR intensity (31.1 vs 25.4 for road)**, near true markings (37.3). **Meaning (nuanced, important):** after Lovász, G2's remaining confusion is driven more by **high LiDAR reflectivity** than by RGB brightness — the *RGB*-brightness shortcut is largely controlled; what remains is intensity-based road↔marking ambiguity. State this honestly; it complicates a naive "bright→marking" claim for the final model.

### 3.7 Reliability / robustness
- **Evaluation seeds:** full-coverage IoU std is tiny (G2 ±0.0005, H0 ±0.0008) — full coverage is near-deterministic.
- **No-voting check** (`coverage_count_by_class.csv` + `confusion_matrix_voted.npy`): per-point **voted** marking IoU (G2 0.528) ≈ patch-accumulated headline (0.518) → the per-patch accumulation does not bias the result. Coverage is uniform across classes (marking median 4 patches/point, same as road/other). Prediction **agreement across overlapping patches**: road 0.977, **marking 0.835**, other 0.991 — marking predictions are the least stable across patches (expected for a thin/boundary class), which also explains the small voted-vs-accumulated difference.
- **Sampled-vs-full cross-check** (`comparisons/crosscheck_sampled_vs_full.csv`): sampled test IoU is ~0.012–0.015 higher than full (G2 0.529 vs 0.517; H0 0.525 vs 0.510) — small, systematic, consistent. This validated comparing full-test to the sampled selection-validation, and motivated also running full-coverage validation.
- **Dominant caveat:** single **training** seed (42). The ~0.008 marking-IoU noise floor means G2≈H0 differences are ties. Evaluation seeds do **not** address training variance.

---

## 4. Complete source map — files, what they are, where they come from

> Rule of thumb: **coverage is the folder** (`…/test/full/…`, `…/test/sampled/…`,
> `…/validation/full/…`). Filenames are identical across coverage; each
> `summary.json` also carries a `"coverage"` field, and **only full runs** contain
> `confusion_matrix_voted.npy` + `coverage_count_by_class.csv`.

### 4.1 Per-model outputs — `results/per_model/<model>/<split>/<coverage>/seed_<S>/`
Produced by `run_test_all.py` → `_sampled_error_engine.py`.

| file | what it is / represents |
| --- | --- |
| `summary.json` | headline metrics block + `coverage`, `has_rgb`, `feature_mode`, raw-remap audit, voted metrics (full only). Source of the single-number metrics. |
| `confusion_matrix.npy` | 3×3 (rows=true, cols=pred; road/marking/other), patch-accumulated. **The source of truth** — every metric is re-derivable from it. |
| `confusion_matrix_voted.npy` | *(full only)* per-point majority-voted 3×3 — the no-voting confirmation. |
| `per_sequence_metrics.csv` | all metrics per test sequence (the macro / per-scene breakdown, incl. 065). |
| `distance_bucket_metrics.csv` | all metrics per range bucket (0–10/10–20/20–30/30–40/40–60/60m+). |
| `rgb_valid_stratified_metrics.csv` | *(RGB models only)* metrics split by rgb-valid vs rgb-invalid (camera-visible or not). |
| `raw_subtype_rgb_stratified_metrics.csv` | recall per marking subtype (raw 8/9/10), × rgb strata. The "all" stratum = subtype recall. |
| `group_feature_summary.csv` | per-outcome (marking_tp, road_to_marking, …) feature profile: intensity, R/G/B, range. The **brightness/intensity fingerprint** source. |
| `frame_error_summary.csv` | per-**patch** row (seq_id, frame_idx, all metrics). Aggregated → per-frame for the qualitative/video pickers and the frame-IoU histogram. |
| `coverage_count_by_class.csv` | *(full only)* per-class coverage-count distribution (median/p10/p90/max) + prediction-agreement rate. |
| `top_frames_by_road_to_marking.csv` / `…_marking_to_road.csv` | worst frames by false-positive / miss count. |
| `README.md` | per-run provenance (checkpoint, steps, seed). |

### 4.2 Aggregated outputs — `results/`
| path | what it is | produced by |
| --- | --- | --- |
| `test_master_table.csv` | all 5 models × all metrics; the two-validation-column headline. | `build_comparison.py` |
| `README.md` | generated provenance + master table in markdown. | `build_comparison.py` |
| `TEST_PLAN.md` | the protocol decision/justification (companion to this doc). | hand-written |
| `comparisons/crosscheck_sampled_vs_full.csv` | sampled vs full test IoU (G2/H0). | `build_comparison.py` |
| `comparisons/rgb_effect__D0_vs_E0/` | `test_metrics.csv` + `deltas_vs_baseline.csv` — the clean RGB on/off step. | `build_comparison.py` |
| `comparisons/system__D0_vs_G2_vs_H0/` | LiDAR baseline vs full systems (metrics + deltas). | `build_comparison.py` |
| `comparisons/shortcut__G2_vs_H0/rgb_valid_gap.csv` | rgb-valid vs invalid precision + `pred/true` + over-prediction gap. | `build_comparison.py` |
| `comparisons/shortcut__G2_vs_H0/brightness_fingerprint.csv` | RGB + intensity profile of road-TP / road→marking-FP / marking-TP. | `build_comparison.py` |
| `comparisons/d0_vs_g2/*.csv` | focused D0-vs-G2 metrics, per-sequence, distance. | `compare_d0_g2.py` |
| `comparisons/video_sequence_suitability_G2.csv` | per-sequence "is this a good video?" stats (marking-frame fraction, IoU spread, consistency). | `pick_video_sequences.py` |

### 4.3 Figures
| set | files | produced by | coverage |
| --- | --- | --- | --- |
| `results/G2_final/` | `fig_per_class_iou`, `fig_marking_metrics`, `fig_confusion` (counts), `fig_by_distance`, `fig_by_sequence`, `fig_by_subtype`, `fig_by_rgb_validity`, `fig_frame_iou_hist` | `model_results_figures.py` | **FULL** (G2 standalone) |
| `comparisons/d0_vs_g2/` | `fig_metrics_bars`, `fig_overprediction`, `fig_confusion_D0/G2`, `fig_error_breakdown`, `fig_per_sequence`, `fig_distance`, `fig_val_vs_test` | `compare_d0_g2.py` | **FULL** (D0 vs G2) |
| `comparisons/plots/` | `progression_marking_iou_test`, `marking_metrics_test`, `per_class_iou_test`, `distance_marking_iou_test`, `shortcut_overprediction_g2_vs_h0`, `val_vs_test_marking_iou` | `build_comparison.py` | **FULL** — except `val_vs_test_marking_iou` **mixes** (E0/F0 bars are sampled selection-val; D0/G2/H0 are full-val). Prefer `d0_vs_g2/fig_val_vs_test`. |

**Colour rule (all figures):** classes blue=road / red=marking / green=other; marking-metric family IoU=red / precision=orange / recall=purple / F1=gold (set in `_suite_style.py`).

### 4.4 Qualitative media (server only — gitignored, not in the repo)
| path | what it is | produced by |
| --- | --- | --- |
| `results/qualitative/<category>/<seq>_f<frame>/<model>/<mode>/*.png` | front-camera GT / prediction / error overlays (opaque), D0 & G2, for auto-picked best / worst / overpredict / night-065 frames. | `run_qualitative.py` → `04_prediction_images.py` |
| `results/videos/<seq>/<model>/images/*.png` + `<seq>_<model>_pred.mp4` | full-sequence prediction fly-through videos (G2; 101, 002, 065 rendered). | `make_sequence_video.py` → `04_prediction_images.py` |

### 4.5 Code — `results/test_suite/`
| script | role |
| --- | --- |
| `_sampled_error_engine.py` | **the engine.** Runs sampled or full-coverage inference for one checkpoint/split, writes the per-model files. Has `--coverage {sampled,full}` + the no-voting diagnostics. |
| `run_test_all.py` | driver: runs the engine for all models × splits × coverages × seeds (the experiment matrix); resume-safe; `--full-val`. |
| `build_comparison.py` | aggregates per-model outputs → master table + comparison folders + `comparisons/plots/` (CPU/file-only). |
| `compare_d0_g2.py` | focused D0-vs-G2 figure/table set. |
| `model_results_figures.py` | standalone final-model figure set (`<model>_final/`); `--model`. |
| `pick_video_sequences.py` | ranks test sequences for fly-through videos. |
| `make_sequence_video.py` | renders a full sequence + stitches an MP4. |
| `run_qualitative.py` | renders spot-frame GT/pred/error overlays. |
| `04_prediction_images.py` | the front-camera overlay viewer (used by the two above). |
| `_suite_style.py` | the colour rule + plot styling (single source). |
| `_suite_paths.py` | path helpers. |
| `01_…`–`07_…`, `run_full_suite.py`, `02_sampled_error_analysis.py` | the original milestone-era reproducible suite scripts (kept for reference; the test phase calls the engine directly, not `02`). |

---

## 5. Where this directs us / further investigation

1. **Clean attribution of the RGB gain (the biggest open item).** D0→G2 mixes
   RGB + Lovász + features + epochs. The rgb-valid stratification (§3.5) is
   suggestive (G2 better where the camera sees), but to *prove* the gain comes
   from RGB, re-run **D0 with the rgb_valid mask recorded** (a small engine change
   — compute the geometric camera-validity for D0 without using RGB as a feature)
   and compare D0-vs-G2 *within* rgb_valid vs rgb_invalid. → look at the engine's
   `rgb_valid` handling + a 1-run D0 re-eval.
2. **Distance degradation.** Marking IoU falls sharply beyond 40m. Worth a focused
   look at whether it's recall (missing far markings) or precision, per subtype. →
   `distance_bucket_metrics.csv` + per-subtype × distance (data exists per patch in
   `frame_error_summary.csv` + the raw labels).
3. **The intensity-not-brightness residual confusion (§3.6).** G2's residual false
   markings are intensity-elevated road, not bright-RGB road. Could be probed
   further with `group_feature_summary.csv` per distance/sequence. Reframes the
   "shortcut" for the final model as intensity-driven.
4. **Hardest scenes (001, 002).** Both are complex urban; qualitative overlays /
   the `002`/`001` videos show *why*. → `results/videos/002/…`, `qualitative/worst/…`.
5. **Per-point logit voting** (currently a disclosed limitation). The voted CM
   already agrees with the headline; a full voting engine would make it exact. →
   `confusion_matrix_voted.npy` (majority-vote) is the starting point.
6. **Single training seed.** The one statistical weakness; the only fix is
   retraining G2/H0 with more seeds (out of scope now, noted as a limitation).

**For any deeper question:** the relevant source is almost always (a) the
`confusion_matrix.npy` of the run in question (everything re-derives from it),
(b) the matching `*_metrics.csv` for a stratified cut, or (c)
`frame_error_summary.csv` for per-frame/per-patch detail. The script that made a
file is in §4.5; re-run it to regenerate or extend.

---

## 6. Caveats / limitations (carry into the writeup)
1. **Single training seed (42)**; ~0.008 marking-IoU noise floor → G2≈H0 is a tie. Evaluation seeds fix only sampling variance.
2. **Patch-accumulated confusion, not per-point logit voting** — backed by the coverage/agreement diagnostics and the voted-CM check, but disclosed.
3. **Select-on-(sampled)-validation, report-on-full-coverage-test** — clean workflow, two named validation columns; selection-val is *not* a result.
4. **D0→G2 is system-level, not a clean RGB ablation** — the clean RGB step is D0→E0.
5. **H0 differs from G2 in training regime** (H0 100ep from scratch; G2 a resume chain) in addition to jitter — noted.
6. **RGB-shortcut evidence is correlational** (group-level fingerprint + rgb-valid stratification; no causal channel ablation).
7. **Night = one sequence (065)** — treat as a case study, not a general low-light claim.
8. Metrics are over the **grid-subsampled evaluation cloud**, not raw PandaSet points; no projection back to the full cloud.
