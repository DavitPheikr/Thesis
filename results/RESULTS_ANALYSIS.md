# Final Results & Test-Set Evaluation — Analysis Document

**Last updated:** 2026-06-24 05:55 CEST.
**Status:** evaluation **COMPLETE**. Chosen final model = **G2** (`LiDAR + RGB + Lovász`).
All numbers below are **full spatial coverage** unless a column/row is explicitly
marked *sampled*.

**Purpose.** End-to-end record of the **final evaluation on the held-out test
split** of the thesis (PandaSet forward-facing road-marking semantic segmentation,
RandLA-Net, LiDAR ± front-camera RGB). It documents what was run, where every
number/file/plot lives, what the findings are and what they *mean*, and where to
look to go deeper. It is the test-phase companion to
[`results/TEST_PLAN.md`](TEST_PLAN.md) (the protocol decision and its
justification) and the test-phase counterpart to the per-milestone
`docs/milestone_*/MILESTONE_*_FINAL_CONTEXT.md` records. It is **not** a
milestone-development record — it covers only the final results discovered *now*,
on the held-out test set.

---

## Provenance / evidence tier (read first)

**This document is at the project's *highest* evidence tier: every headline number
is locally re-derivable from committed raw artifacts.** The full-coverage
confusion matrices (`confusion_matrix.npy`) and the per-stratum CSVs are committed
under `results/per_model/…`; the engine that produced them and the aggregator that
re-derives every metric from the matrices are committed under
`results/test_suite/`. No server access or GPU is needed to reproduce any table
here — only the saved matrices and CSVs.

**Reliability rule used (highest → lowest):**
1. Committed full-coverage run artifacts (`confusion_matrix.npy`, per-stratum CSVs,
   `summary.json`) — the source of truth; every metric re-derives from the matrix.
2. The committed aggregation/comparison CSVs (`test_master_table.csv`,
   `comparisons/…`) produced from (1) by `build_comparison.py`.
3. The committed figures (rendered from 1–2).
4. The protocol record `TEST_PLAN.md` and the milestone `*_FINAL_CONTEXT.md` docs
   (for development-time / validation-era numbers used only as references here).

**Verification performed for this revision (2026-06-24):** every numeric claim in
§0, §2, and §3 was re-checked directly against the committed `.npy`/`.csv` files
(matrices reloaded and metrics recomputed; stratified CSVs read row-by-row;
validation-vs-test recall reconciled across all four protocol/coverage
combinations). Discrepancies found and fixed are listed in §7.

**Labels (constant throughout).** Label mode `road_marking3`: active classes
`0=road, 1=marking, 2=other`; **marking = raw 8 (lane line) + 9 (stop line) + 10
(other road marking)**; in every CSV/JSON, `lane_*` ≡ `marking_*`. Do **not**
compare to Milestone C (strict lane-line labels) without stating the label change.

---

## 0. TL;DR — the findings, each one verified
1. **road / other are essentially solved by geometry; marking is the only
   discriminating class** — road IoU 0.92–0.95 and other IoU 0.98 for *every*
   model. All analysis below is therefore about the marking class.
2. **G2 is the best model**: full-coverage **test marking IoU 0.5170 ± 0.0005**
   (3 eval seeds), a **+0.102** system-level gain over the LiDAR-only baseline D0
   (0.4147).
3. **The D0→G2 gain is entirely false-positive reduction (precision/calibration),
   not better detection.** G2 finds **fewer** true markings than D0
   (TP 1,761,293 vs 1,921,024, recall 0.745→0.683) but makes **far** fewer false
   ones (road→marking 1.75M→0.79M, −55%; other→marking 0.30M→0.04M, −88%);
   over-prediction `pred/true` 1.54→1.00.
4. **Raw RGB *alone* (E0) hurts** (IoU −0.015 vs D0) by *worsening* over-prediction
   (`pred/true` 1.54→1.81) — the brightness shortcut. RGB becomes a net gain only
   after calibration (F0) and the Lovász IoU-surrogate loss (G2). The monotone
   `pred/true` chain **1.54→1.81→1.32→1.00→0.93** (D0→E0→F0→G2→H0) *is* the
   shortcut being progressively tamed.
5. **Jitter (H0) ≈ G2 on IoU** (0.510 vs 0.517 — 0.007 apart, below the ~0.008
   single-training-seed noise floor → a tie); it buys precision (0.70) at a recall
   cost and slightly *under*-predicts (`pred/true` 0.93). A calibration result, not
   an IoU win.
6. **It generalizes** (protocol-matched full-val→full-test IoU gap **0.020**),
   **but recall drops** from ~0.79 (validation) to 0.68 (test); IoU barely moves
   (−0.020) because precision rises (0.63→0.68). On test the model is
   **better-calibrated** (`pred/true` 1.26→1.00), i.e. validation was optimistic on
   recall. Verified across all four val/test × sampled/full combinations.
7. **RGB helps where the camera sees** (G2 IoU 0.524 in camera-visible regions vs
   0.434 where it doesn't) and **even helps at night** (G2 0.531 > D0 0.455 on
   night sequence 065). **Markings degrade with distance** (IoU 0.59 near → 0.35
   beyond 60 m).
8. **On the test set the residual confusion is LiDAR-intensity-driven, not
   RGB-brightness-driven** (G2 false markings: intensity 31.1 vs road 25.4, but
   brightness 0.389 *below* road 0.404) — a notable shift from the
   *validation*-era brightness-shortcut fingerprint (§3.6, §7).

---

## 1. What was run (the experiment matrix)

Five frozen, already-selected checkpoints. Epoch was chosen during development on
the (sampled) training-time validation; **not** re-selected on test (clean
select-on-dev / report-on-test workflow — `TEST_PLAN.md` §1.3).

| code | thesis name | run dir | epoch | feat / in_ch | rgb |
| --- | --- | --- | ---: | --- | --- |
| D0 | LiDAR | `logs/milestone_d/runs/D0_weighted_ce_25ep` | 18 | intensity / 4 | no |
| E0 | LiDAR + RGB | `logs/milestone_e/runs/E0_rgb_front_v1` | 14 | intensity_rgb_front / 8 | yes |
| F0 | LiDAR + RGB (calibrated) | `logs/milestone_f/runs/F0_rgb_soft_weights` | 13 | intensity_rgb_front / 8 | yes |
| G2 | LiDAR + RGB + Lovász | `logs/milestone_g/runs/G2_schedule_extend_100` | 68 | intensity_rgb_front / 8 | yes |
| H0 | LiDAR + RGB + Lovász + Jitter | `logs/milestone_h/runs/H0_rgb_jitter` | 37 | intensity_rgb_front / 8 | yes |

Runs performed (**coverage is a folder level**: `…/full/…` vs `…/sampled/…`):

| Model | test FULL | test SAMPLED (cross-check) | validation FULL |
| --- | --- | --- | --- |
| D0 | ✅ seed 42 | — | ✅ seed 42 |
| E0 | ✅ seed 42 | — | — |
| F0 | ✅ seed 42 | — | — |
| G2 | ✅ seeds 42,1,2 | ✅ seeds 42,1,2 | ✅ seed 42 |
| H0 | ✅ seeds 42,1,2 | ✅ seeds 42,1,2 | ✅ seed 42 |

- **Test split** = 9 sequences (001, 002, 015, 065, 090, 101, 102, 103, 117) × 80
  frames = **720 frames**; **065 is night** (`configs/splits/test.txt`).
- **Full coverage** = spatially-regular sampler over every frame; each point is hit
  ~4–5× via patch overlap → the test confusion matrix totals **172,588,729**
  point-predictions (`rgb_valid_stratified_metrics.csv`, `all` row). **This is the
  protocol for all reported numbers.**
- **Sampled** (G2/H0 only) = 2,160 random patches/seed; exists *only* to verify
  "sampled ≈ full" (the cross-check, §3.7). **Not a reported result.**
- "Point" = a point of the **grid-subsampled evaluation cloud** (identical on val
  and test, so comparisons are fair); predictions are not projected back to the raw
  PandaSet cloud (`TEST_PLAN.md` §2).

Protocol decision, justification, and the sampler bug that was found+fixed (which
invalidated the earlier "test collapse" numbers): see [`TEST_PLAN.md`](TEST_PLAN.md)
§1–§3. Confusion matrices are **patch-accumulated** (not per-point logit voting);
the voted variant is reported as a check (§3.7, `TEST_PLAN.md` §1.5).

---

## 2. Headline results (master table)

Source: [`results/test_master_table.csv`](test_master_table.csv) (re-derivable from
the per-model `confusion_matrix.npy` via `build_comparison.py`). All **full
coverage** except the two columns flagged *sampled*.

| code | sel-val* | full-val | **full-test** | ±std | voted | gap | prec | recall | F1 | mIoU | road IoU | other IoU | pred/true |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | 0.4403* | 0.4296 | **0.4147** | 0.000 | 0.4352 | 0.0149 | 0.4832 | 0.7452 | 0.5863 | 0.7732 | 0.9274 | 0.9774 | 1.542 |
| E0 | 0.4384* | — | **0.3996** | 0.000 | 0.4295 | 0.0389† | 0.4432 | 0.8024 | 0.5710 | 0.7681 | 0.9239 | 0.9809 | 1.811 |
| F0 | 0.4828* | — | **0.4495** | 0.000 | 0.4700 | 0.0332† | 0.5446 | 0.7202 | 0.6202 | 0.7903 | 0.9391 | 0.9823 | 1.322 |
| **G2** | 0.5504* | 0.5368 | **0.5170** | 0.0005 | 0.5277 | 0.0198 | 0.6807 | 0.6825 | 0.6816 | 0.8171 | 0.9492 | 0.9851 | 1.003 |
| H0 | 0.5507* | 0.5369 | **0.5096** | 0.0008 | 0.5186 | 0.0274 | 0.6992 | 0.6527 | 0.6751 | 0.8146 | 0.9495 | 0.9849 | 0.934 |

`*sel-val` = **sampled** training-time validation used only to *select* the
checkpoint (not a performance result; source = each run's `eval_history.csv`
best-epoch row). `full-val`/`full-test` = the reported full-coverage metrics.
`gap` = full-val − full-test where full-val exists; **†** for E0/F0 the gap is
sel-val − full-test (no full-val run), so it is *not* protocol-matched — use it
only as a coarse pointer, never as a generalization measure. `±std` = over the 3
evaluation seeds (G2/H0 only). road/other IoU are 0.93–0.99 for every model —
**marking is the only discriminating class.**

> **Seed note (so figures and table agree).** The 3-seed mean is the headline
> (G2 full-test 0.5170). The focused `comparisons/d0_vs_g2/` and
> `comparisons/system__D0_vs_G2_vs_H0/` artifacts are computed on **seed 42 only**
> (G2 marking IoU 0.51768), so they show the D0→G2 gain as +0.103 rather than the
> 3-seed +0.102. The 0.0007 difference is eval-seed noise — far below any reported
> effect — but the headline uses the 3-seed mean throughout.

---

## 3. Findings and what they mean

*Every subsection states its source file and was re-derived from it on 2026-06-24.*

### 3.1 D0 → G2: the system gain is false-positive reduction, not better detection
Source: [`comparisons/d0_vs_g2/metrics_d0_vs_g2.csv`](comparisons/d0_vs_g2/metrics_d0_vs_g2.csv)
(seed 42); cross-checked against both `confusion_matrix.npy` files.

- Marking IoU **+0.102** (3-seed: 0.4147→0.5170; seed-42 panel: →0.5177),
  precision **+0.198** (0.483→0.681), recall **−0.062** (0.745→0.683),
  `pred/true` **1.542→1.003**.
- **Detection goes *down*, not up:** marking true-positives **1,921,024 (D0) →
  1,761,293 (G2)** — G2 recovers **159,731 fewer** true marking points. Meanwhile
  road→marking false positives **1,751,143 → 786,744 (−55%)** and other→marking
  **303,566 → 37,798 (−88%)**.
- **Meaning:** the full RGB system does **not** find more markings than LiDAR-only
  — it **stops painting false ones** on ambiguous road. The entire +0.102 IoU is a
  precision/calibration gain (fewer FPs + perfect `pred/true`), bought at a small
  recall cost. This is the central, honest thesis narrative: *the contribution is
  calibration, not coverage.*
- **Attribution caveat (important):** D0→G2 is a **system-level** comparison, not a
  clean RGB ablation — G2 also adds the Lovász loss, more features (`dim_features`
  8→16), and more epochs. The clean RGB on/off step is **D0→E0** (§3.2). The
  rgb-valid stratification (§3.5) is the partial evidence that RGB specifically
  contributes.

### 3.2 Raw RGB alone hurts — the brightness shortcut, isolated
Source: [`comparisons/rgb_effect__D0_vs_E0/deltas_vs_baseline.csv`](comparisons/rgb_effect__D0_vs_E0/deltas_vs_baseline.csv).

- D0→E0 (the **only** clean RGB on/off step — loss/epochs essentially fixed):
  IoU **−0.015**, precision **−0.040**, recall **+0.057**, `pred/true` **+0.268**
  (1.54→1.81).
- **Meaning:** naively concatenating camera RGB makes the model paint *more*
  marking (recall up) at *worse* precision — it leans on bright appearance to guess
  "marking", the brightness shortcut. RGB is **not free**; it pays off only once
  over-prediction is controlled: soft class weights (F0, IoU 0.450, `pred/true`
  1.32) and the Lovász loss (G2, IoU 0.517, `pred/true` 1.00). **The `pred/true`
  chain 1.54→1.81→1.32→1.00→0.93 (D0→E0→F0→G2→H0) is the single clearest signature
  of the shortcut being tamed.**
- **Critical caveat:** the −0.015 IoU drop exceeds the ~0.008 noise floor, but
  D0/E0 are **single training seeds** — read it as "RGB-alone does not help and
  plausibly hurts," not a tightly-bounded effect size.

### 3.3 G2 vs H0: jitter is a calibration move, not an IoU win
Source: master table; [`comparisons/shortcut__G2_vs_H0/`](comparisons/shortcut__G2_vs_H0/).

- IoU 0.5170 vs 0.5096 → **0.0074 apart, below the ~0.008 single-training-seed
  noise floor** (`logs/milestone_g/notes/single_seed_limitation.md`) → **a tie.**
- H0 has higher precision (0.699 vs 0.681), lower recall (0.653 vs 0.683), and
  `pred/true` **0.934** (slightly *under*-predicts) vs G2's 1.003 (calibrated).
- **Meaning:** the night-jitter augmentation trades recall for precision and pushes
  the model from perfectly-calibrated to slightly conservative. It does **not**
  improve headline IoU — hence H0 is a *secondary* discussion point and G2 remains
  the final model. (Caveat: H0 also differs in training regime — 100 ep from
  scratch vs G2's resume chain — so even the precision shift is not a clean
  jitter-only effect; `TEST_PLAN.md` §12.5.)

### 3.4 Generalization is healthy; recall (not IoU) is what drops on test
Sources: master table (`full_val_iou`, `full_test_iou`); recall reconciliation
re-derived from `per_model/G2_lidar_rgb_lovasz/{validation/full, test/full,
test/sampled}/` matrices and `logs/milestone_g/runs/G2_.../eval_history.csv`
(ep 68).

Protocol-matched (full-coverage on both sides), G2:

| quantity | full **validation** | full **test** | Δ |
| --- | ---: | ---: | ---: |
| marking IoU | 0.5368 | 0.5170 | **−0.0198** |
| marking precision | 0.6273 | 0.6807 | **+0.0534** |
| marking recall | 0.7883 | 0.6825 | **−0.1058** |
| `pred/true` | 1.257 | 1.003 | −0.254 |

- The **IoU gap is small** (0.020) — healthy generalization. But it is the *net* of
  a **large recall drop (−0.106)** nearly offset by a **precision rise (+0.053)**.
- The recall drop is a **data/generalization effect, not a protocol artifact** —
  verified across all four combinations:
  - validation recall ≈ **0.79–0.80** under *both* sampled (eval_history ep68
    **0.804**) and full coverage (**0.788**);
  - test recall ≈ **0.68** under *both* full (**0.683**) and sampled-3-seed mean
    (**0.689**).
- **Meaning:** the sampled validation was **optimistic on recall**. On unseen data
  the same checkpoint is more conservative and better-calibrated (`pred/true`
  1.26→1.00), so it catches fewer markings but makes proportionally fewer false
  positives — IoU is largely preserved (−0.020) precisely because the precision
  gain cancels most of the recall loss. Report it exactly that way: *"on test the
  model is well-calibrated (pred/true ≈ 1.0) but recalls fewer markings than
  validation suggested (0.68 vs ~0.79), trading sensitivity for precision at
  near-unchanged IoU."*

### 3.5 G2 stratified — where it works and where it fails
All from `per_model/G2_lidar_rgb_lovasz/test/full/seed_42/`.

- **Per class** (`confusion_matrix.npy`): road IoU 0.949, **marking 0.518**, other
  0.985. Geometry alone nearly solves road/other; marking is the hard class.
- **By distance** (`distance_bucket_metrics.csv`): marking IoU **0.588** (0–10 m) →
  0.524 → 0.513 → 0.537 (to 40 m) → **0.438 (40–60 m) → 0.346 (60 m+)**.
  **92.1 %** of true markings lie within 40 m (2,374,404 / 2,577,766); clear
  degradation beyond. Mechanism splits by range: at 40–60 m it is **recall** that
  falls (0.554); at 60 m+ it is **precision** that collapses (0.451), on a sparse
  ~1.4 % of markings (noisy).
- **By sequence** (`per_sequence_metrics.csv`): best **117 (0.648)**, **090
  (0.580)**, **102 (0.578)**; worst **001 (0.302)**, **002 (0.358)**. The two worst
  fail in **opposite** ways — **001 over-predicts** (`pred/true` 1.31, recall 0.54)
  while **002 under-detects** (`pred/true` 0.59, recall 0.42: it *misses* markings,
  205,426 marking→road). Both are complex urban scenes — and are exactly the two
  sequences the discarded sampler-bug smoke accidentally measured (`TEST_PLAN.md`
  §3). Macro per-sequence mean **0.511** ≈ micro **0.517** → the headline is **not**
  carried by a few scenes. (Also note **103**: IoU 0.474 but `pred/true` 1.94 /
  recall 0.945 — a heavy-over-prediction outlier on a low-marking-support scene.)
- **By marking subtype** (`raw_subtype_rgb_stratified_metrics.csv`, `all` stratum;
  here precision = 1.0 by construction so IoU = recall): **lane lines 0.728**
  (best), **stop lines 0.690**, **other road markings 0.591** (worst). The model is
  strongest on the most frequent/regular markings.
- **By RGB-validity** (`rgb_valid_stratified_metrics.csv`): **camera-visible
  (rgb_valid) IoU 0.524, recall 0.692** vs **not-visible (rgb_invalid) IoU 0.434,
  recall 0.575**. **Meaning:** G2 is measurably better where the front camera
  actually sees the road — the cleanest available evidence that the RGB channel
  contributes (a partial answer to "is the D0→G2 gain really RGB?", since D0→G2 is
  not a clean ablation). 87 % of points are rgb_valid (150.3M / 172.6M).
- **Night (065):** G2 IoU **0.531** vs D0 IoU **0.455** (+0.075) — the RGB model
  *beats* LiDAR-only on the one night sequence and sits **above** G2's own
  micro-average. So there is **no night-time collapse** for the final model here.
  Treat as a single case study, not a general low-light claim (`TEST_PLAN.md` §13).

### 3.6 The residual confusion is intensity-driven, not brightness-driven (test)
Source: [`comparisons/shortcut__G2_vs_H0/`](comparisons/shortcut__G2_vs_H0/)
(`rgb_valid_gap.csv`, `brightness_fingerprint.csv`).

- **Over-prediction is still concentrated where the camera sees:** G2 `pred/true`
  **1.011** in rgb_valid vs **0.900** in rgb_invalid — an **over-prediction gap of
  0.111**; H0 trims it to **0.096** (≈14 % less). So jitter reduces, but does not
  remove, the camera-driven over-prediction — consistent with the validation-era
  story.
- **But the residual false markings are *not* the bright pixels.** G2's road→marking
  false positives have mean RGB brightness **0.389** — *below* correctly-classified
  road (**0.404**) and well below true markings (**0.438**) — while their LiDAR
  intensity is **elevated: 31.1 vs 25.4 for road**, partway to true markings (37.3).
- **Meaning (nuanced, and a shift from validation):** after the Lovász loss, what
  remains of the road↔marking confusion on the **held-out test set** is driven by
  **high LiDAR reflectivity**, not RGB brightness — the RGB-brightness shortcut that
  dominated the *validation*-era fingerprint is largely controlled in the final
  model. **This contradicts the Milestone-G validation finding** (`docs/milestone_g/
  MILESTONE_G_FINAL_CONTEXT.md` §G2.6), where G2's road→marking FPs were *brighter*
  than road (median 0.478 vs road 0.392). **The contradiction is real, not a
  statistic mismatch — verified by recomputing the test medians** (from
  `group_feature_summary.csv`): on test the FP brightness median is **~0.367**,
  *below* road TP (**~0.465**) — so by *both* mean and median the test false markings
  are darker than road, the opposite of validation. With the mean-vs-median
  hypothesis ruled out, **exactly one explanation remains: validation and the
  held-out test are different scenes, and on the unseen test set the RGB-brightness
  shortcut has weakened** while LiDAR-intensity ambiguity dominates. **State
  honestly:** on test, intensity-based ambiguity — not an
  RGB-brightness shortcut — is the dominant residual error of the final model.

### 3.7 Reliability / robustness
- **Evaluation seeds:** full-coverage IoU std is tiny (G2 ±0.0005, H0 ±0.0008) —
  full coverage is near-deterministic.
- **No-voting check** (`coverage_count_by_class.csv`, `confusion_matrix_voted.npy`,
  seed 42): per-point **voted** marking IoU (**0.527**) vs the patch-accumulated
  headline (**0.518**, seed 42) — a **+0.009** difference, small and in the
  direction expected if over-covered boundary points are slightly noisier; it does
  **not** overturn the result. Coverage is near-uniform across classes (marking
  median **4** patches/point; road 4, other 5). **Prediction agreement across
  overlapping patches:** road **0.977**, **marking 0.835**, other **0.991** —
  marking predictions are the least stable across patches (expected for a
  thin/boundary class), which also explains the small voted-vs-accumulated gap.
- **Sampled-vs-full cross-check** (`comparisons/crosscheck_sampled_vs_full.csv`):
  sampled test IoU is **+0.012 (G2)** / **+0.015 (H0)** above full — small,
  systematic, consistent. This is what licenses comparing full-coverage test to the
  sampled selection-validation, and it motivated also running full-coverage
  validation.
- **Dominant caveat:** a single **training** seed (42). The ~0.008 marking-IoU
  noise floor means G2≈H0 differences are ties. Evaluation seeds fix only sampling
  variance, **not** training variance.

### 3.8 Cross-model stratified comparison (same full-coverage data, all 5 models)
Source: each model's `distance_bucket_metrics.csv`, `raw_subtype_rgb_stratified_metrics.csv`,
`rgb_valid_stratified_metrics.csv`, `per_sequence_metrics.csv` (test/full/seed 42).
§3.5 figured only G2; the per-model strata were already committed, so this is the
model-to-model read of data we already had.

- **Distance — the far-range weakness is universal, but RGB still helps there.**
  Marking IoU at 60 m+: D0 0.178, E0 0.236, F0 0.191, **G2 0.346**, H0 0.337 — G2 is
  ~2× D0 even at the hardest range, and leads every model at *every* bucket. So
  long-range degradation is LiDAR sparsity, **not** an RGB ceiling.
- **Subtype — the precision gain costs recall on *every* subtype, most on
  stop-lines.** Per-subtype recall *regresses* from LiDAR-only to the full system:
  stop-lines **D0 0.837 → G2 0.690 → H0 0.622**; lane-lines 0.756 → 0.728 → 0.702;
  other road markings 0.712 → 0.591 → 0.561. This is the per-subtype face of the
  val→test recall drop (§3.4): the calibration win is a *uniform* recall trade,
  heaviest on the rare high-intensity stop-lines. **A real, citable limitation.**
- **RGB-validity gap — present but partly confounded.** IoU (camera-visible minus
  not): E0 0.037, F0 0.097, **G2 0.090**, H0 0.089 — the calibrated RGB models do
  clearly better where the camera sees. *Caveat:* rgb_invalid regions are also
  farther / edge-of-frame (geometrically harder), so this is an **upper bound** on
  RGB benefit, not clean attribution (→ §5 #1, the D0-rgb_valid re-run).
- **Night (065) — RGB helps; jitter does not.** IoU: D0 0.455 → E0 0.473 → F0 0.507
  → **G2 0.531** → H0 0.514. RGB improves the night sequence monotonically (via
  precision; recall falls 0.806→0.662), countering the "camera fails in low light"
  prior. H0 (jitter, partly night-motivated) is *below* G2 at night — the
  augmentation gave **no** night-specific test benefit. One sequence ⇒ case study
  (§6.7).

### 3.9 What this evaluation proves / does not prove

**Proves (re-derivable from committed matrices):**
- On the held-out test set, the full RGB+Lovász system (G2) is the best model
  (marking IoU 0.517 ± 0.0005), beating the LiDAR-only baseline by +0.102, and the
  gain is **precision/calibration** (road→marking FPs −55 %, `pred/true` 1.54→1.00),
  not improved detection (recall actually falls).
- Raw RGB alone (D0→E0, clean ablation) does **not** help and plausibly hurts
  (−0.015 IoU); RGB only becomes useful with calibration + Lovász.
- The result generalizes (full-val→full-test IoU gap 0.020); the val→test change is
  a recall drop offset by a precision rise, fully reconciled across protocols.
- RGB contributes specifically where the camera sees (rgb_valid IoU 0.524 vs 0.434)
  and does not collapse at night (G2 065 0.531 > D0 0.455).

**Does not prove:**
- **Not a clean RGB attribution** for the full +0.102 (D0→G2 mixes RGB + Lovász +
  features + epochs); the clean RGB step (D0→E0) is negative. The rgb-valid split is
  suggestive, not a channel ablation.
- **Not training-seed-robust** — single training seed; G2 vs H0 is a tie, and
  E0/F0/D0 single-seed deltas are directional only.
- **Not a general night/low-light conclusion** — one night sequence.
- **Not per-point logit voting** — patch-accumulated (backed by the agreement/voting
  diagnostics, but disclosed).
- **Does not solve long range** (60 m+ IoU 0.35) or the "other road markings"
  subtype (0.591).

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
| `summary.json` | headline metrics block + `coverage`, `has_rgb`, `feature_mode`, raw-remap audit, `voted_metrics`/`voted_confusion_matrix` (full only). Source of the single-number metrics. |
| `confusion_matrix.npy` | 3×3 (rows=true, cols=pred; road/marking/other), patch-accumulated. **The source of truth** — every metric re-derives from it. |
| `confusion_matrix_voted.npy` | *(full only)* per-point majority-voted 3×3 — the no-voting confirmation. |
| `per_sequence_metrics.csv` | all metrics per test sequence (the macro / per-scene breakdown, incl. night 065). |
| `distance_bucket_metrics.csv` | all metrics per range bucket (0–10/10–20/20–30/30–40/40–60/60 m+). |
| `rgb_valid_stratified_metrics.csv` | *(RGB models only)* metrics split by rgb-valid vs rgb-invalid (camera-visible or not); `all` row = the full-coverage totals. |
| `raw_subtype_rgb_stratified_metrics.csv` | recall per marking subtype (raw 8/9/10) × rgb strata; `all` stratum = subtype recall (precision = 1 by construction). |
| `group_feature_summary.csv` | per-outcome (marking_tp, road_to_marking, …) feature profile: intensity, R/G/B, range. The **brightness/intensity fingerprint** source. |
| `frame_error_summary.csv` | per-**patch** row (seq_id, frame_idx, all metrics). Aggregated → per-frame for the qualitative/video pickers and the frame-IoU histogram. |
| `coverage_count_by_class.csv` | *(full only)* per-class coverage-count distribution (median/p10/p90/max) + prediction-agreement rate. |
| `top_frames_by_road_to_marking.csv` / `…_marking_to_road.csv` | worst frames by false-positive / miss count. |
| `README.md` | per-run provenance (checkpoint, steps, seed). |

### 4.2 Aggregated outputs — `results/`
| path | what it is | produced by |
| --- | --- | --- |
| `test_master_table.csv` | all 5 models × all metrics; the two-validation-column headline (§2). | `build_comparison.py` |
| `README.md` | generated provenance + master table in markdown. | `build_comparison.py` |
| `TEST_PLAN.md` | the protocol decision/justification (companion to this doc). | hand-written |
| `comparisons/crosscheck_sampled_vs_full.csv` | sampled vs full test IoU (G2/H0). | `build_comparison.py` |
| `comparisons/rgb_effect__D0_vs_E0/` | `test_metrics.csv` + `deltas_vs_baseline.csv` — the clean RGB on/off step (§3.2). | `build_comparison.py` |
| `comparisons/system__D0_vs_G2_vs_H0/` | LiDAR baseline vs full systems (metrics + deltas; **seed 42**). | `build_comparison.py` |
| `comparisons/shortcut__G2_vs_H0/rgb_valid_gap.csv` | rgb-valid vs invalid precision + `pred/true` + over-prediction gap (§3.6). | `build_comparison.py` |
| `comparisons/shortcut__G2_vs_H0/brightness_fingerprint.csv` | RGB + intensity profile of road-TP / road→marking-FP / marking-TP (§3.6). | `build_comparison.py` |
| `comparisons/d0_vs_g2/*.csv` | focused D0-vs-G2 metrics, per-sequence, distance (**seed 42**). | `compare_d0_g2.py` |
| `comparisons/video_sequence_suitability_G2.csv` | per-sequence "is this a good video?" stats (marking-frame fraction, IoU spread, consistency). | `pick_video_sequences.py` |

### 4.3 Figures (all **22** result figures; every one listed below = every one on disk)
| set | files | produced by | coverage |
| --- | --- | --- | --- |
| `results/G2_final/` | `fig_per_class_iou`, `fig_marking_metrics`, `fig_confusion` (counts), `fig_by_distance`, `fig_by_sequence`, `fig_by_subtype`, `fig_by_rgb_validity`, `fig_frame_iou_hist` | `model_results_figures.py` | **FULL** (G2 standalone) |
| `comparisons/d0_vs_g2/` | `fig_metrics_bars`, `fig_overprediction`, `fig_confusion_D0/G2`, `fig_error_breakdown`, `fig_per_sequence`, `fig_distance`, `fig_val_vs_test` | `compare_d0_g2.py` | **FULL** (D0 vs G2, seed 42) |
| `comparisons/plots/` | `progression_marking_iou_test`, `marking_metrics_test`, `per_class_iou_test`, `distance_marking_iou_test`, `shortcut_overprediction_g2_vs_h0`, `val_vs_test_marking_iou` | `build_comparison.py` | **FULL** — except `val_vs_test_marking_iou` **mixes** (E0/F0 bars are sampled selection-val; D0/G2/H0 are full-val). Prefer `d0_vs_g2/fig_val_vs_test`. |

**Colour rule (all figures):** classes blue=road / red=marking / green=other;
marking-metric family IoU=red / precision=orange / recall=purple / F1=gold
(single source: `_suite_style.py`).

### 4.4 Qualitative media (server only — gitignored, not in the repo)
| path | what it is | produced by |
| --- | --- | --- |
| `results/qualitative/<category>/<seq>_f<frame>/<model>/<mode>/*.png` | front-camera GT / prediction / error overlays (opaque), D0 & G2, for auto-picked best / worst / overpredict / night-065 frames. | `run_qualitative.py` → `04_prediction_images.py` |
| `results/videos/<seq>/<model>/images/*.png` + `<seq>_<model>_pred.mp4` | full-sequence prediction fly-through videos (G2; 101, 002, 065 rendered). | `make_sequence_video.py` → `04_prediction_images.py` |

### 4.4b Methodology / Chapter-3 figures (OUT OF SCOPE here — listed for completeness)
`results/chapter3/` exists but is **not** part of this test-evaluation record: these
are dataset / task-illustration figures for the **methods chapter**, not results,
and are **uncommitted** (local only). Listed so nothing is orphaned.

| file | what it is |
| --- | --- |
| `figures/fig_task_camera.png`, `fig_task_camera_projection.png`, `fig_task_lidar_intensity.png`, `fig_task_lidar_labels.png` | the task illustration (front camera, camera→LiDAR projection, LiDAR intensity, LiDAR GT labels). |
| `figures/fig_intensity_distance.png` | LiDAR intensity vs distance (the input-information motivation). |
| `01_frame011_16_front_camera.png`, `03_frame011_16_gt_projection.png`, `frame11-16-original.png`, `frame-11-16-ground-truth.png` | a single worked example frame (camera / GT projection / raw / GT). |
| scripts `make_chapter3_figures.py`, `frame_figures.py`, `make_intensity_distance_figure.py`, `view_frame.py` | generate the above. |

### 4.5 Code — `results/test_suite/`
| script | role |
| --- | --- |
| `_sampled_error_engine.py` | **the engine.** Runs sampled or full-coverage inference for one checkpoint/split, writes the per-model files. `--coverage {sampled,full}` + the no-voting diagnostics. |
| `run_test_all.py` | driver: runs the engine for all models × splits × coverages × seeds (the experiment matrix); resume-safe; `--full-val`. |
| `build_comparison.py` | aggregates per-model outputs → master table + comparison folders + `comparisons/plots/` (CPU/file-only; re-derives every metric from the matrices). |
| `compare_d0_g2.py` | focused D0-vs-G2 figure/table set (seed 42). |
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
   suggestive (G2 better where the camera sees), but to *prove* the gain comes from
   RGB, re-run **D0 with the rgb_valid mask recorded** (a small engine change —
   compute geometric camera-validity for D0 without using RGB as a feature) and
   compare D0-vs-G2 *within* rgb_valid vs rgb_invalid. → engine's `rgb_valid`
   handling + a 1-run D0 re-eval.
2. **Distance degradation.** Marking IoU falls sharply beyond 40 m, and the
   mechanism flips (recall-limited at 40–60 m, precision-limited at 60 m+). Worth a
   per-subtype × distance cut. → `distance_bucket_metrics.csv` + per-patch data in
   `frame_error_summary.csv`.
3. **The intensity-not-brightness residual confusion (§3.6) — RESOLVED.** On test,
   G2's residual false markings are intensity-elevated road, not bright-RGB road —
   *opposite* to the validation fingerprint. The mean-vs-median worry is now closed
   (test FP brightness median ~0.367 < road ~0.465; confirmed by both statistics).
   Remaining optional depth: the same probe *per distance/sequence*. →
   `group_feature_summary.csv`.
4. **Hardest scenes (001 vs 002) fail differently** — 001 over-predicts, 002
   under-detects. The qualitative overlays / 002 video show why. → `results/videos/
   002/…`, `qualitative/worst/…`.
5. **Per-point logit voting** (currently a disclosed limitation). The voted CM
   already agrees with the headline (+0.009); a full voting engine would make it
   exact. → `confusion_matrix_voted.npy` is the starting point.
6. **Single training seed.** The one statistical weakness; the only fix is
   retraining G2/H0 with more seeds (out of scope now, noted as a limitation).

**For any deeper question:** the relevant source is almost always (a) the
`confusion_matrix.npy` of the run in question (everything re-derives from it),
(b) the matching `*_metrics.csv` for a stratified cut, or (c)
`frame_error_summary.csv` for per-frame/per-patch detail. The script that made a
file is in §4.5; re-run it to regenerate or extend.

---

## 6. Caveats / limitations (carry into the writeup)
1. **Single training seed (42)**; ~0.008 marking-IoU noise floor → G2≈H0 is a tie,
   and E0/F0/D0 single-seed deltas are directional. Evaluation seeds fix only
   sampling variance.
2. **Patch-accumulated confusion, not per-point logit voting** — backed by the
   coverage/agreement diagnostics and the voted-CM check (+0.009), but disclosed.
3. **Select-on-(sampled)-validation, report-on-full-coverage-test** — clean
   workflow, two named validation columns; selection-val is *not* a result.
4. **D0→G2 is system-level, not a clean RGB ablation** — the clean RGB step is
   D0→E0 (and it is negative).
5. **H0 differs from G2 in training regime** (H0 100 ep from scratch; G2 a resume
   chain) in addition to jitter — so even H0's precision shift is not jitter-only.
6. **RGB-shortcut evidence is correlational** (group-level fingerprint + rgb-valid
   stratification; no causal channel ablation).
7. **Night = one sequence (065)** — a case study, not a general low-light claim.
8. Metrics are over the **grid-subsampled evaluation cloud**, not raw PandaSet
   points; no projection back to the full cloud.
9. **E0/F0 have no full-coverage validation run** — their `gap` column is
   sel-val − full-test (not protocol-matched); use only G2/H0 (and D0) for
   generalization-gap statements.

---

## 7. Contradictions / reconciliation points (resolved)

| Point | Resolution |
| --- | --- |
| **D0→G2 gain = +0.102 (table/§0) vs +0.103 (`d0_vs_g2`, §3.1)** | Different seed sets: the master table is the **3-seed mean** (G2 0.5170); `comparisons/d0_vs_g2/` and `system__D0_vs_G2_vs_H0/` are **seed 42** (G2 0.5177). Difference 0.0007 = eval-seed noise. Headline uses the 3-seed mean; figures carry the seed-42 value. Noted in §2. |
| **§3.6: on test, road→marking FPs are *darker* than road (mean 0.389 < 0.404, median 0.367 < 0.465); on Milestone-G validation they were *brighter* than road (median 0.478 > 0.392).** | **The val→test shift is real — one explanation only: validation and the held-out test are different scenes**, and on the unseen test set the RGB-brightness shortcut has weakened, leaving the residual road↔marking confusion **LiDAR-intensity-driven** (FP intensity 31 vs road 25), not brightness-driven. *Hypothesis tested and ruled out:* that the gap was merely a statistic mismatch (validation quoted a brightness **median** 0.478, the original test fingerprint a **mean** 0.389) — recomputing the test **median** (0.367, also below road 0.465) shows the FPs are darker by *both* statistics, so this is **not** a mean-vs-median artifact. |
| **"IoU is preserved on test" (a loose reading) vs IoU drops 0.020 val→test** | IoU **drops** by 0.020 (full-val 0.537 → full-test 0.517). It is *largely* preserved only relative to the much larger recall drop (−0.106), because precision rises (+0.053). Stated precisely in §3.4 — never claim recall and IoU both hold. |
| **"voted ≈ accumulated" vs voted 0.527 > accumulated 0.518** | They differ by **+0.009** (seed 42), small and in the expected direction (over-covered boundary points slightly noisier under hard per-patch argmax). The voted CM **confirms** the headline rather than equalling it exactly; it is a check, not a second headline (§3.7, `TEST_PLAN.md` §1.5). |
| **"RGB collapses at night" (a plausible prior) vs G2 065 0.531 > D0 0.455** | On the single night sequence the RGB model **beats** LiDAR-only and exceeds its own micro-average — no night collapse for the final model. But one sequence ⇒ case study only (§3.5, `TEST_PLAN.md` §13). |
| **Earlier "test collapse" (G2 0.2456)** | A **sampler bug**, not a real result — `--steps` capped the spatially-regular sampler to the first ~50 frames (sequences 001+002, the two genuinely hardest). Discarded; root cause and fix in `TEST_PLAN.md` §3. The real G2 test IoU is 0.517. |
