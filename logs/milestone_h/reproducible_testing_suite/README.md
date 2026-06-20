# Reproducible Testing Suite

One self-contained location for the **entire test/analysis pipeline** of a training
run — inference, metrics CSVs, plots, per-frame prediction images, and the
RGB-shortcut fingerprint. It is currently targeted at **G1 (`G1_schedule_extend`,
epoch 27)**, and is designed to be **copied and retargeted** at a future run (G2,
H0, …) by editing a single file.

These are **cleaned, self-contained copies** of the scripts under
`logs/milestone_g/run_analysis/analysis_code/` (originals untouched). Changes vs the
originals: a robust repo-root finder (works from any location), the G1↔G0 engine
import made local, run-specific paths centralised in `_suite_paths.py`, and the
`group_feature_summary.csv` (needed by the RGB-shortcut step) no longer deleted.

> **Where it runs:** steps **02** (sampled inference), **04** (prediction images)
> and **06** (bias sweep) run the model and need the run's **checkpoint + a GPU** —
> run them on the server. Steps **01, 03, 05, 07** are CPU/file-only and run anywhere.

---

## File map (run order)

| # | file | what it does | produces | needs GPU? |
|---|---|---|---|---|
| — | `_suite_paths.py` | **the one file to edit** — repo root + per-run config | — | no |
| — | `_suite_style.py` | shared semantic palette + thesis typography (imported by all plot steps) | — | no |
| — | `run_full_suite.py` | orchestrator (runs the steps below) | — | — |
| 1 | `01_run_analysis.py` | run-level metrics + integrity; D0…G1 comparison | `summary.csv`, `confusion_breakdown.csv`, `best_vs_final.csv`, `lr_events.csv`, `loss_alignment.csv`, `pred_true_ratio_by_epoch.csv`, `analysis.md`, … | no |
| 2 | `02_sampled_error_analysis.py` | **fresh sampled inference** from the best checkpoint; error breakdowns | `sampled_error_analysis_epoch<best>/` → `summary.json`, `confusion_matrix.npy`, `rgb_valid_stratified_metrics.csv`, `distance_bucket_metrics.csv`, `per_sequence_metrics.csv`, `raw_subtype_rgb_stratified_metrics.csv`, `group_feature_summary.csv`, `top_frames_*` | **yes** |
| — | `_sampled_error_engine.py` | inference engine imported by step 02 (don't run directly) | — | — |
| 3 | `03_plots.py` | all training-curve / confusion / calibration plots; candidate-vs-baseline comparison | `plots/*.png` (incl. `comparison_marking_curves.png`) + `<candidate>_conclusions.md` | no |
| 4 | `04_prediction_images.py` | per-frame front-camera overlays (pred / gt / error) | `front_camera_predictions/<RUN>/<seq>/{pred,gt,error}/*.png` | **yes** |
| 5 | `05_rgb_shortcut.py` | bright-road RGB-shortcut fingerprint (reads step-02 CSVs) | `rgb_shortcut_analysis/` → fingerprint CSV, conclusions, plots | no |
| 6 | `06_bias_sweep.py` | **diagnostic** marking-logit operating-point sweep (one extra inference pass; re-argmax per bias) | `bias_sweep/bias_sweep_metrics.csv`, `bias_sweep_summary.md`, `bias_sweep_operating_point.png` | **yes** |
| 7 | `07_stratified_plots.py` | per-sequence + per-subtype marking plots from step-02 CSVs | `plots/per_sequence_marking_performance.png`, `plots/raw_subtype_marking_recall.png` | no |

> **06 is diagnostic only.** It re-argmaxes the saved scores with a marking-logit
> bias `b` (`pred = argmax(scores + b·e_marking)`) to map the precision/recall
> operating curve. The **official evaluation stays argmax with `b = 0`** (steps
> 01–03). Report it as an operating-point analysis, never as the headline metric.

All outputs land under `<ANALYSIS_OUT>` (= `logs/<MILESTONE>/run_analysis/<RUN_NAME>/`),
except prediction images, which go to a sibling `front_camera_predictions/<RUN_NAME>/`.

---

## Quick start (on the server)

```bash
# core pipeline: run-level analysis + sampled inference + plots + stratified plots (01,02,03,07)
python logs/milestone_g/reproducible_testing_suite/run_full_suite.py --device cuda

# add per-frame prediction overlays (slow, ~1.5h for 5 sequences)
python logs/milestone_g/reproducible_testing_suite/run_full_suite.py --device cuda --images

# add the RGB-shortcut fingerprint (after step 02 has produced group_feature_summary.csv)
python logs/milestone_g/reproducible_testing_suite/run_full_suite.py --skip-core --rgb-shortcut

# add the marking-logit operating-point sweep (diagnostic only; official metrics unchanged)
python logs/milestone_g/reproducible_testing_suite/run_full_suite.py --skip-core --bias-sweep --device cuda
```

You can also run any step directly, e.g.
`python 02_sampled_error_analysis.py --device cuda --steps 2160 --seed 42`,
`python 07_stratified_plots.py`, or
`python 06_bias_sweep.py --device cuda --biases -0.5 -0.25 0 0.25 0.5`.

---

## How to retarget for a NEW run (G2, H0, …)

1. **Copy the whole directory** to the new milestone, e.g.
   `cp -r logs/milestone_g/reproducible_testing_suite logs/milestone_h/reproducible_testing_suite`.
2. **Edit `_suite_paths.py`** — the `PER-RUN CONFIG` block only:
   - `MILESTONE`, `RUN_NAME`, `CONFIG_FILE`, `CACHE_NAME` → point at the new run's
     `logs/<MILESTONE>/runs/<RUN_NAME>/`, `.../configs/<CONFIG_FILE>`,
     `.../run_analysis/<RUN_NAME>/`, `.../cache/<CACHE_NAME>`.
   - `CANDIDATE_LABEL` → the new run's short label (e.g. `"G2"`);
     `BASELINE_LABELS` → the list of run(s) to compare it against in the comparison
     plots, in draw order, primary first (e.g. `["F0", "D0"]`). One or many.
   - `LINEAGE` → add a row for the new run `(label, run-dir, has_lovasz, role,
     official_epoch)`. The candidate and **every** `BASELINE_LABELS` entry must be
     present with role `"best_final"`.
3. **Edit the one run-specific list** (only if needed):
   - `IMAGE_SEQUENCES` in `run_full_suite.py` → the validation sequences to render.

   The comparison labels, run colours (candidate = teal, baselines cycle through
   purple/brown/…), plot titles, CSV/PNG names and the `<candidate>_conclusions.md`
   filename all derive from `CANDIDATE_LABEL` / `BASELINE_LABELS` automatically — no
   per-run edits to `01`/`03`. Example: `CANDIDATE_LABEL="G2"`,
   `BASELINE_LABELS=["F0", "D0"]` gives a G2-vs-F0-vs-D0 comparison.
4. **Smoke-test once** on the server: `python run_full_suite.py --device cuda` and
   confirm `suite_status PASS` + the expected files appear under `<ANALYSIS_OUT>`.

Nothing else hard-codes the run; the repo root is found automatically (it looks for
the parent dir containing `src/thesis_pipeline/` and `logs/`), so the suite works
wherever you place it.

---

## Plot style & colour consistency

All figures are thesis-ready and **mutually consistent**: every plotting step
imports [`_suite_style.py`](./_suite_style.py), the single source of truth for
colours and typography. The rule it enforces is *the same entity is the same
colour in every plot*:

- **classes** — road = blue, marking = orange, other = green;
- **runs** — candidate (the run under analysis) = teal (thick); baselines cycle
  purple → brown → grey in `BASELINE_LABELS` order;
- **marking metrics** — IoU = orange, precision = blue, recall = green, F1 = gold
  (same mapping in the over-epochs, by-distance and bias-sweep figures);
- **calibration** — predicted/true marking ratio = indigo everywhere, with a black
  dotted "calibrated (1.0)" reference;
- **RGB validity** — all = grey, rgb_valid = blue, rgb_invalid = orange;
- **error flows (05)** — road→marking FP owns the suite's single red.

Typography (titles/labels/ticks/legends, DPI, opaque legend boxes that always sit
above the data so no line crosses them) is set once by `S.apply()`. To restyle the
whole suite, edit `_suite_style.py` only — never hard-code a colour in a step.

## Notes / caveats

- Re-running **overwrites** the prior outputs under `<ANALYSIS_OUT>` (deterministic:
  seed 42, fixed steps + checkpoint → same numbers, regenerated). Use a throwaway
  `--out-dir` on step 02 if you want to avoid touching committed outputs.
- Step 02's sampled pass is a *fresh* inference sample (seed 42), not the exact
  training-time validation sample — it agrees with training metrics within ~0.008.
- Step 04 is geometric projection only (no occlusion reasoning) and is slow
  (loads raw frames per step). `front_camera_predictions/` is large — keep it
  gitignored.
- These scripts are copies; the originals in
  `logs/milestone_g/run_analysis/analysis_code/` are unchanged and were used to
  produce the committed G1 results.
- **Single-seed caveat.** All runs use seed 42, so the candidate-vs-baseline IoU
  deltas are a *controlled single-seed* comparison, not a variance estimate. The
  noise floor is ~0.008 IoU; don't claim an improvement smaller than that from IoU
  alone. Read the full rule in
  [`logs/milestone_g/notes/single_seed_limitation.md`](../notes/single_seed_limitation.md)
  before writing up step-03 numbers — rely on *consistent* movement across
  IoU / precision / recall / F1 / road→marking FP / pred-true / stratified /
  qualitative, not a single sub-noise IoU gap.
