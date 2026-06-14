# G0 Analysis Code

Milestone G0 analysis scripts. G0 = F0 with the loss changed to
`weighted_CE + lovasz_lambda * Lovasz-Softmax` (lovasz_lambda = 0.5); everything
else matches F0.

Run the full server-side analysis (after the G0 run finishes) with:

```bash
python logs/milestone_g/run_analysis/analysis_code/run_g0_full_analysis.py \
  --device cuda \
  --steps 2160 \
  --seed 42
```

The orchestrator runs the scripts in this order:

1. `g0_analysis.py`
   - Validates D0/E0/F0/G0 artifacts (counts internally consistent).
   - Validates the G0 config snapshot is **F0 + loss only** (the fairness proof).
   - **Fails loudly** if `loss_components.csv` is missing/empty/inconsistent, and
     tolerance-checks `total == CE + lambda*Lovasz` and
     `eval_history.val_loss == val_total_loss`.
   - Writes `summary.csv`, `confusion_breakdown.csv`, `pred_true_ratio.csv`,
     `artifact_counts.csv`, `analysis.md`.

2. `g0_sampled_error_analysis.py`
   - Fresh sampled validation inference from the G0 best checkpoint.
   - **Best epoch is discovered dynamically** (max raw marking IoU); override
     with `--checkpoint`.
   - Loss-agnostic (predictions are argmax of scores). Writes RGB-valid,
     sequence, distance, raw-subtype, and outcome-group CSVs under
     `sampled_error_analysis_epoch{best}/`.

3. `g0_loss_component_analysis.py`
   - The G-specific stage. Joins `loss_components.csv` and `eval_history.csv`.
   - Characterizes the Lovasz term: scale, stability, IoU-alignment vs CE.
   - Tolerance-checks composition again. Writes `loss_component_summary.csv`,
     `loss_alignment.csv` / `.json`, `loss_component_report.md`, and the
     CE/Lovasz/total plots (incl. the fair F0-vs-G0 CE overlay).

4. `plot_g0_rgb_lovasz_analysis.py`
   - Four-way D0/E0/F0/G0 plots, F0-vs-G0 curves, best-to-final drift, and the
     sampled-error plots. Writes `plots/` and `g0_conclusions.md`.

All outputs are written under:

```text
logs/milestone_g/run_analysis/G0_rgb_lovasz/
```

## RGB-shortcut fingerprint (cross-run, analysis-only)

`rgb_shortcut_fingerprint.py` tests whether residual road->marking false
positives are associated with RGB **brightness/luminance** rather than a specific
marking **color**. It reuses the existing `group_feature_summary.csv` and
`rgb_valid_stratified_metrics.csv` (no inference, no training change). Linear
metrics (brightness, luminance, warmth, per-channel means) are exact group means;
`saturation_proxy`/max/min are flagged aggregate approximations; the
brightness-vs-intensity scatter is group-level (per-point arrays are not saved).

Wording discipline: results are phrased as evidence predictions are *associated
with* a luminance-based RGB shortcut, never as proof the model internally uses
brightness.

```bash
# F0 now (reuses committed F0 sampled outputs):
python logs/milestone_g/run_analysis/analysis_code/rgb_shortcut_fingerprint.py --preset f0
# G0 after it finishes:
python logs/milestone_g/run_analysis/analysis_code/rgb_shortcut_fingerprint.py --preset g0
# F0-vs-G0 symptom-vs-mechanism comparison:
python logs/milestone_g/run_analysis/analysis_code/rgb_shortcut_fingerprint.py --preset f0_vs_g0
```

Outputs per run go to `<run_analysis>/<RUN>/rgb_shortcut_analysis/`
(`rgb_shortcut_fingerprint.csv`, `rgb_shortcut_questions.json`,
`rgb_shortcut_conclusions.md`, `plots/`). Compare mode adds
`rgb_shortcut_compare.csv/.png` and `rgb_shortcut_compare_conclusions.md`.

## Post-G0 decision diagnostics (G1 planning)

Two diagnostics decide whether G0 needs a follow-up run and which one:

- `scheduler_replay.py` (**zero-GPU**, runs anywhere): replays G0's observed
  marking-IoU curve through the real PyTorch `ReduceLROnPlateau` for patience
  6/5/4/3 (with the runner's 3-epoch metric smoothing) to find when LR would
  drop. Writes `scheduler_replay.csv` / `.md` under the G0 run-analysis dir.
- `g0_bias_sweep.py` (**server / GPU**): shifts the marking logit by
  `b ∈ [-0.5..+0.5]` before argmax on the G0 best checkpoint, in a single
  inference pass (re-argmax per bias in numpy — no re-inference). Tells us if G0
  is well-calibrated (best `b≈0`), too conservative (positive `b` best →
  `lambda=0.35` justified), or still overpredicting (negative `b` best). It is a
  **diagnostic only** — a bias-tuned IoU is never reported as a result. Acts only
  if the best bias beats `b=0` by ≥ ~0.008–0.010 IoU (single-seed noise gate).

```bash
python logs/milestone_g/run_analysis/analysis_code/scheduler_replay.py
python logs/milestone_g/run_analysis/analysis_code/g0_bias_sweep.py --device cuda --steps 2160 --seed 42
```

Outputs: `<G0 run_analysis>/scheduler_replay.{csv,md}` and
`<G0 run_analysis>/bias_sweep/bias_sweep_{metrics.csv,summary.md}`.

## Loss-comparison rule (important)

For G0, `eval_history` `train_loss`/`val_loss` are the **TOTAL** loss
(CE + lambda*Lovasz). The only fair loss-vs-loss comparison against F0 is F0
`val_loss` (pure CE) vs G0 `val_ce` (CE component). Never compare F0 `val_loss`
to G0 total loss.

## Notes

- The sampled inference pass is not the exact validation sample seen during the
  best epoch; it is a fresh sampled pass with the supplied seed and step count
  (seed 42, steps 2160, matching F0 for comparability).
- Scripts read files only inside their `main()`; they can be created before the
  G0 run finishes and run afterward.
