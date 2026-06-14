# F0 Analysis Code

This folder contains the Milestone F0 analysis scripts.

Run the full server-side analysis with:

```bash
python logs/milestone_f/run_analysis/analysis_code/run_f0_full_analysis.py \
  --device cuda \
  --steps 2160 \
  --seed 42
```

The orchestrator runs the scripts in this order:

1. `f0_analysis.py`
   - Validates D0/E0/F0 artifacts.
   - Validates F0 config snapshot.
   - Writes run-level comparison CSVs and `analysis.md`.

2. `f0_rgb_valid_coverage.py`
   - Measures validation-set RGB-valid coverage.
   - Does not load the model checkpoint.

3. `f0_sampled_error_analysis.py`
   - Runs a fresh sampled validation inference pass from F0 epoch 13.
   - Writes RGB-valid, sequence, distance, raw subtype, and outcome-group CSVs.

4. `plot_f0_rgb_soft_weights_analysis.py`
   - Generates thesis-oriented plots.
   - Writes `f0_conclusions.md`.

All outputs are written under:

```text
logs/milestone_f/run_analysis/F0_rgb_soft_weights/
```

The sampled inference pass is not the exact validation sample seen during epoch
13 training. It is a new sampled pass with the supplied seed and step count.
