# D0 Run Analysis Code

This folder contains the Milestone D analysis scripts for the run:

```text
logs/milestone_d/runs/D0_weighted_ce_25ep/
```

Run-level plots:

```bash
./panda/bin/python logs/milestone_d/run_analysis/D0_weighted_ce_25ep/analysis_code/plot_d0_run.py
```

Sampled epoch-18 error analysis, preferably on the GPU server:

```bash
python logs/milestone_d/run_analysis/D0_weighted_ce_25ep/analysis_code/analyze_d0_marking_errors.py \
  --steps 2160 \
  --device cuda

python logs/milestone_d/run_analysis/D0_weighted_ce_25ep/analysis_code/plot_d0_marking_error_analysis.py
```

Validation rules:

- class order is treated as `0 road`, `1 marking`, `2 other` after Open3D ignore-label filtering
- CSV `lane_*` metrics are interpreted as `marking_*` for Milestone D
- best D0 checkpoint is selected from `eval_history.csv` by raw marking IoU, not validation loss
- sampled error analysis is explicitly a new sampled inference pass, not the exact epoch-18 validation sample from training
- raw subtype summaries are only generated from raw labels carried through an analysis-only transform path
- all outputs stay under `logs/milestone_d/run_analysis/D0_weighted_ce_25ep/`
