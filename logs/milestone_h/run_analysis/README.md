# Milestone H Run Analysis

Per-run and pre-run H analysis outputs go here.

Current pre-run analysis:

```text
g1_rgb_brightness_sensitivity/
```

This is an eval-only probe using the frozen G1 epoch-27 checkpoint. It scales RGB
inside the analysis batch to test absolute brightness sensitivity. It does not
train, change cache, change validation/test data, or write into the G1 run
directory.

Decision from the probe:

```text
H0_rgb_jitter is justified:
train-only brightness 0.85-1.15, contrast 0.90-1.10
```

Expected H0 layout once a run exists:

```text
logs/milestone_h/run_analysis/
  analysis_code/
  H0_run_name/
    analysis.md
    h0_conclusions.md
    plots/
    sampled_error_analysis_epochXX/
```

Do not mix H outputs into Milestone G directories.
