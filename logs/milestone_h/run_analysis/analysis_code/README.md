# Milestone H Analysis Code

Analysis scripts for Milestone H live here.

Current scripts:

```text
g1_rgb_brightness_sensitivity.py
```

`g1_rgb_brightness_sensitivity.py` is an eval-only probe for frozen G1 epoch 27.
It scales model-input RGB channels in memory before forward pass:

```text
features[..., 4:7] = RGB
features[..., 7]   = rgb_valid
```

The script applies brightness scale only where `rgb_valid >= 0.5`, writes outputs
under `logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/`, and does
not mutate the G1 run directory.

Planned scripts, once H0 exists:

```text
h0_analysis.py
h0_sampled_error_analysis.py
plot_h0_analysis.py
run_h0_full_analysis.py
```

Before H0 training, this folder may also contain dataset/sampler validation
helpers if they are specific to H.
