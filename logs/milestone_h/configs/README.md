# Milestone H Configs

Milestone H configs live here.

Current config:

```text
h0_rgb_jitter.yml
```

H0 is a controlled G1 follow-up:

```text
H0_rgb_jitter = G1 + train-only RGB brightness/contrast jitter
```

The config intentionally keeps G1's architecture, input layout, loss, class
weights, sampler, optimizer, scheduler, split, and RGB cache unchanged. The only
training-affecting change is:

```yaml
pipeline:
  rgb_jitter:
    enabled: true
    train_only: true
    brightness_min: 0.85
    brightness_max: 1.15
    contrast_min: 0.90
    contrast_max: 1.10
    rgb_valid_threshold: 0.5
```

Validation and test data are not jittered.
