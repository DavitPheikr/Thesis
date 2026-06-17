# H0 RGB Jitter Implementation Notes

H0 implements one controlled change from G1:

```text
train-only RGB brightness/contrast jitter
```

Everything else remains G1-identical:

- no z-crop
- no sampler change
- no RGB feature redesign
- no RGB dropout
- no loss or class-weight change
- no scheduler or optimizer change
- unchanged validation/test preprocessing

## Source Evidence

The H0 decision is based on:

```text
logs/milestone_h/run_analysis/g1_rgb_brightness_sensitivity/
```

The eval-only frozen G1 probe showed monotonic sensitivity to RGB brightness:

```text
scale 0.85: road->marking FP 454,798 (-10.32% vs scale 1.00)
scale 1.00: road->marking FP 507,112
scale 1.15: road->marking FP 556,667 (+9.77% vs scale 1.00)
```

Darkening improved precision but lowered recall, so fixed darkening is not used.
H0 instead trains with mild random jitter and evaluates on unchanged validation
data.

## Implementation

Jitter is implemented in the shared training runner and is config-gated by:

```yaml
pipeline.rgb_jitter.enabled: true
```

When disabled or absent, previous D/E/F/G behavior is unchanged.

Model-input feature layout:

```text
[x, y, z, intensity, r, g, b, rgb_valid]
```

Jitter operation:

- runs only in the training loop
- happens after the batch is moved to device and before `model(inputs["data"])`
- modifies `inputs["data"]["features"]` in memory only
- uses RGB columns `features[..., 4:7]`
- uses `rgb_valid` column `features[..., 7]`
- applies only where `rgb_valid >= 0.5`
- leaves x/y/z/intensity/labels/rgb_valid unchanged
- clips RGB to `[0, 1]`
- samples one brightness factor and one contrast factor per batch sample

H0 config:

```text
logs/milestone_h/configs/h0_rgb_jitter.yml
```

H0 wrapper:

```text
tools/train_milestone_h.py
```

Synthetic jitter test:

```text
tools/test_rgb_jitter.py
```

## Recommended Full Run

```bash
python tools/train_milestone_h.py \
  --config logs/milestone_h/configs/h0_rgb_jitter.yml \
  --run-name H0_rgb_jitter \
  --epochs 35 \
  --seed 42 \
  --save-ckpt-freq 1 \
  --device cuda \
  --pin-memory \
  --force
```

Run detached on server with output redirected to:

```text
logs/milestone_h/launch_logs/H0_rgb_jitter_fresh_35.stdout.log
```
