# Class Weight Decision

Generated: 2026-05-05

## Decision

Milestone C C0 keeps the Open3D-native measured-count class-weight policy:

```yaml
dataset:
  class_weights: [119562394.0, 2098182.0, 176504630.0]
```

Class order:

```text
road, lane, other
```

The ignored label `0` is omitted from this list because Open3D filters ignored labels before computing loss.

## Why This Policy

PandaSet lane points are rare in the training split:

```text
road  = 119562394
lane  =   2098182
other = 176504630
```

The lane class is therefore the main minority class. C0 should already include loss-side imbalance handling before any feature or sampler ablations.

This repository previously discovered that local Open3D `SemSegLoss` does not consume direct final CE weights as-is. It transforms `dataset.cfg.class_weights` internally through:

```text
open3d._ml3d.datasets.utils.DataProcessing.get_class_weights(...)
```

Therefore, the config intentionally stores count-like values, not direct inverse-frequency weights.

## Server Verification

The following server check was run in the `panda312` environment using `logs/milestone_c/configs/c0_server_random.yml`:

```text
dataset_cfg_class_weights [119562394.0, 2098182.0, 176504630.0]
expected_effective_ce_weights [2.3753318786621094, 36.98638153076172, 1.6340690851211548]
actual_ce_weights [2.3753318786621094, 36.98638153076172, 1.6340690851211548]
```

The effective runtime loss weights are:

```text
road  = 2.3753
lane  = 36.9864
other = 1.6341
```

Conclusion:

```text
class_weight_policy_ok True
loss_uses_nonuniform_weights True
lane_weight_active True
```

## Caveat For Future Agents

Do not substitute `raw_inverse_frequency` or `sqrt_inverse_frequency` lists directly into `dataset.class_weights` unless the loss path is changed. In this Open3D runtime, those lists would be transformed again, so the final CE weights would not equal the intended direct weights.

If future experiments require direct CE weights, create an explicit loss mode or patch/wrap `SemSegLoss` so the direct tensor is passed to `torch.nn.CrossEntropyLoss(weight=...)` without Open3D's extra transform.
