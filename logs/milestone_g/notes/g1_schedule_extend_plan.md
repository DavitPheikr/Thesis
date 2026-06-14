# G1 Schedule Extension Plan

G1_schedule_extend is an optional continuation experiment, not a replacement for
the official G0 result.

## Purpose

G0 is closed with epoch 18 as the final thesis model. The post-hoc scheduler
replay showed that the original G0 `ReduceLROnPlateau` scheduler did not reduce
the learning rate within the 25-epoch budget. G1 tests one narrow question:

```text
Did G0 simply stop before the unchanged patience-6 scheduler got a lower-LR
refinement phase?
```

## Definition

G1 resumes from:

```text
logs/milestone_g/runs/G1_schedule_extend/checkpoints/ckpt_epoch_00025.pth
```

That checkpoint is created by copying the completed server-side G0 run directory:

```text
logs/milestone_g/runs/G0_rgb_lovasz
```

to:

```text
logs/milestone_g/runs/G1_schedule_extend
```

Then the runner resumes the copied G1 directory with target total epochs set to
35.

## Invariants

Training-affecting settings are intentionally identical to G0:

- same model, RGB features, cache settings, sampler, class weights, and seed
- same loss: `weighted_CE + 0.5 * Lovasz-Softmax`
- same AdamW optimizer settings
- same `ReduceLROnPlateau` settings, including `patience: 6`
- same batch size, point count, neighborhood count, augmentations, and split

The only intended experimental change is:

```text
target total epochs: 25 -> 35
```

The config file `logs/milestone_g/configs/g1_schedule_extend.yml` copies G0's
training settings. Only non-training output labels such as `test_result_folder`
are pointed at G1.

## Safety Rules

- Do not write into `logs/milestone_g/runs/G0_rgb_lovasz`.
- Do not use `--force` with `--resume-latest`.
- Do not delete a pre-existing G1 directory without inspecting it first.
- The G1 directory should be copied from G0 before resume, so the resume command
  appends epochs 26-35 to G1 only.
- The original G0 run and analysis outputs remain the official result unless G1
  produces a clearly defensible improvement.

## Expected Resume Behavior

The shared runner stores and restores:

- model weights
- optimizer state
- scheduler state
- scheduler metric history
- Python, NumPy, PyTorch, and CUDA RNG states where available

With `--epochs 35`, the runner sets `max_epoch` to 34 and continues from the
completed epoch stored in the checkpoint. A copied `ckpt_epoch_00025.pth` should
therefore resume at epoch 26 and write new artifacts through epoch 35 inside the
G1 run directory.

## Interpretation

This experiment is schedule polish. It does not change the G0 result and it does
not test a new loss, class weight, architecture, sampler, or RGB preprocessing
choice. If it helps, it supports the narrow claim that G0 benefited from a longer
unchanged scheduler budget. If it does not help, G0 remains the final model.
