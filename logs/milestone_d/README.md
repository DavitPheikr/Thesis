# Milestone D

Milestone D starts the road-marking segmentation direction.

## Label Definition

C0 remains unchanged:

```text
lane3:
  road  = PandaSet raw 7
  lane  = PandaSet raw 8
  other = all remaining non-ignored raw classes
```

Milestone D adds a new opt-in label mode:

```text
road_marking3:
  road    = PandaSet raw 7
  marking = PandaSet raw 8 + raw 9 + raw 10
  other   = all remaining non-ignored raw classes
```

Ignored raw labels stay:

```text
raw 1, 2, 3, 4 -> ignore label 0
```

The active numeric labels stay the same:

```text
0 = ignore
1 = road
2 = positive class
3 = other
```

For `road_marking3`, label `2` means marking, although existing C-stage metric
columns may still be named `lane_*` until metric/report naming is generalized.

## Current Step

Milestone D currently has the remapping foundation and the statistics script.

Done:

- added opt-in `label_mode: road_marking3`
- kept default `label_mode: lane3` for backward-compatible C0 behavior
- created Milestone D directories for configs, runs, reports, cache, and analysis
- added `tools/compute_milestone_d_training_statistics.py`
- added `tools/train_milestone_d.py`
- added `logs/milestone_d/configs/d0_weighted_ce.yml`

Not done yet:

- no training run
- no RGB projection/features
- no metric-column renaming

## Stage 2 Statistics / Weights

Stage 2 computes the training-split class counts and Open3D-compatible
count-style class weights for `road_marking3`.

Official full run:

```bash
./panda/bin/python tools/compute_milestone_d_training_statistics.py
```

Expected main output:

```text
logs/milestone_d/road_marking3_training_statistics.json
logs/milestone_d/reports/road_marking3_training_statistics.md
```

The config weight list for later D0 configs should come from:

```text
class_weights.sanity_run_recommended_list
```

That list remains count-like because local Open3D-ML transforms
`dataset.cfg.class_weights` internally before building `CrossEntropyLoss`.

Scratch verification example:

```bash
./panda/bin/python tools/compute_milestone_d_training_statistics.py \
  --out-dir /tmp/milestone_d_stats_check \
  --max-sequences 1 \
  --max-frames-per-sequence 2
```

The script refuses limited runs in the official output directory so partial
statistics cannot accidentally replace the real Milestone D artifact.

## D0 Weighted CE Runner

Current D0 uses weighted cross-entropy, not focal loss. The focal-loss patch is
deferred because C0 diagnostics showed that many lane-to-road errors have
road-like intensity values, making aggressive hard-example weighting risky.

Config:

```text
logs/milestone_d/configs/d0_weighted_ce.yml
```

Runner:

```text
tools/train_milestone_d.py
```

Important metric naming note:

```text
lane_iou / lane_f1 / lane_precision / lane_recall = marking metrics in Milestone D
```

Smoke-test pattern:

```bash
rm -rf logs/milestone_d/cache/D0_weighted_ce

./panda/bin/python tools/train_milestone_d.py \
  --config logs/milestone_d/configs/d0_weighted_ce.yml \
  --run-name D0_weighted_ce_smoke \
  --epochs 1 \
  --steps-per-epoch-train 1 \
  --steps-per-epoch-valid 1 \
  --no-cache \
  --force
```

First real segment:

```bash
rm -rf logs/milestone_d/cache/D0_weighted_ce

./panda/bin/python tools/train_milestone_d.py \
  --config logs/milestone_d/configs/d0_weighted_ce.yml \
  --run-name D0_weighted_ce_25ep \
  --epochs 5 \
  --save-ckpt-freq 1 \
  --force
```

Resume after the first 5 epochs if the run is healthy:

```bash
./panda/bin/python tools/train_milestone_d.py \
  --config logs/milestone_d/configs/d0_weighted_ce.yml \
  --run-name D0_weighted_ce_25ep \
  --epochs 25 \
  --save-ckpt-freq 1 \
  --resume-latest
```

`--epochs` is the target total epoch count. On resume, `--epochs 25` continues
from the checkpoint to epoch 25; it does not rerun the completed epochs.
