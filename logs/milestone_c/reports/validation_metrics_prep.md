# Validation Metrics Prep

Generated: 2026-05-04

Post-run note, 2026-05-07: this metric plumbing was exercised by the official full C0 run `logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/`. The run produced 30 rows of per-epoch metrics, 30 per-epoch JSON snapshots, 30 confusion matrices, final plots, and a selected best-lane-F1 checkpoint at epoch 18. See `logs/milestone_c/reports/c0_full_baseline_results.md`.

## 1. Approach Chosen

Chosen integration strategy: **B. Override the validation/training loop in the existing `MilestoneCPipeline` subclass.**

Reasoning:

- Open3D-ML's built-in validation loop computes loss and reduced IoU internally, but it does not persist the thesis-critical artifacts: active-class confusion matrix, per-class precision/recall/F1, and lane recall by distance bucket.
- The needed tensors are available inside Open3D's validation loop immediately after `model.get_loss(...)`: filtered active labels and class scores.
- Distance buckets require ego-frame point ranges before RandLA-Net's validation recentering. The least-invasive solution is a small transform wrapper that attaches `ranges` to the transformed batch before recentering changes the coordinates.
- No Open3D package files were modified. The integration stays in `tools/train_milestone_c.py`.

Investigated:

- `panda/lib/python3.12/site-packages/open3d/_ml3d/torch/pipelines/semantic_segmentation.py`
- `panda/lib/python3.12/site-packages/open3d/_ml3d/torch/models/randlanet.py`
- `panda/lib/python3.12/site-packages/open3d/_ml3d/torch/modules/losses/semseg_loss.py`
- `panda/lib/python3.12/site-packages/open3d/_ml3d/torch/modules/metrics/semseg_metric.py`

## 2. Files Modified Or Created

- `src/thesis_pipeline/eval/__init__.py` — new eval package marker.
- `src/thesis_pipeline/eval/milestone_c_metrics.py` — pure metric functions, JSON/CSV helpers, and mandatory synthetic `_self_test()`.
- `tools/train_milestone_c.py` — added validation metric plumbing to `MilestoneCPipeline`, per-epoch eval artifacts, validation failure handling, and range attachment wrapper.
- `logs/milestone_c/reports/validation_metrics_prep.md` — this report.

## 3. Open3D-ML Internals Touched

No installed Open3D-ML files were edited.

Subclass/interface assumptions:

- `MilestoneCPipeline.run_train()` now mirrors Open3D-ML's `SemanticSegmentation.run_train()` structure so it can collect predictions during validation.
- Validation still uses Open3D-ML `TorchDataloader`, `get_sampler`, `SemSegLoss`, `SemSegMetric`, and RandLA-Net `model.get_loss(...)`.
- `attach_ego_ranges_to_transform(model)` wraps `model.transform` after model construction. It computes ego-frame Euclidean range from pre-augmentation/pre-recenter subsampled points and stores selected patch ranges in `inputs["data"]["ranges"]`.
- Predictions are converted to integer class indices by `torch.argmax(predict_scores, dim=-1)`, where `predict_scores` is the filtered active-class score tensor returned by `model.get_loss(...)`.

## 4. Class Indexing Decision

Verified mapping: the model/metric tensors use active contiguous class indices:

- `0 = road`
- `1 = lane`
- `2 = other`

Why:

- Dataset raw thesis labels are `0=ignore`, `1=road`, `2=lane`, `3=other`.
- Open3D-ML `filter_valid_label(...)` removes ignored label `0` and builds a reducing list that maps raw active labels `1/2/3` to contiguous active indices `0/1/2`.
- RandLA-Net `get_loss(...)` returns this compressed `labels` tensor together with `scores`, so custom metrics are computed in the same class space as Open3D's loss.

The metrics module docstring documents this explicitly.

## 5. Robustness Behavior

- Validation exceptions are caught inside `_run_validation_epoch(...)`.
- On validation failure, a full traceback is written to `validation_errors_epoch_<NNN>.txt`.
- Failed validation epochs skip `eval_history.csv` / JSON / confusion output for that epoch and training continues.
- Empty inputs return a zero 3x3 confusion matrix and NaN metrics rather than crashing.
- Empty class support returns NaN for that class's IoU, precision, recall, and F1.
- JSON snapshots convert NaN to `null`.
- CSV rows write NaN as literal `nan`.
- CUDA validation memory is measured with `torch.cuda.reset_peak_memory_stats()` at validation start and `torch.cuda.max_memory_allocated()` at validation end. If CUDA is unavailable, the value is `0`.
- Validation split is asserted as `validation`; the code comment documents that validation must not use lane-aware sampling or training-only augmentation.

## 6. Synthetic Test Result

Invocation:

```bash
./panda/bin/python -c "from src.thesis_pipeline.eval.milestone_c_metrics import _self_test; _self_test()"
```

Result:

```text
milestone_c_metrics_self_test: PASS
```

Covered cases:

- Known 1000-point confusion matrix with 700 road, 200 lane, 100 other.
- Hand-checked mIoU, per-class IoU, precision, recall, and F1.
- Lane recall by distance bucket: `0-10m`, `10-20m`, `20-30m`, `30+m`.
- All predictions correct.
- No lane predictions and no lane ground truth.
- Empty input arrays.

No expected-vs-actual discrepancies remain.

## 7. Structural Check Results

Commands run, with no actual training:

```bash
./panda/bin/python -m py_compile tools/train_milestone_c.py
```

Result: PASS

```bash
./panda/bin/python -m py_compile src/thesis_pipeline/eval/milestone_c_metrics.py
```

Result: PASS

```bash
./panda/bin/python -c "from src.thesis_pipeline.eval.milestone_c_metrics import _self_test; _self_test()"
```

Result: PASS

```bash
./panda/bin/python -c "import sys; sys.path.insert(0, '.'); from tools.train_milestone_c import MilestoneCPipeline; print([m for m in dir(MilestoneCPipeline) if 'val' in m.lower() or 'eval' in m.lower()])"
```

Result:

```text
['_accumulate_validation_batch', '_run_validation_epoch', '_write_eval_artifacts', '_write_training_log_after_validation_failure']
```

```bash
./panda/bin/python tools/train_milestone_c.py --help
```

Result: PASS, help text printed normally.

## 8. Known Limitations Or Open Questions

- The training entrypoint is now longer than the earlier lightweight-driver target because robust validation required mirroring Open3D-ML's training loop. Current line counts: `tools/train_milestone_c.py` 594 lines, `milestone_c_metrics.py` 339 lines.
- Distance-bucket recall is computed on validation patches after Open3D grid subsampling and patch sampling, not on every original raw point in the frame. This matches what the model actually predicts during validation.
- The range used for buckets is computed before RandLA-Net recentering, but after grid subsampling. That is intentional because recentering would destroy absolute ego distance.
- Full validation metrics have not been runtime-tested inside a real epoch, per instruction not to run training locally.

## 9. What This Enables

Future Milestone C training runs can now produce thesis-relevant validation artifacts after each successful epoch: `eval_history.csv`, `eval_epoch_<NNN>.json`, and `confusion_epoch_<NNN>.npy`. This makes it possible to detect class collapse, measure lane IoU/precision/recall directly, inspect road-vs-lane-vs-other confusion, and evaluate whether lane performance degrades with distance.

## 10. Recommended Next Session

Run a tiny GPU-backed C0 smoke on DigitalOcean with validation enabled, then inspect the first `eval_history.csv` and `confusion_epoch_001.npy` before launching the long baseline.
