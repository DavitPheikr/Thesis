# G2 intensity-binned test diagnostic

Additive diagnostic for thesis Chapter 6.6. Created by
`results/diagnostics/g2_intensity_binned.py`. No existing file is modified.

## Provenance
- Model / run: G2 (`logs/milestone_g/runs/G2_schedule_extend_100`), final reported model.
- Checkpoint: `/home/coder/project/logs/milestone_g/runs/G2_schedule_extend_100/checkpoints/ckpt_epoch_00068.pth` (epoch 68).
- Config: `/home/coder/project/logs/milestone_g/runs/G2_schedule_extend_100/config_snapshot.yml` (the run's frozen `config_snapshot.yml`).
- Split: `test` (held-out test).
- Coverage: **full** (spatially-regular, every frame covered ~8x).
- Seed: 42 (matches the existing seed-42 full-coverage stratified analysis;
  NOT the 3-seed headline mean).
- Inference: **rerun** (no per-point data was saved by the original run; this is a
  fresh forward pass with the identical setup).

## Per-point sources (from the reused engine, `_sampled_error_engine.py`)
- true active class (road=0, marking=1, other=2): `filter_valid_label(...)` output `y_true`.
- predicted class: `argmax(model scores)`.
- raw/un-normalized intensity: `analysis_intensity_z * std + mean`
  (mean=22.471752, std=14.902380); clipped to [0, 114].
- rgb_valid: `analysis_rgb_valid > 0.5`.
- range (m): per-point `ranges`.
- raw->active remap mismatch rate: 0.00e+00 (integrity check).

## IMPORTANT: counts are patch-accumulated, not unique physical points
Full coverage evaluates each point under several overlapping patches, and counts
are accumulated **per patch** (per-patch accumulation, not per-point voting). A
physical point covered N times contributes N to the counts. This is exactly the
convention behind the existing full-coverage confusion matrix
(`results/per_model/G2_lidar_rgb_lovasz/test/full/seed_42/confusion_matrix.npy`),
which is why the summed confusion here matches it exactly. Recall / precision /
overlap percentages are therefore over patch-accumulated active evaluations, on
the same basis as the reported headline metrics.

## Intensity bins
[0.0,10.0)=0-10, [10.0,20.0)=10-20, [20.0,30.0)=20-30, [30.0,40.0)=30-40, [40.0,60.0)=40-60, [60.0,80.0)=60-80, [80.0,114.0]=80-114.
Half-open `[min,max)` except the final bin `[80,114]` (closed). Intensity is clipped
to 114, so no `114+` bin exists.

## Distance buckets
0_10m, 10_20m, 20_30m, 30_40m, 40_60m, 60m_plus (identical to the existing `distance_bucket_metrics.csv`).

## Metric definitions
marking_tp = label==marking & pred==marking; marking_fn = label==marking & pred!=marking;
marking_to_road / marking_to_other = label==marking & pred==road / other;
road_to_marking_fp = label==road & pred==marking; other_to_marking_fp = label==other & pred==marking;
marking_fp_total = pred==marking & label!=marking;
marking_recall = tp/true_marking_count; marking_precision = tp/pred_marking_count;
marking_iou = tp/(tp+marking_fp_total+marking_fn) (within-bin IoU);
pred_marking_rate = pred_marking_count/point_count;
road_to_marking_rate = road_to_marking_fp/true_road_count;
other_to_marking_rate = other_to_marking_fp/true_other_count;
pred_true_ratio_marking = pred_marking_count/true_marking_count.
Undefined divisions are NaN (not zero).

## Validation checks (all required to pass before any CSV is written)
1. bin point_count sum (172588729) == active N (172588729)
   == confusion total (172588729): True.
2. summed 3x3 confusion == existing seed-42 full-test confusion_matrix.npy (exact): True.
3. reconstructed marking IoU=0.5177 (ref ~0.5177),
   precision=0.6811 (ref ~0.6811),
   recall=0.6833 (ref ~0.6833), tol 0.005: True.

## Files
- intensity_bin_metrics.csv          (main: metrics per intensity bin)
- intensity_bin_by_rgb_valid_metrics.csv (intensity bin x rgb_valid/rgb_invalid)
- intensity_bin_by_distance_metrics.csv  (intensity bin x distance bucket)
- intensity_overlap_summary.csv      (class intensity stats + multi-threshold overlap)

## Limitations
- Associational/group-level diagnostic; no causal claim about intensity.
- `marking_iou` per bin is a within-bin IoU (points grouped by their own intensity),
  not a global IoU.
- Counts are patch-accumulated (see above), single training seed (42), single
  evaluation pass; matches the existing seed-42 full-coverage analysis.
