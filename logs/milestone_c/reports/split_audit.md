# Split Audit

Historical note, 2026-05-07: this strict split-audit schema verdict remains a documentation gap, but it did not block the official C0 baseline run. C0 uses the frozen Milestone B train/val/test split and completed as `logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/`. The missing complete per-sequence class-count table for val/test should still be fixed before making a strict ACCEPTABLE/BORDERLINE/NEEDS_REBUILD split-quality claim.

## Headline Verdict

**BLOCKED_SCHEMA**

The requested ACCEPTABLE / BORDERLINE / NEEDS_REBUILD verdict cannot be computed honestly from the requested existing artifacts. The audit is blocked because the available per-sequence analysis table does not contain the required full class totals, and some frozen split sequences are absent from that table.

No split files were modified. No training, validation, inference, checkpoint loading, or Open3D-ML training code was run.

## Blocking Reasons

- sequence_summary.csv missing required columns: total_count (accepted names: ['total_count', 'point_count_total', 'total_point_count', 'total_points']); active_count (accepted names: ['active_count', 'active_point_count', 'active_points']); other_count (accepted names: ['other_count', 'other_point_count', 'other_count_total'])
- Some frozen split sequence IDs are absent from sequence_summary.csv: train: ['028', '030', '032', '033', '035', '069', '070', '139']; val: ['034', '038']

## Schema Check

- `logs/raw_intensity_analysis/tables/sequence_summary.csv`: FOUND
- `logs/raw_intensity_analysis/tables/frame_summary.csv`: FOUND
- `configs/splits/train.txt`: FOUND
- `configs/splits/val.txt`: FOUND
- `configs/splits/test.txt`: FOUND
- `logs/milestone_b_training_statistics.json`: FOUND
- `logs/raw_intensity_analysis/tables/distance_bucket_summary.csv`: FOUND

Additional source search:

- `logs/milestone_b_statistics_progress.jsonl`: FOUND, contains per-sequence remapped class counts, but only for the 58 training sequences. It covers train 58/58, val 0/9, test 0/9.
- `logs/milestone_b_audit_progress.jsonl`: FOUND, covers all split sequences and includes lane/frame counts, but does not include road/other/active/ignore class counts.
- `logs/stats_per_seq/*.npy`: FOUND for training sequences only, but these are intensity samples, not full per-sequence class totals.

Conclusion from source search: the missing data does not appear to be simply renamed or moved. Complete per-sequence class counts exist for train only, not for val/test.

Observed CSV headers:

- `sequence_summary`: `seq_id, frame_count, lane_count_total, road_count_total, raw9_count_total, raw10_count_total, lane_mean_avg, road_mean_avg, lane_median_avg, road_median_avg, lane_minus_road_mean_avg, lane_minus_road_median_avg, cohens_d_lane_vs_road_avg, frac_lane_above_road_p90_avg, auc_lane_vs_road_avg, auc_lane_vs_road_plus_other_marking_avg, lane_forward_median_x_avg, metric_std_lane_minus_road_median, lane_point_count_exact, road_point_count_exact, stop_line_point_count_exact, other_marking_point_count_exact, lane_median_exact, road_median_exact, lane_minus_road_median_exact, auc_lane_vs_road_exact, auc_lane_vs_road_surface_non_lane_exact`
- `frame_summary`: `seq_id, frame_idx, lane_count, road_count, raw9_count, raw10_count, lane_mean, road_mean, lane_median, road_median, lane_minus_road_mean, lane_minus_road_median, cohens_d_lane_vs_road, frac_lane_above_road_p90, auc_lane_vs_road, auc_lane_vs_road_plus_other_marking, lane_forward_median_x, road_p90, insufficient_support`
- `distance_bucket_summary`: `bucket, lane_count, road_count, lane_median, road_median, lane_minus_road_median, auc_lane_vs_road, auc_lane_vs_road_surface_non_lane`

Required logical columns not available:

- `sequence_summary.csv`:
  - total_count (accepted names: ['total_count', 'point_count_total', 'total_point_count', 'total_points'])
  - active_count (accepted names: ['active_count', 'active_point_count', 'active_points'])
  - other_count (accepted names: ['other_count', 'other_point_count', 'other_count_total'])

## Partial Summary Table

These values are included only to show what was safely readable. They are **not sufficient** for the requested verdict rules because `active_count`, `total_count`, and full `other_count` are missing per sequence.

| Split | Seq Count | Present In Sequence Summary | Missing Seq Count | Available Lane Points | Available Road Points | Lane-Bearing Frame Fraction |
|---|---:|---:|---:|---:|---:|---:|
| train | 58 | 50 | 8 | 2098182 | 97744108 | 1.0000 |
| val | 9 | 7 | 2 | 335257 | 11881237 | 1.0000 |
| test | 9 | 9 | 0 | 457416 | 17416298 | 1.0000 |

## Missing Sequence IDs

- `train`: ['028', '030', '032', '033', '035', '069', '070', '139']
- `val`: ['034', '038']
- `test`: none

## Cross-Split Verdict Rules

All requested verdict-rule checks are marked **BLOCKED**, because they require complete per-sequence active/total/other counts and full sequence coverage.

| Check | Status | Reason |
|---|---|---|
| Lane fraction max/min ratio <= 2.0 | BLOCKED | Requires active point counts for every sequence in every split. |
| Every split has >= 3 lane-bearing sequences | BLOCKED | Sequence coverage is incomplete for train/val in `sequence_summary.csv`; non-listed sequences may be zero-lane or simply absent from this analysis table. |
| Every split has >= 50,000 lane points | BLOCKED | Lane counts are missing for 10 frozen split sequences. |
| No sequence contributes >50% of split lane points | BLOCKED | Requires complete lane totals for all sequences in each split. |

## Distance Distribution

- `logs/raw_intensity_analysis/tables/distance_bucket_summary.csv` status: present but global only; cannot derive per-split distance distribution from columns: ['bucket', 'lane_count', 'road_count', 'lane_median', 'road_median', 'lane_minus_road_median', 'auc_lane_vs_road', 'auc_lane_vs_road_surface_non_lane']
- Per-split lane-distance distribution was skipped because the available distance table is not per-sequence or per-split.

## Training Statistics Sanity Check

- `logs/milestone_b_training_statistics.json` class counts: `ignore=398730, road=119562394, lane=2098182, other=176504630`
- This confirms training-split aggregate class counts exist, but it does not provide equivalent validation/test totals and cannot substitute for the missing per-sequence audit fields.

## Recommendation

Do not launch C0 based on this audit alone. First produce or locate a complete per-sequence class-count table covering all 76 frozen split sequences with `total_count`, `active_count`, `lane_count`, `road_count`, and `other_count`. After that table exists, rerun this split audit and apply the ACCEPTABLE / BORDERLINE / NEEDS_REBUILD rules exactly.

## Output Data

- Partial machine-readable data written to `logs/milestone_c/eval/split_audit.csv`
