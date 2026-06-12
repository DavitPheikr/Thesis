# Timestamp Policy Analysis (step 4)

Empirical inputs to the E0 timestamp + ego-motion policy decision.

## Question

How many LiDAR frames receive a valid front-camera match, and how does the
answer depend on the dt threshold?

## How

```bash
./panda/bin/python logs/milestone_e/dataset_analysis/timestamp_policy/analysis_code/scan_dt_per_frame.py
```

Writes:

- `dt_per_frame.csv` -- one row per (sequence, lidar_frame) with same-index
  and nearest-timestamp camera frame and dt values.
- `dt_per_sequence_summary.csv` -- one row per sequence with the number of
  frames failing at the 60 ms candidate threshold.
- `scan_summary.md` -- threshold table + sequence-level failure list.

## Policy doc

The decisions derived from this analysis live in
[`../../notes/timestamp_policy.md`](../../notes/timestamp_policy.md).
