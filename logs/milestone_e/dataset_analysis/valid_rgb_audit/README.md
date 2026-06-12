# Stratified Valid-RGB Audit (step 5)

For every (sequence, lidar_frame) in train/val/test, apply the step-4
timestamp policy and count how many forward-LiDAR points actually receive a
valid RGB sample, broken down by class, distance bucket, marking subtype,
sequence, and split.

## Question

Where will RGB help, and where will it not? Specifically:

- What fraction of marking points have valid RGB, by distance bucket?
- Are any sequences anomalously low on RGB coverage?
- Does any marking subtype (raw 8 / 9 / 10) get systematically less RGB?
- How much does the timestamp policy actually cost us in terms of valid
  points?

## Policy applied

Same as step 4:

```yaml
camera_lookup: nearest_timestamp
rgb_max_dt_s: 0.060
```

A frame is `frame_valid` if its nearest-timestamp camera frame is within
60 ms. A point is `rgb_valid` if its frame is valid AND the projection
lands inside the image AND in front of the camera.

## How

```bash
./panda/bin/python logs/milestone_e/dataset_analysis/valid_rgb_audit/analysis_code/run_audit.py
```

Outputs:

- `per_frame.csv` -- one row per (split, sequence, lidar_frame) with all
  the class x distance counts.
- `aggregate.csv` -- one row per (split, class, distance bucket) with
  total + valid counts and the valid ratio.
- `marking_subtype.csv` -- one row per (split, raw_id) for the marking
  subtypes.
- `per_sequence.csv` -- one row per sequence with overall + marking-specific
  valid ratios.
- `summary.md` -- headline tables.

## Pass criteria

- Marking valid ratio at near distances (0-20 m, where lane lines mostly
  live) should be uniformly high (> 0.85).
- No sequence other than val/054 should be anomalous.
- Marking subtype gap should not be extreme; if raw 9 (stop lines) has
  much lower RGB coverage than raw 8/10, that affects E0 stratification.
