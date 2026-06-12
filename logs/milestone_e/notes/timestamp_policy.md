# E0 Timestamp + Ego-Motion Policy

Final, config-shaped decisions for how Milestone E0 attaches a front-camera
frame to each LiDAR frame, and when a LiDAR point is allowed to receive an
RGB sample.

## Decisions

```yaml
feature_mode: intensity_rgb_front
camera_name: front_camera
camera_lookup: nearest_timestamp
rgb_max_dt_s: 0.060
motion_compensation: none
invalid_rgb_policy: zero_with_valid_flag
```

## Decision 1: camera-frame lookup = nearest-timestamp

Same-index (use `cam[i]` for `lidar[i]`) and nearest-timestamp (use the
camera frame whose timestamp is closest to the lidar frame timestamp) agree
on **98.85 %** of the 6080 lidar frames in the splits. They disagree only on
70 frames, all in `val/054`.

`nearest_timestamp` is strictly safer: it can never produce a worse choice
than `same_index`, and for `val/054` it can return a slightly less-bad
camera frame in some cases. There is no operational cost. Use it.

Source: [`dataset_analysis/timestamp_policy/scan_summary.md`](../dataset_analysis/timestamp_policy/scan_summary.md)

## Decision 2: rgb_max_dt_s = 0.060

Per-frame dt distribution after nearest-timestamp lookup:

| threshold | frames pass | frames fail | pass % | failures concentrated in |
| ---: | ---: | ---: | ---: | --- |
| 40 ms | 0 | 6080 | 0.00 % | all sequences -- threshold is below the systematic offset |
| 60 ms | 6037 | 43 | 99.29 % | val/054 only |
| 80 ms | 6037 | 43 | 99.29 % | val/054 only -- no extra frames recovered |
| 100 ms | 6037 | 43 | 99.29 % | val/054 only -- no extra frames recovered |
| 150 ms | 6047 | 33 | 99.46 % | val/054 only |
| 200 ms | 6056 | 24 | 99.61 % | val/054 only |
| 300 ms | 6068 | 12 | 99.80 % | val/054 only |
| 500 ms | 6078 | 2 | 99.97 % | val/054 only |

Reasoning:

- Most sequences (75 of 76) have a systematic offset of about
  `cam_ts - lid_ts == -50 ms`. A 40 ms threshold rejects all of them and
  is therefore wrong.
- 60 ms covers the systematic offset with margin and rejects only frames
  where the camera-lidar drift is real (val/054 timing anomaly).
- Loosening past 60 ms recovers nothing in the [60, 100] ms band and only
  pulls in more val/054 frames whose drift is visibly large enough to
  misplace markings (confirmed by overlay frame 5: at +450 ms drift, marking
  points are visibly displaced from the painted lines in the image).
- 60 ms is therefore the minimum useful threshold that does not also accept
  pathological frames.

Source: [`dataset_analysis/timestamp_policy/scan_summary.md`](../dataset_analysis/timestamp_policy/scan_summary.md),
[`dataset_analysis/projection_overlay/overlays_class/05__val_054_f00_same_index_class.png`](../dataset_analysis/projection_overlay/overlays_class/05__val_054_f00_same_index_class.png)

## Decision 3: no per-point motion compensation for E0

Even at the systematic -50 ms offset, an ego vehicle moving at typical
urban speeds (40 km/h, 11 m/s) translates ~0.55 m between the LiDAR sweep
and the camera shot. A 12 cm lane line is offset by ~5 line-widths at the
pixel level. In principle, motion-compensating each LiDAR point to the
camera timestamp via pose interpolation would tighten this.

The decision for E0 is to **not** motion-compensate, because:

- The 8-frame class overlay sanity check showed that marking-only points
  visually sit on the painted lines at -50 ms across multiple sequences.
  The aggregate offset is below the threshold where it dominates per-point
  RGB sampling for our subsampled voxel input.
- Motion compensation adds implementation surface (per-point timestamps,
  per-point pose interpolation, sweep-time handling) that we do not need
  to test the core E0 hypothesis: does adding RGB help road/marking
  discrimination at all?
- If E0 succeeds, motion compensation becomes a justifiable later
  refinement (E1 or thesis future-work).
- If E0 fails, motion compensation is unlikely to be the cause and should
  not be confounded into the experiment.

This is recorded as a known limitation in the milestone writeup.

## Decision 4: invalid RGB policy = zero with valid flag

When a LiDAR point cannot receive a valid RGB sample, store:

```text
rgb       = (0.0, 0.0, 0.0)
rgb_valid = 0
```

A point's RGB is invalid if any of:

1. The lidar frame has no camera match within `rgb_max_dt_s`. **All points**
   in the frame get `rgb_valid=0`.
2. The point projects behind the camera (`z_cam <= 0`).
3. The point projects outside the image bounds.
4. (Step 6 may add) the point is occluded by a closer point in the same
   pixel (z-buffer screening). Not enabled in E0 by default.

`rgb_valid` is stored as a separate per-point feature so the model can learn
to discount RGB where it is missing.

## Frames affected at E0 launch

At `rgb_max_dt_s = 0.060`:

- **6037 / 6080 frames (99.29 %)** receive a valid front-camera match.
- 43 frames are invalid. All 43 are in `val/054` (53.75 % of seq 054 fails).
- No `train` or `test` frames are invalidated.
- Seq 054 retains the second half of its frames as valid.

Source: [`dataset_analysis/timestamp_policy/dt_per_sequence_summary.csv`](../dataset_analysis/timestamp_policy/dt_per_sequence_summary.csv)

## Implications for evaluation

- The validation metric will be partially blind to RGB gains on the
  invalidated half of seq 054. Stratifying the eval by `rgb_valid` ratio
  per sequence (and reporting it) makes this honest.
- Marking IoU for `val/054` will look like a partial fallback to D0 on its
  invalidated frames; this is expected and not a sign of E0 failing.

## Implementation notes for the dataset feature builder

```python
# pseudocode
camera_idx = argmin_i |cam_ts[i] - lid_ts[lidar_frame_idx]|
dt = cam_ts[camera_idx] - lid_ts[lidar_frame_idx]
if abs(dt) > rgb_max_dt_s:
    rgb = zeros((N, 3), dtype=float32)
    rgb_valid = zeros((N,), dtype=uint8)
    return rgb, rgb_valid

# else, project all forward-sensor points (in world coords) through the
# selected camera frame and sample colors at the resulting pixels.
uv, _depth, kept_indices = project_world_to_camera(points_world,
                                                   cam_pose[camera_idx],
                                                   intrinsics, W, H)
rgb = zeros((N, 3), dtype=float32)
rgb_valid = zeros((N,), dtype=uint8)
# sample (bilinear or nearest -- pin choice in cache version)
sampled = sample_image(image_array, uv) / 255.0
rgb[kept_indices] = sampled
rgb_valid[kept_indices] = 1
return rgb, rgb_valid
```

Step 7 and step 8 are now pinned in
[`cache_policy.md`](cache_policy.md): E0 uses bilinear sampling and passes
the fractional `rgb_valid` feature through after voxel subsampling.
