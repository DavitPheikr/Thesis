# Milestone E Context Document

End-to-end record of what Milestone E is, why it exists, every decision we
made up to this point, every alternative we considered, every number we
collected, and where the work goes from here.

This is the working memory for the milestone. It is intended to be readable
as a single document by anyone (you, a future you, a reviewer, a future
contributor, a thesis examiner) who wants to understand the milestone
without having to reconstruct it from scattered notes.

---

## 1. Where we came from: the D0 baseline

The Milestone D run `D0_weighted_ce_25ep` is the LiDAR-only road-marking
baseline this milestone tries to improve on. Brief summary of D0 so the rest
of this document makes sense.

### 1.1 Task

PandaSet point clouds, three classes:

```text
road     = raw PandaSet class 7
marking  = raw 8 (lane line) + raw 9 (stop line) + raw 10 (other paint)
other    = everything else not ignored
ignore   = raw 1, 2, 3, 4
```

This is the `road_marking3` label remap. It replaced the earlier strict
`lane3` label scheme from Milestone C because qualitative inspection found
raw 9 and raw 10 are also road paint that the model was being unfairly
penalized for predicting.

### 1.2 Input

```text
points  (N, 3)  ego xyz
feat    (N, 1)  intensity (clipped, standardized)
```

So D0 sees only `xyz + intensity`. 4 input channels through the network's
first MLP.

### 1.3 Result

Best checkpoint = epoch 18 (selected by raw marking IoU after the run).

| metric | epoch 1 | epoch 18 best | epoch 25 final |
| --- | ---: | ---: | ---: |
| marking IoU | 0.202 | **0.440** | 0.410 |
| marking F1 | 0.336 | **0.611** | 0.582 |
| marking precision | 0.230 | **0.515** | 0.451 |
| marking recall | 0.624 | 0.753 | 0.819 |
| mIoU | 0.651 | **0.781** | 0.772 |

### 1.4 D0's failure mode (the reason Milestone E exists)

Diagnostic analysis on the epoch-18 checkpoint identified two-sided
road/marking confusion as the dominant residual error, driven by intensity
overlap between the classes:

| group | median intensity | p25 | p75 |
| --- | ---: | ---: | ---: |
| marking TP (correctly predicted) | 35 | 30 | 47 |
| marking -> road (missed marking) | 29 | 24 | 31 |
| road TP                          | 29 | 26 | 33 |
| road -> marking (false positive) | 31 | 28 | 37 |

Reading: correctly-detected markings are noticeably brighter than road, but
**missed markings have a median intensity of 29 -- the same as correctly
predicted road**. And the road points D0 falsely calls marking are the
bright-end-of-road points, intensity ~31, indistinguishable in feature
space from low-intensity real markings.

This is an **input-information limit**: with LiDAR `xyz + intensity` alone,
many lane-paint points and many bright-asphalt points look identical.
That's the gap E0 is trying to close.

D0 also showed marking overprediction (predicted/true marking ratio = 1.46x
at epoch 18, 1.82x at epoch 25), driven by the strong marking class weight
(effective CE weight 26.88 vs 2.38 for road). This is a separate failure
mode and is **not** Milestone E's target.

Sources:

- `logs/milestone_d/run_analysis/D0_weighted_ce_25ep/reports/d0_conclusions.md`
- `logs/milestone_d/runs/D0_weighted_ce_25ep/eval_history.csv`
- `logs/milestone_d/run_analysis/D0_weighted_ce_25ep/sampled_error_analysis_epoch18/`
- `logs/milestone_d/README.md`

---

## 2. The Milestone E hypothesis

In one line:

> The pixels of a front-camera image attached to each LiDAR point will give
> the model a signal that LiDAR intensity does not -- paint vs asphalt color
> contrast -- and should reduce D0's residual road/marking confusion.

The clean experimental setup is:

```text
D0: x y z intensity
E0: x y z intensity r g b rgb_valid
```

`rgb_valid` is a per-point binary feature recording whether that point
received a valid camera color sample. Points that did not (camera FOV miss,
behind camera, timestamp out of tolerance) get `rgb = (0, 0, 0)` and
`rgb_valid = 0` so the model can learn to discount them. This is the
recommended pattern over any kind of nearest-neighbor color filling, which
would introduce a label-correlated hallucination into the inputs.

D0 stays exactly as it is -- frozen baseline -- so we have a clean before
and after comparison.

---

## 3. The audit-then-decide approach

The hypothesis is plausible, but plenty of things can go wrong upstream of
training and silently invalidate the result:

- a wrong coordinate frame in the projection puts colors on the wrong pixels
- a misjudged timestamp policy attaches colors taken seconds after the LiDAR
  scan
- occlusion attaches colors of foreground objects to background points
- voxel preprocessing averages valid and invalid colors into a confusing
  middle value
- an out-of-scope sequence in the validation split contaminates the metric

If any of those go wrong silently, E0 might "succeed" or "fail" for reasons
unrelated to whether RGB actually helps. The thesis result becomes
unfalsifiable.

So Milestone E was structured as 10 deliberately small steps, each gated by
either an observable result or a written policy decision, before any
training launches:

1. Inventory check
2. Projection function contract
3. Visual overlay sanity check
4. Timestamp + ego-motion policy
5. Stratified valid-RGB audit
6. Occlusion screening
7. Voxel-subsampling rgb_valid policy
8. Cache layout + bake-projection decision
9. Dataset extension implementation + single-sample smoke
10. Tiny training smoke (next)

Steps 1-9 are complete. Step 10 is the gate before launching full E0
training.

---

## 4. Step 1 -- Inventory check

### 4.1 Question

For every sequence in `configs/splits/{train,val,test}.txt`, does the front
camera exist on disk, do all sidecar files (intrinsics, poses, timestamps)
load, and are the counts consistent (images, poses, timestamps, lidar pkls
all equal)?

### 4.2 Approach

Pure filesystem scan: walk every sequence directory, open the JSON
sidecars, open one JPEG header per sequence to read image dimensions. No
image decoding, no LiDAR pickle loads. Runs in seconds, memory footprint
trivial.

Source: `logs/milestone_e/dataset_analysis/front_camera_inventory/analysis_code/inventory_front_camera.py`

### 4.3 Result

| split | sequences | ok | fail |
| --- | ---: | ---: | ---: |
| train | 58 | 58 | 0 |
| val   |  9 |  9 | 0 |
| test  |  9 |  9 | 0 |

Plus, uniform across all 76 sequences:

- image resolution `1920 x 1080`
- intrinsics `fx = 1970.013, fy = 1970.009, cx = 970.000, cy = 483.299`
- 80 images per sequence, 80 lidar pkls, 80 camera poses, 80 timestamps

### 4.4 Side finding

Timestamp delta `cam_ts - lid_ts` was uniformly close to `-50 ms` for 75
sequences. Sequence `054` showed `+450 ms` at the first frame, drifting
back to `-50 ms` by the last frame. This was the only timing anomaly in the
dataset and became the basis for the step-4 decision.

### 4.5 Decision

No decision; this was a gating check. It passed; we moved on.

Sources:

- `logs/milestone_e/dataset_analysis/front_camera_inventory/summary.md`
- `logs/milestone_e/dataset_analysis/front_camera_inventory/inventory.csv`

---

## 5. Step 2 -- Projection function contract

### 5.1 Question

PandaSet's devkit provides `pandaset.geometry.projection`. Before relying
on it, what exactly does it do, and what does it NOT do?

### 5.2 Approach

Read the source. Write a contract document.

Source: `pandaset-devkit/python/pandaset/geometry.py`

### 5.3 Contract (verbatim from the note)

Function signature:

```python
projection(lidar_points, camera_data, camera_pose, camera_intrinsics,
           filter_outliers=True)
```

Inputs:

- `lidar_points` must be in **world coordinates**, NOT lidar-frame. The
  function internally builds `inv(camera_pose_mat) @ lidar_points`, so the
  argument has to be in the same world frame the camera pose lives in. The
  parameter name is misleading.
- `camera_data` is a PIL image. Only `.size` is read (the image bytes are
  not consumed by the projection function itself).
- `camera_pose` is a dict `{heading: {w,x,y,z}, position: {x,y,z}}`.
- `camera_intrinsics` has `fx`, `fy`, `cx`, `cy`. Pinhole only, no
  distortion coefficients.

Output (with `filter_outliers=True`):

- `points2d_camera` (M, 2) -- pixel `(u, v)` of surviving points
- `points3d_camera` (M, 3) -- camera-frame xyz of survivors
- `inliner_indices_arr` (M,) -- indices into the original lidar_points

What it does NOT do:

1. No lens distortion correction. Treats intrinsics as ideal pinhole.
2. No occlusion check. Far points can land on pixels owned by closer
   objects.
3. No timestamp or motion compensation. A single camera pose is used for
   the entire LiDAR sweep.
4. No color sampling. Just gives pixel coordinates.
5. No transform from ego to world. Caller must hand it world-frame points.

### 5.4 Decisions resulting from the contract

| concern | E0 handling |
| --- | --- |
| world vs ego coords | always feed projection world-frame points; keep ego-frame separately for the model input |
| no distortion | accept as a limitation; visual sanity check will reveal if it's an issue at image edges (step 3) |
| no occlusion | measure how big the effect actually is (step 6); if small enough, skip |
| no motion compensation | measure dt distribution (step 4) and decide whether to compensate |
| color sampling | implement ourselves with a pinned policy (step 8) |

Sources:

- `logs/milestone_e/notes/projection_contract.md`
- `pandaset-devkit/python/pandaset/geometry.py`

---

## 6. Step 3 -- Visual overlay sanity check

### 6.1 Question

The projection math is right on paper. Does it actually put LiDAR points on
the right pixels of real photos?

### 6.2 Approach

Eight hand-picked frames covering normal sequences and the seq-054 timing
anomaly. For each frame, project forward-LiDAR points, render dots over the
camera image, and look.

Three iterations of the renderer:

- **v1** -- depth-colored scatter. Too dense to read.
- **v2** -- subsampled depth scatter + three anchor markers labeled at 5 m,
  10 m, 20 m. Anchors picked nearest the camera central axis at the target
  depth. Worked for 6 frames but failed on 2: the anchor algorithm latched
  onto vehicle bodies right ahead of the camera, mislabeling them as "5 m"
  when they were actually 30+ m away.
- **v3** -- same scatter, but anchors **biased to ground-level points**.
  Picks the lowest point at the target depth instead of the most-central
  point. If no ground point exists within the depth window (vehicle in
  front), the anchor is skipped with a note.
- **class overlay** -- separate renderer. Colors projected points by class
  (gray = road, blue = other, **bright yellow = marking**), three panels
  per frame: original | class overlay | marking-only.

Sources:

- `logs/milestone_e/dataset_analysis/projection_overlay/analysis_code/render_overlays.py`
- `logs/milestone_e/dataset_analysis/projection_overlay/analysis_code/render_overlays_v2.py`
- `logs/milestone_e/dataset_analysis/projection_overlay/analysis_code/render_overlays_v3.py`
- `logs/milestone_e/dataset_analysis/projection_overlay/analysis_code/render_class_overlays.py`

### 6.3 Result

User confirmation after looking at v3 + class overlays:

> "the stuff in overlays_class looks fine, not bad. projections are okay.
> everything is ok. only the 5th one is a bit displaced. some points hit
> and some miss fully."

Frame 5 (val/054/0 same_index) was expected to look bad because of the
+450 ms timing drift in seq 054. The other seven frames looked correct.

Frame-level valid-RGB ratios from the per-sequence audit:

| split | overall | road | marking | other |
| --- | ---: | ---: | ---: | ---: |
| train | 0.869 | 0.948 | **0.934** | 0.810 |
| val   | 0.812 | 0.898 | **0.770** | 0.759 |
| test  | 0.870 | 0.938 | **0.917** | 0.821 |

Note: this per-sequence audit only sampled frame 40 of each sequence; the
full audit in step 5 went per-frame. Val being weaker is driven by
seq 054 (see steps 4 and 5).

### 6.4 Decision

Projection passes the visual gate. Move to step 4.

### 6.5 Side findings

- The 5 m / 10 m / 20 m anchors land on the road when ground-biased.
- No corner-of-image drift visible in any frame; lens-distortion absence
  is not a near-image-center problem. Documented as a limitation that
  could matter at long range, not measured further.
- Front camera horizontal FOV is ~52 degrees
  (`2 * atan(960 / 1970)`). This becomes relevant in step 5 to explain
  marking valid-ratio variation by distance.

Sources:

- `logs/milestone_e/dataset_analysis/projection_overlay/summary_v3.md`
- `logs/milestone_e/dataset_analysis/projection_overlay/summary_class.md`
- `logs/milestone_e/dataset_analysis/projection_overlay/per_sequence_audit.md`

---

## 7. Step 4 -- Timestamp + ego-motion policy

### 7.1 Question

Camera and LiDAR don't fire at the same instant. How do we choose a camera
frame for a given LiDAR frame, what time gap counts as "close enough", and
do we motion-compensate the LiDAR points to the camera timestamp?

### 7.2 Approach

Pure JSON scan: read every sequence's camera and lidar timestamps and
compute the time gap for every lidar frame under two lookup policies:

- same-index: use `cam[i]` for `lidar[i]`.
- nearest-timestamp: use the camera frame whose timestamp is closest to the
  lidar frame timestamp.

Then for each candidate threshold T, count frames that pass `|dt| <= T`.

Source: `logs/milestone_e/dataset_analysis/timestamp_policy/analysis_code/scan_dt_per_frame.py`

### 7.3 Result

Distribution over 6080 frames:

| threshold | frames pass | pass % | failures concentrated in |
| ---: | ---: | ---: | --- |
| 40 ms  | 0    | 0.00 % | all sequences -- below the systematic offset |
| 60 ms  | 6037 | **99.29 %** | val/054 only |
| 80 ms  | 6037 | 99.29 % | val/054 only -- no extra frames recovered |
| 100 ms | 6037 | 99.29 % | val/054 only -- no extra frames recovered |
| 150 ms | 6047 | 99.46 % | val/054 only |
| 200 ms | 6056 | 99.61 % | val/054 only |
| 300 ms | 6068 | 99.80 % | val/054 only |
| 500 ms | 6078 | 99.97 % | val/054 only |

Same-index vs nearest-timestamp: agree on 98.85 % of frames (6010 / 6080).
The 70 disagreeing frames are all in val/054.

### 7.4 Decisions

```yaml
camera_lookup:       nearest_timestamp
rgb_max_dt_s:        0.060
motion_compensation: none
invalid_rgb_policy:  zero_with_valid_flag
```

Reasoning per decision:

1. **`nearest_timestamp`**: never worse than same-index, strictly safer
   for val/054, no operational cost.
2. **`60 ms`**: tighter (40 ms) rejects every sequence because the
   systematic offset is `-50 ms`. Looser values recover nothing in
   [60, 100] ms and only pull in pathological val/054 frames where
   the +450 ms drift produces visibly misaligned RGB (the user
   confirmed frame 5 of the overlay looked displaced).
3. **`motion_compensation: none`**: at the -50 ms systematic offset, the
   ego moves ~0.55 m at 40 km/h. A 12 cm lane line is offset by
   ~5 line-widths at the pixel level. But the marking-only overlays
   showed yellow dots sitting on visible paint at this offset, so
   the visual impact is small. Motion compensation would add
   implementation surface (per-point timestamps, pose interpolation,
   sweep-time handling) that we do not need to test the core
   hypothesis. Deferred to E1 / future work.
4. **`zero_with_valid_flag`**: per-point. RGB = (0,0,0), valid = 0
   when invalid. The model learns to discount invalid points via
   the flag, which is better than hallucinating nearest-neighbor
   colors.

### 7.5 Cost of the policy

| outcome | count |
| --- | ---: |
| frames with a valid camera match | 6037 / 6080 (99.29 %) |
| frames invalidated | 43 (0.71 %) |
| splits affected | val only |
| sequences affected | val/054 only |
| val/054 frames invalidated | 43 / 80 (53.75 %) |

The seq 054 anomaly therefore drags down the validation marking-RGB ratio.
This is anticipated and is handled by the evaluation strategy (section 16).

Sources:

- `logs/milestone_e/notes/timestamp_policy.md`
- `logs/milestone_e/dataset_analysis/timestamp_policy/scan_summary.md`
- `logs/milestone_e/dataset_analysis/timestamp_policy/dt_per_frame.csv`

---

## 8. Step 5 -- Stratified valid-RGB audit

### 8.1 Question

Now that we have a timestamp policy, where does the model actually receive
RGB? Specifically: by class, by distance bucket, by marking subtype, by
sequence.

### 8.2 Approach

Apply the step-4 policy to every (sequence, lidar_frame) of train+val+test
(6080 frames). For each frame: project, compute per-point `rgb_valid`,
bin by class and ego-frame planar distance.

Distance buckets match Milestone D's diagnostic analysis:
`0-10, 10-20, 20-30, 30-40, 40-60, 60m+`.

Source: `logs/milestone_e/dataset_analysis/valid_rgb_audit/analysis_code/run_audit.py`

### 8.3 Result -- overall

| split | overall | road | **marking** | other |
| --- | ---: | ---: | ---: | ---: |
| train | 0.867 | 0.946 | **0.907** | 0.810 |
| val   | 0.812 | 0.898 | **0.770** | 0.759 |
| test  | 0.870 | 0.938 | **0.917** | 0.821 |

### 8.4 Result -- marking valid by distance

| split | 0_10m | 10_20m | 20_30m | 30_40m | 40_60m | 60m_plus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.988 | **0.878** | 0.876 | 0.941 | 0.956 | 0.993 |
| val   | 0.842 | 0.716 | 0.802 | 0.854 | 0.861 | 0.912 |
| test  | 0.997 | 0.882 | 0.928 | 0.937 | 0.976 | 0.982 |

10-20 m is the lowest. Reason: camera horizontal FOV ~52 degrees. At
10-20 m, wide markings (stop lines, adjacent-lane content) extend laterally
past the camera edges. At 0-10 m everything is dead ahead in view. At
60+ m only forward-axis content remains.

### 8.5 Result -- marking subtype (raw 8 / 9 / 10)

| split | raw 8 lane | raw 9 stop | raw 10 other |
| --- | ---: | ---: | ---: |
| train | 0.934 | 0.926 | 0.887 |
| val   | 0.845 | **0.458** | 0.711 |
| test  | 0.910 | 0.902 | 0.933 |

Val raw-9 valid is the standout. Reasons:

1. Stop lines extend laterally; many points fall outside the camera FOV.
2. val/054's invalid frames contain stop-line-heavy intersections.

Train and test raw-9 coverage is ~0.9, so the underlying signal is fine;
val just got unlucky on sequence assignment + 054 drift.

### 8.6 Result -- sequence outliers

| split | sequence | marking valid | overall valid |
| --- | --- | ---: | ---: |
| val | 054 | 0.451 | 0.402 |

Only val/054 falls below the 0.6 outlier threshold. Confirms timing-policy
prediction.

### 8.7 Decisions resulting from this audit

- Confirm the eval strategy (section 16): always report val (all 9 seqs)
  alongside val (8 seqs, excluding 054).
- Always report marking IoU broken down by distance bucket. This is where
  the E0 vs D0 comparison lives -- the 10-20 m bucket is both D0's
  worst-case for intensity ambiguity and E0's lowest RGB coverage; the
  net effect is the empirical question.
- Always report per-subtype results. Stop-line gain on val will look
  understated relative to train+test; spell it out.

Sources:

- `logs/milestone_e/dataset_analysis/valid_rgb_audit/summary.md`
- `logs/milestone_e/dataset_analysis/valid_rgb_audit/per_frame.csv`
- `logs/milestone_e/dataset_analysis/valid_rgb_audit/aggregate.csv`
- `logs/milestone_e/dataset_analysis/valid_rgb_audit/marking_subtype.csv`
- `logs/milestone_e/dataset_analysis/valid_rgb_audit/per_sequence.csv`

---

## 9. Step 6 -- Occlusion screening

### 9.1 Question

When a LiDAR point projects "inside the image", is it actually visible from
the camera, or is some closer object in the way (so the pixel we'd sample
shows that other object)?

### 9.2 Approach

Per-frame z-buffer. For every point that's in-image, discretize `(u, v)` to
integer pixel coordinates. For each pixel, keep the nearest point's depth.
Any other point at that pixel deeper than `min_depth + Z_TOLERANCE_M` is
occluded.

`Z_TOLERANCE_M = 0.5` to avoid flagging coplanar surface samples
(road points at slightly different depths at the same pixel) as occluding
each other.

Source: `logs/milestone_e/dataset_analysis/occlusion_audit/analysis_code/run_audit.py`

### 9.3 Result

| split | overall occluded | road | **marking** | other |
| --- | ---: | ---: | ---: | ---: |
| train | 0.0016 | 0.0003 | **0.0003** | 0.0027 |
| val   | 0.0016 | 0.0004 | **0.0002** | 0.0025 |
| test  | 0.0013 | 0.0004 | **0.0004** | 0.0021 |

0.03 % of marking points are occluded. Two orders of magnitude below the
2 % decision threshold.

By distance: monotonically rising 0.00 % at 0-10 m -> 0.16 % at 60+ m.
Still trivial across the board.

### 9.4 Decision rule and outcome

Pre-committed rule:

| marking occluded fraction | policy |
| --- | --- |
| < 2 % | skip occlusion handling, document |
| 2 - 10 % | judgment call |
| > 10 % | mandatory z-buffer in dataset code |

Outcome: **skip occlusion handling for E0.** Documented as a known
limitation in the milestone writeup. Far below threshold; the
implementation complexity is not justified.

### 9.5 Why the number is so small

Markings live on the road. The road is one continuous flat surface
visible to a forward camera. To occlude a marking point, an object must be
physically between the camera and that exact spot of paint -- mostly cars
at intersections. Across a whole dataset of driving, this is uncommon.

The "other" class has 0.27 % occlusion (10x marking), which makes sense
-- "other" includes pedestrians, signs, building parts that often have
foreground occluders. Not Milestone E's concern.

Sources:

- `logs/milestone_e/dataset_analysis/occlusion_audit/summary.md`
- `logs/milestone_e/dataset_analysis/occlusion_audit/aggregate.csv`
- `logs/milestone_e/dataset_analysis/occlusion_audit/marking_subtype.csv`

---

## 10. Step 7 -- Voxel-subsampling rgb_valid policy

### 10.1 Question

The preprocessor voxelizes at `grid_size = 0.04`. Multiple points can fall
into one voxel and their features get averaged. If a voxel contains one
valid contributor (`rgb = red`, `rgb_valid = 1`) and one invalid
(`rgb = 0`, `rgb_valid = 0`), the averaged voxel is `rgb = (red/2, 0, 0)`,
`rgb_valid = 0.5`. That's a value the network has never seen as a clean
signal.

What fraction of marking voxels are mixed?

### 10.2 Approach

Discretize world coordinates to integer voxel indices at the same
`grid_size`. Group points by voxel. For voxels containing at least one
marking point, classify as pure-valid, pure-invalid, or mixed.

Source: `logs/milestone_e/dataset_analysis/voxel_mixing_audit/analysis_code/run_audit.py`

### 10.3 Result

Per-frame on the 8 step-3 frames:

| split/seq | lid | mode | voxels with marking | mixed | mixed % |
| --- | ---: | --- | ---: | ---: | ---: |
| train/003 | 00 | same | 510 | 0 | 0.00 % |
| train/017 | 00 | same | 545 | 0 | 0.00 % |
| train/017 | 40 | same | 2109 | 2 | 0.09 % |
| train/037 | 52 | same | 2808 | 1 | 0.04 % |
| val/054   | 00 | same | 1684 | 0 | 0.00 % |
| val/054   | 00 | near | 1684 | 0 | 0.00 % |
| val/054   | 79 | same | 2011 | 0 | 0.00 % |
| val/106   | 20 | same |  161 | 0 | 0.00 % |

Aggregate: **0.03 %** of marking-containing voxels are mixed.

### 10.4 Decision rule and outcome

Pre-committed rule:

| mixed voxels (marking) | policy |
| --- | --- |
| < 5 % | pass-through fractional flag, document |
| 5 - 15 % | judgment call |
| > 15 % | mask-and-threshold |

Outcome: **pass-through.** Store `rgb_valid` as `float32` so Open3D's
voxel averaging produces a proper fractional value, but accept it as-is.
In practice it is `0.0` or `1.0` in >99.95 % of voxels.

### 10.5 Why the number is so small

Marking points cluster in the camera FOV center (right ahead on the road).
Invalid points cluster at the FOV edges. At 4 cm voxel resolution, those
regions are spatially far apart -- no voxel bridges them.

Source: `logs/milestone_e/dataset_analysis/voxel_mixing_audit/summary.md`

---

## 11. Step 8 -- Cache layout and bake-projection decision

### 11.1 Question

Where does the preprocessing cache live, what does it contain, how do we
invalidate it, and is the projection baked into the cache once or recomputed
every epoch?

### 11.2 Approach

Pure policy decision. No compute. Inputs:

- D0 uses Open3D-ML's cache mechanism via `TorchDataloader(use_cache=...)`.
  The cache stores the result of `preprocess(get_data(idx))` and is
  consulted on every subsequent epoch.
- For E0 we either (a) bake the projection inside `get_data()` so the
  cache stores ready-to-train features, or (b) leave the cache holding
  raw features and run projection at runtime per epoch.

### 11.3 Decisions

```yaml
cache_dir:               logs/milestone_e/cache/E0_rgb_front_v1
cache_mode:              bake_projection_at_build
cache_manifest_file:     <cache_dir>/cache_manifest.json
color_sampling:          bilinear
rgb_normalization:       divide_by_255       # range [0.0, 1.0]
rgb_valid_storage:       uint8 in {0, 1} pre-voxel; float32 post-voxel
rgb_valid_aggregation:   passthrough_fractional
```

### 11.4 Reasoning per decision

**Bake projection at build, not runtime.** The E0 policy is locked
(steps 4, 6, 7). We will not iterate inside the E0 run itself. Baking
saves ~50 ms/frame * 4640 frames * 25 epochs = ~1.6 hours of training-loop
projection. Rebuilding a cache for a future policy variant is one-time
~5 minutes. Tiny next to a 4-hour training run.

**Versioned cache directory (`_v1`).** Any policy change requires a new
versioned dir; never overwrite. The manifest mechanism makes accidental
reuse impossible (it will raise on mismatch).

**Bilinear color sampling.** A projected uv at a paint/asphalt boundary
sits between pixels. Bilinear averages four neighbors, reducing aliasing.
Cost is four lookups instead of one -- negligible.

**Normalize by 255 to [0, 1].** No standardization for E0. We have no
train-split RGB statistics yet, and the first layer of the model can
absorb a bounded range without standardization. If E0 underperforms and
the residual smells like a normalization issue, standardize in E1 and
bump the cache version.

**Pass-through fractional `rgb_valid` post-voxel.** Step 7 measured the
mixing rate at 0.03 %. Threshold logic would add code without changing
inputs the model sees.

### 11.5 Cache manifest

Every relevant config field is written into `cache_manifest.json` at build
time. On every training launch, the dataset class reads this manifest and
compares it field-by-field with the active config. Mismatch -> abort with
a clear diff. Missing manifest -> abort ("looks like a foreign cache").

The manifest covers:

- `label_mode`, `feature_mode`, `camera_name`, `camera_lookup`
- `rgb_max_dt_s`, `color_sampling`, `rgb_normalization`, `motion_compensation`
- `intensity_clip_low`, `intensity_clip_high`, `intensity_mean`, `intensity_std`
- `cache_grid_size`, injected from the active `model.grid_size` by the trainer
- `forward_sensor_id`
- `schema_version`, `created_at`

This is the safety net. Any time you accidentally point an experiment at
a stale cache, training refuses to start instead of silently using wrong
features.

Source: `logs/milestone_e/notes/cache_policy.md`

---

## 12. Step 9 -- Dataset extension + single-sample smoke

### 12.1 Question

Implement the dataset code that produces the new 5-channel feat, build a
cache manifest mechanism, and verify on a single sample that:

- D0 path still works (backwards compatibility)
- E0 path produces correctly-shaped, correctly-typed, correctly-valued
  features for one frame
- Per-class RGB-valid ratios on the test frame match the step-5 audit
- A visual overlay confirms the right colors went to the right points

### 12.2 Code changes

#### A. `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`

Added (existing functions unchanged):

```python
FEATURE_MODE_INTENSITY = "intensity"
FEATURE_MODE_INTENSITY_RGB_FRONT = "intensity_rgb_front"
VALID_FEATURE_MODES = {...}

CAMERA_LOOKUP_NEAREST_TIMESTAMP = "nearest_timestamp"
VALID_CAMERA_LOOKUPS = {...}

COLOR_SAMPLING_BILINEAR = "bilinear"
COLOR_SAMPLING_NEAREST  = "nearest"

def validate_feature_mode(...): ...
def validate_camera_lookup(...): ...
def validate_color_sampling(...): ...

def project_points_to_camera(points_world, camera_pose, intrinsics,
                             image_w, image_h):
    # Returns (uv, depth, in_image_mask), all length N.
    # Identical math to the audit scripts; verified against the
    # devkit projection in step 2.

def bilinear_sample_rgb(image_array, uv, valid_mask):
    # Returns (N, 3) float32 in [0, 1].
def nearest_sample_rgb(image_array, uv, valid_mask):
    # Same shape; nearest pixel.
```

The existing `LABEL_MODE_*`, `validate_label_mode`,
`remap_raw_pandaset_ids`, `build_one_sample` and `inspect_sample_main`
functions were not touched.

#### B. `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py`

New constructor args (all with defaults that reproduce D0):

```python
feature_mode:    str = FEATURE_MODE_INTENSITY,        # D0 default
camera_name:     str = "front_camera",
camera_lookup:   str = CAMERA_LOOKUP_NEAREST_TIMESTAMP,
rgb_max_dt_s:    float = 0.060,
color_sampling:  str = COLOR_SAMPLING_BILINEAR,
rgb_normalization: str = "divide_by_255",
motion_compensation: str = "none",
```

New helper methods:

- `_build_cache_manifest()` -- returns the dict that gets written/checked.
- `_validate_or_write_cache_manifest(cache_dir)` -- enforced at `__init__`
  time when `use_cache` is on AND feature_mode != intensity.
- `_get_camera_metadata(seq_id)` -- lazy per-sequence cache of intrinsics,
  poses, timestamps. JSON-only. No `Camera.load()` call, ever.
- `_compute_rgb_features(seq_id, frame_idx, xyz_world)` -- returns
  `(rgb (N, 3) float32 in [0, 1], rgb_valid (N,) float32 in {0, 1})`.

The only change to `_load_sample` is at the feat-assembly step:

```python
if self.feature_mode == FEATURE_MODE_INTENSITY_RGB_FRONT:
    rgb, rgb_valid = self._compute_rgb_features(seq_id, frame_idx, xyz_world)
    feat = np.concatenate(
        [intensity_col, rgb, rgb_valid[:, None]], axis=1
    ).astype(np.float32, copy=False)
else:
    feat = intensity_col
```

#### C. `logs/milestone_e/configs/e0_rgb_front.yml`

Copy of `logs/milestone_d/configs/d0_weighted_ce.yml` with these changes:

```yaml
dataset:
  feature_mode: intensity_rgb_front
  camera_name: front_camera
  camera_lookup: nearest_timestamp
  rgb_max_dt_s: 0.060
  color_sampling: bilinear
  rgb_normalization: divide_by_255
  motion_compensation: none
  cache_dir: logs/milestone_e/cache/E0_rgb_front_v1
  test_result_folder: logs/milestone_e/test_results/E0_rgb_front
model:
  in_channels: 8                    # xyz (3) + intensity + r + g + b + rgb_valid
pipeline:
  main_log_dir: logs/milestone_e/open3d_logs
  train_sum_dir: logs/milestone_e/tensorboard
```

Everything else (class weights, scheduler, optimizer, augmentations,
sampler, etc.) is identical to D0. The whole point of E0 is to vary
exactly one variable -- input features -- so the result is interpretable.

#### D. Smoke script

`logs/milestone_e/dataset_analysis/sample_smoke/analysis_code/smoke_load_one_sample.py`

Loads one sample (seq 003, frame 0) with each feature mode and:

- asserts shapes (`feat.shape[1] == 1` for D0, `== 5` for E0)
- asserts intensity is finite and roughly N(0,1)
- asserts RGB is in [0, 1]
- asserts `rgb_valid` is binary `{0.0, 1.0}` pre-voxel
- prints per-class valid ratios
- renders a visual overlay of the colored point cloud over the camera image

### 12.3 Smoke result

Ran locally on seq 003 frame 0. Output:

```
=== D0 path (feature_mode=intensity) ===
  D0 feat: shape=(69411, 1)  dtype=float32   # D0 still works

=== E0 path (feature_mode=intensity_rgb_front) ===
  point: shape=(69411, 3) ego xyz
  feat:  shape=(69411, 5)
  intensity: mean=-0.044, std=0.983         # properly standardized
  RGB channel means: R=0.246 G=0.251 B=0.268
  rgb_valid ratio: 0.9095 (0s=6283, 1s=63128)
  per-class:
    road     total= 29636  valid_ratio=0.9761
    marking  total=   560  valid_ratio=1.0000   # every marking point got RGB
    other    total= 39102  valid_ratio=0.8575
```

These per-class ratios match the step-5 audit. The implementation
agrees with the auditing scripts -- there is no silent mismatch.

### 12.4 Backwards-compatibility check

Same script loaded the same frame through the default `feature_mode =
intensity` path and confirmed `feat.shape == (N, 1)`. D0 is intact.

Sources:

- `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`
- `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py`
- `logs/milestone_e/configs/e0_rgb_front.yml`
- `logs/milestone_e/dataset_analysis/sample_smoke/analysis_code/smoke_load_one_sample.py`
- `logs/milestone_e/dataset_analysis/sample_smoke/smoke_report.json`
- `logs/milestone_e/dataset_analysis/sample_smoke/smoke_overlay.png`

---

## 13. Accepted limitations (consolidated)

All the items below were considered, measured, and accepted as known
limitations rather than fixed for E0. Listed here so they don't get
re-litigated later.

### 13.1 Lens distortion not modeled

The devkit projection treats intrinsics as ideal pinhole. Real lenses curve
slightly. Visual sanity check showed no obvious bias in our 8 frames.
Could matter at long range / image corners; not measured. If E0 fails in a
distance-correlated way at the image edges, this is a candidate cause.

### 13.2 No ego-motion compensation

At -50 ms systematic offset and ~40 km/h, ego moves ~0.55 m between LiDAR
and camera. Marking-only panels visually OK. Documented as a future-work
item.

### 13.3 No per-point LiDAR-sweep time

PandaSet's LiDAR sweep is ~100 ms. Different points within a frame have
different timestamps; we use one camera pose for the whole sweep. Same
class of issue as above; not corrected.

### 13.4 No occlusion correction

Measured: 0.03 % marking points occluded. Skipping. (Step 6.)

### 13.5 `rgb_valid` fractional flag passed through

Measured: 0.03 % mixed voxels. Skipping mask-and-threshold. (Step 7.)

### 13.6 val/054 timing anomaly

53.75 % of seq 054 frames invalidated under our policy. Sequence stays in
val for D0 vs E0 comparability. Eval slices report with and without 054
(section 16). On the invalid frames, E0 effectively falls back to
LiDAR-only behavior (rgb_valid=0). This is documented as a feature
(robust degradation) not a bug.

### 13.7 RGB not standardized

Raw [0, 1]. Could matter if the first MLP layer's optimization landscape
hates the scale mismatch with standardized intensity. Easy to revisit in
E1.

### 13.8 E0 vs D0 is not a pure RGB-on/off ablation

E0 adds 4 input channels to the first MLP, so it has slightly more
parameters than D0. Some of any improvement could come from added
capacity, not added information. Clean control would be an E0-shape
network with RGB zeroed-out. Possible follow-up; out of scope for E0
itself.

### 13.9 Lighting / time-of-day coupling

E0 learns RGB appearance from PandaSet's training distribution. Won't
generalize to wildly different lighting without seeing it. Standard
domain-shift limitation; thesis writes this in the limitations section.

### 13.10 Long-range markings get sub-pixel paint

12 cm lane line at 60 m is sub-pixel. JPEG smear dominates. E0 will not
fix the 60+ m bucket regardless of how good projection is. Sensible
thesis claim: "RGB helps in the regime where it can help (close to medium
range)", not "across the board".

---

## 14. Evaluation strategy (locked decisions)

Before any E0 results exist, we commit to the following reporting slices
so we can't be accused of post-hoc cherry-picking:

1. **Headline metric**: marking IoU on **full val (all 9 sequences,
   including 054)**. Honest, includes the hard case, comparable to D0.
2. **Secondary metric**: marking IoU on **val excluding 054**. Shows
   the RGB gain on clean-timing data; the difference between (1) and (2)
   is the cost of seq 054's timing artifact.
3. **Train and test marking IoU**: reported prominently. These do not
   contain the seq-054 anomaly and carry the cleaner signal.
4. **Per-sequence marking IoU** on val: 9 numbers. Lets the reader see
   exactly what 054 looks like.
5. **Marking IoU by distance bucket** (same buckets as D0's analysis).
   This is where the E0 vs D0 story lives -- the 10-20 m bucket is both
   D0's worst intensity ambiguity and E0's lowest RGB coverage; the net
   effect is the empirical question.
6. **Marking IoU by raw subtype** (raw 8 / 9 / 10). Stop-line gain on
   val will look weaker due to lateral FOV + 054 timing; train+test
   carry the real signal.
7. **Marking IoU stratified by per-frame `rgb_valid`** (frames where
   the model had RGB vs frames where it fell back to LiDAR-only). This
   separates "did RGB help?" from "did the fallback degrade gracefully?"
8. **Predicted/true marking ratio** (D0 was 1.46x at epoch 18,
   1.82x at epoch 25). If E0 also drifts, RGB didn't fix the
   loss-weight-driven overprediction; that becomes its own observation.

A reviewer who reads (1)-(8) cannot fairly argue we hid anything.

---

## 15. Code architecture summary

### 15.1 Files changed (existing)

- `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`
  - Added feature/lookup/sampling enums + validators
  - Added `project_points_to_camera`, `bilinear_sample_rgb`,
    `nearest_sample_rgb`
  - Existing label-mode and remap functions untouched
- `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py`
  - Added new `__init__` kwargs (defaulted to D0 behavior)
  - Added cache manifest validation/write at `__init__`
  - Added per-sequence camera-metadata cache
  - Added `_compute_rgb_features`
  - Modified `_load_sample` to branch on feature_mode at the feat
    assembly step
  - Existing intensity-only path is identical to D0

### 15.2 Files created

- `logs/milestone_e/configs/e0_rgb_front.yml` -- E0 trainer config
- `logs/milestone_e/notes/projection_contract.md`
- `logs/milestone_e/notes/timestamp_policy.md`
- `logs/milestone_e/notes/cache_policy.md`
- `logs/milestone_e/dataset_analysis/...` (see sources per step)
- `logs/milestone_e/README.md` -- short status index
- `logs/milestone_e/context.md` -- this document

### 15.3 D0 backwards-compatibility contract

- `feature_mode` defaults to `"intensity"`. All other RGB params have
  benign defaults.
- The manifest mechanism is skipped when `use_cache=False` OR
  `feature_mode == "intensity"`. D0 configs and one-off inspect-sample
  tooling don't pay any cost.
- The intensity-only path in `_load_sample` was not modified.
- D0's existing cache at `logs/milestone_d/cache/D0_weighted_ce/` is
  untouched. E0 builds in a separate dir.
- Verified empirically by the smoke script: D0-config sample still
  returns `feat.shape == (N, 1)`.

---

## 16. Source map (where things live)

```text
logs/milestone_e/
  README.md                         # short status index (steps table)
  context.md                        # THIS FILE
  configs/
    e0_rgb_front.yml                # the E0 trainer config
  notes/                            # short curated policy docs
    projection_contract.md          # step 2
    timestamp_policy.md             # step 4
    cache_policy.md                 # step 8
  dataset_analysis/                 # audits and smokes
    front_camera_inventory/         # step 1
    projection_overlay/             # step 3 (v1/v2/v3 + class + per-seq)
    timestamp_policy/               # step 4 evidence
    valid_rgb_audit/                # step 5
    occlusion_audit/                # step 6
    voxel_mixing_audit/             # step 7
    sample_smoke/                   # step 9
  cache/                            # built at training time
    E0_rgb_front_v1/                # versioned per policy

src/thesis_pipeline/adapters/pandaset_ff_lane3.py
src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py
```

D0 references for comparison:

```text
logs/milestone_d/
  README.md                                            # D0 context
  configs/d0_weighted_ce.yml                           # D0 trainer config
  runs/D0_weighted_ce_25ep/                            # D0 outputs
  run_analysis/D0_weighted_ce_25ep/reports/
    d0_conclusions.md                                  # D0 narrative
    d0_weighted_ce_results.md                          # D0 numbers
  road_marking3_training_statistics.json               # class weights + intensity stats reused for E0
```

---

## 17. Status table

| step | what | status | evidence |
| ---: | --- | --- | --- |
| 1 | Inventory check | done | `dataset_analysis/front_camera_inventory/` |
| 2 | Projection contract | done | `notes/projection_contract.md` |
| 3 | Visual overlay sanity | done | `dataset_analysis/projection_overlay/` |
| 4 | Timestamp + ego-motion policy | done | `notes/timestamp_policy.md`, `dataset_analysis/timestamp_policy/` |
| 5 | Stratified valid-RGB audit | done | `dataset_analysis/valid_rgb_audit/` |
| 6 | Occlusion screening (0.03 %) | done -- skip | `dataset_analysis/occlusion_audit/` |
| 7 | Voxel mixing (0.03 %) | done -- pass-through | `dataset_analysis/voxel_mixing_audit/` |
| 8 | Cache layout + bake policy | done | `notes/cache_policy.md` |
| 9 | Dataset extension + sample smoke | done | `dataset_analysis/sample_smoke/`, code changes in `src/` |
| 10 | Tiny training smoke | next | -- |

---

## 18. Future plan

### 18.1 Step 10 -- tiny training smoke

The last gate before launching full E0 training.

Plan:

1. Delete any existing `logs/milestone_e/cache/E0_rgb_front_v1/` so the
   build is fresh.
2. Use `tools/train_milestone_e.py` with either:
   a. a hand-truncated train index (1-2 sequences) or
   b. a tiny `--epochs 2` run on the full data with `save_ckpt_freq: 1`.
   The E wrapper defaults to `logs/milestone_e/configs/e0_rgb_front.yml` and
   `logs/milestone_e/runs`, and logs `milestone_e_run_complete` on success.
3. Watch for:
   - cache manifest gets written without errors
   - model accepts `in_channels=8`
   - first epoch finishes without NaNs
   - validation loss is finite
   - marking IoU is finite and probably small (only 1-2 epochs)
   - cache contains files after the first epoch
   - manifest re-read on a second run succeeds with identical config and
     fails when one config field is changed
4. If anything dies, fix and re-run. Do not launch the full run until the
   smoke is clean.

Deliverables:

- `logs/milestone_e/launch_logs/E0_smoke.stdout.log`
- `logs/milestone_e/runs/E0_smoke/` (will be deleted before full run)
- a one-page note confirming the smoke passed under
  `logs/milestone_e/notes/training_smoke_result.md`

### 18.2 Full E0 training

Once step 10 passes:

```bash
rm -rf logs/milestone_e/cache/E0_rgb_front_v1
rm -rf logs/milestone_e/runs/E0_rgb_front
mkdir -p logs/milestone_e/launch_logs

nohup python tools/train_milestone_e.py \
  --run-name E0_rgb_front \
  --epochs 25 \
  --save-ckpt-freq 1 \
  --force \
  > logs/milestone_e/launch_logs/E0_rgb_front_fresh_25.stdout.log 2>&1 &
```

Expected wall clock: similar to D0 (~4 hours on the A100). First epoch
will be slower because the cache is being built (projection happens
during the first read of each frame). Subsequent epochs match D0
because the cache returns prebuilt features.

Best checkpoint selection: post-run by raw marking IoU (`lane_iou` column
in the eval history, same alias convention as D0).

### 18.3 E0 analysis

Mirror the D0 analysis structure under
`logs/milestone_e/run_analysis/E0_rgb_front/`.

Required outputs:

- `run_plots/` -- D0 plot suite plus the new RGB-stratified plots
- `reports/e0_results.md` -- run summary with D0 vs E0 headline tables
- `reports/e0_conclusions.md` -- final E0 narrative, mirroring d0_conclusions
- `sampled_error_analysis_epoch<best>/` -- D0-style sampled diagnostic
  plus a new comparison: error rates on rgb_valid vs rgb_invalid points

The eight evaluation slices in section 14 are mandatory in the report.

### 18.4 If E0 succeeds

Conditions for "succeeds":

- E0 marking IoU > D0 marking IoU at best epoch
- E0 marking precision > D0 marking precision (closing the bright-road
  false positives is the hypothesis)
- E0 predicted/true marking ratio closer to 1.0 than D0's 1.46x
- Gains are largest where D0 had the worst intensity ambiguity (10-20 m
  bucket, raw 8 + raw 10 subtypes)

If those hold, the thesis claim is supported.

Possible follow-ups (only if time):

- E0 capacity-control ablation (E0-shape network fed `xyz + intensity +
  0 + 0 + 0 + 0`) to isolate the contribution of RGB vs added capacity.
- Soft-marking-weight ablation (the same idea D1 would have been). If E0
  also drifts toward overprediction late in training, this becomes a
  natural follow-up.

### 18.5 If E0 underperforms

Candidate causes (pre-listed so the debug isn't blind):

1. Misprojection -- caught by the step-3/step-9 visual checks if present;
   revisit if results are weird.
2. Class weight too aggressive given the new signal -- recall stays high,
   precision still drops. Try a softer marking weight.
3. RGB redundant with intensity for the residual D0 errors. The thesis
   would still discuss what this means; E0 stops being a positive result
   but is still informative.
4. Lighting / sequence-distribution coupling. Stratify by lighting
   indicator (sequence-mean image brightness?) post-hoc.
5. Capacity confound -- ablate.

### 18.6 Thesis writeup integration

Milestone E becomes a chapter / extended section after the D0 chapter:

1. **Motivation** -- D0's residual error mode (section 1.4 of this doc).
2. **Hypothesis** -- section 2.
3. **Method** -- everything in steps 1-9, compressed. The auditing
   structure itself is methodology.
4. **Results** -- D0 vs E0 tables and stratifications.
5. **Discussion** -- what RGB did and did not fix, the limitations
   section straight from section 13, future-work straight from
   section 18.

---

## 19. Glossary

- **D0**: the LiDAR-only baseline run, `D0_weighted_ce_25ep`. Frozen.
- **E0**: the LiDAR + front-camera-RGB run. Not yet launched.
- **road_marking3**: the 3-class label remap (road / marking / other).
  `marking` = raw 8 + raw 9 + raw 10.
- **raw 8 / 9 / 10**: PandaSet raw class IDs for lane line / stop line /
  other road marking.
- **forward sensor 1**: PandaSet's forward-facing LiDAR sensor; the one
  D0 and E0 both use.
- **rgb_valid**: per-point binary flag, 1 if the point got a valid pixel
  color, 0 otherwise. Stored as float so it survives voxel averaging.
- **dt**: `cam_ts - lid_ts` for a chosen camera frame; rejected if
  `|dt| > rgb_max_dt_s` (60 ms).
- **same-index vs nearest-timestamp**: two policies for picking a camera
  frame for a given LiDAR frame. We use nearest-timestamp.
- **mixed voxel**: a voxel containing both rgb_valid=1 and rgb_valid=0
  contributors after voxel-grid averaging.
- **bake at build**: store the projected RGB into the cache once, never
  recompute during training.

---

This document is the single source of truth for Milestone E up to step 9.
If you're picking up the work, read sections 1, 2, 14, 17, and 18 first,
then drill into individual step sections as needed.
