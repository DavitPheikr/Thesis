# Projection function contract (PandaSet devkit)

Source: `pandaset-devkit/python/pandaset/geometry.py` (line 17).

```python
projection(lidar_points, camera_data, camera_pose, camera_intrinsics,
           filter_outliers=True)
```

## What goes in

- `lidar_points`: `np.ndarray` of shape `(N, 3)`. **In world coordinates**, not
  ego and not LiDAR-sensor coordinates. The function internally builds
  `trans_lidar_to_camera = inv(camera_pose_mat)` and applies it directly to
  `lidar_points.T`, so the input must be in the same world frame as
  `camera_pose`. The parameter name is misleading; do not pass LiDAR-frame
  points.
- `camera_data`: PIL image object (e.g. `seq.camera['front_camera'][i]`).
  Only `.size` is used, to read `(image_w, image_h)` for the in-image filter.
  No pixel data is consumed.
- `camera_pose`: dict for the same frame index `i` as `camera_data`. Must have
  `{"heading": {w,x,y,z}, "position": {x,y,z}}` in world coordinates.
- `camera_intrinsics`: object with attributes `fx`, `fy`, `cx`, `cy`
  (e.g. `seq.camera['front_camera'].intrinsics`). Pinhole only.
- `filter_outliers`: bool, default `True`.

## What comes out

Returns the tuple `(points2d_camera, points3d_camera, inliner_indices_arr)`.

- `points2d_camera`: `np.ndarray` of shape `(M, 2)`. Pixel coordinates `(u, v)`
  for the surviving points.
- `points3d_camera`: `np.ndarray` of shape `(M, 3)`. The surviving points
  expressed in camera coordinates (z = depth forward).
- `inliner_indices_arr`: `np.ndarray` of shape `(M,)`, dtype int. Indices into
  the original `lidar_points` array; lets you map results back to original
  point ordering.

With `filter_outliers=True` (default):

```text
M <= N
keep iff (z_camera > 0)
       AND (0 < u < image_w)
       AND (0 < v < image_h)
```

Note: the in-image check uses strict `>` against 0 and strict `<` against the
image dimensions, so the boundary pixels at `u==0`, `v==0`, `u==image_w`, and
`v==image_h` are excluded. For our purposes this is fine; just be aware of it
when comparing point counts.

With `filter_outliers=False`:

```text
M == N
no z>0 filter, no in-image filter
points2d_camera may contain values outside the image, and points3d_camera may
contain points behind the camera.
inliner_indices_arr == np.arange(N).
```

For E0 we want both the surviving uv and a per-point validity flag against
the original ordering. Use `filter_outliers=True` and reconstruct the flag:

```python
uv, pcam, idx = projection(points_world, image, cam_pose, cam_intrinsics,
                           filter_outliers=True)
rgb_valid = np.zeros(N, dtype=np.uint8)
rgb_valid[idx] = 1
uv_full = np.full((N, 2), np.nan, dtype=np.float32)
uv_full[idx] = uv.astype(np.float32)
```

## What it does NOT do

1. **No lens distortion.** Only `fx, fy, cx, cy` are used. PandaSet intrinsics
   are treated as a pinhole model; if the front-camera images carry lens
   distortion that has not been pre-rectified, this projection will be off
   near the image edges. To check before relying on it: spot-check the visual
   overlay (step 3) at frame corners, especially curbs and lane edges far
   from image center.
2. **No occlusion check.** A point that lies behind a closer object will still
   receive the closer object's pixel color. This is the well-known z-buffer
   problem. Quantifying it is step 6.
3. **No timestamp/motion compensation.** The function takes one camera pose
   (for the camera frame) and uses the same transform for every LiDAR point.
   PandaSet LiDAR sweeps are ~100 ms long; the ego vehicle moves during the
   sweep, and the camera is a snapshot at a different instant. If we want
   per-point motion compensation, we must do it ourselves before calling
   `projection`: bring each LiDAR point to the camera timestamp using pose
   interpolation, then project. Step 4 decides whether E0 does this or
   accepts the per-frame approximation.
4. **No colour sampling.** The function only returns pixel coordinates.
   Sampling RGB at those coordinates (nearest vs bilinear, sub-pixel
   rounding, sRGB vs linear) is on us. We will pin those choices in the E0
   feature-builder config and version-stamp the cache.
5. **No transform from ego to world.** The dataset stores LiDAR points in
   world coordinates by default in `seq.lidar[i]` (PandaSet convention).
   Confirm before relying: if our adapter already converts to ego before
   feeding the model, we need both copies during projection time -- one
   world-frame array for `projection(...)` and one ego-frame array for the
   network input.

## Verification still needed (do in step 3)

The contract above is read from the source. Two things still need a visual
check before we trust it for E0:

- The "world coords in" rule: project a known forward point using the actual
  saved poses and confirm the resulting `(u, v)` lands where expected on the
  image.
- The "no distortion" assumption: overlay a grid of LiDAR points on the
  image; if edge points sit visibly off their physical features, intrinsics
  may not be pre-rectified.

Both are step 3 (visual overlay sanity).
