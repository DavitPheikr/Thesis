# E0 Cache Layout and Build Policy

Final policy for where the E0 preprocessing cache lives, what it contains,
when to invalidate it, and the small sampling/normalization choices that go
with it.

## Decisions

```yaml
cache_dir:            logs/milestone_e/cache/E0_rgb_front_v1
cache_mode:           bake_projection_at_build
cache_manifest_file:  logs/milestone_e/cache/E0_rgb_front_v1/cache_manifest.json
color_sampling:       bilinear
rgb_normalization:    divide_by_255    # range [0.0, 1.0]
rgb_valid_storage:    uint8 in {0, 1}  # passed through after voxel averaging
rgb_valid_dtype_in_voxel: float32       # fractional after averaging, no thresholding
```

## Cache location and naming

```text
logs/milestone_e/cache/<run_or_policy_name>_v<N>/
```

E0 uses `E0_rgb_front_v1`. The `_v1` suffix is mandatory: any policy change
that affects preprocessing requires bumping to `_v2`, never overwriting
`_v1` in place. This is the same lesson Milestone D recorded after its own
cache drama.

## What gets cached

Per frame (after grid subsampling at `grid_size = 0.04`):

```text
points        float32   (N_sub, 3)   ego-frame xyz
feat          float32   (N_sub, 5)   [intensity, r, g, b, rgb_valid]
labels        int32     (N_sub,)     road_marking3 labels (0..3)
```

Plus, written once at the cache root:

- `cache_manifest.json` -- see below.

## Bake projection at build, not at runtime

For E0 we bake the projection into the cache:

- Cache build: project every frame once, sample RGB, write features.
- Training: load preprocessed features as-is, no per-epoch projection.

Why:

1. Projection policy is locked for E0 (steps 4, 6, 7). We are not iterating
   on it inside the E0 run itself.
2. A baked cache saves roughly 50 ms per frame on data loading. Over 25
   epochs of 4640 train frames, that is ~1.6 hours.
3. Rebuild for a future policy change (motion compensation, threshold tweak,
   nearest sampling) is a one-time ~5 minute job. Tiny next to a 4-hour
   training run.
4. Determinism: same RGB every epoch, no risk of subtle non-determinism
   from intermediate libraries during loading.

## Cache manifest

A `cache_manifest.json` is written at build time and read on every training
launch. If anything in this manifest does not match the active config, the
loader aborts and tells the user to bump the version.

```json
{
  "schema_version": 1,
  "created_at": "<ISO timestamp>",
  "label_mode": "road_marking3",
  "feature_mode": "intensity_rgb_front",
  "forward_sensor_id": 1,
  "cache_grid_size": 0.04,

  "camera_name": "front_camera",
  "camera_lookup": "nearest_timestamp",
  "rgb_max_dt_s": 0.060,
  "motion_compensation": "none",

  "color_sampling": "bilinear",
  "rgb_normalization": "divide_by_255",
  "rgb_invalid_fill": [0.0, 0.0, 0.0],
  "rgb_valid_policy": "zero_with_valid_flag",
  "rgb_valid_aggregation": "passthrough_fractional",

  "intensity_clip_low": 0.0,
  "intensity_clip_high": 114.0,
  "intensity_mean": 22.471752166748047,
  "intensity_std": 14.902379989624023
}
```

## When to invalidate the cache

Any change to a manifest field requires building a new cache version. Do
NOT overwrite. The fields above are the complete list of things that, if
changed, invalidate the cache. The most common causes:

- different `rgb_max_dt_s`
- adding motion compensation
- switching `color_sampling` between bilinear and nearest
- changing `rgb_normalization`
- switching to a different camera (e.g. side cameras)
- changing `label_mode`, `cache_grid_size` / `model.grid_size`, or `forward_sensor_id`

Augmentations that operate on already-cached points (rotation, scale, noise)
do not invalidate the cache. Augmentations that affect what gets stored
(e.g. RandomDropout in a pre-cache step) would, but D0 already ruled
RandomDropout out, so this is not an E0 concern.

## Color sampling: bilinear

Each LiDAR point projects to floating-point `(u, v)` pixel coordinates.
Bilinear sampling takes a weighted average of the four neighbor pixels.

Why bilinear over nearest:

- Reduces aliasing on marking edges where the paint/asphalt boundary
  passes through the projected coordinate.
- Almost-free in cost (4 lookups + 4 multiplies per point).
- Standard practice for image-to-pointcloud feature attachment.

Why not something fancier (Lanczos, bicubic):

- Diminishing returns at 4 cm voxel input.
- Adds dependency or implementation surface for no measurable gain at the
  scale of road-marking segmentation.

## RGB normalization: divide by 255

For E0:

```text
rgb_normalized = rgb_pixel.astype(float32) / 255.0
```

Both R, G, B are independently in `[0, 1]`.

Why not zero-mean/unit-variance standardize:

- We have no train-split RGB statistics yet, and computing them is one
  more pass.
- The model first layer can easily absorb a known bounded range; this is
  not where E0's success or failure will live.
- Avoids coupling to a specific normalization that would need to be
  recomputed if we later swap cameras.

A future E1 could standardize if needed; cache version would bump.

## rgb_valid handling after voxel averaging

Per step 7's audit (mixing rate 0.03 % of marking voxels):

- Pre-voxel: per-point `rgb_valid` is a binary `uint8`, 0 or 1.
- After Open3D's grid subsampling: the averaged value may be fractional,
  but in practice will be 0 or 1 in >99.95 % of voxels.
- Pass through the fractional value as `float32`. Do not threshold.
- Do not implement masked RGB averaging. The mixing rate does not justify
  the code complexity.

If E0's marking IoU is suspiciously low and the residual error correlates
with fractional `rgb_valid` voxels, revisit. Otherwise leave alone.

## Storage cost estimate

D0 cache stores `(N_sub, 1)` intensity per frame. E0 stores `(N_sub, 5)`.
Five times the feature bytes, roughly. With 4640 train frames + 720 val
frames + 720 test frames and per-frame point counts around 30k-50k after
voxelization, the cache is a few hundred MB to ~1 GB extra over D0. Fine.

## What happens if you reuse the wrong cache

The dataset loader must check the cache manifest on every launch. If the
manifest is missing or any field disagrees with the active config, the
loader prints the diff and aborts. Do not silently coerce. This will be
implemented in step 9.

## Summary in one screen

- Cache: `logs/milestone_e/cache/E0_rgb_front_v1/`.
- Bake projection at cache build, not runtime.
- Bilinear color sampling.
- RGB scaled to `[0, 1]` by dividing by 255. No standardization.
- `rgb_valid` is a uint8 flag pre-voxel, float32 fractional post-voxel,
  not thresholded.
- Cache manifest pinned. Mismatch -> abort with a clear error.
- Version bump on any preprocessing change. Never overwrite.
