# Occlusion Audit (step 6)

Measures how often a LiDAR point that successfully projects into the front
camera image is actually occluded by a closer point at the same image pixel.

## Why

The PandaSet projection only checks that a point is in front of the camera
and inside the image. It does NOT check whether the point is visible from
the camera. If a marking point on the road has a parked car between it and
the camera, the projection still maps the marking point onto the car's
silhouette pixels, and our RGB sampler would attach the car's color to a
marking point. That is the wrong color, and it confuses the model.

The audit quantifies the size of this effect for E0.

## Approach

For each frame:

1. Apply the step-4 timestamp policy (nearest_ts + 60 ms).
2. Project forward-LiDAR points.
3. Round (u, v) to integer pixel coordinates.
4. For each pixel, keep the nearest point (smallest camera z).
5. Any point whose depth exceeds the nearest depth at its pixel by more
   than `Z_TOLERANCE_M` is flagged as **occluded**.
6. Count occluded points by class, by distance bucket, by marking subtype.

A point with `rgb_valid = 0` from step 5 (frame invalidated, behind camera,
or out of image) is not counted as occluded; it was never going to receive
RGB.

## Tolerance

`Z_TOLERANCE_M = 0.5` (meters).

Reason: LiDAR points on the same planar surface (e.g. the road) at the same
pixel typically differ in depth by centimeters because of view angle. We
don't want to flag those as occluding each other. A real occlusion (car in
front of road) differs by at least meters.

## How

```bash
./panda/bin/python logs/milestone_e/dataset_analysis/occlusion_audit/analysis_code/run_audit.py
```

Outputs:

- `per_frame.csv` -- one row per frame with occlusion counts.
- `aggregate.csv` -- per split x class x distance bucket.
- `marking_subtype.csv` -- per split x raw 8/9/10.
- `summary.md` -- headline tables and the E0 decision rule outcome.

## Decision rule for E0

- marking occluded fraction < 2 %    => skip occlusion handling for E0,
                                       document as limitation.
- marking occluded fraction in 2-10% => judgment call.
- marking occluded fraction > 10 %   => add z-buffer to E0 dataset code.
