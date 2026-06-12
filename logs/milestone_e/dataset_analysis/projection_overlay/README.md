# Projection Overlay Sanity Check (step 3)

Visual gate for the Milestone E projection pipeline. If overlays look wrong
here, every later step is built on a coordinate/pose/intrinsic bug, so this
step has to pass before we move on.

## Frames

| # | split / seq / frame | camera lookup | why |
| ---: | --- | --- | --- |
| 1 | train/003/0 | same-index | known-good early train frame |
| 2 | train/017/0 | same-index | Codex-audited, ~0.92 valid expected |
| 3 | train/017/40 | same-index | middle frame, dynamic scene |
| 4 | train/037/52 | same-index | Codex-audited |
| 5 | val/054/0 | same-index | dt drift +0.45 s -- this overlay should look misaligned on moving objects |
| 6 | val/054/0 | nearest-timestamp | same lidar frame; nearest camera is still ~0.45 s away, so final E0 policy invalidates RGB for this frame |
| 7 | val/054/79 | same-index | end of seq 054, dt back to -50 ms -- should look fine |
| 8 | val/106/20 | same-index | another val sequence for diversity |

## How

```bash
./panda/bin/python logs/milestone_e/dataset_analysis/projection_overlay/analysis_code/render_overlays.py
```

Outputs:

- `overlays/<NN>__<split>_<seq>_f<frame>_<mode>.png` -- one PNG per frame
  with projected forward-LiDAR points colored by camera-frame depth.
- `overlays.csv` -- one row per frame with valid ratio, point count, and the
  dt actually used.
- `summary.md` -- per-frame quick-look table.

## Pass criteria (visual)

For each PNG:

- Forward LiDAR points densely cover the road surface in the lower portion
  of the image.
- Points on car/pedestrian/pole silhouettes coincide with those silhouettes
  in the image, not offset.
- No systematic shift of points toward one image corner (would indicate lens
  distortion not being accounted for).
- For frames 5 and 6 (`val/054/0`), the closest available camera timestamp
  is still about 0.45 s away. These overlays are useful as a drift example,
  but final E0 training should mark this lidar frame as `rgb_valid=0` for all
  points under the 60 ms timestamp policy.

If any of frames 1-4, 7, or 8 look misaligned, stop. Do not proceed to step 4
until the cause is understood and fixed.
