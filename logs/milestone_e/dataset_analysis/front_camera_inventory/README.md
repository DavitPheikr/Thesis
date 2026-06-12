# Front-Camera Inventory Check

This is step 1 of the Milestone E pre-training audit.

## Question

For every sequence in `configs/splits/{train,val,test}.txt`, do all of the
following load via the PandaSet devkit without error:

- the front-camera image directory
- the intrinsics (fx, fy, cx, cy)
- per-image poses (position + heading)
- per-image timestamps

And: is the count of images, poses, and timestamps consistent within each
sequence, and is the LiDAR frame count consistent with what Milestone D
trained on (80 frames per sequence)?

## How

```bash
./panda/bin/python logs/milestone_e/dataset_analysis/front_camera_inventory/analysis_code/inventory_front_camera.py
```

Writes:

- `inventory.csv` -- one row per split sequence
- `summary.md` -- aggregate counts and any failures

## Done when

A script prints counts and the first/last timestamps for every sequence and
nothing is missing.
