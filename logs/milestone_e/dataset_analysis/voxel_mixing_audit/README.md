# Voxel-Mixing Audit (step 7)

When the dataset preprocessor groups nearby LiDAR points into 4 cm voxels,
multiple points get merged into one. Their per-point features
(`rgb`, `rgb_valid`) get averaged. If one contributor is RGB-valid and
another is not, the merged voxel ends up with a fractional `rgb_valid` and
a halved RGB color.

## Question

What fraction of voxels containing at least one marking point have mixed
valid/invalid contributors?

## Decision rule

| mixed voxels (marking) | policy |
| --- | --- |
| < 5 % | pass-through fractional flag, document |
| 5 - 15 % | judgment call |
| > 15 % | mask-and-threshold: average RGB only over valid contributors, snap flag to {0, 1} |

## Approach

For the 8 step-3 frames:

1. Load forward LiDAR + semseg.
2. Apply step-4 timestamp policy + project to compute per-point `rgb_valid`.
3. Discretize world coordinates at `grid_size = 0.04` to get a voxel index.
4. For each voxel, count contributors and the fraction that are RGB-valid.
5. Tabulate, for voxels containing at least one marking point:
   - how many voxels are pure-valid (all contributors valid)
   - how many are pure-invalid (none valid)
   - how many are mixed (some valid, some invalid)

## How

```bash
./panda/bin/python logs/milestone_e/dataset_analysis/voxel_mixing_audit/analysis_code/run_audit.py
```

Outputs `per_frame.csv`, `summary.md`, and an auto-stamped decision at the
bottom of the summary.
