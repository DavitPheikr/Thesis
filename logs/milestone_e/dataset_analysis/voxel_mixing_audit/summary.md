# Voxel-Mixing Audit

Grid size: 0.04 m. 8 frames sampled.

## Per-frame mixing of marking voxels

| split/seq | lid | mode | voxels w/ marking | mixed | mixed % |
| --- | ---: | --- | ---: | ---: | ---: |
| train/003 | 00 | same_index | 510 | 0 | 0.00% |
| train/017 | 00 | same_index | 545 | 0 | 0.00% |
| train/017 | 40 | same_index | 2109 | 2 | 0.09% |
| train/037 | 52 | same_index | 2808 | 1 | 0.04% |
| val/054 | 00 | same_index | 1684 | 0 | 0.00% |
| val/054 | 00 | nearest_ts | 1684 | 0 | 0.00% |
| val/054 | 79 | same_index | 2011 | 0 | 0.00% |
| val/106 | 20 | same_index | 161 | 0 | 0.00% |

## Aggregate across the 8 frames

- total voxels containing marking: 11512
- of those, mixed valid/invalid contributors: 3
- aggregate mixed fraction: **0.03%**

## E0 decision

**SKIP** — pass-through the fractional rgb_valid flag. Mixed voxels are rare; not worth the implementation complexity. Document as a known limitation.

