# Occlusion Audit

Computed in 530.4s over all 6080 frames with Z_TOLERANCE_M = 0.5.

Counts are restricted to points that already pass the step-5 valid-RGB filter (in front of camera, inside image, frame within 60 ms). Occluded fraction = occluded / in_image.

## Overall occlusion by split and class

| split | overall | road occl | marking occl | other occl |
| --- | ---: | ---: | ---: | ---: |
| train | 0.0016 | 0.0003 | 0.0003 | 0.0027 |
| val | 0.0016 | 0.0004 | 0.0002 | 0.0025 |
| test | 0.0013 | 0.0004 | 0.0004 | 0.0021 |

## Marking occlusion by distance bucket

| split | 0_10m | 10_20m | 20_30m | 30_40m | 40_60m | 60m_plus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.0000 | 0.0001 | 0.0005 | 0.0009 | 0.0018 | 0.0016 |
| val | 0.0000 | 0.0001 | 0.0004 | 0.0006 | 0.0009 | 0.0007 |
| test | 0.0001 | 0.0001 | 0.0011 | 0.0012 | 0.0016 | 0.0009 |

## Marking occlusion by subtype (raw 8 / 9 / 10)

| split | raw 8 lane | raw 9 stop | raw 10 other |
| --- | ---: | ---: | ---: |
| train | 0.0002 | 0.0006 | 0.0003 |
| val | 0.0002 | 0.0008 | 0.0003 |
| test | 0.0003 | 0.0008 | 0.0005 |

## E0 decision rule

Train marking occluded ratio = 0.0003 (0.03%).

**Decision: SKIP** occlusion handling for E0. Below 2% threshold. Document as a known limitation.

