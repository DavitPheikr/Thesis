# Stratified Valid-RGB Audit

Computed in 254.0s over all 6080 frames.

## Overall RGB coverage by split and class

| split | overall valid | road valid | marking valid | other valid |
| --- | ---: | ---: | ---: | ---: |
| train | 0.867 | 0.946 | 0.907 | 0.810 |
| val | 0.812 | 0.898 | 0.770 | 0.759 |
| test | 0.870 | 0.938 | 0.917 | 0.821 |

## Marking valid ratio by distance bucket

| split | 0_10m | 10_20m | 20_30m | 30_40m | 40_60m | 60m_plus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.988 | 0.878 | 0.876 | 0.941 | 0.956 | 0.993 |
| val | 0.842 | 0.716 | 0.802 | 0.854 | 0.861 | 0.912 |
| test | 0.997 | 0.882 | 0.928 | 0.937 | 0.976 | 0.982 |

## Marking valid by subtype (raw 8 / 9 / 10)

| split | raw 8 lane line | raw 9 stop line | raw 10 other road marking |
| --- | ---: | ---: | ---: |
| train | 0.934 | 0.926 | 0.887 |
| val | 0.845 | 0.458 | 0.711 |
| test | 0.910 | 0.902 | 0.933 |

## Sequences with marking valid ratio < 0.6

| split | sequence | n_marking | marking valid | overall valid |
| --- | --- | ---: | ---: | ---: |
| val | 054 | 216200 | 0.451 | 0.402 |

## Cost of the timestamp policy

- frames total: 6080
- frames invalidated (|dt| > 60 ms): 43 (0.71%)

