# Per-Sequence Projection Audit

Frame audited per sequence: index `40` (camera frame chosen by nearest timestamp).

## Coverage across 76 ok sequences (of 76)

| metric | mean | min | p10 | median | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall valid | 0.869 | 0.809 | 0.841 | 0.870 | 0.898 | 0.921 |
| road valid | 0.948 | 0.874 | 0.905 | 0.952 | 0.985 | 1.000 |
| marking valid | 0.934 | 0.634 | 0.825 | 0.967 | 1.000 | 1.000 |

## Timestamp policy outcomes

- sequences with |dt| > 0.06 s at frame 40: **1**

| split | sequence | dt(s) |
| --- | --- | ---: |
| val | 054 | -0.1499 |

## Sequences with marking valid ratio < 0.6

None.

