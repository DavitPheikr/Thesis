# Per-Frame dt Scan

Total frames scanned (76 sequences x 80 frames): 6080.

## Frames passing under each candidate threshold (nearest-timestamp camera lookup)

| threshold | frames pass | frames fail | pass % |
| ---: | ---: | ---: | ---: |
| 40 ms | 0 | 6080 | 0.00% |
| 60 ms | 6037 | 43 | 99.29% |
| 80 ms | 6037 | 43 | 99.29% |
| 100 ms | 6037 | 43 | 99.29% |
| 150 ms | 6047 | 33 | 99.46% |
| 200 ms | 6056 | 24 | 99.61% |
| 300 ms | 6068 | 12 | 99.80% |
| 500 ms | 6078 | 2 | 99.97% |

## Sequences with any frame failing at the 60 ms candidate

| split | sequence | failing frames / 80 | % |
| --- | --- | ---: | ---: |
| val | 054 | 43 | 53.8% |

## Same-index vs nearest-timestamp

- frames where same_index == nearest_ts: 6010 (98.85%)
- frames where they differ: 70

Sequences where the two policies disagree on at least one frame:

| split | sequence | frames disagreeing |
| --- | --- | ---: |
| val | 054 | 70 |

