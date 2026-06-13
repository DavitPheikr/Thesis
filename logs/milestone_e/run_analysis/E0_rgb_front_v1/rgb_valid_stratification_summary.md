# E0 RGB-Valid Coverage Summary

This is a dataset coverage diagnostic, not a checkpoint inference report.
It tells us where E0 had RGB available in the validation split.

- validation frames: `720`
- generated in: `347.4s`

## Overall by Class

| class | total points | valid RGB points | valid ratio |
| --- | ---: | ---: | ---: |
| road | 17550109 | 15753162 | 0.897610 |
| marking | 747235 | 575677 | 0.770410 |
| other | 27623043 | 20966030 | 0.759005 |

## Sequences With Low Marking RGB Coverage

| sequence | marking points | marking valid ratio | mean frame valid ratio |
| --- | ---: | ---: | ---: |
| 054 | 216200 | 0.450791 | 0.398917 |

## Interpretation Boundary

This script can identify low-RGB-coverage sequences, but it cannot answer
whether E0 false positives occur mostly where `rgb_valid=1` or
`rgb_valid=0`. That requires a fresh sampled inference pass with the
epoch-14 checkpoint.
