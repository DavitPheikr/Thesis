# Milestone D Training Statistics

## Scope

- split: training only
- label mode: road_marking3
- positive class: raw 8 + raw 9 + raw 10 road markings

## Class Counts

| class | count | active share |
| --- | ---: | ---: |
| road | 119,562,394 | 40.099378% |
| marking | 5,129,328 | 1.720297% |
| other | 173,473,484 | 58.180324% |
| ignore | 398,730 | n/a |

## Recommended Config Weights

Use this count-like list in the Open3D-ML config:

```text
[119562394.0, 5129328.0, 173473484.0]
```

Open3D-ML transforms that list internally into effective CE weights:

```text
[2.3753321170806885, 26.87957191467285, 1.6616727113723755]
```

## Intensity Normalization

```text
clip_low_p0p5  0.000000
clip_high_p99p5 114.000000
clip_mean      22.471752
clip_std       14.902380
```

## Run Health

```text
sequences_completed 58
frames_processed    4640
frames_failed       0
skip_rate           0.000000
```
