# Raw Intensity Notes

## Core Takeaway

Raw LiDAR intensity is useful, but it is not enough by itself.

It helps separate lane markings from road/asphalt, especially near the sensor
and in local neighborhoods. It does not cleanly separate lane-line markings from
other painted road markings, because those paint classes have very similar raw
intensity.

Practical conclusion:

```text
Keep intensity as an input feature, but do not treat raw intensity as a lane
detector. The model needs geometry, local context, and lane-aware sampling.
```

## Run Scope

- Sequences analyzed: `66`
- Lane-bearing frames analyzed: `4901`
- Sensor: forward-facing LiDAR only
- Intensity source: raw `pc_df["i"]`
- Preprocessing used for this analysis: none
- Main output folder: `logs/raw_intensity_analysis/`

## Most Important Facts

### 1. Lane Is Brighter Than Road Globally

| Class             | Raw ID | Median |    Mean |  IQR |
| ----------------- | -----: | -----: | ------: | ---: |
| Lane Line Marking |    `8` |   `33` | `37.77` | `19` |
| Road              |    `7` |   `28` | `26.36` |  `6` |

Key fact:

```text
lane median - road median = +5
```

This means lane markings are typically brighter than road/asphalt.

Source:

- `tables/raw_id_summary.csv`
- `reports/stage1_global_raw_id_report.md`

### 2. Lane Is Not Separated From Other Road Paint By Median Intensity

| Class              | Raw ID | Median |
| ------------------ | -----: | -----: |
| Lane Line Marking  |    `8` |   `33` |
| Stop Line Marking  |    `9` |   `33` |
| Other Road Marking |   `10` |   `33` |

Key fact:

```text
lane, stop-line, and other-road-marking medians are all about 33
```

This means raw intensity helps with paint-vs-asphalt, but not reliably with
lane-paint-vs-other-paint.

Source:

- `tables/raw_id_summary.csv`
- `plots/stage1/raw_id_histograms.png`
- `plots/stage1/raw_id_kde.png`

### 3. Some Sequences Have Strong Intensity Signal

Top examples:

| Sequence | Lane points | Lane-road median shift | AUC lane vs road | AUC lane vs road-surface non-lane |
| -------- | ----------: | ---------------------: | ---------------: | --------------------------------: |
| `023`    |    `49,342` |                  `+23` |          `0.858` |                           `0.858` |
| `042`    |    `20,066` |                  `+17` |          `0.830` |                           `0.826` |
| `053`    |    `47,695` |                   `+6` |          `0.827` |                           `0.818` |
| `109`    |    `40,057` |                  `+17` |          `0.825` |                           `0.826` |
| `016`    |   `118,863` |                  `+11` |          `0.817` |                           `0.816` |

Interpretation:

Some sequences are very intensity-friendly. Lane visibility is not uniform
across the dataset.

Source:

- `tables/sequence_summary.csv`
- `reports/stage2_sequence_frame_report.md`

### 4. Some Individual Frames Are Extremely Clean

Example:

```text
sequence 103, frame 50
lane count = 86
road count = 21,923
lane median = 69.5
road median = 29
lane-road median shift = +40.5
AUC lane vs road = 0.984
```

Interpretation:

There are frames where raw intensity makes lane markings very visible. These
are useful for visual inspection and understanding best-case behavior.

Source:

- `tables/frame_summary.csv`
- `reports/stage2_sequence_frame_report.md`

### 5. Local Contrast Is Strong

At radius `0.5 m`:

```text
delta_mean_avg = 12.46
delta_median_avg = 12.56
frac_positive_delta_mean = 0.816
frac_above_local_p90 = 0.598
```

Meaning:

- lane points are about `12.46` raw intensity units brighter than nearby
  non-lane road-surface points on average
- about `81.6%` of evaluated lane points are brighter than their local
  neighborhood mean
- about `59.8%` are brighter than the local neighborhood P90

Interpretation:

The useful intensity signal is often local/contextual. This is good for a
neighborhood-based point-cloud model, but it also means training patches must
actually contain lane neighborhoods.

Source:

- `tables/local_contrast_summary.csv`
- `reports/stage3_local_contrast_report.md`
- `plots/stage3/local_contrast_delta_mean_r0p5.png`

### 6. Distance Weakens The Signal

| Distance  |  Lane count | Lane median | Road median | Median shift | AUC lane vs road |
| --------- | ----------: | ----------: | ----------: | -----------: | ---------------: |
| `0-10 m`  | `1,013,079` |        `40` |        `29` |        `+11` |          `0.780` |
| `10-20 m` |    `14,925` |        `14` |         `7` |         `+7` |          `0.660` |
| `20-30 m` |       `421` |         `1` |         `0` |         `+1` |          `0.593` |
| `30-40 m` |        `13` |         `0` |         `0` |          `0` |          `0.507` |

Interpretation:

Raw intensity is mostly useful near range. After `20 m`, lane evidence becomes
very sparse and the intensity signal is weak.

Source:

- `tables/distance_bucket_summary.csv`
- `reports/stage4_distance_report.md`
- `plots/stage4/distance_auc.png`

### 7. Intensity Alone Is Weak For The Full Road-Paint Problem

| Task                               | ROC AUC |  PR AUC | Best F1 |
| ---------------------------------- | ------: | ------: | ------: |
| `lane_vs_road`                     | `0.705` | `0.768` | `0.667` |
| `lane_vs_road_plus_other_markings` | `0.585` | `0.345` | `0.406` |

Interpretation:

Raw intensity alone is moderately useful for lane vs asphalt. It becomes weak
when stop lines and other road markings are included.

Source:

- `tables/benchmark_summary.csv`
- `reports/stage5_benchmark_report.md`

## Metric Definitions

`median`

The middle value. Used heavily because raw LiDAR intensity has outliers.

`IQR`

Interquartile range: `P75 - P25`. Describes the spread of the middle half of
the values.

`lane_minus_road_median`

```text
median lane intensity - median road intensity
```

Positive means lane is typically brighter than road.

`AUC`

Area under the ROC curve. In this analysis:

```text
Pick one random lane point and one random comparison point.
What is the probability that the lane point has higher raw intensity?
```

Interpretation:

- `0.5`: random/no useful separation
- `0.6`: weak
- `0.7`: useful
- `0.8+`: strong

`local contrast`

For each lane point, compare its raw intensity to nearby non-lane road-surface
points in ego `x/y` space.

## Trust And Provenance

Code used:

- `src/thesis_pipeline/analysis/raw_intensity_lane_visibility.py`
- `tools/analyze_raw_lane_intensity.py`

Primary generated data:

- `tables/raw_id_summary.csv`
- `tables/sequence_summary.csv`
- `tables/frame_summary.csv`
- `tables/local_contrast_summary.csv`
- `tables/distance_bucket_summary.csv`
- `tables/benchmark_summary.csv`

Internal count checks:

```text
lane raw-ID count = 2,890,855
sequence lane total = 2,890,855
frame lane total = 2,890,855

road raw-ID count = 127,041,643
sequence road total = 127,041,643
frame road total = 127,041,643

stop-line raw-ID count = 192,099
sequence stop-line total = 192,099
frame stop-line total = 192,099

other-road-marking raw-ID count = 3,033,954
sequence other-road-marking total = 3,033,954
frame other-road-marking total = 3,033,954
```

These matching totals mean the raw-ID summary, sequence summary, and frame
summary are counting the same aligned points.

Output shape checks:

```text
sequence rows = 66
frame rows = 4901
local contrast rows = 19604
distance rows = 5
benchmark rows = 2
```

`19604 = 4901 frames * 4 radii`, so local contrast ran for every analyzed frame
at every configured radius.

## Caveats

- This analysis measures raw intensity signal, not final segmentation quality.
- Stage 5 is diagnostic and should not be treated as a final classifier result.
- The analysis supports using intensity, but it also shows why intensity alone is
  not enough.
- The next modeling decision should account for geometry, local context,
  sparse lane points, distance effects, and confusion with other road markings.
