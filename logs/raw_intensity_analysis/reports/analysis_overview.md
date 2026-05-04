# Raw Intensity Analysis Overview

## Purpose

This analysis asks whether **raw PandaSet LiDAR intensity** can help separate
lane-line markings from road/asphalt and other road-surface markings. It uses
raw intensity directly from the PandaSet LiDAR `i` column. No training-time
clipping, standardization, gamma correction, percentile normalization, rank
normalization, or visualization-only intensity transform is used in the core
metrics.

The analysis includes both:

- raw PandaSet semantic IDs, especially `7`, `8`, `9`, and `10`
- the thesis remapped classes: `ignore`, `road`, `lane`, and `other`

The key raw IDs are:

- `7 = Road`
- `8 = Lane Line Marking`
- `9 = Stop Line Marking`
- `10 = Other Road Marking`

The most important practical question is not only whether lane markings are
brighter than asphalt. It is also whether lane-line markings are distinguishable
from other painted road markings. The results show that these are different
questions.

## Run Summary

- Sequences analyzed: `66`
- Lane-bearing frames analyzed: `4901`
- Scope: all lane-bearing semseg-enabled sequences from the Milestone B manifest
- Sensor: forward-facing LiDAR only
- Intensity source: raw `pc_df["i"]`

Main artifact folders:

- CSV tables: `logs/raw_intensity_analysis/tables/`
- Plots: `logs/raw_intensity_analysis/plots/`
- Stage reports: `logs/raw_intensity_analysis/reports/`

## How To Read The Metrics

`mean` is the arithmetic average. It is useful, but it can be pulled upward by
rare very bright points.

`median` is the middle value. It is usually the better first summary for raw
LiDAR intensity because the intensity distribution has outliers and long tails.
For example, raw intensity can reach `255`, but that does not mean most points
in a class look that bright.

`p5`, `p25`, `p50`, `p75`, and `p95` describe the distribution shape. `p50` is
the same as the median.

`iqr` means interquartile range, computed as `p75 - p25`. It describes the
spread of the middle half of the points.

`lane_minus_road_median` means:

```text
median lane intensity - median road intensity
```

Positive values mean lane markings are typically brighter than road. Zero means
the typical lane and road intensities are similar. Negative values mean lane is
typically darker than road.

`AUC` means area under the ROC curve. In this analysis, it can be read as:

```text
Pick one random lane point and one random non-lane comparison point.
What is the probability that the lane point has higher raw intensity?
```

Interpretation:

- `0.5`: no useful intensity separation
- `0.6`: weak separation
- `0.7`: useful separation
- `0.8+`: strong separation
- `< 0.5`: the intensity direction is reversed or misleading

`auc_lane_vs_road` compares lane-line marking points against road points only.

`auc_lane_vs_road_surface_non_lane` compares lane-line marking points against
road plus stop-line marking plus other road marking. This is the harder and more
realistic road-surface comparison.

## Stage 1: Global Raw Intensity By Raw ID

Main files:

- `tables/raw_id_summary.csv`
- `tables/remapped_label_summary.csv`
- `tables/global_intensity_summary.csv`
- `plots/stage1/raw_id_histograms.png`
- `plots/stage1/raw_id_kde.png`
- `reports/stage1_global_raw_id_report.md`

Headline raw-ID statistics:

| Raw class               |         Count |    Mean | Median |  P25 |  P75 |  P95 |  IQR |
| ----------------------- | ------------: | ------: | -----: | ---: | ---: | ---: | ---: |
| `8 Lane Line Marking`   |   `2,890,855` | `37.77` |   `33` | `28` | `47` | `75` | `19` |
| `7 Road`                | `127,041,643` | `26.36` |   `28` | `25` | `31` | `38` |  `6` |
| `9 Stop Line Marking`   |     `192,099` | `34.23` |   `33` | `27` | `43` | `67` | `16` |
| `10 Other Road Marking` |   `3,033,954` | `35.18` |   `33` | `28` | `43` | `67` | `15` |

Important observations:

- Lane-line marking median intensity is `33`.
- Road median intensity is `28`.
- Lane is globally brighter than road by `+5` median intensity units.
- Stop-line marking median intensity is also `33`.
- Other-road-marking median intensity is also `33`.
- Lane has a wider intensity distribution than road: lane IQR is `19`, road IQR is `6`.

Interpretation:

Raw intensity contains useful information for separating lane paint from road or
asphalt. However, raw intensity alone does not clearly separate lane-line
marking from other painted road markings. This means the hard problem is not
only paint versus asphalt. It is also lane paint versus other road paint.

The KDE plot is a smoothed version of the histogram. It shows how the raw
intensity values are distributed for each road-surface raw ID. If two KDE curves
overlap strongly, those classes are hard to separate using raw intensity alone.

## Stage 2: Sequence-Level And Frame-Level Separability

Main files:

- `tables/sequence_summary.csv`
- `tables/frame_summary.csv`
- `plots/stage2/frame_auc_distribution.png`
- `plots/stage2/lane_count_vs_auc.png`
- `plots/stage2/distance_proxy_vs_median_shift.png`
- `plots/stage2/road_p90_vs_lane_median.png`
- `reports/stage2_sequence_frame_report.md`

Stage 2 asks whether raw-intensity separability is consistent across the
dataset. It is not. Some sequences and frames have very strong lane-road
intensity separation, while others are weaker.

Most intensity-friendly sequences by exact aggregated AUC:

| Sequence | Lane points | Lane-road median shift | AUC lane vs road | AUC lane vs road-surface non-lane |
| -------- | ----------: | ---------------------: | ---------------: | --------------------------------: |
| `023`    |    `49,342` |                  `+23` |          `0.858` |                           `0.858` |
| `042`    |    `20,066` |                  `+17` |          `0.830` |                           `0.826` |
| `053`    |    `47,695` |                   `+6` |          `0.827` |                           `0.818` |
| `109`    |    `40,057` |                  `+17` |          `0.825` |                           `0.826` |
| `046`    |    `29,398` |                   `+8` |          `0.799` |                           `0.794` |
| `066`    |    `62,225` |                   `+7` |          `0.803` |                           `0.793` |
| `016`    |   `118,863` |                  `+11` |          `0.817` |                           `0.816` |
| `044`    |    `27,313` |                  `+16` |          `0.804` |                           `0.804` |
| `103`    |    `21,140` |                  `+23` |          `0.798` |                           `0.798` |
| `024`    |   `104,112` |                   `+9` |          `0.805` |                           `0.802` |

Important observations:

- Several sequences have strong intensity separation, with AUC around `0.8` or higher.
- Sequence `023` is especially clean: lane median is `52`, road median is `29`, and the exact lane-vs-road AUC is about `0.858`.
- Sequence `016` has the largest lane-point count among the top sequences, with `118,863` lane points and strong AUC.
- Some sequences remain strong even when other road markings are included in the negative set. This suggests that in those sequences lane-line markings are not only brighter than asphalt but also separable from other road-surface markings.

Top individual frames are even stronger than the best sequence averages. For
example, frame `103/50` has:

- lane count: `86`
- road count: `21,923`
- lane median: `69.5`
- road median: `29`
- lane-road median shift: `+40.5`
- AUC lane vs road: `0.984`
- AUC lane vs road plus other markings: `0.984`

Interpretation:

Raw intensity is strongly useful in some frames and sequences. The signal is
not uniform across the dataset, so any training strategy should expect
sequence/frame variability. This also means visual inspection should include
both high-AUC and low-AUC examples, not only the nicest frames.

## Stage 3: Local Contrast

Main files:

- `tables/local_contrast_summary.csv`
- `tables/local_contrast_sequence_summary.csv`
- `plots/stage3/local_contrast_delta_mean_r0p5.png`
- `reports/stage3_local_contrast_report.md`

Stage 3 asks whether lane points are brighter than nearby non-lane road-surface
points. This is different from asking whether lane is globally brighter. Local
contrast is closer to what a model can exploit through neighborhoods.

Radius sweep summary:

|   Radius | Mean local delta | Median local delta | Fraction positive | Fraction above local P90 |
| -------: | ---------------: | -----------------: | ----------------: | -----------------------: |
| `0.25 m` |         `10.996` |           `11.170` |           `0.770` |                  `0.587` |
| `0.50 m` |         `12.457` |           `12.563` |           `0.816` |                  `0.598` |
| `0.75 m` |         `12.956` |           `12.932` |           `0.836` |                  `0.608` |
| `1.00 m` |         `13.159` |           `13.081` |           `0.842` |                  `0.610` |

Definitions:

- `delta_mean`: lane intensity minus nearby non-lane road-surface mean intensity
- `delta_median`: lane intensity minus nearby non-lane road-surface median intensity
- `frac_positive_delta_mean`: fraction of lane points brighter than their local neighborhood mean
- `frac_above_local_p90`: fraction of lane points brighter than the local neighborhood 90th percentile

Interpretation:

Local contrast is strong. At `0.5 m`, about `81.6%` of evaluated lane points are
brighter than the local non-lane road-surface mean, and about `59.8%` are above
the local neighborhood P90. This suggests the useful intensity signal is often
contextual. A model that sees local neighborhoods may use intensity better than
a simple global threshold.

This supports preserving intensity and designing sampling/training so lane
neighborhoods survive subsampling and patch selection.

## Stage 4: Distance-Conditioned Analysis

Main files:

- `tables/distance_bucket_summary.csv`
- `plots/stage4/distance_auc.png`
- `reports/stage4_distance_report.md`

Distance bucket results:

| Distance bucket |  Lane count |   Road count | Lane median | Road median | Median shift | AUC lane vs road | AUC lane vs road-surface non-lane |
| --------------- | ----------: | -----------: | ----------: | ----------: | -----------: | ---------------: | --------------------------------: |
| `0-10 m`        | `1,013,079` | `61,106,276` |        `40` |        `29` |        `+11` |          `0.780` |                           `0.776` |
| `10-20 m`       |    `14,925` |    `731,681` |        `14` |         `7` |         `+7` |          `0.660` |                           `0.635` |
| `20-30 m`       |       `421` |     `38,237` |         `1` |         `0` |         `+1` |          `0.593` |                           `0.593` |
| `30-40 m`       |        `13` |        `769` |         `0` |         `0` |          `0` |          `0.507` |                           `0.512` |
| `40+ m`         |         `0` |         `47` |       `NaN` |         `0` |        `NaN` |            `NaN` |                             `NaN` |

Important observations:

- Most lane points in this analysis are within `0-10 m`.
- The intensity signal is strongest close to the sensor.
- The signal weakens quickly with distance.
- Beyond `20 m`, lane point counts become very small, and separability approaches weak or random.

Interpretation:

Raw intensity is most useful for near-range lane markings. Farther away, there
are fewer lane points and the intensity signal becomes much weaker. This matters
for training because patch selection and subsampling may determine whether the
model sees enough useful lane evidence.

## Stage 5: Intensity-Only Benchmark

Main files:

- `tables/benchmark_summary.csv`
- `reports/stage5_benchmark_report.md`

This stage treats raw intensity as a simple one-dimensional classifier. It is
diagnostic only. It is not meant to replace RandLA-Net or prove final model
performance.

Benchmark results:

| Task                               | ROC AUC |  PR AUC | Best threshold | Best F1 | Recall at precision 0.90 | Recall at precision 0.95 |
| ---------------------------------- | ------: | ------: | -------------: | ------: | -----------------------: | -----------------------: |
| `lane_vs_road`                     | `0.705` | `0.768` |            `0` | `0.667` |                  `0.369` |                  `0.321` |
| `lane_vs_road_plus_other_markings` | `0.585` | `0.345` |           `26` | `0.406` |               `0.000775` |               `0.000525` |

Important observations:

- Raw intensity alone is moderately useful for lane versus road.
- Raw intensity alone becomes much weaker when other road markings are included as negatives.
- High-precision lane recall is extremely low for the harder task.
- The `lane_vs_road` best threshold is not practically meaningful as a detector because it accepts too many road points.

Interpretation:

Intensity contains useful signal, but simple thresholding is not enough for the
real task. The model needs geometry, local context, and likely better sampling
of lane-containing neighborhoods.

## Overall Findings

The strongest conclusion is:

```text
Raw intensity helps distinguish lane markings from road/asphalt, but it is not
enough by itself to reliably distinguish lane-line markings from other painted
road markings.
```

More detailed conclusions:

- Lane-line markings are globally brighter than road: median `33` vs `28`.
- Lane-line markings have similar median intensity to stop-line and other road markings: all around `33`.
- The lane intensity distribution is wider than the road distribution, so median and percentile views are more informative than mean alone.
- Some sequences have very strong raw-intensity lane signal, especially `023`, `042`, `053`, `109`, `016`, `044`, `103`, and `024`.
- Some individual frames have extremely strong intensity separation, with AUC near `0.98`.
- Local contrast is stronger than global absolute intensity. Lane points often stand out relative to nearby non-lane road-surface points.
- Distance matters a lot. The signal is strongest in the `0-10 m` bucket and weakens sharply after that.
- Intensity-only thresholding is too crude for the real lane segmentation task, especially when other road markings are treated as negatives.

## What This Means For The Thesis Pipeline

Intensity should remain an important input feature. The analysis gives evidence
that raw intensity carries useful lane information, especially near the sensor
and in local neighborhoods.

However, intensity should not be treated as a complete solution. It is likely
most useful when combined with:

- local geometry
- neighborhood context
- lane-aware sampling
- careful handling of sparse lane points
- explicit awareness that other road markings can look intensity-similar to lane-line markings

The results support the Milestone B carry-forward concern about lane-aware patch
oversampling. If lane points are sparse and the strongest signal is local, then
training patches must actually contain enough lane neighborhoods for the model
to learn them.

## Recommended Next Reads

Read these artifacts in order:

1. `reports/stage1_global_raw_id_report.md`
2. `plots/stage1/raw_id_histograms.png`
3. `plots/stage1/raw_id_kde.png`
4. `reports/stage2_sequence_frame_report.md`
5. `tables/sequence_summary.csv`
6. `tables/frame_summary.csv`
7. `reports/stage3_local_contrast_report.md`
8. `reports/stage4_distance_report.md`
9. `reports/stage5_benchmark_report.md`

For visual follow-up, inspect high-signal and low-signal frames in the point
cloud viewer rather than only looking at the top-ranked examples. This will help
separate true data difficulty from metric artifacts.
