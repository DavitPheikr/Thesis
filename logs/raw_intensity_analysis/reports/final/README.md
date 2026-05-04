# Final Raw Intensity Plots

This folder contains the tight visual set worth keeping from the raw-intensity
analysis. The goal is to show the main evidence without carrying every
diagnostic plot forward.

## Keep These

1. `01_raw_id_kde.png`
   - Best first plot for class distribution overlap.
   - Shows lane is brighter than road, but overlaps with other road paint.

2. `02_raw_id_histogram.png`
   - Histogram version of the same raw-ID comparison.
   - Useful when you want to see raw bin-level distribution.

3. `03_sequence_auc_ranked.png`
   - Shows the most intensity-friendly sequences.
   - Compares lane-vs-road AUC against the harder lane-vs-road+markings AUC.

4. `04_sequence_median_shift_ranked.png`
   - Shows which sequences have the largest lane-road median intensity shift.
   - Good for picking sequences to inspect visually.

5. `05_frame_auc_distribution.png`
   - Shows frame-level variability.
   - Useful evidence that lane visibility is not uniform across frames.

6. `06_lane_count_vs_auc.png`
   - Shows whether frame separability relates to lane point count.
   - Useful as a diagnostic for sparse-lane frames.

7. `07_local_contrast_radius_sweep.png`
   - Most compact Stage 3 plot.
   - Shows local contrast and local brightness fractions across radius.

8. `08_local_contrast_distribution_0p5m.png`
   - Distribution of local contrast at the key `0.5 m` radius.
   - Positive values mean lane points are brighter than nearby road-surface points.

9. `09_distance_auc.png`
   - Shows distance decay in intensity separability.
   - Important for near-range vs far-range interpretation.

10. `10_distance_lane_count.png`
    - Shows lane evidence collapses with distance.
    - Complements the distance AUC plot.

11. `11_benchmark_auc_bar.png`
    - Shows the intensity-only diagnostic benchmark.
    - The key point is that lane-vs-road is much easier than lane-vs-road+markings.

## Main Message

Raw intensity is useful, especially near range and locally. It is not enough by
itself because lane-line markings overlap strongly with other painted road
markings.
