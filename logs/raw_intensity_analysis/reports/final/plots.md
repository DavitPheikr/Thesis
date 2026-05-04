# Plot Guide

`01_raw_id_kde.png`

X-axis: raw LiDAR intensity. Y-axis: smoothed density, meaning how common that intensity is for each class. If curves overlap, intensity cannot separate those classes well. Main takeaway: lane is shifted brighter than road, but overlaps strongly with other road paint.

`02_raw_id_histogram.png`

X-axis: raw LiDAR intensity bins. Y-axis: density/frequency of points in each bin. This is the non-smoothed version of the raw-ID comparison. Main takeaway: road is concentrated lower, while lane and other markings have wider/brighter distributions.

`03_sequence_auc_ranked.png`

X-axis: AUC score. Y-axis: sequence ID. Higher AUC means raw intensity separates lane points better. Compare the two bars: if the road+markings bar drops, other painted markings are confusing intensity.

`04_sequence_median_shift_ranked.png`

X-axis: sequence ID. Y-axis: lane median intensity minus road median intensity. Bigger positive values mean lane is typically much brighter than road in that sequence. Use this to pick clear sequences for visual inspection.

`05_frame_auc_distribution.png`

X-axis: frame-level AUC. Y-axis: number of frames. Values near `0.5` mean weak separation; values closer to `1.0` mean strong separation. A wide spread means some frames are easy and others are hard.

`06_lane_count_vs_auc.png`

X-axis: lane point count in a frame. Y-axis: frame AUC for lane vs road+markings. Each dot is one frame. Low-AUC frames with many lane points are important hard cases because lanes exist but intensity does not separate them well.

`07_local_contrast_radius_sweep.png`

X-axis: neighborhood radius in meters. Left y-axis plot: raw intensity delta between lane and nearby road-surface points. Right y-axis plot: fraction of lane points brighter than local neighbors. Main takeaway: lane intensity often stands out locally.

`08_local_contrast_distribution_0p5m.png`

X-axis: lane intensity minus nearby road-surface intensity at `0.5 m`. Y-axis: number of frames/entries in that delta range. Values above zero mean lane is locally brighter. Main takeaway: much of the distribution is positive, so local context is useful.

`09_distance_auc.png`

X-axis: forward distance bucket from the sensor. Y-axis: AUC. If the line drops toward `0.5`, intensity becomes close to random. Main takeaway: intensity separability is strongest near the sensor and weakens with distance.

`10_distance_lane_count.png`

X-axis: forward distance bucket from the sensor. Y-axis: number of lane points. Main takeaway: lane evidence collapses after near range, so far-range lane learning has much less data.

`11_benchmark_auc_bar.png`

X-axis: benchmark task. Y-axis: score from `0` to `1`. Higher is better. Compare ROC AUC and PR AUC bars. Main takeaway: intensity alone helps with asphalt vs paint, but not enough for the full lane-marking task.
