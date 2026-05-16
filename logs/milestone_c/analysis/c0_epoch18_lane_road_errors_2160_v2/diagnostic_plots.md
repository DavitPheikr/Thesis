# Diagnostic Plots

These plots summarize the C0 epoch-18 sampled inference analysis. They are meant to explain why true lane points are often predicted as road.

## `distance_lane_outcomes.png`

Purpose: shows what happens to true lane points at each distance range.

How to read it: each bar is one distance bucket. The bar is split into true lane points predicted as lane, road, or other. Higher red means more lane points were missed as road. The blue line shows true-lane support, so buckets with more lane points carry more evidence.

Finding: the largest lane support is `10_20m` with 157,816 true lane points and 54.0% lane-to-road error. The highest lane-to-road rate is `10_20m` at 54.0%.

## `distance_lane_intensity_medians.png`

Purpose: compares median intensity for correctly detected lane, lane predicted as road, and correctly detected road within each distance bucket.

How to read it: green is true lane predicted as lane, red is true lane predicted as road, and gray dashed is true road predicted as road. If red tracks gray more closely than green, the missed lane points are road-like in intensity at that distance.

Finding: this is the distance-controlled test of the main hypothesis. Use it to check whether missed lane intensity follows road intensity, not just whether missed lane is lower than detected lane.

## `frame_lane_to_road_scatter.png`

Purpose: identifies whether the lane-to-road problem is spread evenly or concentrated in specific bad frames.

How to read it: each dot is one sampled frame. X-axis is true lane support in that sampled frame. Y-axis is the percent of those true lane points predicted as road. Red outlined dots are the top 10 frames by lane-to-road count.

Finding: the worst sampled frame is `124/68` with 641 true lane points, 468 lane points predicted as road, and 73.0% lane-to-road error.

## `confusion_matrix_row_normalized.png`

Purpose: summarizes sampled model behavior across road, lane, and other.

How to read it: rows are true classes and columns are predicted classes. The diagonal is correct prediction. Off-diagonal cells are errors. Each cell shows row percentage and raw count.

Finding: road recall is 98.3%, other recall is 92.6%, and lane recall is 50.3%. For true lane points, 47.6% are predicted as road, so the main lane failure mode is lane-to-road confusion.

## `intensity_hist_by_outcome.png`

Purpose: shows full intensity distributions for prediction outcomes.

How to read it: compare the lane-to-road curve with correct-lane and correct-road curves. If lane-to-road overlaps road, missed lane points are intensity-wise similar to road.

Finding: the histogram gives the distribution-level version of the median-intensity result. Correct lane is generally brighter, while missed lane overlaps strongly with road.
