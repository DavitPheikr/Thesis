# Diagnostic Plots

These plots summarize the C0 epoch-18 sampled inference analysis. They are meant to explain why true lane points are often predicted as road.

## `distance_lane_outcomes.png`

Purpose: shows what happens to true lane points at each distance range.

How to read it: each bar is one distance bucket. The bar is split into true lane points predicted as lane, road, or other. Higher red means more lane points were missed as road. The blue line shows how many true lane points were available in that bucket, so low-support buckets should be interpreted more carefully.

Finding: the lane-to-road error is not only a far-distance problem. The strongest bucket is 10-20 m, where only 45.5% of lane points were detected as lane and 54.0% were predicted as road. Far range is also weak: 60 m+ has 42.0% lane recall and 49.0% lane-to-road error.

## `distance_lane_intensity_medians.png`

Purpose: compares intensity of correctly detected lane points against lane points missed as road, per distance bucket.

How to read it: the green line is true lane predicted as lane. The red line is true lane predicted as road. A large gap means the model detects higher-intensity lane points better and misses lower-intensity lane points.

Finding: missed lane points have much lower median intensity in every distance bucket. Examples: at 10-20 m, correct lane median intensity is 47 while missed-lane-as-road is 30. At 40-60 m, correct lane is 27 while missed lane is 5. This strongly supports the explanation that weak lane returns overlap with road intensity.

## `frame_lane_to_road_scatter.png`

Purpose: identifies whether the problem is spread evenly or concentrated in specific bad frames.

How to read it: each dot is one sampled frame. X-axis is how many true lane points were sampled. Y-axis is the percent of those lane points predicted as road. Red outlined dots are the top 10 frames by lane-to-road count. Labels show `seq_id/frame_idx`.

Finding: some frames are severe outliers. The worst sampled frame is sequence 124 frame 68: 641 true lane points, 468 predicted as road, lane-to-road rate 73.0%. Other high-error examples include 124/54, 123/8, 124/65, and 123/16. These are the best candidates for visual inspection.

## `confusion_matrix_row_normalized.png`

Purpose: summarizes the sampled model behavior across road, lane, and other.

How to read it: rows are true classes and columns are predicted classes. Each cell shows row percentage and raw count. The diagonal is correct prediction. Off-diagonal cells are errors.

Finding: road and other are strong, but lane is much weaker. Road recall is 98.3%, other recall is 92.6%, and lane recall is 50.3%. For true lane points, 47.6% are predicted as road and only 2.0% as other, so the main lane failure mode is specifically lane-to-road confusion.

## `intensity_hist_by_outcome.png`

Purpose: shows the full intensity distributions for prediction outcomes.

How to read it: compare the lane-to-road curve with correct-lane and correct-road curves. If lane-to-road overlaps road, missed lane points are intensity-wise similar to road.

Finding: the histogram matches the table result: correctly detected lane points are generally brighter, while missed lane points overlap strongly with road. Overall medians are: correct lane 44, lane predicted as road 29, correct road 29.
