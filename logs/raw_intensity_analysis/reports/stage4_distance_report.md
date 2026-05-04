# Stage 4 Distance-Conditioned Report

## Distance bucket summary

 bucket  lane_count  road_count  lane_median  road_median  lane_minus_road_median  auc_lane_vs_road  auc_lane_vs_road_surface_non_lane
  0_10m     1013079    61106276         40.0         29.0                    11.0          0.780380                           0.775565
 10_20m       14925      731681         14.0          7.0                     7.0          0.660087                           0.634982
 20_30m         421       38237          1.0          0.0                     1.0          0.593087                           0.592690
 30_40m          13         769          0.0          0.0                     0.0          0.507402                           0.511778
40_infm           0          47          NaN          0.0                     NaN               NaN                                NaN

- Falling AUC with distance suggests the raw intensity signal weakens as lanes get farther from the sensor.
