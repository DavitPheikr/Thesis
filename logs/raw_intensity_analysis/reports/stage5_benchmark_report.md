# Stage 5 Intensity-Only Benchmark Report

## Threshold benchmark

                            task  roc_auc   pr_auc  best_threshold  best_f1  recall_at_precision_0p90  recall_at_precision_0p95  false_positive_rate_at_best_f1
                    lane_vs_road 0.705279 0.768316             0.0 0.666667                  0.368800                  0.321070                        1.000000
lane_vs_road_plus_other_markings 0.584796 0.345108            26.0 0.406054                  0.000775                  0.000525                        0.784805

- This is a diagnostic baseline only. Strong scores indicate raw intensity contains signal; weak scores indicate intensity alone is not enough.
