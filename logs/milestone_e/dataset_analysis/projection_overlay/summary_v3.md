# Projection Overlay (v3) Quick Look

Anchors are now constrained to ground-level points (lowest y_cam within +/-1.5 m of target depth and +/-1.2 m lateral). If no ground point exists in that window (e.g. a vehicle directly ahead occludes the road) the anchor is skipped and the figure notes which depths were missing.

| # | split/seq | lid | cam | mode | dt(s) | valid | anchors missing | file |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| 1 | train/003 | 00 | 00 | same_index | -0.0499 | 0.909 | 5 m | `overlays_v3/01__train_003_f00_same_index.png` |
| 2 | train/017 | 00 | 00 | same_index | -0.0498 | 0.920 | 5 m | `overlays_v3/02__train_017_f00_same_index.png` |
| 3 | train/017 | 40 | 40 | same_index | -0.0498 | 0.892 | 5 m | `overlays_v3/03__train_017_f40_same_index.png` |
| 4 | train/037 | 52 | 52 | same_index | -0.0499 | 0.853 | 5 m | `overlays_v3/04__train_037_f52_same_index.png` |
| 5 | val/054 | 00 | 00 | same_index | +0.4499 | 0.775 | 5 m | `overlays_v3/05__val_054_f00_same_index.png` |
| 6 | val/054 | 00 | 00 | nearest_ts | +0.4499 | 0.775 | 5 m | `overlays_v3/06__val_054_f00_nearest_ts.png` |
| 7 | val/054 | 79 | 79 | same_index | -0.0499 | 0.854 | 5 m | `overlays_v3/07__val_054_f79_same_index.png` |
| 8 | val/106 | 20 | 20 | same_index | -0.0497 | 0.896 | 5 m | `overlays_v3/08__val_106_f20_same_index.png` |

