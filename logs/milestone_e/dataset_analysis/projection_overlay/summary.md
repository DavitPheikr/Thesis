# Projection Overlay Quick Look

| # | split/seq | lid | cam | mode | dt(s) | valid | pts proj / in | file |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| 1 | train/003 | 00 | 00 | same_index | -0.0499 | 0.909 | 63128 / 69411 | `overlays/01__train_003_f00_same_index.png` |
| 2 | train/017 | 00 | 00 | same_index | -0.0498 | 0.920 | 51846 / 56347 | `overlays/02__train_017_f00_same_index.png` |
| 3 | train/017 | 40 | 40 | same_index | -0.0498 | 0.892 | 48168 / 54022 | `overlays/03__train_017_f40_same_index.png` |
| 4 | train/037 | 52 | 52 | same_index | -0.0499 | 0.853 | 58664 / 68780 | `overlays/04__train_037_f52_same_index.png` |
| 5 | val/054 | 00 | 00 | same_index | +0.4499 | 0.775 | 53551 / 69141 | `overlays/05__val_054_f00_same_index.png` |
| 6 | val/054 | 00 | 00 | nearest_ts | +0.4499 | 0.775 | 53551 / 69141 | `overlays/06__val_054_f00_nearest_ts.png` |
| 7 | val/054 | 79 | 79 | same_index | -0.0499 | 0.854 | 54887 / 64285 | `overlays/07__val_054_f79_same_index.png` |
| 8 | val/106 | 20 | 20 | same_index | -0.0497 | 0.896 | 55409 / 61839 | `overlays/08__val_106_f20_same_index.png` |

Open each PNG and check that LiDAR points land on the correct
pixels (road on road, car-silhouette points on the car body, etc.).
Frame 5 (val/054/0 same_index) is expected to look misaligned on
moving objects; frame 6 (nearest_ts) should look much better.

