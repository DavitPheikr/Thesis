# Projection Overlay (v2) Quick Look

Sparser overlays (every 30th point) with 5 m / 10 m / 20 m anchor circles on the right panel.

| # | split/seq | lid | cam | mode | dt(s) | valid | shown / proj | file |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| 1 | train/003 | 00 | 00 | same_index | -0.0499 | 0.909 | 2105 / 63128 | `overlays_v2/01__train_003_f00_same_index.png` |
| 2 | train/017 | 00 | 00 | same_index | -0.0498 | 0.920 | 1729 / 51846 | `overlays_v2/02__train_017_f00_same_index.png` |
| 3 | train/017 | 40 | 40 | same_index | -0.0498 | 0.892 | 1606 / 48168 | `overlays_v2/03__train_017_f40_same_index.png` |
| 4 | train/037 | 52 | 52 | same_index | -0.0499 | 0.853 | 1956 / 58664 | `overlays_v2/04__train_037_f52_same_index.png` |
| 5 | val/054 | 00 | 00 | same_index | +0.4499 | 0.775 | 1786 / 53551 | `overlays_v2/05__val_054_f00_same_index.png` |
| 6 | val/054 | 00 | 00 | nearest_ts | +0.4499 | 0.775 | 1786 / 53551 | `overlays_v2/06__val_054_f00_nearest_ts.png` |
| 7 | val/054 | 79 | 79 | same_index | -0.0499 | 0.854 | 1830 / 54887 | `overlays_v2/07__val_054_f79_same_index.png` |
| 8 | val/106 | 20 | 20 | same_index | -0.0497 | 0.896 | 1847 / 55409 | `overlays_v2/08__val_106_f20_same_index.png` |

## How to read these

Each red circle is a single LiDAR point chosen to be roughly
at 5 m / 10 m / 20 m directly in front of the camera. If the
projection is correct, the 5 m circle sits on the road just
ahead of the car; 10 m further down the road; 20 m further
still. If they land in the sky, on a building, or off the road,
the projection is wrong.

The colored dots are a 1-in-30 sample. They should sit on the
road and on the silhouettes of cars / poles / signs, not float
above the road or appear in the sky.

