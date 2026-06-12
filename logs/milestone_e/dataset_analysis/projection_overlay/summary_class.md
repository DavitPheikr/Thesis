# Class-Colored Projection Overlays

Three panels per frame: original | class overlay (subsampled) | marking-only (all marking points, drawn large).

Color key in the class overlay:

- light gray = road (raw 7)
- blue = other
- bright yellow = marking (raw 8 + 9 + 10)
- ignore class (raw 1-4) is hidden.

Key check: in the right panel, do the yellow dots sit on visible white road paint in the photo, or do they sit on plain asphalt next to it? If they consistently land on the paint, the projection is good enough to attach per-point RGB for E0.

| # | split/seq | lid | cam | mode | dt(s) | valid | marking proj/total | road proj/total | file |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| 1 | train/003 | 00 | 00 | same_index | -0.0499 | 0.909 | 560 / 560 (1.000) | 28927 / 29636 (0.976) | `overlays_class/01__train_003_f00_same_index_class.png` |
| 2 | train/017 | 00 | 00 | same_index | -0.0498 | 0.920 | 719 / 756 (0.951) | 13283 / 13301 (0.999) | `overlays_class/02__train_017_f00_same_index_class.png` |
| 3 | train/017 | 40 | 40 | same_index | -0.0498 | 0.892 | 2525 / 2723 (0.927) | 17098 / 18517 (0.923) | `overlays_class/03__train_017_f40_same_index_class.png` |
| 4 | train/037 | 52 | 52 | same_index | -0.0499 | 0.853 | 4076 / 4218 (0.966) | 18728 / 19238 (0.973) | `overlays_class/04__train_037_f52_same_index_class.png` |
| 5 | val/054 | 00 | 00 | same_index | +0.4499 | 0.775 | 1646 / 1910 (0.862) | 21880 / 27344 (0.800) | `overlays_class/05__val_054_f00_same_index_class.png` |
| 6 | val/054 | 00 | 00 | nearest_ts | +0.4499 | 0.775 | 1646 / 1910 (0.862) | 21880 / 27344 (0.800) | `overlays_class/06__val_054_f00_nearest_ts_class.png` |
| 7 | val/054 | 79 | 79 | same_index | -0.0499 | 0.854 | 2167 / 2257 (0.960) | 24875 / 26019 (0.956) | `overlays_class/07__val_054_f79_same_index_class.png` |
| 8 | val/106 | 20 | 20 | same_index | -0.0497 | 0.896 | 202 / 202 (1.000) | 23845 / 24267 (0.983) | `overlays_class/08__val_106_f20_same_index_class.png` |

