# G0 ReduceLROnPlateau Replay (zero-GPU)

Replays the observed G0 marking-IoU curve through the real PyTorch
ReduceLROnPlateau (mode=max, factor=0.5, threshold=0.01 abs, cooldown=1, min_lr=1e-06, smoothing_window=3, base_lr=0.0014).
Observed marking-IoU peak epoch = 18.

| patience | first LR drop epoch | drop before peak? | all drop epochs | final lr |
| ---: | ---: | :---: | --- | ---: |
| 6 | never (within 25 ep) | no | none | 0.0014 |
| 5 | 25 | no | 25 | 0.0007 |
| 4 | 18 | no | 18, 24 | 0.00035 |
| 3 | 17 | YES | 17, 23 | 0.00035 |

## Reading

- **patience=6 (the run's actual config):** first LR drop at epoch `None` — i.e. within the 25-epoch budget the scheduler did NOT enter a lower-LR phase.

Interpretation:
- If patience=6 drops at/after the peak but only near epoch ~24-25, a plain
  **epochs-only extension** (e.g. 25->35, patience unchanged) gives the model
  the lower-LR refinement phase it never really used — the clean single-variable
  schedule test.
- Any patience whose 'drop before peak?' = YES is risky: it would cut LR before
  epoch 18 and could lower the peak. Prefer such patiences only if you explicitly
  want to test earlier annealing.

Caveat: patience<6 rows are 'what the scheduler would do on the OBSERVED curve';
a real run with earlier LR drops would follow a different curve.
