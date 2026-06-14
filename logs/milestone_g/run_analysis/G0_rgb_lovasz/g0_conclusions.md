# G0 Conclusions

## Main Conclusion

G0 = F0 with the loss changed to `weighted_CE + lambda*Lovasz-Softmax`.
The official G0 checkpoint is epoch `18`, selected by maximum
raw marking IoU (same rule as D0/E0/F0).

| metric | D0 ep18 | E0 ep14 | F0 ep13 | G0 ep18 |
| --- | ---: | ---: | ---: | ---: |
| marking IoU | 0.440294 | 0.438439 | 0.482770 | 0.523048 |
| F1 | 0.611394 | 0.609604 | 0.651173 | 0.686844 |
| precision | 0.514788 | 0.470298 | 0.549249 | 0.629949 |
| recall | 0.752636 | 0.866170 | 0.799543 | 0.755036 |
| mIoU | 0.780829 | 0.780658 | 0.798682 | 0.817407 |

## Success Criteria (vs F0)

| criterion | F0 | G0 | G beats F0? |
| --- | ---: | ---: | :---: |
| best marking IoU | 0.482770 | 0.523048 | YES |
| best-to-final IoU drift | -0.032787 | -0.044455 | NO |
| best pred/true ratio | 1.456 | 1.199 | YES |

## Fair CE Comparison

F0 `val_loss` is pure weighted CE; G0 `val_ce` is the CE component (comparable). G0 `val_loss` is the TOTAL and is not compared.

- F0 best val CE: `0.118228`
- G0 best val CE: `0.112872`
- G0 best val total: `0.209156`
- G0 best val Lovasz (raw): `0.192567`

## Lovasz Term Health

- scale: **dominant** (Lovasz share of total ~0.460)
- stability: **stable** (val Lovasz CV 0.103)
- alignment: **Lovasz aligns with marking IoU better than CE**
  - corr(val CE, IoU) = -0.910; corr(val Lovasz, IoU) = -0.975
  - argmin val Lovasz epoch 18, argmin val CE epoch 24, argmax IoU epoch 18

## Best vs Final Epoch (drift)

| run | metric | best | final | delta |
| --- | --- | ---: | ---: | ---: |
| F0 | marking IoU | 0.482770 | 0.449982 | -0.032787 |
| G0 | marking IoU | 0.523048 | 0.478593 | -0.044455 |
| F0 | pred/true | 1.456 | 1.782 | +0.326 |
| G0 | pred/true | 1.199 | 1.424 | +0.226 |

## Sampled Best-Epoch Residuals

```text
all:         pred/true 1.189, IoU 0.522980
rgb_valid:   pred/true 1.240, IoU 0.525595  (F0 was 1.550)
rgb_invalid: pred/true 1.004, IoU 0.512477  (F0 was 1.148)
```

## Provenance

- sampled analysis checkpoint: `/home/coder/project/logs/milestone_g/runs/G0_rgb_lovasz/checkpoints/ckpt_epoch_00018.pth`
- sampled steps: `2160`
- sampled seed: `42`
- sampled split: `validation`
