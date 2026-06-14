# G0 Loss-Component Analysis

Loss = `weighted_CE + 0.5 * Lovasz-Softmax`. This report characterizes the
Lovasz term: scale, stability, and whether it aligns with marking IoU better
than CE did.

## Composition Provenance (tolerance checks passed)

- max |val_total - (val_ce + 0.5*val_lovasz)|: `8.71e-10`
- max |train_total - (train_ce + 0.5*train_lovasz)|: `6.84e-10`
- max |val_total - eval_history val_loss|: `0.00e+00`

## Scale: is Lovasz tiny, comparable, or dominant?

At the best marking-IoU epoch (18):

- val CE: `0.112872`
- val Lovasz (raw): `0.192567`
- val Lovasz (scaled, x0.5): `0.096284`
- val total: `0.209156`
- Lovasz share of total: `0.460`  -> **dominant**

Mean Lovasz share across training: `0.448`.

## Stability: is Lovasz noisy across epochs?

- val Lovasz coefficient of variation: `0.103`  -> **stable**
- val CE coefficient of variation: `0.271`

## Alignment: does Lovasz track marking IoU better than CE?

Lower loss should accompany higher IoU, so a stronger NEGATIVE correlation with
IoU means better alignment.

- corr(val CE, marking IoU): `-0.910`
- corr(val Lovasz, marking IoU): `-0.975`
- epoch of min val CE: `24`
- epoch of min val Lovasz: `18`
- epoch of max marking IoU: `18`

Verdict: **Lovasz aligns with marking IoU better than CE**

In F0, validation CE kept improving after the marking-IoU peak (the CE/IoU
mismatch). The key question for G is whether the Lovasz minimum coincides with
the IoU peak more tightly than the CE minimum does. Here the Lovasz minimum is
0 epoch(s)
from the IoU peak, versus 6
epoch(s) for CE.

## Outputs

- `loss_component_summary.csv` (per-epoch CE/Lovasz/total/share/IoU)
- `loss_alignment.csv` (the scalar diagnostics above)
- `plots/train_loss_components.png`, `plots/val_loss_components.png`
- `plots/val_ce_vs_marking_iou.png`, `plots/val_lovasz_vs_marking_iou.png`, `plots/val_total_vs_marking_iou.png`
- `plots/f0_vs_g0_val_ce.png`
