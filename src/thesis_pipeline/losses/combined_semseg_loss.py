"""Combined semantic-segmentation loss: weighted CE + lambda * Lovász-Softmax.

This wraps the stock Open3D-ML ``SemSegLoss`` so it is a drop-in for the
``RandLANet.get_loss`` call site, which only does:

    loss = Loss.weighted_CrossEntropyLoss(scores, labels)

Design guarantees:

- The cross-entropy term is the *exact* stock weighted CE: we instantiate a
  stock ``SemSegLoss`` internally and reuse its ``weighted_CrossEntropyLoss``.
  So class weights (the count-list transformed by
  ``DataProcessing.get_class_weights``) apply ONLY to the CE term.
- The Lovász term is UNWEIGHTED (``classes="present"``); it is not class-weighted
  again. The rare-class emphasis comes entirely from the CE weights.
- CE receives raw logits; Lovász receives ``softmax(scores)``.
- Both terms use the same already-filtered ``(scores, labels)`` that
  ``RandLANet.get_loss`` passes (ignored labels already removed, labels already
  compressed to active indices).
- Per-call CE / Lovász / total scalars are stashed on the instance for logging.

``build_loss`` is config-gated: with no loss block or ``name == "weighted_ce"``
it returns the stock ``SemSegLoss`` unchanged, so existing D/E/F runs behave
exactly as before.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from open3d._ml3d.torch.modules.losses import SemSegLoss

from .lovasz import lovasz_softmax


class CombinedSemSegLoss:
    """Duck-typed ``SemSegLoss``: weighted CE + ``lovasz_lambda`` * Lovász."""

    def __init__(
        self,
        pipeline,  # noqa: ANN001
        model,  # noqa: ANN001
        dataset,  # noqa: ANN001
        device,  # noqa: ANN001
        lovasz_lambda: float = 0.5,
        lovasz_classes: str = "present",
    ) -> None:
        # Build the stock loss so the CE term is byte-for-byte identical to the
        # current weighted-CE behavior (same class-weight transform, same device).
        self._stock = SemSegLoss(pipeline, model, dataset, device)
        self._ce = self._stock.weighted_CrossEntropyLoss
        self.lovasz_lambda = float(lovasz_lambda)
        self.lovasz_classes = lovasz_classes
        # Per-call component values (floats), for logging. None until first call.
        self.last_ce: float | None = None
        self.last_lovasz: float | None = None
        self.last_total: float | None = None

    def weighted_CrossEntropyLoss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return weighted CE + lambda * Lovász on filtered logits/labels.

        Named to match the attribute ``RandLANet.get_loss`` calls. ``scores`` are
        logits; ``labels`` are active contiguous indices.
        """
        ce = self._ce(scores, labels)  # weighted CE on logits
        probas = F.softmax(scores, dim=-1)  # probabilities for Lovász
        lov = lovasz_softmax(probas, labels, classes=self.lovasz_classes)
        total = ce + self.lovasz_lambda * lov

        self.last_ce = float(ce.detach().item())
        self.last_lovasz = float(lov.detach().item())
        self.last_total = float(total.detach().item())
        return total


def build_loss(
    pipeline,  # noqa: ANN001
    model,  # noqa: ANN001
    dataset,  # noqa: ANN001
    device,  # noqa: ANN001
    loss_cfg=None,  # noqa: ANN001
):
    """Config-gated loss factory.

    Returns the stock ``SemSegLoss`` (current behavior) when ``loss_cfg`` is
    falsy or names ``weighted_ce``. Returns a ``CombinedSemSegLoss`` when it
    names ``weighted_ce_lovasz``.
    """
    name = None
    if loss_cfg:
        name = loss_cfg.get("name") if hasattr(loss_cfg, "get") else None

    if not loss_cfg or name in (None, "weighted_ce"):
        return SemSegLoss(pipeline, model, dataset, device)

    if name == "weighted_ce_lovasz":
        lovasz_lambda = float(loss_cfg.get("lovasz_lambda", 0.5))
        lovasz_classes = loss_cfg.get("lovasz_classes", "present")
        return CombinedSemSegLoss(
            pipeline,
            model,
            dataset,
            device,
            lovasz_lambda=lovasz_lambda,
            lovasz_classes=lovasz_classes,
        )

    raise ValueError(
        f"Unknown pipeline.loss.name={name!r}; expected 'weighted_ce' or "
        "'weighted_ce_lovasz'."
    )
