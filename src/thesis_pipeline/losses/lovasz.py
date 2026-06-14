"""Lovász-Softmax loss (multiclass), vendored locally.

This is a minimal, dependency-free implementation of the Lovász-Softmax loss
adapted from the reference implementation by Maxim Berman et al.,
"The Lovász-Softmax loss: A tractable surrogate for the optimization of the
intersection-over-union measure in neural networks" (CVPR 2018). Reference
code: https://github.com/bermanmaxim/LovaszSoftmax (MIT License).

Adaptation notes for this project:

- We operate on already-flattened, already-filtered tensors. In the Open3D-ML
  RandLA-Net path, ``RandLANet.get_loss`` calls ``filter_valid_label`` before
  the loss, which removes ignored labels and compresses labels to active
  contiguous indices. So this module takes ``probas`` of shape ``[P, C]`` and
  ``labels`` of shape ``[P]`` and does NOT do any ignore handling or
  per-image splitting itself.
- Input ``probas`` must be SOFTMAX PROBABILITIES (not logits).
- ``classes="present"`` means only classes that actually appear in ``labels``
  contribute, so absent classes never add a spurious term.
- Everything stays on the input tensors' device (CPU or CUDA); there are no
  ``.cpu()`` / NumPy conversions on the differentiable path.
"""

from __future__ import annotations

import torch


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovász extension w.r.t. sorted errors.

    Args:
        gt_sorted: 1D tensor of binary ground truth (0/1) for one class, sorted
            in the order of descending prediction error.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover the case where the foreground is a single pixel
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
) -> torch.Tensor:
    """Multiclass Lovász-Softmax on flat, pre-filtered tensors.

    Args:
        probas: softmax probabilities, shape ``[P, C]``.
        labels: ground-truth active indices, shape ``[P]`` with values in
            ``[0, C)``.
        classes: ``"all"`` to average over every class, ``"present"`` to average
            only over classes that appear in ``labels``, or a list of class
            indices to restrict to.

    Returns:
        A scalar tensor with a valid gradient path (zero, with grad, when no
        eligible class is present or the input is empty).
    """
    if probas.numel() == 0:
        # No valid points in this batch; return 0 with a grad path.
        return probas.sum() * 0.0

    C = probas.size(1)
    losses = []
    class_iter = list(range(C)) if classes in ("all", "present") else classes
    for c in class_iter:
        fg = (labels == c).type_as(probas)  # foreground mask for class c
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm.data]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if len(losses) == 0:
        # No eligible class present; return 0 with a grad path.
        return probas.sum() * 0.0
    return torch.stack(losses).mean()
