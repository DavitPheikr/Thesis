#!/usr/bin/env python
"""Safety/equivalence tests for the Milestone G combined loss.

Runs without training. Verifies:

1. default-path equivalence: no loss block / name=weighted_ce -> stock SemSegLoss
2. combined factory: name=weighted_ce_lovasz -> CombinedSemSegLoss
3. composition: total == CE + lambda * Lovász  (independent recompute)
4. CE component equals stock weighted CE on the same logits/labels
5. Lovász on (near) perfect prediction is ~0
6. single-class and present-class cases do not crash and are finite
7. CPU/CUDA finite gradient (CUDA only if available)

Emits `script_status PASS` as the final line on success.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from open3d._ml3d.torch.modules.losses import SemSegLoss  # noqa: E402

from thesis_pipeline.losses import build_loss  # noqa: E402
from thesis_pipeline.losses.combined_semseg_loss import CombinedSemSegLoss  # noqa: E402
from thesis_pipeline.losses.lovasz import lovasz_softmax  # noqa: E402


class _Cfg(dict):
    """Minimal dict that also supports attribute access (like Open3D ConfigDict)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(key) from exc


# F0 capped-weight count list: marking effective 15.0.
CLASS_WEIGHTS = [119562394.0, 14344000.0, 173473484.0]
NUM_CLASSES = 3


def _stub_dataset():
    return _Cfg(cfg=_Cfg(class_weights=list(CLASS_WEIGHTS)))


def _fake_batch(device, n=4096, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.randn(n, NUM_CLASSES, generator=g).to(device)
    labels = torch.randint(0, NUM_CLASSES, (n,), generator=g).to(device)
    return scores, labels


def _check(cond, msg):
    if not cond:
        print(f"FAIL {msg}")
        raise SystemExit(1)
    print(f"ok   {msg}")


def run(device):
    print(f"=== device={device} ===")
    ds = _stub_dataset()

    # 1. default-path equivalence
    none_loss = build_loss(None, None, ds, device, None)
    wce_loss = build_loss(None, None, ds, device, {"name": "weighted_ce"})
    _check(type(none_loss) is SemSegLoss, "no loss block -> stock SemSegLoss")
    _check(type(wce_loss) is SemSegLoss, "name=weighted_ce -> stock SemSegLoss")

    # 2. combined factory
    combined = build_loss(
        None, None, ds, device,
        {"name": "weighted_ce_lovasz", "lovasz_lambda": 0.5, "lovasz_classes": "present"},
    )
    _check(isinstance(combined, CombinedSemSegLoss), "name=weighted_ce_lovasz -> CombinedSemSegLoss")

    # 3 + 4. composition and CE-component equality vs stock weighted CE
    stock = SemSegLoss(None, None, ds, device)
    scores, labels = _fake_batch(device)
    total = combined.weighted_CrossEntropyLoss(scores, labels)
    stock_ce = stock.weighted_CrossEntropyLoss(scores, labels)
    lov = lovasz_softmax(F.softmax(scores, dim=-1), labels, classes="present")
    expected = stock_ce + 0.5 * lov
    _check(torch.allclose(total, expected, atol=1e-6), "total == CE + 0.5*Lovasz")
    _check(abs(combined.last_ce - float(stock_ce.item())) < 1e-6, "CE component == stock weighted CE")
    _check(abs(combined.last_lovasz - float(lov.item())) < 1e-6, "Lovasz component matches standalone")
    _check(math.isfinite(combined.last_total), "total is finite")

    # 5. Lovász ~0 on (near) perfect prediction
    perfect_logits = (F.one_hot(labels, NUM_CLASSES).float() * 20.0).to(device)
    perfect_lov = lovasz_softmax(F.softmax(perfect_logits, dim=-1), labels, classes="present")
    _check(float(perfect_lov.item()) < 1e-3, f"Lovasz perfect-pred ~0 (got {float(perfect_lov.item()):.2e})")

    # 6. single-class + present-class robustness
    one_class_labels = torch.zeros_like(labels)
    lov_one = lovasz_softmax(F.softmax(scores, dim=-1), one_class_labels, classes="present")
    _check(math.isfinite(float(lov_one.item())), "single-class batch finite (present excludes absent)")
    # labels with class 1 (marking) absent
    no_marking = torch.where(labels == 1, torch.zeros_like(labels), labels)
    lov_nomark = lovasz_softmax(F.softmax(scores, dim=-1), no_marking, classes="present")
    _check(math.isfinite(float(lov_nomark.item())), "marking-absent batch finite")

    # 7. finite gradient
    grad_scores = scores.clone().detach().requires_grad_(True)
    g_total = combined.weighted_CrossEntropyLoss(grad_scores, labels)
    g_total.backward()
    _check(grad_scores.grad is not None and torch.isfinite(grad_scores.grad).all(),
           "combined loss produces finite gradients")


def main():
    run("cpu")
    if torch.cuda.is_available():
        run("cuda")
    else:
        print("note: CUDA not available; skipped CUDA checks")
    print("script_status PASS")


if __name__ == "__main__":
    main()
