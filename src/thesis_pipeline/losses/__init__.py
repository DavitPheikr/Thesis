"""Loss extensions for the thesis pipeline.

`build_loss` is the config-gated factory used by the training runner. With no
loss config (or `name: weighted_ce`) it returns the stock Open3D `SemSegLoss`,
preserving the existing weighted-CE behavior exactly.
"""

from .combined_semseg_loss import CombinedSemSegLoss, build_loss
from .lovasz import lovasz_softmax

__all__ = ["CombinedSemSegLoss", "build_loss", "lovasz_softmax"]
