"""Custom loss functions for attribute extraction training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeExtractionLoss(nn.Module):
    """Weighted cross-entropy loss for multi-attribute extraction.

    Applies different weights to different attributes based on their
    importance and difficulty. Used only with the BERT multi-head model.
    """

    def __init__(
        self,
        attribute_weights: dict[str, float] | None = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.attribute_weights = attribute_weights or {
            "style": 1.5,
            "primary_material": 1.2,
            "color_family": 1.0,
        }
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute weighted sum of per-attribute losses.

        Args:
            logits: Dict of {attr_name: [batch, num_classes]} tensors.
            targets: Dict of {attr_name: [batch]} target class indices.

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=next(iter(logits.values())).device)

        for attr_name, attr_logits in logits.items():
            if attr_name not in targets:
                continue

            attr_targets = targets[attr_name]
            weight = self.attribute_weights.get(attr_name, 1.0)

            loss = F.cross_entropy(
                attr_logits,
                attr_targets,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + weight * loss

        return total_loss
