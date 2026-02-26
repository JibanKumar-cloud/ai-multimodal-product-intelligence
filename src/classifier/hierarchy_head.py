"""Hierarchical Classification Heads.

Multiple heads predict taxonomy at each level:
  z → level1 ("Furniture")
  z → level2 ("Living Room Furniture")
  z → level3 ("Chairs")
  z → leaf   ("Accent Chairs")

Loss: weighted sum of CE at each level with focal loss for rare classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    FL(p) = -alpha * (1-p)^gamma * log(p)
    Down-weights easy examples, focuses on hard/rare ones.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits
            targets: [B] class indices

        Returns:
            scalar loss
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
            weight=self.weight,
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class HierarchyHead(nn.Module):
    """Single classification head for one taxonomy level."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, input_dim] fused embedding

        Returns:
            logits: [B, num_classes]
        """
        return self.head(z)


class HierarchicalClassifier(nn.Module):
    """Multi-level hierarchical classification.

    Predicts taxonomy at each level independently from z.
    Each head is lightweight and independently replaceable.
    """

    def __init__(self, input_dim: int = 768,
                 num_classes_per_level: dict = None,
                 level_weights: list = None,
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        """
        Args:
            input_dim: dimension of fused embedding z
            num_classes_per_level: {"level1": 10, "level2": 50, ...}
            level_weights: loss weight per level [0.1, 0.2, 0.3, 0.4]
            use_focal_loss: use focal loss for class imbalance
            focal_gamma: focal loss gamma
            label_smoothing: label smoothing epsilon
        """
        super().__init__()

        if num_classes_per_level is None:
            num_classes_per_level = {}

        self.level_names = sorted(num_classes_per_level.keys())
        self.num_classes_per_level = num_classes_per_level

        # Default weights: leaf gets most weight
        if level_weights is None:
            n = len(self.level_names)
            level_weights = [(i + 1) / sum(range(1, n + 1)) for i in range(n)]
        self.level_weights = level_weights

        # Create a head for each level
        self.heads = nn.ModuleDict()
        for level_name in self.level_names:
            num_classes = num_classes_per_level[level_name]
            self.heads[level_name] = HierarchyHead(input_dim, num_classes)

        # Loss functions
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.criterion = FocalLoss(
                gamma=focal_gamma, label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing)

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: [B, input_dim] fused embedding

        Returns:
            dict of {level_name: logits [B, num_classes]}
        """
        return {name: head(z) for name, head in self.heads.items()}

    def compute_loss(self, logits_dict: dict, labels_dict: dict) -> tuple:
        """Compute weighted hierarchical loss.

        Args:
            logits_dict: {level_name: logits [B, C]}
            labels_dict: {level_name: labels [B]}

        Returns:
            total_loss: weighted sum of level losses
            level_losses: dict of individual level losses
        """
        total_loss = torch.tensor(0.0, device=next(iter(logits_dict.values())).device)
        level_losses = {}

        for i, level_name in enumerate(self.level_names):
            if level_name not in labels_dict:
                continue
            logits = logits_dict[level_name]
            labels = labels_dict[level_name]

            # Skip invalid labels (-1 means unknown)
            valid = labels >= 0
            if not valid.any():
                continue

            loss = self.criterion(logits[valid], labels[valid])
            weight = self.level_weights[i] if i < len(self.level_weights) else 1.0
            total_loss = total_loss + weight * loss
            level_losses[level_name] = loss.item()

        return total_loss, level_losses

    def predict(self, z: torch.Tensor) -> dict:
        """Get predictions with confidence.

        Args:
            z: [B, input_dim]

        Returns:
            dict with predictions, probabilities, confidence per level
        """
        logits_dict = self.forward(z)
        result = {}

        for name, logits in logits_dict.items():
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)
            top2 = probs.topk(min(2, probs.size(-1)), dim=-1)
            margin = (top2.values[:, 0] - top2.values[:, -1]) if probs.size(-1) > 1 else confidence

            result[name] = {
                "predicted": predicted,
                "confidence": confidence,
                "margin": margin,
                "probs": probs,
                "logits": logits,
            }

        return result
