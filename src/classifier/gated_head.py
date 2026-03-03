"""Shared 2-Way Gated Head.

Used by both AttributePredictor and TaxonomyPredictor.
Each head learns its own gate: how much to trust image vs text.

When text is empty/zero → gate learns to put w_img≈1.0
When image is zero     → gate learns to put w_txt≈1.0
When both available    → gate balances per attribute/level
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedHead(nn.Module):
    """2-way gated classification head (image vs text).

    Automatically adapts when one modality is missing (zeros).
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_factor: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // hidden_factor),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // hidden_factor, num_classes),
        )

    def forward(self, e_img, e_txt):
        """
        Args:
            e_img: [B, D] image embedding (zeros if no image)
            e_txt: [B, D] text embedding (zeros if no text)

        Returns:
            {"logits": [B, C], "gate_weights": [B, 2]}
        """
        concat = torch.cat([e_img, e_txt], dim=-1)
        gate_weights = F.softmax(self.gate(concat), dim=-1)
        z = gate_weights[:, 0:1] * e_img + gate_weights[:, 1:2] * e_txt
        logits = self.classifier(z)
        return {"logits": logits, "gate_weights": gate_weights}