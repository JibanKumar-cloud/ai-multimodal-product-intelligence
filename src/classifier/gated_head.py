"""Shared 2-Way Gated Head with Temperature + Entropy Regularization.

Used by both AttributePredictor and TaxonomyPredictor.
Each head learns its own gate: how much to trust image vs text.

Gate collapse prevention:
  - Temperature > 1.0 softens softmax → prevents [1.0, 0.0]
  - compute_gate_entropy() for regularization loss
    → maximizing entropy prevents any gate from ignoring a modality

When text is empty/zero → gate learns to put w_img≈1.0
When image is zero     → gate learns to put w_txt≈1.0
When both available    → gate balances per attribute/level
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedHead(nn.Module):
    """2-way gated classification head (image vs text).

    Temperature scaling prevents gate collapse to [1.0, 0.0].
    With temperature=2.0, softmax output range is roughly [0.12, 0.88]
    even with large gate logit differences.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_factor: int = 4, gate_temperature: float = 1.0):
        super().__init__()
        self.gate_temperature = gate_temperature
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
        gate_logits = self.gate(concat)
        # Temperature > 1.0 softens the distribution, prevents collapse
        gate_weights = F.softmax(gate_logits / self.gate_temperature, dim=-1)
        z = gate_weights[:, 0:1] * e_img + gate_weights[:, 1:2] * e_txt
        logits = self.classifier(z)
        return {"logits": logits, "gate_weights": gate_weights}


def compute_gate_entropy(outputs_dict, eps=1e-8):
    """Compute mean gate entropy across all heads in a dict.

    Returns negative entropy (add to loss with positive lambda to
    maximize entropy and prevent gate collapse).

    Max entropy for 2-way gate = ln(2) ~ 0.693

    Args:
        outputs_dict: {"head_name": {"logits": ..., "gate_weights": [B, 2]}}

    Returns:
        neg_entropy: scalar tensor
    """
    total = 0.0
    n = 0
    for name, head_out in outputs_dict.items():
        gw = head_out["gate_weights"]  # [B, 2]
        entropy = -(gw * (gw + eps).log()).sum(-1).mean()
        total += entropy
        n += 1
    if n == 0:
        return torch.tensor(0.0)
    return -(total / n)  # negative: minimizing this maximizes entropy