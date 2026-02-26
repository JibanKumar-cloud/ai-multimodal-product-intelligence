"""Attribute Prediction with Per-Attribute Gating.

Each attribute head gets raw modality embeddings and learns
its OWN gate — which modality to trust for THIS attribute.

  color_head:    [e_img; e_txt; e_attr] → gate → z_color → "blue"
  assembly_head: [e_img; e_txt; e_attr] → gate → z_assembly → "yes"

After training with adversarial augmentation:
  color_head gate:    w_img=0.7, w_txt=0.2, w_attr=0.1  (visual)
  assembly_head gate: w_img=0.0, w_txt=0.3, w_attr=0.7  (textual)
  style_head gate:    w_img=0.4, w_txt=0.4, w_attr=0.2  (both)

Each head learns from data what to trust. No shared z.
If confidence is low → fallback to parsed product_features value.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .attribute_encoder import COMMON_VALUES


PREDICTED_ATTRIBUTES = {
    "color_family": {
        "values": COMMON_VALUES["color_family"],
        "type": "single_label",
        "loss_weight": 1.0,
    },
    "primary_material": {
        "values": COMMON_VALUES["primary_material"],
        "type": "single_label",
        "loss_weight": 1.0,
    },
    "style": {
        "values": COMMON_VALUES["style"],
        "type": "single_label",
        "loss_weight": 1.5,     # needs model most
    },
    "room_type": {
        "values": COMMON_VALUES["room_type"],
        "type": "multi_label",
        "loss_weight": 1.0,
    },
    "assembly_required": {
        "values": COMMON_VALUES["assembly_required"],
        "type": "single_label",
        "loss_weight": 0.3,
    },
    "shape": {
        "values": COMMON_VALUES["shape"],
        "type": "single_label",
        "loss_weight": 0.5,
    },
}


class GatedAttributeHead(nn.Module):
    """Single attribute head with its own modality gate.

    Gets [e_img, e_txt, e_attr] → learns own weighting → predicts.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 attr_type: str = "single_label"):
        super().__init__()
        self.attr_type = attr_type

        # Per-attribute gate: learns which modality matters for THIS attribute
        # Input: concat of all three → softmax over 3 weights
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # [w_img, w_txt, w_attr]
        )

        # Classification head: from gated embedding → prediction
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 4, num_classes),
        )

    def forward(self, e_img: torch.Tensor,
                e_txt: torch.Tensor,
                e_attr: torch.Tensor) -> dict:
        """
        Args:
            e_img: [B, D] image embedding
            e_txt: [B, D] text embedding
            e_attr: [B, D] attribute embedding

        Returns:
            {"logits": [B, C], "gate_weights": [B, 3]}
        """
        # Compute per-attribute gate weights
        concat = torch.cat([e_img, e_txt, e_attr], dim=-1)  # [B, 3D]
        gate_weights = F.softmax(self.gate(concat), dim=-1)  # [B, 3]

        # Gated fusion specific to THIS attribute
        w_img = gate_weights[:, 0:1]   # [B, 1]
        w_txt = gate_weights[:, 1:2]
        w_attr = gate_weights[:, 2:3]

        z_attr = w_img * e_img + w_txt * e_txt + w_attr * e_attr  # [B, D]

        # Classify
        logits = self.classifier(z_attr)  # [B, C]

        return {"logits": logits, "gate_weights": gate_weights}


class AttributePredictor(nn.Module):
    """Multi-task attribute prediction with per-attribute gating.

    Each attribute has its OWN gate that learns from adversarial
    training which modality to trust.

    At inference:
      confidence HIGH → use model prediction (can correct wrong metadata)
      confidence LOW  → fallback to parsed product_features value
    """

    def __init__(self, input_dim: int = 768):
        super().__init__()

        self.attributes = OrderedDict()
        self.heads = nn.ModuleDict()
        self.value_to_idx = {}
        self.idx_to_value = {}

        for attr_name, attr_info in PREDICTED_ATTRIBUTES.items():
            values = attr_info["values"]
            num_classes = len(values) + 1  # +1 for UNK
            attr_type = attr_info["type"]

            self.attributes[attr_name] = attr_info
            self.heads[attr_name] = GatedAttributeHead(
                input_dim, num_classes, attr_type)

            v2i = {"<UNK>": 0}
            i2v = {0: "<UNK>"}
            for i, val in enumerate(values):
                v2i[val] = i + 1
                i2v[i + 1] = val
            self.value_to_idx[attr_name] = v2i
            self.idx_to_value[attr_name] = i2v

    def forward(self, e_img: torch.Tensor,
                e_txt: torch.Tensor,
                e_attr: torch.Tensor) -> dict:
        """
        Args:
            e_img: [B, D] image embedding
            e_txt: [B, D] text embedding
            e_attr: [B, D] attribute embedding

        Returns:
            dict of {attr_name: {"logits": [B,C], "gate_weights": [B,3]}}
        """
        return {
            name: head(e_img, e_txt, e_attr)
            for name, head in self.heads.items()
        }

    def compute_loss(self, logits_dict: dict,
                     labels_dict: dict) -> tuple:
        """Compute weighted attribute prediction loss.

        Args:
            logits_dict: {attr_name: {"logits": [B,C], "gate_weights": [B,3]}}
            labels_dict: {attr_name: [B] labels, -1 for missing}

        Returns:
            total_loss, per_attr_losses
        """
        device = list(logits_dict.values())[0]["logits"].device
        total_loss = torch.tensor(0.0, device=device)
        per_attr_losses = {}

        for attr_name, head_output in logits_dict.items():
            if attr_name not in labels_dict:
                continue

            logits = head_output["logits"]
            labels = labels_dict[attr_name]
            valid = labels >= 0

            if not valid.any():
                continue

            attr_info = self.attributes[attr_name]
            weight = attr_info["loss_weight"]

            if attr_info["type"] == "multi_label":
                loss = F.binary_cross_entropy_with_logits(
                    logits[valid], labels[valid].float(),
                    reduction="mean")
            else:
                loss = F.cross_entropy(
                    logits[valid], labels[valid],
                    reduction="mean", label_smoothing=0.1)

            total_loss = total_loss + weight * loss
            per_attr_losses[attr_name] = loss.item()

        return total_loss, per_attr_losses

    def predict(self, e_img: torch.Tensor,
                e_txt: torch.Tensor,
                e_attr: torch.Tensor,
                parsed_values: dict = None,
                confidence_threshold: float = 0.5) -> list:
        """Get attribute predictions with confidence gating.

        High confidence → use model prediction
        Low confidence  → fallback to parsed value

        Args:
            e_img: [B, D]
            e_txt: [B, D]
            e_attr: [B, D]
            parsed_values: dict of {attr_name: str} from product_features
            confidence_threshold: below this → use parsed value

        Returns:
            list of dicts (one per batch item):
            {attr_name: {"value": str, "confidence": float,
                         "source": "model"|"parsed",
                         "gate_weights": [w_img, w_txt, w_attr]}}
        """
        logits_dict = self.forward(e_img, e_txt, e_attr)
        B = e_img.shape[0]
        parsed_values = parsed_values or {}

        results = [{} for _ in range(B)]

        for attr_name, head_output in logits_dict.items():
            logits = head_output["logits"]
            gate_w = head_output["gate_weights"]
            attr_info = self.attributes[attr_name]

            if attr_info["type"] == "multi_label":
                probs = torch.sigmoid(logits)
                for i in range(B):
                    active = (probs[i] > 0.5).nonzero(as_tuple=True)[0]
                    values = [
                        self.idx_to_value[attr_name].get(j.item(), "<UNK>")
                        for j in active if j.item() > 0
                    ]
                    conf = (probs[i][active].mean().item()
                            if len(active) > 0 else 0.0)

                    # Confidence gating
                    if conf >= confidence_threshold and values:
                        results[i][attr_name] = {
                            "value": values,
                            "confidence": conf,
                            "source": "model",
                            "gate_weights": gate_w[i].tolist(),
                        }
                    else:
                        # Fallback to parsed value
                        parsed = parsed_values.get(attr_name)
                        if parsed:
                            results[i][attr_name] = {
                                "value": [parsed] if isinstance(
                                    parsed, str) else parsed,
                                "confidence": 1.0,
                                "source": "parsed",
                                "gate_weights": gate_w[i].tolist(),
                            }
            else:
                probs = F.softmax(logits, dim=-1)
                confidence, predicted = probs.max(dim=-1)

                for i in range(B):
                    pred_idx = predicted[i].item()
                    value = self.idx_to_value[attr_name].get(
                        pred_idx, "<UNK>")
                    conf = confidence[i].item()

                    # Confidence gating
                    if conf >= confidence_threshold and value != "<UNK>":
                        results[i][attr_name] = {
                            "value": value,
                            "confidence": conf,
                            "source": "model",
                            "gate_weights": gate_w[i].tolist(),
                        }
                    else:
                        # Fallback to parsed value
                        parsed = parsed_values.get(attr_name)
                        if parsed:
                            results[i][attr_name] = {
                                "value": parsed,
                                "confidence": 1.0,
                                "source": "parsed",
                                "gate_weights": gate_w[i].tolist(),
                            }

        return results

    def encode_labels(self, parsed_features: dict) -> dict:
        """Convert parsed feature dict to label indices."""
        labels = {}
        for attr_name in self.attributes:
            value = parsed_features.get(attr_name)
            if value is None:
                labels[attr_name] = -1
            else:
                value = str(value).lower().strip()
                v2i = self.value_to_idx[attr_name]
                labels[attr_name] = v2i.get(value, 0)
        return labels
