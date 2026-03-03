"""Attribute Prediction with Per-Attribute 2-Way Gating.

7 heads, each with [e_img, e_txt] -> own gate -> prediction.

Handles all modality combinations:
  image + text  → gate balances both
  image only    → gate shifts to w_img≈1.0
  text only     → gate shifts to w_txt≈1.0

Expected gate convergence after training:
  primary_color:      w_img=0.85  w_txt=0.15  (visual)
  secondary_color:    w_img=0.80  w_txt=0.20  (visual)
  shape:              w_img=0.90  w_txt=0.10  (visual)
  primary_material:   w_img=0.65  w_txt=0.35  (texture + text)
  secondary_material: w_img=0.60  w_txt=0.40  (texture + text)
  style:              w_img=0.45  w_txt=0.55  (both)
  assembly:           w_img=0.05  w_txt=0.95  (pure text)
"""
import json
import torch
import torch.nn.functional as F
from collections import OrderedDict

from .gated_head import GatedHead


DEFAULT_VOCAB = {
    "primary_color": [
        "beige", "black", "blue", "brown", "clear", "gold_metal",
        "gray", "green", "multi", "orange", "other", "pink",
        "purple", "red", "silver", "white", "yellow",
    ],
    "secondary_color": [
        "beige", "black", "blue", "brown", "clear", "gold_metal",
        "gray", "green", "multi", "orange", "pink", "purple",
        "red", "silver", "white", "yellow",
    ],
    "primary_material": [
        "ceramic", "fabric", "foam", "glass", "leather", "metal",
        "natural_fiber", "other", "plastic", "stone", "synthetics", "wood",
    ],
    "secondary_material": [
        "ceramic", "fabric", "foam", "glass", "leather", "metal",
        "mixed", "natural_fiber", "plastic", "stone", "synthetics", "wood",
    ],
    "style": [
        "bohemian", "coastal", "farmhouse", "glam", "industrial",
        "mid-century modern", "modern", "other", "rustic",
        "scandinavian", "traditional", "transitional",
    ],
    "shape": [
        "hexagon", "irregular", "l-shaped", "other", "oval",
        "rectangular", "round", "runner", "square", "u-shaped",
    ],
    "assembly": ["full", "none", "partial"],
}

LOSS_WEIGHTS = {
    "primary_color": 1.5,
    "secondary_color": 0.8,
    "primary_material": 1.5,
    "secondary_material": 0.8,
    "style": 1.5,
    "shape": 0.5,
    "assembly": 0.3,
}


class AttributePredictor(torch.nn.Module):
    """7 attribute heads with per-attribute 2-way gating."""

    def __init__(self, input_dim: int = 768, vocab_path: str = None):
        super().__init__()

        if vocab_path:
            with open(vocab_path) as f:
                vocab = json.load(f)
        else:
            vocab = DEFAULT_VOCAB

        self.attributes = OrderedDict()
        self.heads = torch.nn.ModuleDict()
        self.value_to_idx = {}
        self.idx_to_value = {}

        for attr_name, values in vocab.items():
            num_classes = len(values) + 1  # +1 for UNK
            self.attributes[attr_name] = {
                "values": values,
                "loss_weight": LOSS_WEIGHTS.get(attr_name, 1.0),
            }
            self.heads[attr_name] = GatedHead(input_dim, num_classes)

            v2i = {"<UNK>": 0}
            i2v = {0: "<UNK>"}
            for i, val in enumerate(values):
                v2i[val] = i + 1
                i2v[i + 1] = val
            self.value_to_idx[attr_name] = v2i
            self.idx_to_value[attr_name] = i2v

    def forward(self, e_img, e_txt):
        return {name: head(e_img, e_txt)
                for name, head in self.heads.items()}

    def compute_loss(self, logits_dict, labels_dict):
        device = next(iter(logits_dict.values()))["logits"].device
        total_loss = torch.tensor(0.0, device=device)
        per_attr = {}

        for attr_name, head_out in logits_dict.items():
            if attr_name not in labels_dict:
                continue
            labels = labels_dict[attr_name]
            valid = labels >= 0
            if not valid.any():
                continue

            weight = self.attributes[attr_name]["loss_weight"]
            loss = F.cross_entropy(
                head_out["logits"][valid], labels[valid],
                reduction="mean", label_smoothing=0.1)
            total_loss = total_loss + weight * loss
            per_attr[attr_name] = loss.item()

        return total_loss, per_attr

    def predict(self, e_img, e_txt, confidence_threshold=0.5):
        logits_dict = self.forward(e_img, e_txt)
        B = e_img.shape[0]
        results = [{} for _ in range(B)]

        for attr_name, head_out in logits_dict.items():
            probs = F.softmax(head_out["logits"], dim=-1)
            conf, pred = probs.max(dim=-1)
            gw = head_out["gate_weights"]

            for i in range(B):
                value = self.idx_to_value[attr_name].get(
                    pred[i].item(), "<UNK>")
                c = conf[i].item()
                results[i][attr_name] = {
                    "value": value if c >= confidence_threshold else None,
                    "confidence": c,
                    "needs_vlm": c < confidence_threshold,
                    "gate_weights": gw[i].tolist(),
                }
        return results

    def get_gate_summary(self, e_img, e_txt):
        logits_dict = self.forward(e_img, e_txt)
        return {
            name: {"w_img": out["gate_weights"].mean(0)[0].item(),
                   "w_txt": out["gate_weights"].mean(0)[1].item()}
            for name, out in logits_dict.items()
        }