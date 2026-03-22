"""Attribute Prediction with Modality-Specific Head Types.

Each attribute uses the head type that matches its information source:

  COLOR (primary/secondary):
    ConcatHead — concatenates e_img + e_txt, no gate.
    Both modalities always contribute. Text says "espresso",
    image confirms dark brown tone. They complement, not compete.

  SHAPE:
    Image-only — fed (e_img, zeros). Shape is purely visual.

  ASSEMBLY:
    Text-only — fed (zeros, e_txt). Can't see "needs screwdriver" in a photo.

  MATERIAL, STYLE:
    GatedHead — learned gate balances image vs text.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .gated_head import GatedHead


class ConcatHead(nn.Module):
    """Classification head that concatenates both modalities (no gate).

    Input: concat(e_img, e_txt) → [B, 2*D]
    Always uses both — no modality competition.
    Better for attributes where text names the value and image confirms it.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        concat_dim = input_dim * 2  # 1536
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 4),  # 1536 → 384
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(concat_dim // 4, num_classes),
        )

    def forward(self, e_img, e_txt):
        z = torch.cat([e_img, e_txt], dim=-1)
        logits = self.classifier(z)
        # Return compatible format with GatedHead
        B = e_img.shape[0]
        fake_gate = torch.tensor([[0.5, 0.5]],
                                 device=e_img.device).expand(B, -1)
        return {"logits": logits, "gate_weights": fake_gate}


LOSS_WEIGHTS = {
    "primary_color": 1.5,
    "secondary_color": 0.8,
    "primary_material": 1.5,
    "secondary_material": 0.8,
    "style": 1.5,
    "shape": 0.5,
    "assembly": 0.3,
}

# Which head type for each attribute
HEAD_TYPES = {
    "primary_color": "concat",      # text says "espresso", image shows dark tone
    "secondary_color": "concat",    # text + image
    "primary_material": "concat",   # text says "oak", image shows wood grain
    "secondary_material": "concat", # text says "wood frame", image shows texture
    "style": "concat",              # text says "farmhouse", image shows aesthetic
    "shape": "image_only",          # pure image — round, rectangular, L-shaped
    "assembly": "concat",           # image shows complexity, text has product type
}


class AttributePredictor(nn.Module):
    """7 attribute heads with modality-specific architectures.

    Colors: ConcatHead (both modalities always used)
    Shape: image-only (GatedHead fed zeros for text)
    Assembly: text-only (GatedHead fed zeros for image)
    Material, Style: GatedHead (learned balance)
    """

    ATTR_KEYS = list(HEAD_TYPES.keys())

    def __init__(self, input_dim: int = 768, vocab_path: str = None):
        super().__init__()

        if vocab_path:
            with open(vocab_path) as f:
                vocab = json.load(f)
        else:
            vocab = {}

        self.attributes = OrderedDict()
        self.heads = nn.ModuleDict()
        self.head_types = {}
        self.value_to_idx = {}
        self.idx_to_value = {}

        for attr_name in self.ATTR_KEYS:
            values = vocab.get(attr_name, [])
            num_classes = len(values) + 1  # +1 for UNK

            self.attributes[attr_name] = {
                "values": values,
                "loss_weight": LOSS_WEIGHTS.get(attr_name, 1.0),
            }

            head_type = HEAD_TYPES[attr_name]
            self.head_types[attr_name] = head_type

            if head_type == "concat":
                self.heads[attr_name] = ConcatHead(input_dim, num_classes)
            else:
                # gated, image_only, text_only all use GatedHead
                # (modality zeroing happens in forward)
                self.heads[attr_name] = GatedHead(input_dim, num_classes)

            v2i = {"<UNK>": 0}
            i2v = {0: "<UNK>"}
            for i, val in enumerate(values):
                v2i[val] = i + 1
                i2v[i + 1] = val
            self.value_to_idx[attr_name] = v2i
            self.idx_to_value[attr_name] = i2v

    def forward(self, e_img, e_txt):
        results = {}
        for name, head in self.heads.items():
            head_type = self.head_types[name]

            if head_type == "concat":
                # Color: always uses both modalities (ConcatHead)
                results[name] = head(e_img, e_txt)

            elif head_type == "text_only":
                # Assembly: zero out image → gate forced to use text
                results[name] = head(torch.zeros_like(e_img), e_txt)

            elif head_type == "image_only":
                # Shape: zero out text → gate forced to use image
                results[name] = head(e_img, torch.zeros_like(e_txt))

            else:
                # Material, Style: learned gate
                results[name] = head(e_img, e_txt)

        return results

    def compute_class_weights(self, products, smoothing=0.1):
        """Compute inverse-frequency class weights from training data.

        Call once after creating model:
            model.attribute_heads.compute_class_weights(queue_products)

        Uses sqrt(inverse frequency) to avoid over-weighting extreme rarities.
        """
        import torch
        from collections import Counter

        self.class_weights = {}

        for attr_name, v2i in self.value_to_idx.items():
            num_classes = len(v2i)  # includes <UNK>

            # Count per-class frequency
            counts = Counter()
            for p in products:
                val = p.get(attr_name)
                if val and val in v2i:
                    counts[v2i[val]] += 1
                elif val:
                    counts[0] += 1  # <UNK>

            total = sum(counts.values())
            if total == 0:
                continue

            # Inverse frequency with sqrt smoothing
            # weight_i = sqrt(total / (num_classes * count_i))
            weights = torch.ones(num_classes)
            for cls_idx in range(num_classes):
                c = counts.get(cls_idx, 0)
                if c > 0:
                    weights[cls_idx] = (total / (num_classes * c)) ** 0.5
                else:
                    weights[cls_idx] = 1.0  # default for unseen

            # Normalize so mean weight = 1.0
            weights = weights / weights.mean()
            self.class_weights[attr_name] = weights

        print(f"  Class weights computed for {len(self.class_weights)} attributes")
        for attr_name, w in self.class_weights.items():
            print(f"    {attr_name:25s}: min={w.min():.2f}, max={w.max():.2f}, "
                  f"range={w.max()/w.min():.1f}x")

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

            attr_weight = self.attributes[attr_name]["loss_weight"]

            # Per-class weights (inverse frequency)
            cls_weight = None
            if hasattr(self, 'class_weights') and attr_name in self.class_weights:
                cls_weight = self.class_weights[attr_name].to(device)

            loss = F.cross_entropy(
                head_out["logits"][valid], labels[valid],
                weight=cls_weight,
                reduction="mean", label_smoothing=0.1)
            total_loss = total_loss + attr_weight * loss
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
                    "head_type": self.head_types[attr_name],
                }
        return results

    def get_gate_summary(self, e_img, e_txt):
        logits_dict = self.forward(e_img, e_txt)
        summary = {}
        for name, out in logits_dict.items():
            ht = self.head_types[name]
            gw = out["gate_weights"].mean(0)
            summary[name] = {
                "w_img": gw[0].item(),
                "w_txt": gw[1].item(),
                "type": ht,
            }
        return summary