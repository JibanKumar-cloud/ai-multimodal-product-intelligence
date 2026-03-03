"""Taxonomy + Product Class Prediction with 2-Way Gating.

Predicts:
  level_1: "Furniture"              (54 classes)
  level_2: "Living Room Furniture"  (126 classes)
  level_3: "Cabinets & Chests"     (440 classes)
  level_4: "Brown Cabinets"         (785 classes)
  level_5+: deeper levels           (variable)
  product_class: "Accent Chests / Cabinets"  (~580 classes)

Taxonomy levels are independent 2-way gated heads.
Product class is CONDITIONED on taxonomy — it receives the probability
distributions from all taxonomy heads as additional context, so it knows
where in the taxonomy tree the product sits before predicting class.

Gradient flows back: product_class loss → taxonomy probs → taxonomy heads.
This creates mutual reinforcement between taxonomy and product_class.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .gated_head import GatedHead


class TaxonomyConditionedClassHead(nn.Module):
    """Product class head conditioned on taxonomy distributions.

    Takes the standard gated fusion z = w_img * e_img + w_txt * e_txt
    PLUS a projected taxonomy context vector built from the probability
    distributions of all taxonomy level heads.

    Architecture:
        tax_probs (all levels) → concat → project → tax_emb [256]
        gate(e_img, e_txt) → z [768]
        concat(z, tax_emb) [1024] → MLP → product_class logits
    """

    def __init__(self, input_dim: int, num_classes: int,
                 tax_probs_dim: int, tax_proj_dim: int = 256):
        super().__init__()

        # Gate: same 2-way gate as other heads
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        # Taxonomy context projector
        # tax_probs_dim = sum of all taxonomy level class counts
        self.tax_projector = nn.Sequential(
            nn.Linear(tax_probs_dim, tax_proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Classifier takes fused embedding + taxonomy context
        fused_dim = input_dim + tax_proj_dim  # 768 + 256 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),  # 1024 → 512
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, e_img, e_txt, tax_probs_concat):
        """
        Args:
            e_img:             [B, D] image embedding
            e_txt:             [B, D] text embedding
            tax_probs_concat:  [B, sum(tax_classes)] concatenated taxonomy
                               probability distributions from all levels

        Returns:
            {"logits": [B, C], "gate_weights": [B, 2]}
        """
        # Standard 2-way gating
        concat = torch.cat([e_img, e_txt], dim=-1)
        gate_weights = F.softmax(self.gate(concat), dim=-1)
        z = gate_weights[:, 0:1] * e_img + gate_weights[:, 1:2] * e_txt

        # Project taxonomy context
        tax_emb = self.tax_projector(tax_probs_concat)

        # Fuse and classify
        class_input = torch.cat([z, tax_emb], dim=-1)
        logits = self.classifier(class_input)

        return {"logits": logits, "gate_weights": gate_weights}


class TaxonomyPredictor(nn.Module):
    """Hierarchical taxonomy + taxonomy-conditioned product_class prediction.

    Taxonomy levels: independent 2-way gated heads.
    Product class: conditioned on taxonomy probability distributions.
        tax level probs → project → concat with gated fusion → predict class.

    Gradient from product_class loss flows back through taxonomy probs,
    creating mutual reinforcement between taxonomy and class predictions.
    """

    LEVEL_WEIGHTS = {
        "level_1": 2.0, "level_2": 1.5, "level_3": 1.0,
        "level_4": 0.8, "level_5": 0.5, "level_6": 0.3,
        "level_7": 0.2, "level_8": 0.1,
    }
    PRODUCT_CLASS_WEIGHT = 1.5

    def __init__(self, input_dim: int = 768, taxonomy_path: str = None):
        super().__init__()

        with open(taxonomy_path) as f:
            tax = json.load(f)

        # ── Taxonomy level heads ──
        self.level_heads = nn.ModuleDict()
        self.level_v2i = {}
        self.level_i2v = {}
        self.level_keys = []
        self.level_num_classes = {}  # track class counts for projector sizing

        for level_key, values in sorted(tax.get("level_values", {}).items()):
            num_classes = len(values) + 1  # +1 for <UNK>
            self.level_heads[level_key] = GatedHead(
                input_dim, num_classes, hidden_factor=2)
            self.level_keys.append(level_key)
            self.level_num_classes[level_key] = num_classes

            v2i = {"<UNK>": 0}
            i2v = {0: "<UNK>"}
            for i, val in enumerate(values):
                v2i[val] = i + 1
                i2v[i + 1] = val
            self.level_v2i[level_key] = v2i
            self.level_i2v[level_key] = i2v

        # ── Product class head (conditioned on taxonomy) ──
        class_values = tax.get("product_classes", [])
        if not class_values:
            class_values = []
        self.has_class_head = len(class_values) > 0

        if self.has_class_head:
            num_class_labels = len(class_values) + 1  # +1 for <UNK>

            # Total dimension of concatenated taxonomy probability vectors
            # e.g. level_1 has 55 classes + level_2 has 127 + ... = ~1545
            tax_probs_dim = sum(self.level_num_classes.values())

            self.class_head = TaxonomyConditionedClassHead(
                input_dim=input_dim,
                num_classes=num_class_labels,
                tax_probs_dim=tax_probs_dim,
                tax_proj_dim=256,
            )

            self.class_v2i = {"<UNK>": 0}
            self.class_i2v = {0: "<UNK>"}
            for i, val in enumerate(class_values):
                self.class_v2i[val] = i + 1
                self.class_i2v[i + 1] = val

    def forward(self, e_img, e_txt):
        """Forward pass: taxonomy levels first, then conditioned product_class.

        Returns:
            dict of {head_name: {"logits": [B, C], "gate_weights": [B, 2]}}
        """
        results = {}

        # Step 1: Run all taxonomy level heads
        tax_probs_list = []
        for k in self.level_keys:
            head_out = self.level_heads[k](e_img, e_txt)
            results[k] = head_out

            # Collect probability distributions for product_class conditioning
            # Use softmax on logits — these are differentiable, so gradients
            # from product_class loss flow back into taxonomy heads
            probs = F.softmax(head_out["logits"], dim=-1)  # [B, C_level]
            tax_probs_list.append(probs)

        # Step 2: Run product_class head conditioned on taxonomy distributions
        if self.has_class_head:
            # Concatenate all taxonomy probs: [B, sum(C_levels)]
            tax_probs_concat = torch.cat(tax_probs_list, dim=-1)
            results["product_class"] = self.class_head(
                e_img, e_txt, tax_probs_concat)

        return results

    def compute_loss(self, logits_dict, labels_dict):
        """Compute weighted multi-task loss for taxonomy + product_class."""
        device = next(iter(logits_dict.values()))["logits"].device
        total_loss = torch.tensor(0.0, device=device)
        per_head = {}

        # Taxonomy level losses
        for level_key, head_out in logits_dict.items():
            if level_key == "product_class":
                continue
            label_key = f"tax_{level_key}"
            if label_key not in labels_dict:
                continue
            labels = labels_dict[label_key]
            valid = labels >= 0
            if not valid.any():
                continue

            weight = self.LEVEL_WEIGHTS.get(level_key, 0.5)
            loss = F.cross_entropy(
                head_out["logits"][valid], labels[valid],
                reduction="mean", label_smoothing=0.1)
            total_loss = total_loss + weight * loss
            per_head[level_key] = loss.item()

        # Product class loss (gradients flow back through taxonomy probs)
        if "product_class" in logits_dict and "product_class" in labels_dict:
            labels = labels_dict["product_class"]
            valid = labels >= 0
            if valid.any():
                loss = F.cross_entropy(
                    logits_dict["product_class"]["logits"][valid],
                    labels[valid],
                    reduction="mean", label_smoothing=0.1)
                total_loss = total_loss + self.PRODUCT_CLASS_WEIGHT * loss
                per_head["product_class"] = loss.item()

        return total_loss, per_head

    def predict(self, e_img, e_txt, confidence_threshold=0.5):
        """Predict taxonomy levels and product_class with confidence scores."""
        logits_dict = self.forward(e_img, e_txt)
        B = e_img.shape[0]
        results = [{} for _ in range(B)]

        for level_key in self.level_keys:
            head_out = logits_dict[level_key]
            probs = F.softmax(head_out["logits"], dim=-1)
            conf, pred = probs.max(dim=-1)

            for i in range(B):
                value = self.level_i2v[level_key].get(
                    pred[i].item(), "<UNK>")
                c = conf[i].item()
                results[i][level_key] = {
                    "value": value,
                    "confidence": c,
                    "needs_vlm": c < confidence_threshold,
                }

        if self.has_class_head and "product_class" in logits_dict:
            head_out = logits_dict["product_class"]
            probs = F.softmax(head_out["logits"], dim=-1)
            conf, pred = probs.max(dim=-1)
            for i in range(B):
                value = self.class_i2v.get(pred[i].item(), "<UNK>")
                c = conf[i].item()
                results[i]["product_class"] = {
                    "value": value,
                    "confidence": c,
                    "needs_vlm": c < confidence_threshold,
                }

        return results

    def get_gate_summary(self, e_img, e_txt):
        """Get average gate weights for monitoring convergence."""
        logits_dict = self.forward(e_img, e_txt)
        return {
            name: {"w_img": out["gate_weights"].mean(0)[0].item(),
                   "w_txt": out["gate_weights"].mean(0)[1].item()}
            for name, out in logits_dict.items()
        }