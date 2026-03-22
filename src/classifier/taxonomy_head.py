"""Cascading Taxonomy + Conditioned Product Class Prediction.

Architecture:
  level_1: gate(e_img, e_txt) → "Furniture"
  level_2: gate(e_img, e_txt) + level_1_probs → "Living Room"     (narrows ~20)
  level_3: gate(e_img, e_txt) + level_1+2_probs → "Sofas"         (narrows ~50)
  level_4: gate(e_img, e_txt) + level_1+2+3_probs → "Sectionals"
  ...
  product_class: gate(e_img, e_txt) + all_level_probs → "Sectional Sofas"

Each deeper level receives accumulated probability context from all
previous levels, so it knows WHERE in the taxonomy tree to look.

Gradients flow back through all softmax probs — deeper level losses
improve shallower predictions through backprop.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gated_head import GatedHead


class CascadingHead(nn.Module):
    """Taxonomy head conditioned on parent level distributions.

    Uses concatenation (not gating) for e_img + e_txt — same fix as attributes.
    CLIP is too dominant for learned gating.

    level_1: concat(e_img, e_txt) → classify
    level_2+: concat(e_img, e_txt, parent_probs_proj) → classify
    """

    def __init__(self, input_dim: int, num_classes: int,
                 parent_probs_dim: int = 0, proj_dim: int = 128):
        super().__init__()
        self.has_parent = parent_probs_dim > 0

        # Concat both modalities (no gate)
        base_dim = input_dim * 2  # 1536

        if self.has_parent:
            self.parent_projector = nn.Sequential(
                nn.Linear(parent_probs_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            classifier_input_dim = base_dim + proj_dim  # 1536 + 128
        else:
            classifier_input_dim = base_dim  # 1536

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_input_dim // 4, num_classes),
        )

    def forward(self, e_img, e_txt, parent_probs_concat=None):
        """
        Args:
            e_img: [B, D]
            e_txt: [B, D]
            parent_probs_concat: [B, sum(parent_classes)] or None for level_1
        """
        # Always use both modalities
        z = torch.cat([e_img, e_txt], dim=-1)  # [B, 1536]

        if self.has_parent and parent_probs_concat is not None:
            parent_emb = self.parent_projector(parent_probs_concat)
            z = torch.cat([z, parent_emb], dim=-1)  # [B, 1536 + proj_dim]

        logits = self.classifier(z)

        # Return compatible format
        B = e_img.shape[0]
        fake_gate = torch.tensor([[0.5, 0.5]],
                                 device=e_img.device).expand(B, -1)
        return {"logits": logits, "gate_weights": fake_gate}


class TaxonomyPredictor(nn.Module):
    """Cascading taxonomy + conditioned product_class prediction.

    Each level receives probability distributions from ALL previous levels.
    This narrows the search space — if level_1 says "Furniture" (high prob),
    level_2 focuses on furniture subcategories, not lighting subcategories.

    Product class receives ALL level probs.

    Gradient flow: deeper losses → parent probs → parent heads.
    All levels reinforce each other through backprop.
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

        # ── Taxonomy level heads (cascading) ──
        self.level_heads = nn.ModuleDict()
        self.level_v2i = {}
        self.level_i2v = {}
        self.level_keys = []
        self.level_num_classes = {}

        # Build heads — each knows the cumulative parent dim
        cumulative_parent_dim = 0
        for level_key, values in sorted(tax.get("level_values", {}).items()):
            num_classes = len(values) + 1  # +1 for <UNK>

            # proj_dim scales with parent context size, clamped
            proj_dim = min(128, max(32, cumulative_parent_dim // 2)) \
                if cumulative_parent_dim > 0 else 0

            self.level_heads[level_key] = CascadingHead(
                input_dim=input_dim,
                num_classes=num_classes,
                parent_probs_dim=cumulative_parent_dim,  # 0 for level_1
                proj_dim=proj_dim if proj_dim > 0 else 128,
            )

            self.level_keys.append(level_key)
            self.level_num_classes[level_key] = num_classes
            cumulative_parent_dim += num_classes

            v2i = {"<UNK>": 0}
            i2v = {0: "<UNK>"}
            for i, val in enumerate(values):
                v2i[val] = i + 1
                i2v[i + 1] = val
            self.level_v2i[level_key] = v2i
            self.level_i2v[level_key] = i2v

        # ── Product class head (conditioned on ALL taxonomy probs) ──
        class_values = tax.get("product_classes", [])
        if not class_values:
            class_values = []
        self.has_class_head = len(class_values) > 0

        if self.has_class_head:
            num_class_labels = len(class_values) + 1
            total_tax_probs_dim = sum(self.level_num_classes.values())

            self.class_head = CascadingHead(
                input_dim=input_dim,
                num_classes=num_class_labels,
                parent_probs_dim=total_tax_probs_dim,
                proj_dim=256,
            )

            self.class_v2i = {"<UNK>": 0}
            self.class_i2v = {0: "<UNK>"}
            for i, val in enumerate(class_values):
                self.class_v2i[val] = i + 1
                self.class_i2v[i + 1] = val

    def forward(self, e_img, e_txt):
        """Cascading forward: each level feeds into the next.

        Returns:
            dict of {head_name: {"logits": [B, C], "gate_weights": [B, 2]}}
        """
        results = {}
        all_probs = []

        # Run levels in order — each receives cumulative parent probs
        parent_probs_concat = None
        for k in self.level_keys:
            head_out = self.level_heads[k](e_img, e_txt, parent_probs_concat)
            results[k] = head_out

            # Softmax probs (differentiable — gradients flow back)
            probs = F.softmax(head_out["logits"], dim=-1)
            all_probs.append(probs)

            # Build cumulative context for next level
            parent_probs_concat = torch.cat(all_probs, dim=-1)

        # Product class: conditioned on ALL level probs
        if self.has_class_head:
            all_tax_probs = torch.cat(all_probs, dim=-1)
            results["product_class"] = self.class_head(
                e_img, e_txt, all_tax_probs)

        return results

    def compute_class_weights(self, products, smoothing=0.1):
        """Compute inverse-frequency class weights from training data.

        Call once: model.taxonomy_heads.compute_class_weights(queue_products)
        """
        from collections import Counter

        self.class_weights = {}

        # Taxonomy level weights
        for level_key, v2i in self.level_v2i.items():
            num_classes = len(v2i)
            counts = Counter()
            level_idx = int(level_key.split("_")[1]) - 1  # "level_1" → 0

            for p in products:
                tax = p.get("taxonomy", [])
                if level_idx < len(tax):
                    val = tax[level_idx]
                    idx = v2i.get(val, 0)
                    counts[idx] += 1

            total = sum(counts.values())
            if total == 0:
                continue

            weights = torch.ones(num_classes)
            for cls_idx in range(num_classes):
                c = counts.get(cls_idx, 0)
                if c > 0:
                    weights[cls_idx] = (total / (num_classes * c)) ** 0.5

            weights = weights / weights.mean()
            self.class_weights[level_key] = weights

        # Product class weights
        if self.has_class_head:
            counts = Counter()
            for p in products:
                pc = p.get("product_class")
                if pc and pc in self.class_v2i:
                    counts[self.class_v2i[pc]] += 1

            total = sum(counts.values())
            if total > 0:
                num_classes = len(self.class_v2i)
                weights = torch.ones(num_classes)
                for cls_idx in range(num_classes):
                    c = counts.get(cls_idx, 0)
                    if c > 0:
                        weights[cls_idx] = (total / (num_classes * c)) ** 0.5
                weights = weights / weights.mean()
                self.class_weights["product_class"] = weights

        print(f"  Taxonomy class weights computed for {len(self.class_weights)} heads")

    def compute_loss(self, logits_dict, labels_dict):
        """Compute weighted multi-task loss for taxonomy + product_class."""
        device = next(iter(logits_dict.values()))["logits"].device
        total_loss = torch.tensor(0.0, device=device)
        per_head = {}

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

            level_weight = self.LEVEL_WEIGHTS.get(level_key, 0.5)

            cls_weight = None
            if hasattr(self, 'class_weights') and level_key in self.class_weights:
                cls_weight = self.class_weights[level_key].to(device)

            loss = F.cross_entropy(
                head_out["logits"][valid], labels[valid],
                weight=cls_weight,
                reduction="mean", label_smoothing=0.1)
            total_loss = total_loss + level_weight * loss
            per_head[level_key] = loss.item()

        if "product_class" in logits_dict and "product_class" in labels_dict:
            labels = labels_dict["product_class"]
            valid = labels >= 0
            if valid.any():
                cls_weight = None
                if hasattr(self, 'class_weights') and "product_class" in self.class_weights:
                    cls_weight = self.class_weights["product_class"].to(device)

                loss = F.cross_entropy(
                    logits_dict["product_class"]["logits"][valid],
                    labels[valid],
                    weight=cls_weight,
                    reduction="mean", label_smoothing=0.1)
                total_loss = total_loss + self.PRODUCT_CLASS_WEIGHT * loss
                per_head["product_class"] = loss.item()

        return total_loss, per_head

    def predict(self, e_img, e_txt, confidence_threshold=0.5):
        """Predict taxonomy levels and product_class with confidence."""
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