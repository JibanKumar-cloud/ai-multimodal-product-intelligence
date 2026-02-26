"""Multi-Tower Product Classifier.

Full model combining:
  - Image encoder (frozen CLIP ViT) + Attention Pooler
  - Text encoder (frozen DistilBERT)
  - Attribute encoder (trainable MLP)
  - Gated fusion
  - Hierarchical classification heads
  - Mismatch detector

Supports both:
  - Live mode: raw images/text → full forward pass
  - Cached mode: precomputed embeddings → train only small layers
"""
import os
import json
from typing import Optional
from dataclasses import asdict

import torch
import torch.nn as nn
import numpy as np
from loguru import logger

from .config import ClassifierConfig
from .attention_pooler import AttentionPooler
from .attribute_encoder import AttributeEncoder
from .fusion import GatedFusion
from .hierarchy_head import HierarchicalClassifier
from .mismatch_detector import MismatchDetector, MismatchResult
from .attribute_head import AttributePredictor


class MultiTowerClassifier(nn.Module):
    """Full multi-tower product classification model.

    Architecture:
        images [B, K, 3, 224, 224] → ViT (frozen) → [B, K, 768]
            → AttentionPooler → e_img [B, 768]
        text [B, T] → DistilBERT (frozen) → e_txt [B, 768]
        attrs [B, A] → MLP → e_attr [B, 768]
        → GatedFusion → z [B, 768]
        → HierarchicalHeads → logits per level
        → MismatchDetector → mismatch flags
    """

    def __init__(self, config: ClassifierConfig,
                 taxonomy: dict = None):
        """
        Args:
            config: classifier configuration
            taxonomy: dict with label mappings per level, e.g.:
                {"level1": {"Furniture": 0, ...},
                 "level2": {"Living Room": 0, ...}, ...}
        """
        super().__init__()
        self.config = config
        self.taxonomy = taxonomy or {}

        # Store num classes from taxonomy (variable depth)
        num_classes = {}
        for level_name, mapping in self.taxonomy.items():
            num_classes[level_name] = len(mapping)

        self.level_names = sorted(
            [k for k in num_classes if k != "leaf"]) + ["leaf"]
        logger.info(f"Taxonomy levels: {self.level_names} "
                     f"({[num_classes.get(l, 0) for l in self.level_names]} classes)")

        # ── Trainable components ──
        self.attention_pooler = AttentionPooler(
            embed_dim=config.image_dim,
            num_heads=config.pooler_heads,
        )

        self.attribute_encoder = AttributeEncoder(
            output_dim=config.attr_output_dim,
            hidden_dim=config.attr_hidden_dim,
        )

        self.fusion = GatedFusion(
            embed_dim=config.fusion_dim,
            dropout=config.fusion_dropout,
        )

        self.hierarchy = HierarchicalClassifier(
            input_dim=config.fusion_dim,
            num_classes_per_level=num_classes,
            level_weights=config.hierarchy_weights,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_loss_gamma,
            label_smoothing=config.label_smoothing,
        )

        # Mismatch detector uses leaf class count
        num_leaf = num_classes.get("leaf", 100)
        self.mismatch_detector = MismatchDetector(
            embed_dim=config.fusion_dim,
            num_leaf_classes=num_leaf,
            kl_threshold=config.mismatch_kl_threshold,
            confidence_threshold=config.mismatch_confidence_threshold,
            margin_threshold=config.mismatch_margin_threshold,
        )

        # Attribute predictor (color, style, material, etc.)
        self.attribute_predictor = AttributePredictor(
            input_dim=config.fusion_dim,
        )

        # Log parameter counts
        self._log_params()

    def _log_params(self):
        """Log trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MultiTowerClassifier: {total:,} total, "
                     f"{trainable:,} trainable params")

        for name, module in [
            ("AttentionPooler", self.attention_pooler),
            ("AttributeEncoder", self.attribute_encoder),
            ("GatedFusion", self.fusion),
            ("HierarchyHeads", self.hierarchy),
            ("MismatchDetector", self.mismatch_detector),
            ("AttributePredictor", self.attribute_predictor),
        ]:
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            logger.info(f"  {name}: {n:,} trainable params")

    def forward(self, image_embeddings: torch.Tensor,
                image_mask: torch.Tensor,
                text_embeddings: torch.Tensor,
                attr_indices: torch.Tensor,
                labels: Optional[dict] = None):
        """Forward pass using precomputed image/text embeddings.

        Args:
            image_embeddings: [B, K_MAX, 768] precomputed ViT embeddings
            image_mask: [B, K_MAX] binary mask
            text_embeddings: [B, 768] precomputed DistilBERT embeddings
            attr_indices: [B, num_attr_keys] attribute index tensor
            labels: optional dict {"level1": [B], "level2": [B], ...}

        Returns:
            dict with logits, loss, predictions, mismatch results
        """
        B = image_embeddings.shape[0]
        has_image = image_mask.sum(dim=1) > 0  # [B]

        # ── Attention pooling over images ──
        e_img, attention_weights = self.attention_pooler(
            image_embeddings, image_mask)  # [B, 768], [B, K]

        # ── Text embedding (already precomputed) ──
        e_txt = text_embeddings  # [B, 768]

        # ── Attribute encoding ──
        e_attr = self.attribute_encoder(attr_indices)  # [B, 768]

        # ── Gated fusion ──
        z, gate_weights = self.fusion(
            e_img, e_txt, e_attr, has_image)  # [B, 768], [B, 3]

        # ── Hierarchical classification ──
        logits_dict = self.hierarchy(z)

        # ── Attribute prediction (per-attribute gating from raw embeddings) ──
        # NOT from z — each attribute head has its OWN gate
        attr_logits = self.attribute_predictor(e_img, e_txt, e_attr)

        # ── Build output ──
        output = {
            "logits": logits_dict,
            "attr_logits": attr_logits,
            "z": z,
            "e_img": e_img,
            "e_txt": e_txt,
            "e_attr": e_attr,
            "gate_weights": gate_weights,
            "attention_weights": attention_weights,
            "has_image": has_image,
        }

        # ── Compute loss if labels provided (training) ──
        if labels is not None:
            # Main hierarchical loss
            main_loss, level_losses = self.hierarchy.compute_loss(
                logits_dict, labels)

            # Attribute prediction loss (with modality probe losses)
            attr_labels = labels.get("attributes", {})
            if attr_labels:
                attr_loss, attr_level_losses = self.attribute_predictor.compute_loss(
                    attr_logits, attr_labels)
            else:
                attr_loss = torch.tensor(0.0, device=z.device)
                attr_level_losses = {}

            # Mismatch detector auxiliary loss (trains modality-specific heads)
            if "leaf" in labels:
                mismatch_loss = self.mismatch_detector.compute_loss(
                    e_img, e_txt, labels["leaf"], has_image)
            else:
                mismatch_loss = torch.tensor(0.0, device=z.device)

            total_loss = main_loss + 0.3 * attr_loss + 0.1 * mismatch_loss

            output["loss"] = total_loss
            output["main_loss"] = main_loss
            output["attr_loss"] = attr_loss
            output["mismatch_loss"] = mismatch_loss
            output["level_losses"] = level_losses
            output["attr_level_losses"] = attr_level_losses

        return output

    def predict(self, image_embeddings: torch.Tensor,
                image_mask: torch.Tensor,
                text_embeddings: torch.Tensor,
                attr_indices: torch.Tensor) -> dict:
        """Predict with confidence and mismatch detection.

        Args:
            Same as forward() but no labels.

        Returns:
            dict with predictions, confidence, mismatch results
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                image_embeddings, image_mask,
                text_embeddings, attr_indices)

            # Get predictions from hierarchy
            predictions = self.hierarchy.predict(output["z"])

            # Get attribute predictions (per-attribute gating)
            attr_predictions = self.attribute_predictor.predict(
                output["e_img"], output["e_txt"], output["e_attr"])

            # Run mismatch detection
            leaf_logits = output["logits"].get(
                "leaf", list(output["logits"].values())[-1])
            mismatch_results = self.mismatch_detector(
                output["e_img"], output["e_txt"],
                leaf_logits,
                output["attention_weights"],
                output["has_image"],
            )

            return {
                "predictions": predictions,
                "attr_predictions": attr_predictions,
                "gate_weights": output["gate_weights"],
                "attention_weights": output["attention_weights"],
                "mismatch_results": mismatch_results,
                "z": output["z"],
            }

    def save(self, save_dir: str):
        """Save model checkpoint and config."""
        os.makedirs(save_dir, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))

        # Save config
        config_dict = {k: v for k, v in vars(self.config).items()}
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Save taxonomy
        with open(os.path.join(save_dir, "taxonomy.json"), "w") as f:
            json.dump(self.taxonomy, f, indent=2)

        logger.info(f"Model saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str, device: str = "cpu") -> "MultiTowerClassifier":
        """Load model from checkpoint."""
        # Load config
        with open(os.path.join(save_dir, "config.json")) as f:
            config_dict = json.load(f)
        config = ClassifierConfig(**{
            k: v for k, v in config_dict.items()
            if k in ClassifierConfig.__dataclass_fields__
        })

        # Load taxonomy
        with open(os.path.join(save_dir, "taxonomy.json")) as f:
            taxonomy = json.load(f)

        model = cls(config=config, taxonomy=taxonomy)
        state_dict = torch.load(
            os.path.join(save_dir, "model.pt"),
            map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded from {save_dir}")
        return model
