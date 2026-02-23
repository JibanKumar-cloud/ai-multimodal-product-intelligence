"""BERT-based text classifier baseline for attribute extraction.

A simpler baseline that treats each attribute as a separate classification task.
Uses a shared BERT backbone with per-attribute classification heads.
This serves as a mid-range baseline between rule-based and VLM approaches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from src.models.attribute_extractor import BaseAttributeExtractor
from src.data.feature_parser import ATTRIBUTE_SCHEMA


class BERTClassificationHead(nn.Module):
    """Classification head for a single attribute."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled_output)


class BERTAttributeModel(nn.Module):
    """Multi-head BERT model for attribute extraction.

    Shared BERT backbone with separate classification heads per attribute.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        attribute_classes: Optional[dict[str, list[str]]] = None,
    ):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Default attribute classes (can be overridden with actual WANDS classes)
        self.attribute_classes = attribute_classes or {
            "style": [
                "mid-century modern", "contemporary", "traditional", "farmhouse",
                "industrial", "scandinavian", "bohemian", "coastal", "rustic",
                "transitional", "other",
            ],
            "primary_material": [
                "solid wood", "engineered wood", "metal", "glass", "fabric",
                "leather", "ceramic", "plastic", "stone", "other",
            ],
            "color_family": [
                "brown", "gray", "white", "black", "blue", "green", "beige",
                "red", "multi-color", "other",
            ],
        }

        # Create classification heads
        self.heads = nn.ModuleDict()
        for attr_name, classes in self.attribute_classes.items():
            self.heads[attr_name] = BERTClassificationHead(
                hidden_size=hidden_size,
                num_classes=len(classes),
            )

    def forward(self, input_ids, attention_mask=None) -> dict[str, torch.Tensor]:
        """Forward pass through backbone + all classification heads."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [batch, hidden_size]

        logits = {}
        for attr_name, head in self.heads.items():
            logits[attr_name] = head(pooled)

        return logits


class BERTExtractor(BaseAttributeExtractor):
    """BERT-based attribute extractor (text-only baseline)."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        checkpoint_path: Optional[str | Path] = None,
        device: str = "auto",
    ):
        super().__init__(model_name=model_name)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device_str = device
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load BERT model and optional checkpoint."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BERTAttributeModel(model_name=self.model_name)

        if self.checkpoint_path and self.checkpoint_path.exists():
            state_dict = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True,
            )
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")

        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device_str == "auto"
            else torch.device(self.device_str)
        )
        self.model.to(device)
        self.model.eval()
        self._is_loaded = True

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"BERT model loaded: {total_params:,} parameters")

    @torch.inference_mode()
    def extract(
        self,
        product_name: str,
        product_description: Optional[str] = None,
        product_class: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> dict:
        """Extract attributes using BERT classification heads."""
        if not self._is_loaded:
            self.load()

        # Build input text
        text = product_name
        if product_class:
            text += f" [SEP] {product_class}"
        if product_description:
            text += f" [SEP] {str(product_description)[:300]}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True,
        ).to(next(self.model.parameters()).device)

        # Forward pass
        logits = self.model(**inputs)

        # Convert logits to predictions
        result = {}
        for attr_name, attr_logits in logits.items():
            pred_idx = attr_logits.argmax(dim=-1).item()
            classes = self.model.attribute_classes[attr_name]
            result[attr_name] = classes[pred_idx]

        # Fill in missing attributes
        for key in ["secondary_material", "room_type", "product_type", "assembly_required"]:
            if key not in result:
                result[key] = None

        return result
