"""Attribute Encoder for structured product features.

Parses WANDS product_features (pipe-delimited key:value pairs)
and encodes them into a fixed-dim embedding via MLP.

Input: "Color:Beige | Material:Polyester | Style:Contemporary"
Output: e_attr [768]
"""
import json
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


# Canonical attribute keys we encode
ATTRIBUTE_KEYS = [
    "style", "primary_material", "secondary_material", "color_family",
    "room_type", "product_type", "assembly_required",
    "brand", "finish", "shape", "number_of_pieces",
]

# Common values for each key (built from WANDS analysis)
COMMON_VALUES = {
    "style": [
        "contemporary", "modern", "traditional", "transitional", "rustic",
        "industrial", "coastal", "farmhouse", "mid-century modern", "bohemian",
        "glam", "scandinavian", "cottage", "tropical", "asian",
    ],
    "primary_material": [
        "wood", "metal", "fabric", "leather", "plastic", "glass", "stone",
        "wicker", "rattan", "bamboo", "ceramic", "marble", "steel",
        "aluminum", "iron", "polyester", "cotton", "velvet", "linen",
    ],
    "color_family": [
        "white", "black", "brown", "gray", "blue", "green", "red", "beige",
        "cream", "gold", "silver", "navy", "pink", "orange", "yellow",
        "purple", "tan", "ivory", "walnut", "espresso", "natural",
    ],
    "room_type": [
        "living room", "bedroom", "dining room", "bathroom", "kitchen",
        "office", "outdoor", "entryway", "hallway", "nursery",
    ],
    "assembly_required": ["yes", "no"],
    "shape": [
        "rectangular", "round", "square", "oval", "l-shaped", "u-shaped",
    ],
}


class AttributeEncoder(nn.Module):
    """Encodes structured product attributes into embedding."""

    def __init__(self, output_dim: int = 768, hidden_dim: int = 256):
        super().__init__()

        # Build vocabulary for each attribute key
        self.vocab = OrderedDict()
        self.vocab_sizes = OrderedDict()
        total_features = 0

        for key in ATTRIBUTE_KEYS:
            values = COMMON_VALUES.get(key, [])
            # +1 for unknown, +1 for missing/null
            self.vocab[key] = {v: i + 2 for i, v in enumerate(values)}
            self.vocab[key]["<UNK>"] = 1
            self.vocab[key]["<NULL>"] = 0
            self.vocab_sizes[key] = len(values) + 2
            total_features += 1

        # Embedding for each attribute key
        self.embeddings = nn.ModuleDict()
        embed_dim_per_key = 16
        for key in ATTRIBUTE_KEYS:
            vocab_size = self.vocab_sizes[key]
            self.embeddings[key] = nn.Embedding(vocab_size, embed_dim_per_key)

        # MLP to project concatenated attribute embeddings
        input_dim = len(ATTRIBUTE_KEYS) * embed_dim_per_key
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        logger.info(
            f"AttributeEncoder: {len(ATTRIBUTE_KEYS)} keys, "
            f"input_dim={input_dim}, output_dim={output_dim}"
        )

    def parse_features(self, feature_string: str) -> dict:
        """Parse WANDS product_features string into normalized dict.

        Args:
            feature_string: "Color:Beige | Material:Polyester | ..."

        Returns:
            dict mapping canonical keys to values
        """
        if not feature_string or str(feature_string) == "nan":
            return {}

        result = {}
        parts = str(feature_string).split("|")
        for part in parts:
            part = part.strip()
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip().lower()
            value = value.strip().lower()

            # Map raw keys to canonical keys
            key_mapping = {
                "color": "color_family", "primary color": "color_family",
                "finish color": "color_family", "finish": "color_family",
                "material": "primary_material", "frame material": "primary_material",
                "top material": "primary_material",
                "seat material": "secondary_material",
                "upholstery material": "secondary_material",
                "upholstery": "secondary_material",
                "style": "style", "design style": "style",
                "room": "room_type", "room type": "room_type",
                "product type": "product_type", "type": "product_type",
                "assembly required": "assembly_required",
                "assembly": "assembly_required",
                "brand": "brand",
                "shape": "shape",
                "number of pieces": "number_of_pieces",
            }
            canonical = key_mapping.get(key)
            if canonical and canonical not in result:
                result[canonical] = value

        return result

    def encode_features(self, parsed: dict) -> np.ndarray:
        """Convert parsed features dict to integer indices.

        Args:
            parsed: dict from parse_features()

        Returns:
            numpy array of indices [num_keys]
        """
        indices = []
        for key in ATTRIBUTE_KEYS:
            value = parsed.get(key)
            if value is None:
                indices.append(0)  # <NULL>
            else:
                vocab = self.vocab[key]
                idx = vocab.get(value, vocab["<UNK>"])
                indices.append(idx)
        return np.array(indices, dtype=np.int64)

    def precompute_product(self, product_id: str, feature_string: str,
                           output_dir: str) -> str:
        """Precompute attribute indices for one product.

        Args:
            product_id: product identifier
            feature_string: raw WANDS product_features string
            output_dir: directory to save

        Returns:
            path to saved file
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{product_id}_attr.npy")
        if os.path.exists(save_path):
            return save_path

        parsed = self.parse_features(feature_string)
        indices = self.encode_features(parsed)
        np.save(save_path, indices)
        return save_path

    @staticmethod
    def load_cached(product_id: str, embeddings_dir: str) -> np.ndarray:
        """Load cached attribute indices."""
        import os
        path = os.path.join(embeddings_dir, f"{product_id}_attr.npy")
        if os.path.exists(path):
            return np.load(path)
        return np.zeros(len(ATTRIBUTE_KEYS), dtype=np.int64)

    def forward(self, attr_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attr_indices: [B, num_keys] integer indices per attribute

        Returns:
            e_attr: [B, output_dim]
        """
        parts = []
        for i, key in enumerate(ATTRIBUTE_KEYS):
            idx = attr_indices[:, i]  # [B]
            emb = self.embeddings[key](idx)  # [B, embed_dim_per_key]
            parts.append(emb)

        concatenated = torch.cat(parts, dim=-1)  # [B, num_keys * 16]
        return self.mlp(concatenated)  # [B, output_dim]
