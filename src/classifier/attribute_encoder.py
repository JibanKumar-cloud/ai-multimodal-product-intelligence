"""Attribute Encoder - encodes structured product attributes into embedding.

Updated to match 7-attribute vocab from prepare_classifier_data.py:
  primary_color, secondary_color, primary_material, secondary_material,
  style, shape, assembly

COMMON_VALUES matches attribute_vocab.json exactly.
"""
import torch
import torch.nn as nn
from collections import OrderedDict


# ════════════════════════════════════════════════════════════════
# Attribute keys we extract from product_features
# ════════════════════════════════════════════════════════════════

ATTRIBUTE_KEYS = [
    "primary_color", "secondary_color",
    "primary_material", "secondary_material",
    "style", "shape", "assembly",
]

# Common normalized values for each attribute
# Matches attribute_vocab.json from prepare_classifier_data.py
COMMON_VALUES = {
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
        "natural_fiber", "other", "plastic", "stone", "synthetics",
        "wood",
    ],
    "secondary_material": [
        "ceramic", "fabric", "foam", "glass", "leather", "metal",
        "mixed", "natural_fiber", "plastic", "stone", "synthetics",
        "wood",
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
    "assembly": [
        "full", "none", "partial",
    ],
}


class AttributeEncoder(nn.Module):
    """Encodes structured product attributes into a single embedding.

    Each attribute key has an embedding table. Values are looked up,
    summed, and projected to output_dim.

    This is the e_attr branch in the multi-tower architecture:
        IMAGE  → ViT   → e_img [768]
        TEXT   → BERT  → e_txt [768]
        ATTRS  → THIS  → e_attr [768]  ← structured metadata
    """

    def __init__(self, output_dim: int = 768, hidden_dim: int = 256):
        super().__init__()

        self.vocab = OrderedDict()
        self.vocab_sizes = OrderedDict()
        total_features = 0

        for key in ATTRIBUTE_KEYS:
            values = COMMON_VALUES.get(key, [])
            # idx 0 = padding/unknown, 1..N = known values
            vocab = {"<PAD>": 0}
            for i, val in enumerate(values):
                vocab[val] = i + 1
            self.vocab[key] = vocab
            self.vocab_sizes[key] = len(vocab)
            total_features += 1

        # One embedding table per attribute key
        self.embeddings = nn.ModuleDict({
            key: nn.Embedding(size, hidden_dim, padding_idx=0)
            for key, size in self.vocab_sizes.items()
        })

        # Project concatenated attribute embeddings to output_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * len(ATTRIBUTE_KEYS), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_values(self, attr_dict: dict) -> dict:
        """Convert string attribute values to indices.

        Args:
            attr_dict: {"primary_color": "blue", "style": "modern", ...}

        Returns:
            {"primary_color": 5, "style": 7, ...}  (0 for unknown/missing)
        """
        indices = {}
        for key in ATTRIBUTE_KEYS:
            value = attr_dict.get(key)
            if value is None:
                indices[key] = 0  # PAD
            else:
                value = str(value).lower().strip()
                indices[key] = self.vocab[key].get(value, 0)
        return indices

    def forward(self, attr_indices: dict) -> torch.Tensor:
        """
        Args:
            attr_indices: {key: tensor [B]} of attribute index tensors

        Returns:
            e_attr: [B, output_dim]
        """
        parts = []
        for key in ATTRIBUTE_KEYS:
            idx = attr_indices.get(key)
            if idx is None:
                # Create zero tensor if key missing
                B = next(iter(attr_indices.values())).shape[0]
                device = next(iter(attr_indices.values())).device
                parts.append(torch.zeros(B, self.embeddings[key].embedding_dim,
                                         device=device))
            else:
                parts.append(self.embeddings[key](idx))  # [B, hidden_dim]

        concat = torch.cat(parts, dim=-1)  # [B, hidden_dim * num_keys]
        return self.projection(concat)  # [B, output_dim]


def load_vocab_from_json(vocab_path: str) -> dict:
    """Load attribute vocab from attribute_vocab.json.

    Can be used to verify COMMON_VALUES matches the data prep output.
    """
    import json
    with open(vocab_path) as f:
        vocab = json.load(f)
    return vocab