"""Dataset for Multi-Tower Classifier Training.

Supports two modes:
  - Cached: load precomputed ViT/BERT embeddings from disk (fast training)
  - Live: process raw images/text on the fly (slower, for inference)

Handles:
  - Variable image counts (0 to K_MAX) with padding/masking
  - Random image sampling for products with > K_MAX images
  - WANDS taxonomy parsing into hierarchical labels
  - Train/val/test splits with stratification
"""
import os
import json
import random
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class ClassifierDataset(Dataset):
    """Dataset for multi-tower classifier training."""

    def __init__(self, products_df: pd.DataFrame,
                 taxonomy: dict,
                 embeddings_dir: str = "data/embeddings",
                 image_manifest: str = None,
                 k_max: int = 5,
                 mode: str = "cached",
                 split: str = "train",
                 text_augment: bool = True):
        """
        Args:
            products_df: DataFrame with product_id, product_name, product_class,
                        category_hierarchy, product_description, product_features
            taxonomy: label mappings {"level1": {"Furniture": 0}, ...}
            embeddings_dir: directory with precomputed .npy files
            image_manifest: path to image manifest JSON
            k_max: max images per product
            mode: "cached" (precomputed embeddings) or "live"
            split: "train", "val", "test"
            text_augment: enable text/modality augmentation during training
        """
        self.products_df = products_df.reset_index(drop=True)
        self.taxonomy = taxonomy
        self.embeddings_dir = embeddings_dir
        self.k_max = k_max
        self.mode = mode
        self.split = split
        self.text_augment = text_augment and (split == "train")

        # Parse hierarchy for each product
        self._parse_hierarchy()

        # Load image manifest if available
        self.image_manifest = {}
        if image_manifest and os.path.exists(image_manifest):
            with open(image_manifest) as f:
                self.image_manifest = json.load(f)

        aug_str = "ON" if self.text_augment else "OFF"
        logger.info(
            f"ClassifierDataset ({split}): {len(self)} products, "
            f"mode={mode}, k_max={k_max}, text_augment={aug_str}"
        )

    def _parse_hierarchy(self):
        """Parse category_hierarchy into variable-depth level labels.

        Handles taxonomies of different depths:
          "Furniture / Living Room / Sofas / Sectional"  → 4 levels
          "Lighting / Table Lamps"                        → 2 levels
          "Decor / Wall Art / Canvas / Abstract / Modern" → 5 levels

        Strategy:
          - Parse ALL hierarchy levels from data
          - Determine MAX_DEPTH from the deepest path
          - Products with fewer levels get -1 for missing levels
          - product_class (leaf) is ALWAYS the deepest prediction target
          - Intermediate levels may overlap with leaf for shallow trees

        Example:
          "Lighting / Table Lamps" with product_class="Table Lamps"
            level1: "Lighting" → valid
            level2: "Table Lamps" → valid (same as leaf, that's OK)
            level3: -1 (doesn't exist)
            level4: -1 (doesn't exist)
            leaf:   "Table Lamps" → valid (always present)
        """
        # First pass: determine max depth across all products
        all_depths = []
        parsed_levels = []

        for _, row in self.products_df.iterrows():
            hierarchy = str(row.get("category_hierarchy", "")).strip()
            if "/" in hierarchy:
                levels = [l.strip() for l in hierarchy.split("/") if l.strip()]
            elif ">" in hierarchy:
                levels = [l.strip() for l in hierarchy.split(">") if l.strip()]
            else:
                levels = [hierarchy] if hierarchy else []

            all_depths.append(len(levels))
            parsed_levels.append(levels)

        self.max_depth = max(all_depths) if all_depths else 1
        depth_dist = {}
        for d in all_depths:
            depth_dist[d] = depth_dist.get(d, 0) + 1
        logger.info(f"Taxonomy depths: {depth_dist}")
        logger.info(f"Max depth: {self.max_depth}")

        # Determine which levels to use as classification targets
        # We use up to max_depth intermediate levels + leaf (product_class)
        self.level_names = [f"level{i+1}" for i in range(self.max_depth)] + ["leaf"]

        # Second pass: assign labels
        self.labels = []
        for idx, (_, row) in enumerate(self.products_df.iterrows()):
            levels = parsed_levels[idx]
            leaf = str(row.get("product_class", "")).strip()

            label = {}
            # Intermediate levels
            for i in range(self.max_depth):
                level_name = f"level{i+1}"
                if i < len(levels) and levels[i]:
                    label[level_name] = self.taxonomy.get(
                        level_name, {}).get(levels[i], -1)
                else:
                    label[level_name] = -1  # doesn't exist for this product

            # Leaf (always present — it's product_class)
            label["leaf"] = self.taxonomy.get("leaf", {}).get(leaf, -1)

            self.labels.append(label)

    def __len__(self):
        return len(self.products_df)

    def __getitem__(self, idx):
        row = self.products_df.iloc[idx]
        pid = str(row["product_id"])

        # ── Image embeddings ──
        if self.mode == "cached":
            img_embeds, img_mask = self._load_cached_images(pid)
        else:
            img_embeds, img_mask = self._load_live_images(pid)

        # ── Text embedding ──
        if self.mode == "cached":
            txt_embed = self._load_cached_text(pid)
        else:
            txt_embed = np.zeros(768, dtype=np.float32)  # placeholder

        # ── Attribute indices (INPUT to encoder) ──
        attr_indices = self._load_attributes(pid, row)

        # ── Attribute labels (TARGET for prediction heads) ──
        attr_labels = self._get_attribute_labels(row)

        # ── CRITICAL: Attribute dropout to prevent leakage ──
        # Without this, attr_encoder INPUT and attr_head TARGET are IDENTICAL.
        # Model just learns to copy. Dropout forces learning from text/images.
        if self.text_augment:
            attr_indices, attr_labels = self._attribute_dropout(
                attr_indices, attr_labels)

        # ── Modality augmentation (training only) ──
        # Forces the gate to learn dynamic modality weighting.
        if self.text_augment:
            txt_embed, attr_indices, img_embeds, img_mask = (
                self._augment_modalities(
                    txt_embed, attr_indices, img_embeds, img_mask)
            )

        # ── Taxonomy labels ──
        labels = self.labels[idx]

        item = {
            "product_id": pid,
            "image_embeddings": torch.tensor(img_embeds, dtype=torch.float32),
            "image_mask": torch.tensor(img_mask, dtype=torch.float32),
            "text_embeddings": torch.tensor(txt_embed, dtype=torch.float32),
            "attr_indices": torch.tensor(attr_indices, dtype=torch.long),
            "label_leaf": labels["leaf"],
            "attr_labels": attr_labels,
        }

        # Add variable depth level labels
        for level_name in self.level_names:
            if level_name != "leaf":
                item[f"label_{level_name}"] = labels.get(level_name, -1)

        return item

    def _get_attribute_labels(self, row: pd.Series) -> dict:
        """Parse product_features into attribute prediction labels."""
        from .attribute_encoder import AttributeEncoder
        from .attribute_head import AttributePredictor

        feature_str = str(row.get("product_features", ""))
        # Quick parse
        parsed = {}
        if feature_str and feature_str != "nan":
            parts = feature_str.split("|")
            key_mapping = {
                "color": "color_family", "primary color": "color_family",
                "finish color": "color_family", "finish": "color_family",
                "material": "primary_material", "frame material": "primary_material",
                "seat material": "secondary_material",
                "upholstery material": "secondary_material",
                "style": "style", "design style": "style",
                "room": "room_type", "room type": "room_type",
                "assembly required": "assembly_required",
                "shape": "shape",
            }
            for part in parts:
                part = part.strip()
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                key = key.strip().lower()
                value = value.strip().lower()
                canonical = key_mapping.get(key)
                if canonical and canonical not in parsed:
                    parsed[canonical] = value

        # Convert to label indices using AttributePredictor vocab
        predictor = AttributePredictor.__new__(AttributePredictor)
        labels = {}
        from .attribute_head import PREDICTED_ATTRIBUTES, COMMON_VALUES
        for attr_name, attr_info in PREDICTED_ATTRIBUTES.items():
            value = parsed.get(attr_name)
            if value is None:
                labels[attr_name] = -1  # missing → skip in loss
            else:
                values_list = attr_info["values"]
                # +1 because 0 is UNK
                v2i = {v: i + 1 for i, v in enumerate(values_list)}
                v2i["<UNK>"] = 0
                labels[attr_name] = v2i.get(value, 0)

        return labels

    def _attribute_dropout(self, attr_indices, attr_labels):
        """Drop attributes from INPUT while keeping them as LABELS.

        Prevents data leakage: without this, model just copies
        input attributes to output. With dropout, model must learn
        to predict attributes from text + images.

        Strategy per attribute (independent coin flip):
          50% — KEEP in input, KEEP as label (easy: can copy)
          30% — DROP from input, KEEP as label (must infer from text/img)
          20% — DROP from input, DROP as label (unknown: skip in loss)

        This means for any given training step, ~30% of attributes
        must be predicted purely from text + image embeddings.
        """
        from .attribute_encoder import ATTRIBUTE_KEYS
        from .attribute_head import PREDICTED_ATTRIBUTES

        # Build mapping: which attr_indices positions map to which attr_labels keys
        # attr_indices is ordered by ATTRIBUTE_KEYS
        # attr_labels is keyed by PREDICTED_ATTRIBUTES keys
        attr_key_to_idx = {k: i for i, k in enumerate(ATTRIBUTE_KEYS)}

        # Map predicted attribute names to encoder index positions
        pred_to_encoder_idx = {
            "color_family": attr_key_to_idx.get("color_family"),
            "primary_material": attr_key_to_idx.get("primary_material"),
            "style": attr_key_to_idx.get("style"),
            "room_type": attr_key_to_idx.get("room_type"),
            "assembly_required": attr_key_to_idx.get("assembly_required"),
            "shape": attr_key_to_idx.get("shape"),
        }

        new_indices = attr_indices.copy()
        new_labels = attr_labels.copy()

        for attr_name in PREDICTED_ATTRIBUTES:
            if new_labels.get(attr_name, -1) == -1:
                # No label exists — nothing to dropout
                continue

            r = random.random()

            if r < 0.50:
                # KEEP: input has it, label has it (model CAN copy)
                pass

            elif r < 0.80:
                # DROP from input, KEEP as label (model MUST infer)
                encoder_idx = pred_to_encoder_idx.get(attr_name)
                if encoder_idx is not None and encoder_idx < len(new_indices):
                    new_indices[encoder_idx] = 0  # 0 = <NULL> in encoder vocab

            else:
                # DROP both: pretend attribute doesn't exist
                encoder_idx = pred_to_encoder_idx.get(attr_name)
                if encoder_idx is not None and encoder_idx < len(new_indices):
                    new_indices[encoder_idx] = 0
                new_labels[attr_name] = -1  # skip in loss

        return new_indices, new_labels

    def _load_cached_images(self, pid: str):
        """Load precomputed image embeddings with padding."""
        path = os.path.join(self.embeddings_dir, f"{pid}_img.npy")
        embed_dim = 768

        if os.path.exists(path):
            raw = np.load(path)
            n_images = raw.shape[0]
        else:
            raw = np.zeros((0, embed_dim), dtype=np.float32)
            n_images = 0

        padded = np.zeros((self.k_max, embed_dim), dtype=np.float32)
        mask = np.zeros(self.k_max, dtype=np.float32)

        if n_images > 0:
            if n_images > self.k_max and self.split == "train":
                # Random sample during training (regularization)
                indices = np.random.choice(n_images, self.k_max, replace=False)
                padded[:] = raw[indices]
                mask[:] = 1.0
            else:
                n_use = min(n_images, self.k_max)
                padded[:n_use] = raw[:n_use]
                mask[:n_use] = 1.0

        return padded, mask

    def _load_live_images(self, pid: str):
        """Load and encode images on the fly (for inference)."""
        # Placeholder — would use ImageEncoder in live mode
        padded = np.zeros((self.k_max, 768), dtype=np.float32)
        mask = np.zeros(self.k_max, dtype=np.float32)

        manifest_entry = self.image_manifest.get(pid, {})
        # In live mode, images would be encoded here
        return padded, mask

    def _load_cached_text(self, pid: str):
        """Load precomputed text embedding."""
        path = os.path.join(self.embeddings_dir, f"{pid}_txt.npy")
        if os.path.exists(path):
            return np.load(path)
        return np.zeros(768, dtype=np.float32)

    def _load_attributes(self, pid: str, row: pd.Series):
        """Load or compute attribute indices."""
        # Try cached first
        path = os.path.join(self.embeddings_dir, f"{pid}_attr.npy")
        if os.path.exists(path):
            return np.load(path)

        # Compute on the fly from feature_parser
        from .attribute_encoder import AttributeEncoder, ATTRIBUTE_KEYS
        encoder = AttributeEncoder.__new__(AttributeEncoder)
        # Quick parse without full init
        feature_str = str(row.get("product_features", ""))
        parsed = encoder.parse_features(feature_str) if hasattr(encoder, 'parse_features') else {}
        return np.zeros(len(ATTRIBUTE_KEYS), dtype=np.int64)

    def _augment_modalities(self, txt_embed, attr_indices, img_embeds, img_mask):
        """Augment modalities during training to force dynamic gate learning.

        Without this, BERT dominates and the image tower is useless.
        Same principle as adversarial training in VLM — degrade text
        so the model learns to rely on images when text is bad.

        Distribution:
          20% — original (all modalities intact)
          25% — vague text (add noise to text embedding)
          15% — no text (zero out text embedding entirely)
          10% — no attributes (zero out attribute indices)
          10% — adversarial text (random embedding, simulates wrong text)
          10% — text + attrs degraded (only images should matter)
           5% — no images (drop images, text must carry)
           5% — only images (drop text + attrs)
        """
        r = random.random()

        if r < 0.20:
            # Original — no augmentation
            pass

        elif r < 0.45:
            # Vague text: add gaussian noise to weaken text signal
            noise_scale = random.uniform(0.3, 0.8)
            noise = np.random.randn(*txt_embed.shape).astype(np.float32)
            txt_embed = txt_embed * (1 - noise_scale) + noise * noise_scale
            norm = np.linalg.norm(txt_embed)
            if norm > 0:
                txt_embed = txt_embed / norm * np.sqrt(768)

        elif r < 0.60:
            # No text: zero out entirely
            txt_embed = np.zeros_like(txt_embed)

        elif r < 0.70:
            # No attributes: zero out attr indices
            attr_indices = np.zeros_like(attr_indices)

        elif r < 0.80:
            # Adversarial text: random embedding (wrong description)
            txt_embed = np.random.randn(*txt_embed.shape).astype(np.float32)
            norm = np.linalg.norm(txt_embed)
            if norm > 0:
                txt_embed = txt_embed / norm * np.sqrt(768)

        elif r < 0.90:
            # Text + attrs both degraded: only images should matter
            txt_embed = np.zeros_like(txt_embed)
            attr_indices = np.zeros_like(attr_indices)

        elif r < 0.95:
            # No images: force text to carry (bidirectional learning)
            img_embeds = np.zeros_like(img_embeds)
            img_mask = np.zeros_like(img_mask)

        else:
            # Only images: drop text AND attrs
            # Most extreme case — image tower must work alone
            txt_embed = np.zeros_like(txt_embed)
            attr_indices = np.zeros_like(attr_indices)

        return txt_embed, attr_indices, img_embeds, img_mask


def build_taxonomy(products_df: pd.DataFrame) -> dict:
    """Build taxonomy label mappings from WANDS products.

    Handles variable-depth hierarchies automatically.

    Args:
        products_df: DataFrame with category_hierarchy and product_class

    Returns:
        dict with label mappings per level:
        {"level1": {"Furniture": 0, ...}, "level2": {...}, ..., "leaf": {...}}
    """
    # Parse all hierarchies to find max depth
    level_sets = {}  # level_name → set of values
    max_depth = 0

    for _, row in products_df.iterrows():
        hierarchy = str(row.get("category_hierarchy", "")).strip()
        if "/" in hierarchy:
            levels = [l.strip() for l in hierarchy.split("/") if l.strip()]
        elif ">" in hierarchy:
            levels = [l.strip() for l in hierarchy.split(">") if l.strip()]
        else:
            levels = [hierarchy] if hierarchy else []

        max_depth = max(max_depth, len(levels))

        for i, level_val in enumerate(levels):
            level_name = f"level{i+1}"
            if level_name not in level_sets:
                level_sets[level_name] = set()
            if level_val:
                level_sets[level_name].add(level_val)

    # Leaf classes (product_class)
    leaf_set = set()
    for _, row in products_df.iterrows():
        leaf = str(row.get("product_class", "")).strip()
        if leaf:
            leaf_set.add(leaf)

    # Build taxonomy with sorted, deterministic mappings
    taxonomy = {}
    for i in range(max_depth):
        level_name = f"level{i+1}"
        if level_name in level_sets:
            taxonomy[level_name] = {
                v: idx for idx, v in enumerate(sorted(level_sets[level_name]))
            }
    taxonomy["leaf"] = {v: idx for idx, v in enumerate(sorted(leaf_set))}

    # Log summary
    logger.info(f"Taxonomy built (max_depth={max_depth}):")
    for level_name, mapping in taxonomy.items():
        logger.info(f"  {level_name}: {len(mapping)} classes")

    return taxonomy


def create_splits(products_df: pd.DataFrame,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42) -> tuple:
    """Create stratified train/val/test splits.

    Stratifies by product_class to ensure all classes in all splits.

    Returns:
        (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    # Stratify by product_class
    labels = products_df["product_class"].fillna("Unknown")

    # Handle classes with too few samples for stratification
    class_counts = labels.value_counts()
    rare_classes = class_counts[class_counts < 3].index
    # Group rare classes together for stratification
    strat_labels = labels.copy()
    strat_labels[strat_labels.isin(rare_classes)] = "__RARE__"

    # Split train+val vs test
    train_val_df, test_df = train_test_split(
        products_df, test_size=test_ratio,
        stratify=strat_labels, random_state=seed)

    # Split train vs val
    strat_tv = train_val_df["product_class"].fillna("Unknown").copy()
    rare_tv = strat_tv.value_counts()
    rare_tv = rare_tv[rare_tv < 2].index
    strat_tv[strat_tv.isin(rare_tv)] = "__RARE__"

    val_fraction = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_fraction,
        stratify=strat_tv, random_state=seed)

    logger.info(f"Splits: train={len(train_df)}, val={len(val_df)}, "
                f"test={len(test_df)}")

    return train_df, val_df, test_df


def collate_fn(batch):
    """Custom collate for DataLoader. Handles variable taxonomy depth."""
    # Collect attribute labels
    attr_label_keys = batch[0]["attr_labels"].keys()
    attr_labels = {
        key: torch.tensor([b["attr_labels"][key] for b in batch])
        for key in attr_label_keys
    }

    # Collect taxonomy level labels (variable depth)
    level_labels = {}
    for key in batch[0].keys():
        if key.startswith("label_level"):
            level_name = key.replace("label_", "")
            level_labels[level_name] = torch.tensor(
                [b[key] for b in batch])

    # Leaf is always present
    level_labels["leaf"] = torch.tensor([b["label_leaf"] for b in batch])

    return {
        "product_id": [b["product_id"] for b in batch],
        "image_embeddings": torch.stack([b["image_embeddings"] for b in batch]),
        "image_mask": torch.stack([b["image_mask"] for b in batch]),
        "text_embeddings": torch.stack([b["text_embeddings"] for b in batch]),
        "attr_indices": torch.stack([b["attr_indices"] for b in batch]),
        "labels": {
            **level_labels,
            "attributes": attr_labels,
        },
    }
