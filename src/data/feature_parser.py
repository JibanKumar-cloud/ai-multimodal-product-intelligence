"""Parse WANDS product_features into structured attribute dictionaries.

The product_features column in WANDS contains pipe-delimited key:value pairs like:
    "Color:Beige | Material:Polyester | Style:Contemporary | Assembly Required:Yes"

This module parses these into structured JSON suitable for training and evaluation.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from loguru import logger


# Canonical attribute names and their normalized forms
ATTRIBUTE_SCHEMA = {
    "style": {
        "source_keys": [
            "style", "design style", "style/pattern", "furniture style",
            "decor style", "design", "theme",
        ],
        "type": "categorical",
    },
    "primary_material": {
        "source_keys": [
            "material", "primary material", "frame material", "top material",
            "base material", "main material", "construction material",
        ],
        "type": "categorical",
    },
    "secondary_material": {
        "source_keys": [
            "secondary material", "seat material", "cushion material",
            "upholstery material", "fabric", "upholstery",
        ],
        "type": "categorical",
    },
    "color_family": {
        "source_keys": [
            "color", "color family", "primary color", "finish color",
            "finish", "color/finish",
        ],
        "type": "categorical",
    },
    "room_type": {
        "source_keys": [
            "room", "room type", "room use", "suitable room",
            "recommended room", "room recommendation",
        ],
        "type": "multi_label",
    },
    "product_type": {
        "source_keys": ["product type", "type", "category", "subcategory"],
        "type": "categorical",
    },
    "assembly_required": {
        "source_keys": [
            "assembly required", "assembly", "requires assembly",
            "assembly needed",
        ],
        "type": "boolean",
    },
}


def _build_key_lookup() -> dict[str, str]:
    """Build a reverse lookup from source keys to canonical names."""
    lookup = {}
    for canonical, info in ATTRIBUTE_SCHEMA.items():
        for key in info["source_keys"]:
            lookup[key.lower().strip()] = canonical
    return lookup


_KEY_LOOKUP = _build_key_lookup()


def parse_feature_string(feature_str: str) -> dict[str, str]:
    """Parse a raw WANDS feature string into key-value pairs.

    Args:
        feature_str: Raw string like "Color:Beige | Material:Wood"

    Returns:
        Dictionary of raw key-value pairs.
    """
    if not feature_str or pd.isna(feature_str):
        return {}

    pairs = {}
    # Split by pipe delimiter
    parts = feature_str.split("|")
    for part in parts:
        part = part.strip()
        if ":" not in part:
            continue
        # Split on first colon only (values might contain colons)
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            pairs[key] = value

    return pairs


def normalize_attributes(raw_pairs: dict[str, str]) -> dict[str, Optional[str | list[str] | bool]]:
    """Normalize raw feature pairs into canonical attribute schema.

    Args:
        raw_pairs: Dictionary from parse_feature_string.

    Returns:
        Normalized attribute dictionary matching ATTRIBUTE_SCHEMA.
    """
    normalized = {key: None for key in ATTRIBUTE_SCHEMA}

    for raw_key, raw_value in raw_pairs.items():
        canonical = _KEY_LOOKUP.get(raw_key.lower().strip())
        if canonical is None:
            continue  # Skip attributes not in our schema

        attr_type = ATTRIBUTE_SCHEMA[canonical]["type"]

        if attr_type == "boolean":
            normalized[canonical] = _normalize_boolean(raw_value)
        elif attr_type == "multi_label":
            existing = normalized[canonical] or []
            new_values = _normalize_multi_label(raw_value)
            normalized[canonical] = list(set(existing + new_values))
        else:
            # Categorical: take the first non-None value
            if normalized[canonical] is None:
                normalized[canonical] = _normalize_categorical(raw_value)

    return normalized


def _normalize_categorical(value: str) -> str:
    """Normalize a categorical value."""
    value = value.strip().lower()
    # Remove common suffixes/noise
    value = re.sub(r"\s*\(.*?\)\s*", "", value)  # Remove parentheticals
    value = value.strip()
    return value


def _normalize_boolean(value: str) -> Optional[bool]:
    """Normalize a boolean value."""
    value = value.strip().lower()
    if value in ("yes", "true", "1", "required"):
        return True
    elif value in ("no", "false", "0", "not required", "none"):
        return False
    return None


def _normalize_multi_label(value: str) -> list[str]:
    """Normalize a multi-label value."""
    # Split on common delimiters
    items = re.split(r"[,/;&]", value)
    return [item.strip().lower() for item in items if item.strip()]


def process_products_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process a products DataFrame, adding parsed attribute columns.

    Args:
        df: Products DataFrame with 'product_features' column.

    Returns:
        DataFrame with additional columns for each canonical attribute.
    """
    logger.info(f"Parsing features for {len(df)} products...")

    # Parse all features
    parsed = df["product_features"].apply(
        lambda x: normalize_attributes(parse_feature_string(x))
    )

    # Add each attribute as a column
    attrs_df = pd.DataFrame(parsed.tolist(), index=df.index)

    result = pd.concat([df, attrs_df], axis=1)

    # Log coverage
    for attr in ATTRIBUTE_SCHEMA:
        non_null = result[attr].notna().sum()
        pct = non_null / len(result) * 100
        logger.info(f"  {attr}: {non_null} / {len(result)} ({pct:.1f}%)")

    return result


def create_training_example(row: pd.Series) -> dict:
    """Convert a product row into a training example.

    Args:
        row: A row from the processed products DataFrame.

    Returns:
        Training example dictionary with input and target fields.
    """
    # Build the input text
    input_text = f"Product: {row.get('product_name', 'Unknown')}"
    if pd.notna(row.get("product_class")):
        input_text += f"\nCategory: {row['product_class']}"
    if pd.notna(row.get("product_description")):
        desc = str(row["product_description"])[:500]  # Truncate long descriptions
        input_text += f"\nDescription: {desc}"

    # Build the target attributes (ground truth)
    target = {}
    for attr in ATTRIBUTE_SCHEMA:
        value = row.get(attr)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            target[attr] = value

    return {
        "product_id": row.get("product_id"),
        "input_text": input_text,
        "target_attributes": target,
        "product_class": row.get("product_class"),
    }
