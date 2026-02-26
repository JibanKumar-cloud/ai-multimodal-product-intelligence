"""VLM Output Normalizer.

Converts VLM (LLaVA) free-text output into the same ProductResult format
as the classifier. One system, one output — user never knows which model ran.

VLM output keys → Classifier output keys:
  color_family      → color
  primary_material  → material
  style             → style
  room_type         → room
  assembly_required → assembly
  product_type      → used for taxonomy inference

Also infers taxonomy from VLM output + classifier's partial prediction.
"""
from typing import Optional
from loguru import logger


# Map VLM output keys to clean classifier output keys
VLM_TO_CLASSIFIER_KEYS = {
    "color_family": "color",
    "primary_material": "material",
    "secondary_material": "secondary_material",
    "style": "style",
    "room_type": "room",
    "product_type": "type",
    "assembly_required": "assembly",
}

# Map VLM string values to classifier vocabulary
COLOR_NORMALIZE = {
    "grey": "gray", "off-white": "cream", "ivory": "cream",
    "tan": "beige", "espresso": "brown", "walnut": "brown",
    "chocolate": "brown", "navy": "navy", "charcoal": "gray",
    "slate": "gray", "graphite": "gray", "silver": "silver",
    "champagne": "gold", "bronze": "gold",
}

MATERIAL_NORMALIZE = {
    "faux leather": "leather", "pu leather": "leather",
    "bonded leather": "leather", "solid wood": "wood",
    "engineered wood": "wood", "mdf": "wood",
    "plywood": "wood", "particle board": "wood",
    "stainless steel": "steel", "wrought iron": "iron",
    "cast iron": "iron", "cast aluminum": "aluminum",
    "microfiber": "polyester", "chenille": "fabric",
    "tweed": "fabric", "canvas": "cotton",
}

STYLE_NORMALIZE = {
    "mid century": "mid-century modern",
    "mid-century": "mid-century modern",
    "mcm": "mid-century modern",
    "scandi": "scandinavian",
    "boho": "bohemian",
    "country": "farmhouse",
    "modern farmhouse": "farmhouse",
    "retro": "mid-century modern",
    "minimalist": "modern",
    "classic": "traditional",
}


def normalize_vlm_output(vlm_raw: dict,
                         classifier_taxonomy: list = None,
                         classifier_attrs: dict = None) -> dict:
    """Convert VLM output to match classifier's ProductResult format.

    Args:
        vlm_raw: raw VLM output dict (from PostProcessor)
            e.g. {"style": "modern", "color_family": "blue", ...}
        classifier_taxonomy: partial taxonomy from classifier (if available)
            e.g. ["Furniture", "Living Room", "Chairs", "Accent Chairs"]
        classifier_attrs: partial attributes from classifier (if available)

    Returns:
        dict matching ProductResult format:
        {
            "taxonomy": [...],
            "attributes": {"color": "blue", "material": "velvet", ...},
            "confidence": 0.85,
        }
    """
    if not vlm_raw:
        return {
            "taxonomy": classifier_taxonomy or [],
            "attributes": classifier_attrs or {},
            "confidence": 0.5,
        }

    # ── Normalize attributes ──
    attributes = {}

    for vlm_key, clean_key in VLM_TO_CLASSIFIER_KEYS.items():
        value = vlm_raw.get(vlm_key)
        if value is None:
            continue

        # Normalize value
        if isinstance(value, bool):
            value = "yes" if value else "no"
        elif isinstance(value, list):
            value = ", ".join(str(v).lower().strip() for v in value if v)
        else:
            value = str(value).lower().strip()

        # Apply vocabulary normalization
        if clean_key == "color":
            value = COLOR_NORMALIZE.get(value, value)
        elif clean_key == "material":
            value = MATERIAL_NORMALIZE.get(value, value)
        elif clean_key == "style":
            value = STYLE_NORMALIZE.get(value, value)

        # Skip empty/none values
        if value and value not in ("none", "null", "n/a", "unknown", ""):
            attributes[clean_key] = value

    # ── Merge with classifier's partial predictions ──
    # Classifier ran first but was uncertain. Use its predictions
    # for attributes VLM didn't produce.
    if classifier_attrs:
        for key, value in classifier_attrs.items():
            if key not in attributes and value:
                attributes[key] = value

    # ── Infer taxonomy ──
    # Strategy: use classifier's taxonomy (it ran first)
    # VLM doesn't directly output hierarchy, but product_type can help
    taxonomy = classifier_taxonomy or []

    product_type = vlm_raw.get("product_type", "")
    if product_type and not taxonomy:
        # If classifier gave no taxonomy, use product_type as leaf
        taxonomy = [str(product_type).title()]

    # ── Confidence ──
    # VLM doesn't give numeric confidence, estimate from completeness
    n_attrs = len(attributes)
    if n_attrs >= 5:
        confidence = 0.90
    elif n_attrs >= 3:
        confidence = 0.80
    elif n_attrs >= 1:
        confidence = 0.70
    else:
        confidence = 0.50

    return {
        "taxonomy": taxonomy,
        "attributes": attributes,
        "confidence": confidence,
    }
