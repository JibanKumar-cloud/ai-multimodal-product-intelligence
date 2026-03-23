"""Unified Post-Processing for Classifier, VLM, and Hybrid outputs.

Architecture:
  BasePostProcessor         — shared vocab, normalization, output format
    ├── ClassifierPostProcessor  — confidence scores, taxonomy validation
    ├── VLMPostProcessor         — JSON parsing, key mapping, value normalization
    └── HybridPostProcessor      — merges classifier + VLM by confidence

Usage:
    # Classifier
    cls_pp = ClassifierPostProcessor(taxonomy_path="data/processed/taxonomy_tree.json",
                                     queue_path="data/processed/image_queue_with_images.json")
    result = cls_pp.process(model_output, confidence_threshold=0.5)

    # VLM
    vlm_pp = VLMPostProcessor()
    result = vlm_pp.process(raw_text)

    # Hybrid
    hybrid_pp = HybridPostProcessor(cls_pp, vlm_pp)
    result = hybrid_pp.merge(cls_result, vlm_result, confidence_threshold=0.5)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch


# ════════════════════════════════════════════════════════════════
# ATTRIBUTE FAMILIES
# ════════════════════════════════════════════════════════════════

COLOR_FAMILIES = {
    "white": ["white", "snow", "alabaster", "eggshell"],
    "cream": ["cream", "ivory", "off-white", "pearl", "bone", "ecru"],
    "black": ["black", "jet", "onyx", "ebony"],
    "dark_gray": ["charcoal", "graphite", "slate", "dark gray"],
    "gray": ["gray", "grey", "pewter"],
    "light_gray": ["light gray", "ash", "heather", "smoke"],
    "dark_brown": ["espresso", "chocolate", "cocoa", "coffee", "mahogany",
                   "cherry", "cognac", "amber", "dark brown"],
    "brown": ["brown", "walnut", "chestnut", "sienna", "saddle"],
    "light_brown": ["tan", "khaki", "camel", "fawn", "light brown"],
    "beige": ["beige", "sand", "taupe", "mushroom", "champagne"],
    "natural": ["natural", "oatmeal", "wheat"],
    "navy": ["navy", "midnight blue", "indigo", "dark blue"],
    "blue": ["blue", "royal blue", "cobalt", "sapphire", "denim"],
    "light_blue": ["sky blue", "baby blue", "powder blue", "cerulean"],
    "teal": ["teal", "turquoise", "aqua"],
    "green": ["green", "emerald", "forest", "hunter", "jade"],
    "sage": ["sage", "olive", "moss", "pistachio", "mint", "seafoam"],
    "red": ["red", "scarlet", "crimson", "ruby", "brick", "cranberry"],
    "burgundy": ["burgundy", "maroon", "wine", "oxblood"],
    "pink": ["pink", "blush", "rose", "salmon", "coral", "fuchsia"],
    "orange": ["orange", "tangerine", "peach", "apricot"],
    "rust": ["rust", "terracotta", "burnt orange", "copper"],
    "yellow": ["yellow", "lemon", "canary", "sunflower"],
    "gold": ["gold", "mustard", "honey", "golden"],
    "purple": ["purple", "violet", "eggplant", "amethyst", "plum",
               "lavender", "lilac"],
    "silver": ["silver", "chrome", "platinum", "stainless", "nickel"],
    "gold_metal": ["brass", "bronze", "antique gold", "rose gold"],
    "clear": ["clear", "transparent", "glass", "crystal"],
    "multi": ["multi", "multicolor", "rainbow", "assorted"],
}

MATERIAL_GROUPS = {
    "light_wood": ["oak", "pine", "maple", "birch", "bamboo", "cedar",
                   "rubberwood", "acacia", "beech", "poplar"],
    "dark_wood": ["walnut", "mahogany", "teak", "cherry wood", "rosewood",
                  "ebony wood"],
    "wood": ["wood", "solid wood", "hardwood"],
    "manufactured_wood": ["manufactured wood", "mdf", "particle board",
                          "plywood", "engineered wood", "laminate"],
    "metal": ["metal", "steel", "aluminum", "stainless steel", "chrome", "tin"],
    "iron": ["iron", "wrought iron", "cast iron"],
    "brass_metal": ["brass", "bronze", "copper"],
    "velvet": ["velvet"],
    "linen": ["linen", "cotton", "canvas", "tweed", "burlap"],
    "microfiber": ["polyester", "microfiber", "chenille", "satin", "silk"],
    "leather": ["leather", "genuine leather", "top grain", "full grain"],
    "faux_leather": ["faux leather", "bonded leather", "pu leather",
                     "vegan leather", "leatherette"],
    "plastic": ["plastic", "resin", "acrylic", "polycarbonate", "pvc", "vinyl"],
    "glass": ["glass", "tempered glass", "frosted glass"],
    "ceramic": ["ceramic", "porcelain", "terracotta", "stoneware"],
    "stone": ["marble", "granite", "quartz", "stone", "concrete", "slate"],
    "natural_fiber": ["wool", "jute", "sisal", "rattan", "wicker", "cane",
                      "hemp", "raffia"],
    "foam": ["foam", "memory foam", "gel foam"],
    "synthetics": ["synthetics", "synthetic", "olefin"],
}

STYLE_GROUPS = {
    "modern": ["modern", "contemporary", "minimalist",
               "modern & contemporary", "scandinavian", "transitional"],
    "traditional": ["traditional", "classic", "glam", "glamorous"],
    "mid-century modern": ["mid-century modern", "mid century", "midcentury"],
    "farmhouse": ["farmhouse", "country", "cottage", "farmhouse / country"],
    "rustic": ["rustic", "lodge", "cabin"],
    "industrial": ["industrial", "urban"],
    "coastal": ["coastal", "nautical", "beach", "tropical"],
    "bohemian": ["bohemian", "boho", "eclectic", "global inspired"],
}

SHAPE_GROUPS = {
    "rectangular": ["rectangular", "rectangle"],
    "square": ["square"],
    "round": ["round", "circle", "circular"],
    "oval": ["oval", "oblong"],
    "l-shaped": ["l-shaped", "l-shape", "l shape"],
    "u-shaped": ["u-shaped", "u-shape"],
    "runner": ["runner"],
    "irregular": ["irregular", "novelty", "freeform", "organic"],
    "hexagon": ["hexagon", "hexagonal"],
}

ASSEMBLY_GROUPS = {
    "full": ["full assembly", "full", "yes", "true"],
    "partial": ["partial assembly", "partial", "light"],
    "none": ["none", "no", "no assembly", "false"],
}


# ════════════════════════════════════════════════════════════════
# STANDARD OUTPUT FORMAT
# ════════════════════════════════════════════════════════════════

ATTR_KEYS = [
    "primary_color", "secondary_color",
    "primary_material", "secondary_material",
    "style", "shape", "assembly",
]

TAXONOMY_PREFIX = "level_"

# Map attribute → which family for normalization
ATTR_FAMILY_TYPE = {
    "primary_color": "color",
    "secondary_color": "color",
    "primary_material": "material",
    "secondary_material": "material",
    "style": "style",
    "shape": "shape",
    "assembly": "assembly",
}

ALL_FAMILIES = {
    "color": COLOR_FAMILIES,
    "material": MATERIAL_GROUPS,
    "style": STYLE_GROUPS,
    "shape": SHAPE_GROUPS,
    "assembly": ASSEMBLY_GROUPS,
}


# ════════════════════════════════════════════════════════════════
# BASE POST-PROCESSOR
# ════════════════════════════════════════════════════════════════

class BasePostProcessor:
    """Shared normalization logic for all model outputs."""

    def __init__(self):
        # Build reverse lookup: keyword → group name
        self._reverse_maps = {}
        for family_name, families in ALL_FAMILIES.items():
            mapping = {}
            for group, keywords in families.items():
                mapping[group] = group
                for kw in keywords:
                    mapping[kw.lower()] = group
            self._reverse_maps[family_name] = mapping

    def normalize_value(self, value, family_type):
        """Map raw value to canonical family group.

        Examples:
            normalize_value("espresso", "color") → "dark_brown"
            normalize_value("oak", "material") → "light_wood"
            normalize_value("modern & contemporary", "style") → "modern"
        """
        if not value:
            return None

        mapping = self._reverse_maps.get(family_type, {})
        v = str(value).lower().strip()

        # Direct match
        if v in mapping:
            return mapping[v]

        # Partial match
        for keyword, group in mapping.items():
            if keyword in v:
                return group

        return value  # return as-is if no match

    def empty_result(self):
        """Return a blank standardized result."""
        return {
            "taxonomy": {},
            "product_class": None,
            "attributes": {k: None for k in ATTR_KEYS},
            "sources": {k: None for k in ATTR_KEYS},
            "vlm_needed": [],
            "latency_ms": 0,
        }


# ════════════════════════════════════════════════════════════════
# CLASSIFIER POST-PROCESSOR
# ════════════════════════════════════════════════════════════════

class ClassifierPostProcessor(BasePostProcessor):
    """Process raw classifier output into standardized result.

    Handles:
      - Taxonomy hierarchy validation (stops at invalid children)
      - Confidence scoring and VLM routing
      - Standard output format
    """

    def __init__(self, taxonomy_path=None, queue_path=None):
        super().__init__()
        self.valid_children = {}
        if queue_path and Path(queue_path).exists():
            self._build_taxonomy_hierarchy(queue_path)

    def _build_taxonomy_hierarchy(self, queue_path):
        """Build parent→valid_children map from training data."""
        with open(queue_path) as f:
            queue = json.load(f)

        for p in queue:
            taxonomy = p.get("taxonomy", [])
            for i in range(len(taxonomy) - 1):
                parent_key = (f"level_{i+1}", taxonomy[i])
                child_val = taxonomy[i + 1]
                if parent_key not in self.valid_children:
                    self.valid_children[parent_key] = set()
                self.valid_children[parent_key].add(child_val)

    def process(self, model, model_output, confidence_threshold=0.5):
        """Process classifier forward output into standardized result.

        Args:
            model: ProductClassifier instance (for label lookups)
            model_output: dict from model.forward() with "taxonomy" and "attributes"
            confidence_threshold: below this → flag for VLM

        Returns:
            Standardized result dict
        """
        result = self.empty_result()
        result["sources"] = {}
        vlm_needed = []

        # ── Taxonomy ──
        raw_taxonomy = {}
        for lk in model.taxonomy_heads.level_keys:
            if lk not in model_output["taxonomy"]:
                continue
            head_out = model_output["taxonomy"][lk]
            probs = torch.softmax(head_out["logits"], dim=-1)
            conf, pred = probs.max(-1)
            value = model.taxonomy_heads.level_i2v[lk].get(
                pred[0].item(), "<UNK>")
            c = conf[0].item()
            raw_taxonomy[lk] = {"value": value, "confidence": round(c, 3)}
            if c < confidence_threshold:
                vlm_needed.append(lk)

        # Validate taxonomy chain
        result["taxonomy"] = self._validate_taxonomy_chain(raw_taxonomy)

        # Remove invalid levels from vlm_needed
        valid_levels = set(result["taxonomy"].keys())
        vlm_needed = [f for f in vlm_needed if f in valid_levels]

        # ── Product class ──
        if (model.taxonomy_heads.has_class_head and
                "product_class" in model_output["taxonomy"]):
            head_out = model_output["taxonomy"]["product_class"]
            probs = torch.softmax(head_out["logits"], dim=-1)
            conf, pred = probs.max(-1)
            value = model.taxonomy_heads.class_i2v.get(
                pred[0].item(), "<UNK>")
            c = conf[0].item()
            result["product_class"] = {
                "value": value, "confidence": round(c, 3)}
            if c < confidence_threshold:
                vlm_needed.append("product_class")

        # ── Attributes ──
        for attr, head_out in model_output["attributes"].items():
            probs = torch.softmax(head_out["logits"], dim=-1)
            conf, pred = probs.max(-1)
            value = model.attribute_heads.idx_to_value[attr].get(
                pred[0].item(), "<UNK>")
            c = conf[0].item()
            result["attributes"][attr] = {
                "value": value, "confidence": round(c, 3)}
            result["sources"][attr] = "classifier"
            if c < confidence_threshold:
                vlm_needed.append(attr)

        result["vlm_needed"] = vlm_needed
        return result

    def _validate_taxonomy_chain(self, taxonomy_dict):
        """Validate taxonomy predictions — stop at first broken parent→child link."""
        if not taxonomy_dict or not self.valid_children:
            return taxonomy_dict

        validated = {}
        sorted_levels = sorted(
            ((k, v) for k, v in taxonomy_dict.items()
             if k.startswith("level_")),
            key=lambda x: int(x[0].split("_")[1])
        )

        prev_level = None
        prev_value = None

        for lk, info in sorted_levels:
            value = info["value"]

            if value == "<UNK>":
                break

            # level_1: always accept
            if prev_level is None:
                validated[lk] = info
                prev_level = lk
                prev_value = value
                continue

            # Check valid parent→child
            parent_key = (prev_level, prev_value)
            children = self.valid_children.get(parent_key, set())

            if children and value not in children:
                break  # invalid child

            if not children:
                break  # no known children — taxonomy ends here

            if info.get("confidence", 0) < 0.15:
                break  # random guess

            validated[lk] = info
            prev_level = lk
            prev_value = value

        return validated


# ════════════════════════════════════════════════════════════════
# VLM POST-PROCESSOR
# ════════════════════════════════════════════════════════════════

# Key aliases: map VLM output keys → our standard keys
VLM_KEY_ALIASES = {
    "color_family": "primary_color",
    "color": "primary_color",
    "primary_color": "primary_color",
    "secondary_color": "secondary_color",
    "accent_color": "secondary_color",
    "material": "primary_material",
    "primary_material": "primary_material",
    "secondary_material": "secondary_material",
    "frame_material": "secondary_material",
    "style": "style",
    "design_style": "style",
    "shape": "shape",
    "assembly": "assembly",
    "assembly_required": "assembly",
}


class VLMPostProcessor(BasePostProcessor):
    """Process raw VLM text output into standardized result.

    Handles:
      - JSON and key:value parsing
      - Legacy key mapping (color_family → primary_color)
      - Value normalization (espresso → dark_brown)
    """

    def process(self, raw_output):
        """Parse raw VLM output string into standardized result.

        Args:
            raw_output: raw text from VLM generation

        Returns:
            Standardized result dict (same format as classifier)
        """
        result = self.empty_result()

        if not raw_output or not raw_output.strip():
            return result

        parsed = self._parse_raw(raw_output)
        if not parsed:
            return result

        # Normalize keys and values
        for raw_key, raw_val in parsed.items():
            clean_key = str(raw_key).lower().strip().replace(" ", "_")
            our_key = VLM_KEY_ALIASES.get(clean_key)
            if not our_key or our_key not in ATTR_KEYS:
                continue

            if raw_val is None or str(raw_val).lower() in (
                    "none", "null", "n/a", ""):
                continue

            # Handle booleans
            if isinstance(raw_val, bool):
                if our_key == "assembly":
                    raw_val = "full" if raw_val else "none"
                else:
                    continue

            # Normalize to family
            family_type = ATTR_FAMILY_TYPE.get(our_key)
            if family_type:
                normalized = self.normalize_value(str(raw_val), family_type)
            else:
                normalized = str(raw_val)

            result["attributes"][our_key] = {
                "value": normalized,
                "confidence": None,  # VLM doesn't give confidence
            }
            result["sources"][our_key] = "vlm"

        return result

    def _parse_raw(self, text):
        """Parse raw VLM output — tries JSON first, then key:value."""
        text = text.strip()

        # Truncate at repetitions
        if "\n\n" in text:
            text = text.split("\n\n")[0]
        text = text[:500]

        # Try JSON
        try:
            d = json.loads(text)
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r'\{[^{}]+\}', text)
        if match:
            try:
                d = json.loads(match.group())
                if isinstance(d, dict):
                    return d
            except (json.JSONDecodeError, ValueError):
                pass

        # Try key: value parsing
        parsed = {}
        for line in text.split("\n"):
            for part in re.split(r';\s*', line):
                if ":" not in part:
                    continue
                key, _, val = part.partition(":")
                key = key.strip().lower().replace(" ", "_")
                val = val.strip().strip('"').strip("'")
                if val and val.lower() not in ("none", "null", "n/a", ""):
                    parsed[key] = val

        return parsed if parsed else None


# ════════════════════════════════════════════════════════════════
# HYBRID POST-PROCESSOR
# ════════════════════════════════════════════════════════════════

class HybridPostProcessor:
    """Merge classifier + VLM results by confidence threshold.

    Rules:
      - Taxonomy: always from classifier (VLM doesn't predict taxonomy)
      - High-confidence attributes: from classifier
      - Low-confidence attributes: from VLM if available
    """

    def __init__(self, classifier_pp: ClassifierPostProcessor,
                 vlm_pp: VLMPostProcessor):
        self.classifier_pp = classifier_pp
        self.vlm_pp = vlm_pp

    def get_vlm_attrs(self, cls_result):
        """Get list of attribute names that need VLM fallback.

        Only attributes — never taxonomy or product_class.
        """
        return [f for f in cls_result.get("vlm_needed", [])
                if f in ATTR_KEYS]

    def merge(self, cls_result, vlm_result):
        """Merge classifier and VLM results.

        Args:
            cls_result: output from ClassifierPostProcessor.process()
            vlm_result: output from VLMPostProcessor.process()

        Returns:
            Merged result with source annotations
        """
        merged = {
            "taxonomy": cls_result.get("taxonomy", {}),
            "product_class": cls_result.get("product_class"),
            "attributes": {},
            "sources": {},
            "vlm_needed": cls_result.get("vlm_needed", []),
        }

        vlm_attrs = self.get_vlm_attrs(cls_result)

        for attr in ATTR_KEYS:
            cls_info = cls_result.get("attributes", {}).get(attr)
            vlm_info = vlm_result.get("attributes", {}).get(attr)

            if attr in vlm_attrs and vlm_info and vlm_info.get("value"):
                # Low confidence — use VLM
                merged["attributes"][attr] = vlm_info
                merged["sources"][attr] = "vlm"
            elif cls_info:
                # High confidence — use classifier
                merged["attributes"][attr] = cls_info
                merged["sources"][attr] = "classifier"
            else:
                merged["attributes"][attr] = {"value": None, "confidence": None}
                merged["sources"][attr] = None

        return merged