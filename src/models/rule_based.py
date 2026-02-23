"""Rule-based / regex baseline for attribute extraction.

The simplest baseline — uses keyword matching and regex patterns.
Establishes the absolute floor for comparison.
"""

from __future__ import annotations

import re
from typing import Optional

from src.models.attribute_extractor import BaseAttributeExtractor


# Keyword dictionaries for each attribute
STYLE_KEYWORDS = {
    "mid-century modern": ["mid-century", "mid century", "mcm", "retro modern"],
    "contemporary": ["contemporary", "modern", "sleek", "minimalist"],
    "traditional": ["traditional", "classic", "timeless", "elegant"],
    "farmhouse": ["farmhouse", "country", "barn", "shiplap", "rustic charm"],
    "industrial": ["industrial", "factory", "loft", "pipe", "raw metal"],
    "scandinavian": ["scandinavian", "nordic", "scandi", "hygge"],
    "bohemian": ["bohemian", "boho", "eclectic", "gypsy", "macrame"],
    "coastal": ["coastal", "beach", "nautical", "ocean", "seaside"],
    "rustic": ["rustic", "reclaimed", "distressed", "weathered", "log"],
    "transitional": ["transitional", "blend", "updated classic"],
}

MATERIAL_KEYWORDS = {
    "solid wood": ["solid wood", "hardwood", "oak", "walnut", "maple", "cherry", "pine", "teak", "mahogany", "birch", "acacia"],
    "engineered wood": ["engineered wood", "mdf", "particle board", "plywood", "laminate", "veneer"],
    "metal": ["metal", "steel", "iron", "aluminum", "brass", "copper", "chrome", "stainless"],
    "glass": ["glass", "tempered glass", "frosted glass", "crystal"],
    "fabric": ["fabric", "polyester", "cotton", "linen", "velvet", "microfiber", "tweed", "chenille"],
    "leather": ["leather", "genuine leather", "faux leather", "bonded leather", "pu leather", "vegan leather"],
    "ceramic": ["ceramic", "porcelain", "stoneware", "terracotta"],
    "plastic": ["plastic", "acrylic", "polypropylene", "resin", "abs"],
    "stone": ["stone", "marble", "granite", "slate", "quartz", "travertine"],
}

COLOR_KEYWORDS = {
    "brown": ["brown", "walnut", "espresso", "chocolate", "mocha", "tan", "chestnut", "umber", "coffee"],
    "gray": ["gray", "grey", "charcoal", "slate", "silver", "ash", "graphite", "pewter"],
    "white": ["white", "ivory", "cream", "off-white", "snow", "pearl", "alabaster"],
    "black": ["black", "ebony", "onyx", "jet", "midnight"],
    "blue": ["blue", "navy", "cobalt", "teal", "aqua", "turquoise", "denim", "sapphire", "indigo"],
    "green": ["green", "olive", "sage", "emerald", "forest", "mint", "hunter"],
    "beige": ["beige", "sand", "khaki", "oatmeal", "linen", "natural", "wheat"],
    "red": ["red", "burgundy", "maroon", "crimson", "ruby", "wine", "scarlet"],
}


class RuleBasedExtractor(BaseAttributeExtractor):
    """Rule-based attribute extractor using keyword matching."""

    def __init__(self):
        super().__init__(model_name="rule-based-regex")
        self._is_loaded = True  # No model to load

    def load(self) -> None:
        """No-op — no model to load."""
        self._is_loaded = True

    def extract(
        self,
        product_name: str,
        product_description: Optional[str] = None,
        product_class: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> dict:
        """Extract attributes using keyword matching.

        Searches through product text for known keywords.
        """
        # Combine all text, lowercase
        text_parts = [product_name or ""]
        if product_description:
            text_parts.append(str(product_description))
        if product_class:
            text_parts.append(str(product_class))
        full_text = " ".join(text_parts).lower()

        return {
            "style": self._match_keywords(full_text, STYLE_KEYWORDS),
            "primary_material": self._match_keywords(full_text, MATERIAL_KEYWORDS),
            "secondary_material": None,  # Too complex for rules
            "color_family": self._match_keywords(full_text, COLOR_KEYWORDS),
            "room_type": self._extract_room_type(full_text),
            "product_type": self._extract_from_class(product_class),
            "assembly_required": self._extract_assembly(full_text),
        }

    @staticmethod
    def _match_keywords(text: str, keyword_dict: dict[str, list[str]]) -> Optional[str]:
        """Find the best matching category based on keyword count."""
        best_match = None
        best_count = 0

        for category, keywords in keyword_dict.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_match = category

        return best_match

    @staticmethod
    def _extract_room_type(text: str) -> Optional[list[str]]:
        """Extract room types from text."""
        rooms = {
            "living room": ["living room", "family room", "lounge", "sitting room"],
            "bedroom": ["bedroom", "master bedroom", "guest room"],
            "dining room": ["dining room", "dining area", "eat-in"],
            "kitchen": ["kitchen", "kitchenette"],
            "bathroom": ["bathroom", "bath", "powder room"],
            "office": ["office", "study", "workspace", "home office"],
            "outdoor": ["outdoor", "patio", "deck", "garden", "balcony"],
            "entryway": ["entryway", "foyer", "mudroom", "hallway"],
        }

        found = []
        for room, keywords in rooms.items():
            if any(kw in text for kw in keywords):
                found.append(room)

        return found if found else None

    @staticmethod
    def _extract_from_class(product_class: Optional[str]) -> Optional[str]:
        """Extract product type from product_class field."""
        if not product_class:
            return None
        # Product class is often a good proxy for product type
        return product_class.strip().lower()

    @staticmethod
    def _extract_assembly(text: str) -> Optional[bool]:
        """Extract assembly requirement from text."""
        if re.search(r"assembly required|requires assembly|some assembly", text):
            return True
        if re.search(r"no assembly|fully assembled|ready to use", text):
            return False
        return None
