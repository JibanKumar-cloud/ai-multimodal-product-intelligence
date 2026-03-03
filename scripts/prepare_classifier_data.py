"""Prepare WANDS data for classifier training + image sourcing.

V4: Claude API integration + fine-grained color/material
  - If --api-key provided: Claude extracts attributes (best quality)
  - Otherwise: rule-based extraction (free, instant, ~90% accurate)
  - Fine-grained colors (~28) and materials (~18)
  - text_input field restored for training
  - Shape blacklist, smart upholstery detection

Usage:
    # Rule-based (free)
    python scripts/prepare_classifier_data.py

    # Claude-enhanced (~$3 for 10k products)
    python scripts/prepare_classifier_data.py --api-key sk-ant-...

    # Resume interrupted Claude run
    python scripts/prepare_classifier_data.py --api-key sk-ant-... --resume
"""
import pandas as pd
import json
import re
import os
import sys
import time
import argparse
from collections import defaultdict, Counter
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# COLOR FAMILIES (~28 visually distinct groups)
# ════════════════════════════════════════════════════════════════

COLOR_FAMILIES = {
    "white": ["white", "snow", "alabaster", "eggshell"],
    "cream": ["cream", "ivory", "off-white", "off white",
              "linen", "pearl", "bone", "ecru"],
    "black": ["black", "jet", "onyx", "ebony"],
    "dark_gray": ["charcoal", "graphite", "slate", "dark gray", "dark grey"],
    "gray": ["gray", "grey", "pewter"],
    "light_gray": ["light gray", "light grey", "ash", "heather",
                   "smoke", "silver gray"],
    "dark_brown": ["espresso", "chocolate", "cocoa", "coffee",
                   "umber", "dark brown"],
    "brown": ["brown", "walnut", "chestnut", "sienna",
              "saddle", "russet"],
    "light_brown": ["tan", "khaki", "camel", "fawn", "light brown"],
    "mahogany": ["mahogany", "cherry", "auburn"],
    "cognac": ["cognac", "amber", "cinnamon"],
    "beige": ["beige", "sand", "taupe", "mushroom", "parchment",
              "champagne", "buff", "latte"],
    "natural": ["natural", "oatmeal", "wheat"],
    "navy": ["navy", "midnight blue", "indigo", "dark blue"],
    "blue": ["blue", "royal blue", "cobalt", "sapphire",
             "denim", "cornflower"],
    "light_blue": ["sky blue", "baby blue", "powder blue",
                   "cerulean", "periwinkle", "light blue"],
    "teal": ["teal", "turquoise", "aqua"],
    "green": ["green", "emerald", "forest", "hunter",
              "kelly", "jade", "fern"],
    "sage": ["sage", "olive", "moss", "pistachio", "mint",
             "seafoam", "light green"],
    "red": ["red", "scarlet", "crimson", "ruby", "brick",
            "cranberry", "garnet"],
    "burgundy": ["burgundy", "maroon", "wine", "oxblood"],
    "pink": ["pink", "blush", "rose", "salmon",
             "dusty rose", "hot pink", "raspberry"],
    "coral": ["coral", "fuchsia", "magenta", "mauve"],
    "orange": ["orange", "tangerine", "peach", "apricot"],
    "rust": ["rust", "terracotta", "burnt orange", "copper"],
    "yellow": ["yellow", "lemon", "canary", "sunflower"],
    "gold": ["gold", "mustard", "honey", "golden"],
    "purple": ["purple", "violet", "eggplant", "amethyst",
               "grape", "plum"],
    "lavender": ["lavender", "lilac", "orchid"],
    "silver": ["silver", "chrome", "platinum", "metallic silver",
               "brushed silver", "stainless", "nickel"],
    "gold_metal": ["brass", "bronze", "antique gold",
                   "rose gold", "brushed gold"],
    "clear": ["clear", "transparent", "glass", "crystal"],
    "multi": ["multi", "multicolor", "multicolored",
              "rainbow", "assorted", "various"],
}

# ════════════════════════════════════════════════════════════════
# MATERIAL GROUPS (~18 visually distinct groups)
# ════════════════════════════════════════════════════════════════

MATERIAL_GROUPS = {
    "light_wood": ["oak", "pine", "maple", "birch", "bamboo",
                   "cedar", "rubberwood", "acacia", "ash wood",
                   "beech", "poplar", "light wood"],
    "dark_wood": ["walnut", "mahogany", "teak", "cherry wood",
                  "rosewood", "ebony wood", "dark wood",
                  "espresso wood"],
    "wood": ["wood", "solid wood", "hardwood",
             "solid + manufactured wood",
             "manufactured wood + solid wood"],
    "manufactured_wood": ["manufactured wood", "mdf",
                          "particle board", "plywood",
                          "engineered wood", "laminate",
                          "wood veneer"],
    "metal": ["metal", "steel", "aluminum", "stainless steel",
              "zinc", "chrome", "tin"],
    "iron": ["iron", "wrought iron", "cast iron"],
    "brass_metal": ["brass", "bronze", "copper"],
    "velvet": ["velvet"],
    "linen": ["linen", "cotton", "100 % cotton", "canvas",
              "tweed", "burlap", "muslin"],
    "microfiber": ["polyester", "microfiber / polyester",
                   "microfiber", "chenille", "satin", "silk",
                   "nylon fabric"],
    "leather": ["leather", "genuine leather", "top grain",
                "full grain"],
    "faux_leather": ["faux leather", "bonded leather",
                     "pu leather", "vegan leather",
                     "leatherette", "faux leather/leatherette"],
    "plastic": ["plastic", "resin/plastic", "resin",
                "acrylic", "polycarbonate", "abs",
                "polypropylene", "pvc", "vinyl"],
    "glass": ["glass", "tempered glass", "frosted glass",
              "mirror glass", "crystal glass"],
    "ceramic": ["ceramic", "porcelain", "terracotta",
                "earthenware", "stoneware"],
    "stone": ["marble", "granite", "quartz", "stone",
              "concrete", "slate", "travertine",
              "limestone", "terrazzo"],
    "natural_fiber": ["wool", "jute", "sisal", "seagrass",
                      "rattan", "wicker", "cane", "hemp", "raffia"],
    "foam": ["foam", "memory foam", "gel foam",
             "polyurethane foam"],
    "synthetics": ["synthetics", "synthetic", "olefin",
                   "polypropylene fiber"],
}

SURFACE_MATERIALS = {"velvet", "linen", "microfiber", "leather",
                     "faux_leather", "natural_fiber", "synthetics"}
FRAME_MATERIALS = {"wood", "light_wood", "dark_wood",
                   "manufactured_wood", "metal", "iron",
                   "brass_metal", "plastic", "foam"}

# ════════════════════════════════════════════════════════════════
# SHAPE, STYLE, ASSEMBLY
# ════════════════════════════════════════════════════════════════

SHAPE_GROUPS = {
    "rectangular": ["rectangular", "rectangle", "rect"],
    "square": ["square"],
    "round": ["round", "circle", "circular"],
    "oval": ["oval", "round/oval", "oblong"],
    "l-shaped": ["l-shaped", "l-shape", "l shape"],
    "u-shaped": ["u-shaped", "u-shape", "u shape"],
    "runner": ["runner"],
    "irregular": ["irregular", "novelty", "freeform", "organic"],
    "hexagon": ["hexagon", "hexagonal"],
}

SHAPE_INVALID_KEYWORDS = {
    "chair", "sofa", "couch", "loveseat", "sectional",
    "recliner", "futon", "settee", "chaise", "rocker",
    "glider", "stool", "bench",
    "bed", "headboard", "crib", "bassinet",
    "dresser", "wardrobe", "armoire", "nightstand", "chest",
    "bookcase", "cabinet", "vanity", "hutch", "sideboard",
    "buffet", "credenza",
    "lamp", "chandelier", "sconce", "pendant", "lantern",
    "faucet", "toilet", "sink", "shower",
    "fan", "heater", "air conditioner",
    "curtain", "drape", "blind", "shade",
}

ASSEMBLY_GROUPS = {
    "full": ["full assembly needed", "full assembly", "full", "yes"],
    "partial": ["partial assembly", "partial", "light"],
    "none": ["none", "no", "no assembly"],
}

STYLE_GROUPS = {
    "modern": ["modern & contemporary", "modern", "contemporary",
               "minimalist"],
    "traditional": ["traditional", "classic"],
    "mid-century modern": ["mid-century modern", "mid century",
                           "midcentury"],
    "farmhouse": ["farmhouse / country", "farmhouse",
                  "country/cottage", "cottage / country",
                  "modern farmhouse", "country"],
    "rustic": ["rustic", "lodge", "cabin"],
    "industrial": ["industrial", "urban"],
    "coastal": ["coastal", "nautical", "beach", "tropical"],
    "glam": ["glam", "glamorous", "hollywood regency"],
    "bohemian": ["bohemian", "boho", "eclectic",
                 "global inspired"],
    "transitional": ["transitional"],
    "scandinavian": ["scandinavian", "nordic"],
}

UPHOLSTERED_CLASSES = {
    "Accent Chairs", "Arm Chairs", "Armchairs", "Club Chairs",
    "Barrel Chairs", "Wingback Chairs", "Parsons Chairs",
    "Recliners", "Rocking Chairs", "Swivel Chairs",
    "Dining Chairs", "Side Chairs",
    "Sofas", "Couches", "Loveseats", "Sectional Sofas",
    "Sleeper Sofas", "Futons", "Settees", "Chaise Lounges",
    "Ottomans", "Poufs", "Benches",
    "Headboards", "Upholstered Beds",
    "Bar Stools", "Counter Stools",
}


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _normalize_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f" {s} "


def _build_keyword_table(families):
    table = []
    for label, keywords in families.items():
        for kw in keywords:
            kw_norm = _normalize_text(kw).strip()
            if kw_norm:
                table.append((kw_norm, label))
    table.sort(key=lambda x: len(x[0]), reverse=True)
    return table


_COLOR_TABLE = _build_keyword_table(COLOR_FAMILIES)
_MATERIAL_TABLE = _build_keyword_table(MATERIAL_GROUPS)
_STYLE_TABLE = _build_keyword_table(STYLE_GROUPS)
_SHAPE_TABLE = _build_keyword_table(SHAPE_GROUPS)
_ASSEMBLY_TABLE = _build_keyword_table(ASSEMBLY_GROUPS)


def normalize_value(raw, group_map):
    if not raw:
        return None
    raw_lower = raw.lower().strip()
    for group, keywords in group_map.items():
        for kw in keywords:
            if raw_lower == kw:
                return group
    for group, keywords in group_map.items():
        for kw in sorted(keywords, key=len, reverse=True):
            if kw in raw_lower:
                return group
    return "other"


def infer_from_text(text, table):
    t = _normalize_text(text)
    hits = []
    seen = set()
    for kw, label in table:
        if kw and kw in t and label not in seen:
            hits.append(label)
            seen.add(label)
    return hits


def parse_features_multi(features_str):
    if pd.isna(features_str) or not features_str:
        return {}
    attrs = defaultdict(list)
    for pair in str(features_str).split("|"):
        pair = pair.strip()
        if ":" in pair:
            key, val = pair.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            if val and val.lower() not in ("", "nan", "none"):
                attrs[key].append(val)
    return dict(attrs)


def parse_hierarchy(hierarchy_str):
    if pd.isna(hierarchy_str) or not hierarchy_str:
        return []
    h = str(hierarchy_str).strip().strip('"')
    return [l.strip() for l in h.split("/") if l.strip()]


def build_text_input(product_name, product_class, description):
    """Build text_input for training: name [SEP] class [SEP] description."""
    parts = [str(product_name).strip()]
    if product_class and str(product_class).strip() and str(product_class) != "nan":
        parts.append(str(product_class).strip())
    if description and str(description).strip() and str(description) != "nan":
        # Truncate description to ~200 chars
        desc = str(description).strip()
        if len(desc) > 200:
            desc = desc[:200].rsplit(" ", 1)[0] + "..."
        parts.append(desc)
    return " [SEP] ".join(parts)


# ════════════════════════════════════════════════════════════════
# RULE-BASED ATTRIBUTE EXTRACTION (V3 logic)
# ════════════════════════════════════════════════════════════════

SAFE_NAME_COLORS = {
    "white", "black", "gray", "grey", "brown", "blue",
    "green", "red", "pink", "orange", "yellow",
    "purple", "beige", "navy", "teal", "burgundy",
    "silver", "charcoal", "cream", "ivory", "tan",
    "gold", "espresso", "walnut", "mahogany",
    "coral", "lavender", "sage", "olive", "rust",
    "turquoise", "indigo", "slate",
}


def extract_color_from_name(product_name):
    name_lower = product_name.lower()
    in_match = re.search(r'\bin\s+(\w+)\s*$', name_lower.strip())
    if in_match:
        word = in_match.group(1)
        color = normalize_value(word, COLOR_FAMILIES)
        if color and color != "other":
            return [color]
    in_match2 = re.search(r'\bin\s+(\w+\s+\w+)\s*$', name_lower.strip())
    if in_match2:
        phrase = in_match2.group(1)
        color = normalize_value(phrase, COLOR_FAMILIES)
        if color and color != "other":
            return [color]
    slash_match = re.findall(
        r'(?:dark\s+|light\s+)?(\w+)\s*/\s*(?:dark\s+|light\s+)?(\w+)',
        name_lower)
    if slash_match:
        colors = []
        for c1, c2 in slash_match:
            n1 = normalize_value(c1, COLOR_FAMILIES)
            n2 = normalize_value(c2, COLOR_FAMILIES)
            if n1 and n1 != "other" and n1 not in colors:
                colors.append(n1)
            if n2 and n2 != "other" and n2 != n1 and n2 not in colors:
                colors.append(n2)
        if colors:
            return colors
    found = []
    for word in name_lower.split():
        if word in SAFE_NAME_COLORS:
            color = normalize_value(word, COLOR_FAMILIES)
            if color and color != "other" and color not in found:
                found.append(color)
    return found


def extract_colors_smart(features_dict, product_name, product_class,
                         taxonomy_text="", description=""):
    name_colors = extract_color_from_name(product_name)
    feat_colors = []
    for key in ["color", "basecolor", "finishcolor"]:
        vals = features_dict.get(key, [])
        if not isinstance(vals, list):
            vals = [vals]
        for v in vals:
            norm = normalize_value(v, COLOR_FAMILIES)
            if norm and norm != "other" and norm not in feat_colors:
                feat_colors.append(norm)
    for v in features_dict.get("dswoodtone", []):
        norm = normalize_value(v, COLOR_FAMILIES)
        if norm and norm != "other" and norm not in feat_colors:
            feat_colors.append(norm)
    tax_colors = infer_from_text(taxonomy_text, _COLOR_TABLE) if taxonomy_text else []
    desc_colors = infer_from_text(description, _COLOR_TABLE) if description else []

    if name_colors:
        primary = name_colors[0]
    elif feat_colors:
        primary = feat_colors[0]
    elif tax_colors:
        primary = tax_colors[0]
    elif desc_colors:
        primary = desc_colors[0]
    else:
        return None, None

    all_colors = []
    for c in name_colors + feat_colors + tax_colors:
        if c not in all_colors:
            all_colors.append(c)
    secondary = None
    for c in all_colors:
        if c != primary:
            secondary = c
            break
    if name_colors and feat_colors and feat_colors[0] != primary:
        if secondary is None:
            secondary = feat_colors[0]
    return primary, secondary


def is_upholstered(product_name, product_class, features_text):
    combined = f"{product_name} {product_class} {features_text}".lower()
    if product_class in UPHOLSTERED_CLASSES:
        return True
    signals = ["upholster", "tufted", "cushion", "padded",
               "velvet", "linen seat", "chenille", "microfiber",
               "fabric seat", "leather seat", "faux leather"]
    return any(kw in combined for kw in signals)


def extract_materials_smart(features_dict, product_name, product_class,
                            description=""):
    features_text = " ".join(
        " ".join(v) if isinstance(v, list) else str(v)
        for v in features_dict.values())
    upholstered = is_upholstered(product_name, product_class, features_text)

    all_materials = []
    seen = set()
    for key in ["primarymaterial", "material", "framematerial",
                "topmaterial", "seatmaterial", "fabricdetails",
                "upholsterymaterial", "additionalframematerialdetails",
                "legmaterial"]:
        vals = features_dict.get(key, [])
        if not isinstance(vals, list):
            vals = [vals]
        for v in vals:
            norm = normalize_value(v, MATERIAL_GROUPS)
            if norm and norm != "other" and norm not in seen:
                all_materials.append((key, norm))
                seen.add(norm)
    for m in infer_from_text(features_text, _MATERIAL_TABLE):
        if m not in seen:
            all_materials.append(("text_infer", m))
            seen.add(m)
    if not all_materials:
        for m in infer_from_text(f"{product_name} {description}",
                                  _MATERIAL_TABLE):
            if m not in seen:
                all_materials.append(("name_desc", m))
                seen.add(m)
    if not all_materials:
        return None, None

    if upholstered:
        surface = [m for _, m in all_materials if m in SURFACE_MATERIALS]
        frame = [m for _, m in all_materials if m in FRAME_MATERIALS]
        if not surface:
            name_lower = f"{product_name} {product_class}".lower()
            if "leather" in name_lower:
                surface = ["leather"]
            elif "faux leather" in name_lower:
                surface = ["faux_leather"]
            elif "velvet" in name_lower:
                surface = ["velvet"]
            elif "linen" in name_lower:
                surface = ["linen"]
            elif any(w in name_lower for w in
                     ["upholster", "tufted", "cushion", "padded", "fabric"]):
                surface = ["microfiber"]
        if surface:
            primary = surface[0]
            secondary = frame[0] if frame else None
        else:
            primary = all_materials[0][1]
            secondary = all_materials[1][1] if len(all_materials) > 1 else None
    else:
        primary = all_materials[0][1]
        secondary = all_materials[1][1] if len(all_materials) > 1 else None
    if secondary == primary:
        secondary = all_materials[2][1] if len(all_materials) > 2 else None
    return primary, secondary


def extract_shape_validated(features_dict, product_class, product_name):
    combined = f"{(product_class or '').lower()} {(product_name or '').lower()}"
    if any(kw in combined for kw in SHAPE_INVALID_KEYWORDS):
        return None
    for key in ["shape", "headboardshape", "topshape",
                "mirrorshape", "rugshape", "tableshape"]:
        vals = features_dict.get(key, [])
        if not isinstance(vals, list):
            vals = [vals]
        for v in vals:
            norm = normalize_value(v, SHAPE_GROUPS)
            if norm and norm != "other":
                return norm
    return None


def extract_assembly(features_dict):
    for key in ["levelofassembly", "adultassemblyrequired"]:
        vals = features_dict.get(key, [])
        if not isinstance(vals, list):
            vals = [vals]
        for v in vals:
            norm = normalize_value(v, ASSEMBLY_GROUPS)
            if norm and norm != "other":
                return norm
    return None


def extract_style(features_dict, taxonomy_text="", description=""):
    for key in ["style", "dsprimaryproductstyle"]:
        vals = features_dict.get(key, [])
        if not isinstance(vals, list):
            vals = [vals]
        for v in vals:
            norm = normalize_value(v, STYLE_GROUPS)
            if norm and norm != "other":
                return norm
    if taxonomy_text:
        hits = infer_from_text(taxonomy_text, _STYLE_TABLE)
        if hits:
            return hits[0]
    if description:
        hits = infer_from_text(description, _STYLE_TABLE)
        if hits:
            return hits[0]
    return None


def rule_based_extract(row, desc_col):
    """Extract all attributes using rule-based logic. Returns dict."""
    features = parse_features_multi(row.get("product_features", ""))
    pname = str(row.get("product_name", ""))
    pclass = str(row.get("product_class", ""))
    tax_text = str(row.get("taxonomy_full", ""))
    desc = str(row.get(desc_col, ""))
    if desc == "nan":
        desc = ""

    pc, sc = extract_colors_smart(features, pname, pclass, tax_text, desc)
    pm, sm = extract_materials_smart(features, pname, pclass, desc)
    shape = extract_shape_validated(features, pclass, pname)
    assembly = extract_assembly(features)
    style = extract_style(features, tax_text, desc)

    return {
        "primary_color": pc,
        "secondary_color": sc,
        "primary_material": pm,
        "secondary_material": sm,
        "style": style,
        "shape": shape,
        "assembly": assembly,
    }


# ════════════════════════════════════════════════════════════════
# CLAUDE API ATTRIBUTE EXTRACTION
# ════════════════════════════════════════════════════════════════

# Valid values for Claude to return (validated against our taxonomy)
VALID_COLORS = sorted(COLOR_FAMILIES.keys())
VALID_MATERIALS = sorted(MATERIAL_GROUPS.keys())
VALID_STYLES = sorted(STYLE_GROUPS.keys())
VALID_SHAPES = sorted(SHAPE_GROUPS.keys())
VALID_ASSEMBLIES = sorted(ASSEMBLY_GROUPS.keys())

CLAUDE_SYSTEM_PROMPT = f"""You are a product attribute extractor for a furniture/home goods catalog.

Given product information, return a JSON object with these exact fields:
- primary_color: the DOMINANT visible color (what a camera sees)
- secondary_color: a secondary accent color, or null
- primary_material: the DOMINANT visible material (what a camera sees, not internal frame)
- secondary_material: internal/frame material, or null
- style: design style
- shape: visual shape (only for rugs, tables, mirrors, trays, pillows — null for chairs, sofas, beds, cabinets, lamps)
- assembly: assembly level required

VALID VALUES:
  primary_color / secondary_color: {json.dumps(VALID_COLORS)}
  primary_material / secondary_material: {json.dumps(VALID_MATERIALS)}
  style: {json.dumps(VALID_STYLES)}
  shape: {json.dumps(VALID_SHAPES)}
  assembly: {json.dumps(VALID_ASSEMBLIES)}

CRITICAL RULES:
1. For upholstered furniture (chairs, sofas), primary_material = the FABRIC/LEATHER surface, NOT the wood frame inside
2. For wood furniture (tables, dressers), primary_material = the visible wood type
3. "espresso" color → dark_brown, "walnut" color → brown, "charcoal" → dark_gray
4. If product name says a color (e.g., "chair in gray"), that color WINS over feature data
5. Shape is null for chairs, sofas, beds, cabinets, lamps, faucets, etc.
6. Use null (not "other") when you cannot determine an attribute
7. Return ONLY valid values from the lists above, or null

Return ONLY the JSON object, no explanation, no markdown fences."""


def claude_extract_batch(client, products_batch, model="claude-haiku-4-5-20251001"):
    """Call Claude API to extract attributes for a batch of products.

    Sends up to 5 products per call to reduce API overhead.
    Returns list of attribute dicts.
    """
    # Build product descriptions for the batch
    product_texts = []
    for p in products_batch:
        features_str = p.get("product_features", "")
        # Truncate features to key fields
        features = parse_features_multi(features_str)
        key_features = {}
        for k in ["color", "basecolor", "primarymaterial", "framematerial",
                   "material", "style", "shape", "levelofassembly",
                   "seatmaterial", "upholsterymaterial", "fabricdetails",
                   "dswoodtone", "finishcolor", "topmaterial"]:
            if k in features:
                key_features[k] = features[k]

        desc = str(p.get("product_description", ""))
        if desc == "nan":
            desc = ""
        if len(desc) > 300:
            desc = desc[:300] + "..."

        product_texts.append(
            f"Product ID: {p['product_id']}\n"
            f"Name: {p['product_name']}\n"
            f"Class: {p.get('product_class', '')}\n"
            f"Category: {p.get('taxonomy_full', '')}\n"
            f"Key features: {json.dumps(key_features)}\n"
            f"Description: {desc}"
        )

    if len(products_batch) == 1:
        user_msg = product_texts[0] + "\n\nReturn the JSON object."
    else:
        user_msg = (
            f"Extract attributes for these {len(products_batch)} products. "
            f"Return a JSON ARRAY of {len(products_batch)} objects, one per product, "
            f"in the same order.\n\n"
            + "\n---\n".join(product_texts)
        )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            system=CLAUDE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)

        parsed = json.loads(text)

        # Normalize to list
        if isinstance(parsed, dict):
            parsed = [parsed]

        # Validate each result
        results = []
        for item in parsed:
            validated = {}
            for key in ["primary_color", "secondary_color"]:
                v = item.get(key)
                if v and v in VALID_COLORS:
                    validated[key] = v
                else:
                    validated[key] = None
            for key in ["primary_material", "secondary_material"]:
                v = item.get(key)
                if v and v in VALID_MATERIALS:
                    validated[key] = v
                else:
                    validated[key] = None
            v = item.get("style")
            validated["style"] = v if v and v in VALID_STYLES else None
            v = item.get("shape")
            validated["shape"] = v if v and v in VALID_SHAPES else None
            v = item.get("assembly")
            validated["assembly"] = v if v and v in VALID_ASSEMBLIES else None
            results.append(validated)

        # Pad if Claude returned fewer items
        while len(results) < len(products_batch):
            results.append({k: None for k in [
                "primary_color", "secondary_color",
                "primary_material", "secondary_material",
                "style", "shape", "assembly"]})

        return results

    except Exception as e:
        print(f"    ⚠ Claude API error: {e}")
        return [{k: None for k in [
            "primary_color", "secondary_color",
            "primary_material", "secondary_material",
            "style", "shape", "assembly"]} for _ in products_batch]


def claude_extract_all(df, api_key, desc_col, batch_size=5,
                       progress_path=None, model="claude-haiku-4-5-20251001"):
    """Extract attributes for all products using Claude API.

    Resume-safe: saves progress to progress_path after each batch.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # Load progress if resuming
    progress = {}
    if progress_path and os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        print(f"  Resuming: {len(progress)} products already done")

    # Initialize columns
    for col in ["primary_color", "secondary_color",
                "primary_material", "secondary_material",
                "style", "shape", "assembly"]:
        df[col] = None

    # Apply already-completed results
    for pid, attrs in progress.items():
        mask = df["product_id"].astype(str) == pid
        if mask.any():
            idx = df[mask].index[0]
            for k, v in attrs.items():
                df.at[idx, k] = v

    # Find remaining products
    remaining = []
    for idx, row in df.iterrows():
        pid = str(row["product_id"])
        if pid not in progress:
            remaining.append((idx, row))

    total = len(remaining)
    if total == 0:
        print("  All products already processed!")
        return

    # Estimate cost
    est_calls = (total + batch_size - 1) // batch_size
    est_cost = total * 0.0003  # rough Haiku estimate
    print(f"\n  Products remaining: {total}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Est. API calls:     {est_calls}")
    print(f"  Est. cost:          ~${est_cost:.2f}")
    print(f"  Est. time:          ~{est_calls * 1.5 / 60:.0f} min")
    print(f"  Model:              {model}")
    print()

    done = 0
    errors = 0
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch_items = remaining[batch_start:batch_start + batch_size]
        batch_rows = [row.to_dict() for _, row in batch_items]

        results = claude_extract_batch(client, batch_rows, model=model)

        for i, (idx, row) in enumerate(batch_items):
            pid = str(row["product_id"])
            attrs = results[i] if i < len(results) else {}

            for k, v in attrs.items():
                df.at[idx, k] = v

            progress[pid] = attrs
            if any(v is None for v in attrs.values()):
                errors += 1

        done += len(batch_items)

        # Save progress
        if progress_path and done % (batch_size * 5) == 0:
            with open(progress_path, "w") as f:
                json.dump(progress, f)

        # Rate limit
        time.sleep(0.5)

        # Progress report
        if done % 50 == 0 or done == total:
            elapsed = time.time() - start_time
            rate = done / max(elapsed, 1) * 3600
            eta = (total - done) / max(rate / 3600, 0.001)
            print(f"  [{done}/{total}] {rate:.0f}/hr, "
                  f"ETA {eta:.1f}min, errors={errors}")

    # Final save
    if progress_path:
        with open(progress_path, "w") as f:
            json.dump(progress, f)
        print(f"  Progress saved to {progress_path}")

    print(f"  Claude extraction complete: {done} products, {errors} partial errors")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prepare WANDS data for classifier training")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Anthropic API key for Claude attribute extraction")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                        help="Claude model to use (default: haiku)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Products per Claude API call")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted Claude extraction")
    args = parser.parse_args()

    products_csv = "data/raw/WANDS/dataset/product.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(products_csv, sep="\t")
    mode = "CLAUDE API" if args.api_key else "RULE-BASED"
    print(f"{'='*70}")
    print(f"WANDS DATA PREPARATION FOR CLASSIFIER (V4)")
    print(f"  Mode: {mode}")
    if args.api_key:
        print(f"  Model: {args.model}")
        print(f"  Batch: {args.batch_size}")
    print(f"{'='*70}")
    print(f"Total products: {len(df)}\n")

    desc_col = "product_description"

    # ── Step 1: Taxonomy ──
    print("Step 1: Parsing taxonomy...")
    hier_col = None
    for col in df.columns:
        if "hierarchy" in col.lower() or "category" in col.lower():
            hier_col = col
            break

    all_levels = []
    depths = []
    for _, row in df.iterrows():
        levels = parse_hierarchy(row.get(hier_col, ""))
        all_levels.append(levels)
        depths.append(len(levels))

    max_depth = max(depths) if depths else 0
    for d in range(max_depth):
        df[f"level_{d+1}"] = [
            levels[d] if d < len(levels) else None for levels in all_levels]
    df["taxonomy_depth"] = depths
    df["taxonomy_full"] = [" / ".join(levels) for levels in all_levels]

    valid_mask = (df["taxonomy_depth"] > 0)
    if "level_1" in df.columns:
        valid_mask = valid_mask & (
            ~df["level_1"].str.contains("Browse By Brand", na=True))
    valid_df = df[valid_mask].copy()
    print(f"  Valid: {len(valid_df)}, Invalid: {len(df) - len(valid_df)}")

    # ── Step 2: Extract attributes ──
    print(f"\n{'='*70}")
    print(f"Step 2: Attribute extraction ({mode})...")

    if args.api_key:
        # Claude API extraction
        progress_path = os.path.join(output_dir, "claude_progress.json")
        if not args.resume and os.path.exists(progress_path):
            os.remove(progress_path)
        claude_extract_all(df, args.api_key, desc_col,
                           batch_size=args.batch_size,
                           progress_path=progress_path,
                           model=args.model)
    else:
        # Rule-based extraction
        for col in ["primary_color", "secondary_color",
                     "primary_material", "secondary_material",
                     "style", "shape", "assembly"]:
            df[col] = None

        for idx, row in df.iterrows():
            attrs = rule_based_extract(row, desc_col)
            for k, v in attrs.items():
                df.at[idx, k] = v

    # Coverage report
    print(f"\n  Attribute coverage:")
    for col in ["primary_color", "secondary_color", "primary_material",
                "secondary_material", "style", "shape", "assembly"]:
        cov = df[col].notna().sum()
        uniq = df[col].dropna().nunique()
        print(f"    {col:25s}: {cov:5d} ({cov/len(df)*100:.1f}%), {uniq:3d} unique")

    for col in ["primary_color", "primary_material", "style", "shape"]:
        print(f"\n    {col}:")
        for val, cnt in df[col].value_counts().head(15).items():
            tot = df[col].notna().sum()
            print(f"      {val:25s} {cnt:5d} ({cnt/tot*100:.1f}%)")

    # ── Step 3: Richness scores ──
    print(f"\n{'='*70}")
    print(f"Step 3: Richness scores...")
    scores = []
    for _, row in df.iterrows():
        s = 0
        if pd.notna(row.get("primary_color")): s += 2
        if pd.notna(row.get("secondary_color")): s += 0.5
        if pd.notna(row.get("primary_material")): s += 2
        if pd.notna(row.get("secondary_material")): s += 0.5
        if pd.notna(row.get("style")): s += 1.5
        if pd.notna(row.get("shape")): s += 1
        if pd.notna(row.get("assembly")): s += 0.5
        if row.get("taxonomy_depth", 0) >= 4: s += 1
        scores.append(s)
    df["attr_score"] = scores

    # ── Step 4: Build queue ──
    print(f"\n{'='*70}")
    print(f"Step 4: Building queue...")

    valid_df = valid_df.copy()
    for acol in ["primary_color", "secondary_color", "primary_material",
                 "secondary_material", "style", "shape", "assembly",
                 "attr_score"]:
        if acol in df.columns:
            valid_df[acol] = df.loc[valid_df.index, acol]

    JUNK = {"Sale", "Protection Plans", "Shop Product Type", "Clips",
            "Cash Handling", "Display Cases",
            "Stages, Risers and Accessories", "Learning Resources",
            "Physical Education Equipment",
            "Partition & Panel Hardware Accessories",
            "Meeting & Collaborative Spaces", "Ergonomic Accessories",
            "Early Education", "Desk Parts", "Buffet Accessories"}
    if "level_1" in valid_df.columns:
        before = len(valid_df)
        valid_df = valid_df[~valid_df["level_1"].isin(JUNK)]
        print(f"  Filtered {before - len(valid_df)} junk products")

    valid_df["class_clean"] = valid_df["product_class"].apply(
        lambda x: str(x).split("|")[0].strip() if pd.notna(x) else "Unknown")
    cc = valid_df["class_clean"].value_counts()
    large = set(cc[cc >= 100].index)
    medium = set(cc[(cc >= 20) & (cc < 100)].index)
    small = set(cc[(cc >= 5) & (cc < 20)].index)
    tiny = set(cc[cc < 5].index)

    valid_df = valid_df.sort_values("attr_score", ascending=False)

    queue = []
    for cat_set, max_n, tier in [(large, 30, "large"), (medium, 50, "medium"),
                                  (small, 999, "small"), (tiny, 999, "tiny")]:
        for cat in sorted(cat_set):
            for _, row in valid_df[valid_df["class_clean"] == cat].head(max_n).iterrows():
                levels = parse_hierarchy(row.get(hier_col, ""))

                def sv(c):
                    v = row.get(c)
                    return None if v is None or pd.isna(v) else str(v)

                # Build text_input
                desc = str(row.get(desc_col, ""))
                if desc == "nan":
                    desc = ""
                text_input = build_text_input(
                    row["product_name"],
                    row.get("product_class", ""),
                    desc)

                queue.append({
                    "product_id": str(row["product_id"]),
                    "product_name": str(row["product_name"]),
                    "product_class": str(row.get("product_class", "")),
                    "taxonomy": levels,
                    "taxonomy_depth": len(levels),
                    "text_input": text_input,
                    "primary_color": sv("primary_color"),
                    "secondary_color": sv("secondary_color"),
                    "primary_material": sv("primary_material"),
                    "secondary_material": sv("secondary_material"),
                    "style": sv("style"),
                    "shape": sv("shape"),
                    "assembly": sv("assembly"),
                    "attr_score": float(row.get("attr_score", 0)),
                    "tier": tier,
                })

    total_q = len(queue)
    print(f"\n  QUEUE: {total_q} products")
    for attr in ["primary_color", "secondary_color", "primary_material",
                 "secondary_material", "style", "shape", "assembly"]:
        c = sum(1 for q in queue if q.get(attr) is not None)
        print(f"    {attr:25s}: {c:5d} ({c/total_q*100:.0f}%)")

    # ── Step 5: Validation ──
    print(f"\n{'='*70}")
    print(f"Step 5: Validation...")

    chairs = [q for q in queue if "chair" in q["product_name"].lower()]
    ch_wood = [q for q in chairs if q.get("primary_material") in
               ("wood", "light_wood", "dark_wood", "manufactured_wood")]
    ch_fab = [q for q in chairs if q.get("primary_material") in
              ("velvet", "linen", "microfiber", "leather", "faux_leather")]
    print(f"\n  Chairs: {len(chairs)}")
    print(f"    wood*:    {len(ch_wood)} ({len(ch_wood)/max(len(chairs),1)*100:.0f}%)")
    print(f"    fab/lth*: {len(ch_fab)} ({len(ch_fab)/max(len(chairs),1)*100:.0f}%)")
    for q in chairs[:5]:
        print(f"    {q['product_name'][:42]:42s} "
              f"c={str(q.get('primary_color')):14s} "
              f"m={str(q.get('primary_material')):15s}")

    # Sample text_input
    print(f"\n  Sample text_input:")
    for q in queue[:3]:
        ti = q["text_input"]
        if len(ti) > 100:
            ti = ti[:100] + "..."
        print(f"    {ti}")

    # ── Step 6: Save ──
    print(f"\n{'='*70}")
    print(f"Step 6: Saving...")

    df.to_csv(os.path.join(output_dir, "classifier_products.tsv"),
              index=False, sep="\t")

    taxonomy_data = {
        "max_depth": max_depth,
        "level_counts": {f"level_{d+1}": int(df[f"level_{d+1}"].dropna().nunique())
                         for d in range(max_depth) if f"level_{d+1}" in df.columns},
        "all_paths": sorted(list(set(" / ".join(l) for l in all_levels if l))),
    }
    with open(os.path.join(output_dir, "taxonomy_tree.json"), "w") as f:
        json.dump(taxonomy_data, f, indent=2)

    attr_vocab = {col: sorted(df[col].dropna().unique().tolist())
                  for col in ["primary_color", "secondary_color",
                              "primary_material", "secondary_material",
                              "style", "shape", "assembly"]}
    vp = os.path.join(output_dir, "attribute_vocab.json")
    with open(vp, "w") as f:
        json.dump(attr_vocab, f, indent=2)
    print(f"  {vp}")
    for a, v in attr_vocab.items():
        print(f"    {a}: {v}")

    with open(os.path.join(output_dir, "image_queue.json"), "w") as f:
        json.dump(queue, f, indent=2, default=str)
    print(f"  image_queue.json ({total_q} products)")

    summary = {
        "mode": mode,
        "total_products": len(df),
        "valid_products": len(valid_df),
        "attributes": {
            attr: {
                "coverage": int(df[attr].notna().sum()),
                "pct": round(df[attr].notna().sum()/len(df)*100, 1),
                "unique": int(df[attr].dropna().nunique()),
            }
            for attr in ["primary_color", "secondary_color",
                         "primary_material", "secondary_material",
                         "style", "shape", "assembly"]},
        "queue_size": total_q,
    }
    with open(os.path.join(output_dir, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"DONE! Mode: {mode}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()