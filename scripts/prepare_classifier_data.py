#!/usr/bin/env python3
"""Prepare WANDS data for classifier training + image sourcing.

Reads REAL taxonomy from "category hierarchy" column (variable depth).
No hardcoding. Normalizes messy attribute values.
Builds prioritized image sourcing queue.

Usage:
    python scripts/prepare_classifier_data.py

Outputs:
    data/processed/classifier_products.tsv  ← cleaned product data
    data/processed/taxonomy_tree.json       ← full taxonomy from data
    data/processed/attribute_vocab.json     ← normalized value mappings
    data/processed/image_queue.json         ← prioritized sampling queue
    data/processed/data_summary.json        ← distribution stats
"""
import pandas as pd
import json
import re
import os
from collections import defaultdict, Counter
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# ATTRIBUTE NORMALIZATION MAPS
# ════════════════════════════════════════════════════════════════

COLOR_FAMILIES = {
    "white": ["white", "ivory", "cream", "off-white", "off white",
              "snow", "pearl", "alabaster", "eggshell", "linen"],
    "black": ["black", "jet", "onyx", "ebony"],
    "gray": ["gray", "grey", "charcoal", "slate",
             "dark gray", "light gray", "dark grey", "light grey",
             "graphite", "pewter", "smoke", "ash", "heather"],
    "brown": ["brown", "chocolate", "espresso", "mocha", "walnut",
              "mahogany", "chestnut", "cocoa", "coffee", "umber",
              "sienna", "cognac", "saddle", "russet", "auburn"],
    "beige": ["beige", "tan", "khaki", "sand", "taupe", "camel",
              "natural", "oatmeal", "wheat", "buff", "latte",
              "mushroom", "fawn", "parchment", "champagne",
              "bone", "ecru"],
    "blue": ["blue", "navy", "royal blue", "sky blue", "baby blue",
             "teal", "turquoise", "aqua", "cobalt", "sapphire",
             "indigo", "denim", "powder blue", "cerulean",
             "periwinkle", "cornflower", "midnight blue"],
    "green": ["green", "sage", "olive", "emerald", "forest",
              "mint", "lime", "hunter", "seafoam", "jade",
              "moss", "fern", "pistachio", "kelly"],
    "red": ["red", "burgundy", "crimson", "maroon", "scarlet",
            "wine", "cherry", "ruby", "brick", "cranberry",
            "garnet", "oxblood"],
    "pink": ["pink", "blush", "rose", "coral", "salmon",
             "fuchsia", "magenta", "mauve", "dusty rose",
             "hot pink", "raspberry"],
    "orange": ["orange", "rust", "terracotta", "burnt orange",
               "peach", "apricot", "tangerine", "amber",
               "copper", "cinnamon"],
    "yellow": ["yellow", "gold", "mustard", "lemon",
               "honey", "canary", "golden", "sunflower"],
    "purple": ["purple", "lavender", "violet", "plum",
               "lilac", "eggplant", "amethyst", "orchid", "grape"],
    "silver": ["silver", "chrome", "platinum", "metallic silver",
               "brushed silver", "stainless", "nickel"],
    "gold_metal": ["brass", "bronze", "antique gold",
                   "rose gold", "brushed gold"],
    "clear": ["clear", "transparent", "glass", "crystal"],
    "multi": ["multi", "multicolor", "multicolored",
              "rainbow", "assorted", "various"],
}

MATERIAL_GROUPS = {
    "wood": ["solid wood", "manufactured wood", "wood",
             "manufactured wood + solid wood",
             "solid + manufactured wood", "mdf",
             "particle board", "plywood", "hardwood",
             "engineered wood", "rubberwood", "pine",
             "oak", "bamboo", "teak", "cedar", "birch",
             "maple", "acacia"],
    "metal": ["metal", "steel", "iron", "aluminum",
              "stainless steel", "brass", "chrome",
              "wrought iron", "cast iron", "zinc"],
    "fabric": ["polyester", "microfiber / polyester",
               "cotton", "100 % cotton", "linen",
               "microfiber", "chenille", "tweed",
               "canvas", "satin", "silk", "velvet"],
    "leather": ["leather", "genuine leather", "faux leather",
                "bonded leather", "top grain", "full grain",
                "vegan leather", "pu leather", "leatherette"],
    "plastic": ["plastic", "resin/plastic", "resin",
                "acrylic", "polycarbonate", "abs",
                "polypropylene", "pvc", "vinyl"],
    "stone": ["marble", "granite", "quartz", "stone",
              "concrete", "slate", "travertine",
              "limestone", "terrazzo"],
    "ceramic": ["ceramic", "porcelain", "terracotta",
                "earthenware", "stoneware"],
    "glass": ["glass", "tempered glass", "frosted glass",
              "mirror glass", "crystal"],
    "natural_fiber": ["wool", "jute", "sisal", "seagrass",
                      "rattan", "wicker", "cane", "hemp"],
    "foam": ["foam", "memory foam", "gel foam",
             "polyurethane foam"],
    "synthetics": ["synthetics", "synthetic", "nylon", "olefin"],
}

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


def normalize_value(raw, group_map):
    """Normalize a raw value using a group mapping."""
    if not raw:
        return None
    raw_lower = raw.lower().strip()
    # Exact match first
    for group, keywords in group_map.items():
        for kw in keywords:
            if raw_lower == kw:
                return group
    # Contains match
    for group, keywords in group_map.items():
        for kw in keywords:
            if kw in raw_lower:
                return group
    return "other"


def parse_features(features_str):
    """Parse product_features string into dict."""
    if pd.isna(features_str) or not features_str:
        return {}
    attrs = {}
    for pair in str(features_str).split("|"):
        pair = pair.strip()
        if ":" in pair:
            key, val = pair.split(":", 1)
            attrs[key.strip()] = val.strip()
    return attrs


def parse_hierarchy(hierarchy_str):
    """Parse 'category hierarchy' into list of levels."""
    if pd.isna(hierarchy_str) or not hierarchy_str:
        return []
    # Handle quoted strings with dimensions like 28"-33"
    h = str(hierarchy_str).strip().strip('"')
    levels = [l.strip() for l in h.split("/") if l.strip()]
    return levels


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    products_csv = "data/raw/WANDS/dataset/product.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(products_csv, sep="\t")
    print(f"{'='*70}")
    print(f"WANDS DATA PREPARATION FOR CLASSIFIER")
    print(f"{'='*70}")
    print(f"Total products: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    # ── Step 1: Parse REAL taxonomy from data ──
    print("Step 1: Parsing taxonomy from 'category hierarchy' column...")

    # Find the hierarchy column (might have space in name)
    hier_col = None
    for col in df.columns:
        if "hierarchy" in col.lower() or "category" in col.lower():
            hier_col = col
            break
    if hier_col is None:
        # Try with space
        for col in df.columns:
            if "category" in col.lower():
                hier_col = col
                break
    print(f"  Using column: '{hier_col}'")

    all_levels = []
    depths = []
    for _, row in df.iterrows():
        levels = parse_hierarchy(row.get(hier_col, ""))
        all_levels.append(levels)
        depths.append(len(levels))

    max_depth = max(depths) if depths else 0
    print(f"  Max depth: {max_depth}")

    depth_dist = Counter(depths)
    for d in sorted(depth_dist.keys()):
        print(f"    Depth {d}: {depth_dist[d]:5d} products "
              f"({depth_dist[d]/len(df)*100:.1f}%)")

    # Add level columns dynamically
    for d in range(max_depth):
        col_name = f"level_{d+1}"
        df[col_name] = [
            levels[d] if d < len(levels) else None
            for levels in all_levels
        ]

    df["taxonomy_depth"] = depths
    df["taxonomy_full"] = [" / ".join(levels) for levels in all_levels]

    # Filter out bad hierarchies (depth 0 or "Browse By Brand")
    valid_mask = (df["taxonomy_depth"] > 0)
    if "level_1" in df.columns:
        valid_mask = valid_mask & (
            ~df["level_1"].str.contains("Browse By Brand", na=True))
    valid_df = df[valid_mask].copy()
    invalid_df = df[~valid_mask].copy()

    print(f"\n  Valid products:   {len(valid_df)}")
    print(f"  Invalid (no hier):{len(invalid_df)}")

    # Taxonomy stats per level
    for d in range(max_depth):
        col = f"level_{d+1}"
        if col in valid_df.columns:
            n_unique = valid_df[col].dropna().nunique()
            coverage = valid_df[col].notna().sum()
            print(f"  Level {d+1}: {n_unique:4d} unique classes, "
                  f"{coverage:5d} products have it")

    # Build taxonomy tree
    taxonomy_tree = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(set)))))

    for levels in all_levels:
        if len(levels) < 2:
            continue
        node = taxonomy_tree
        for level in levels:
            if isinstance(node, dict):
                node = node[level]

    # Simpler: just collect unique paths per depth
    paths_per_depth = defaultdict(set)
    for levels in all_levels:
        for d in range(1, len(levels) + 1):
            path = " / ".join(levels[:d])
            paths_per_depth[d].add(path)

    print(f"\n  Unique taxonomy paths per depth:")
    for d in sorted(paths_per_depth.keys()):
        print(f"    Depth {d}: {len(paths_per_depth[d])} paths")

    # Level 1 distribution
    if "level_1" in valid_df.columns:
        print(f"\n  Level 1 (department) distribution:")
        for dept, count in valid_df["level_1"].value_counts().head(15).items():
            print(f"    {dept:40s} {count:5d}")

    # ── Step 2: Parse and normalize attributes ──
    print(f"\n{'='*70}")
    print(f"Step 2: Parsing and normalizing attributes...")

    # Discover all attribute keys
    all_attr_keys = Counter()
    for features_str in df["product_features"].dropna():
        for pair in str(features_str).split("|"):
            if ":" in pair:
                key = pair.split(":")[0].strip()
                all_attr_keys[key] += 1

    print(f"  All attribute keys found ({len(all_attr_keys)}):")
    for key, count in all_attr_keys.most_common(20):
        print(f"    {key:40s} {count:5d} ({count/len(df)*100:.1f}%)")

    # Explicit key mapping (verified from data analysis)
    # Picked highest-coverage key for each target attribute:
    #   color:    'color' (26,289)     not 'basecolor' (8,145)
    #   style:    'style' (19,221)     not 'dsprimaryproductstyle' (18,227)
    #   material: 'primarymaterial' (10,817) not 'framematerial' (10,396)
    #   shape:    'shape' (9,122)      not 'headboardshape' or 'topshape'
    #   assembly: 'levelofassembly'
    EXPLICIT_KEY_MAP = {
        "color": ["color"],
        "material": ["primarymaterial", "material"],
        "style": ["style"],
        "shape": ["shape"],
        "assembly": ["levelofassembly", "adultassemblyrequired"],
    }

    # Find actual keys (case-insensitive match)
    all_keys_lower = {k.lower().strip(): k for k in all_attr_keys}
    target_map = {}
    for target, candidates in EXPLICIT_KEY_MAP.items():
        for candidate in candidates:
            actual = all_keys_lower.get(candidate.lower())
            if actual:
                target_map[target] = actual
                break

    print(f"\n  Attribute key mapping (explicit):")
    for target, actual in target_map.items():
        count = all_attr_keys.get(actual, 0)
        print(f"    {target:15s} ← '{actual}' ({count} products)")

    # Normalization functions per attribute
    normalizers = {
        "color": (COLOR_FAMILIES, target_map.get("color")),
        "material": (MATERIAL_GROUPS, target_map.get("material")),
        "style": (STYLE_GROUPS, target_map.get("style")),
        "shape": (SHAPE_GROUPS, target_map.get("shape")),
        "assembly": (ASSEMBLY_GROUPS, target_map.get("assembly")),
    }

    # Normalize for every product
    for attr_name, (group_map, source_key) in normalizers.items():
        if source_key is None:
            df[f"{attr_name}_raw"] = None
            df[f"{attr_name}_norm"] = None
            continue

        raw_vals = []
        norm_vals = []
        for _, row in df.iterrows():
            features = parse_features(row.get("product_features", ""))
            raw = features.get(source_key)
            if raw:
                raw_vals.append(raw)
                norm_vals.append(normalize_value(raw, group_map))
            else:
                raw_vals.append(None)
                norm_vals.append(None)

        df[f"{attr_name}_raw"] = raw_vals
        df[f"{attr_name}_norm"] = norm_vals

    # ── Step 2b: Mine taxonomy for missing attributes ──
    # Taxonomy leaves often encode attributes, e.g.:
    #   "Grey Cabinets & Chests"     → color = gray
    #   "Brown Throw Pillows"        → color = brown
    #   "Metal Shelving"             → material = metal
    #   "Rustic Bed Frames"          → style = rustic
    # We only backfill when the attribute column is currently null.

    print(f"\n  Step 2b: Mining taxonomy for missing attributes...")

    # Build reverse lookups: keyword → normalized value
    # Only mine color, material, style (shape/assembly rarely in taxonomy)
    TAXONOMY_MINERS = {
        "color": COLOR_FAMILIES,
        "material": MATERIAL_GROUPS,
        "style": STYLE_GROUPS,
    }

    # Pre-build keyword→group reverse maps for fast lookup
    reverse_maps = {}
    for attr_name, group_map in TAXONOMY_MINERS.items():
        rmap = {}
        for group, keywords in group_map.items():
            for kw in keywords:
                rmap[kw.lower()] = group
        reverse_maps[attr_name] = rmap

    # Get all taxonomy level columns
    level_cols = sorted([c for c in df.columns if c.startswith("level_")])

    mined_counts = {attr: 0 for attr in TAXONOMY_MINERS}

    for idx, row in df.iterrows():
        # Collect all taxonomy text (focus on leaf → root order)
        tax_texts = []
        for lcol in reversed(level_cols):
            val = row.get(lcol)
            if pd.notna(val) and val:
                tax_texts.append(str(val).lower())

        if not tax_texts:
            continue

        for attr_name, rmap in reverse_maps.items():
            norm_col = f"{attr_name}_norm"
            # Only backfill if currently null
            if pd.notna(row.get(norm_col)):
                continue

            # Search taxonomy levels (leaf first) for attribute keywords
            found = None
            for text in tax_texts:
                # Try multi-word matches first (longer keywords first)
                sorted_kws = sorted(rmap.keys(), key=len, reverse=True)
                for kw in sorted_kws:
                    if kw in text:
                        found = rmap[kw]
                        break
                if found:
                    break

            if found:
                df.at[idx, norm_col] = found
                if pd.isna(row.get(f"{attr_name}_raw")) or not row.get(f"{attr_name}_raw"):
                    df.at[idx, f"{attr_name}_raw"] = f"[mined:{tax_texts[0]}]"
                mined_counts[attr_name] += 1

    # Report mining results
    print(f"  Taxonomy mining results (backfilled nulls):")
    for attr_name in TAXONOMY_MINERS:
        mined = mined_counts[attr_name]
        norm_col = f"{attr_name}_norm"
        new_total = df[norm_col].notna().sum()
        print(f"    {attr_name:15s}: +{mined:5d} mined → "
              f"{new_total:5d} total ({new_total/len(df)*100:.1f}%)")

    # Attribute stats
    print(f"\n  Normalized attribute coverage:")
    for attr_name in ["color", "material", "style", "shape", "assembly"]:
        col = f"{attr_name}_norm"
        coverage = df[col].notna().sum()
        unique = df[col].dropna().nunique()
        print(f"    {attr_name:15s}: {coverage:5d} products "
              f"({coverage/len(df)*100:.1f}%), "
              f"{unique:3d} unique values")

        # Show distribution
        for val, count in df[col].value_counts().head(10).items():
            total_attr = df[col].notna().sum()
            print(f"      {val:25s} {count:5d} "
                  f"({count/total_attr*100:.1f}%)")

    # ── Step 3: Compute label richness score ──
    print(f"\n{'='*70}")
    print(f"Step 3: Computing label richness scores...")

    scores = []
    for _, row in df.iterrows():
        score = 0
        if pd.notna(row.get("color_norm")):
            score += 2
        if pd.notna(row.get("material_norm")):
            score += 2
        if pd.notna(row.get("style_norm")):
            score += 1.5
        if pd.notna(row.get("shape_norm")):
            score += 1
        if pd.notna(row.get("assembly_norm")):
            score += 0.5
        # Bonus for deeper taxonomy
        if row.get("taxonomy_depth", 0) >= 4:
            score += 1
        scores.append(score)

    df["attr_score"] = scores

    score_dist = Counter()
    for s in scores:
        bucket = "0" if s == 0 else f"{int(s)}-{int(s)+1}"
        score_dist[bucket] += 1

    print(f"  Score distribution:")
    for bucket in sorted(score_dist.keys()):
        print(f"    Score {bucket:6s}: {score_dist[bucket]:5d} products")

    # ── Step 4: Build smart sampling queue ──
    print(f"\n{'='*70}")
    print(f"Step 4: Building image sourcing queue...")

    # Use product_class for grouping (leaf category)
    valid_df = valid_df.copy()
    valid_df["attr_score"] = df.loc[valid_df.index, "attr_score"]
    for acol in ["color_norm", "material_norm", "style_norm", "shape_norm", "assembly_norm"]:
        if acol in df.columns:
            valid_df[acol] = df.loc[valid_df.index, acol]

    # Filter out junk departments (not real product categories)
    JUNK_DEPARTMENTS = {
        "Sale", "Protection Plans", "Shop Product Type",
        "Clips", "Cash Handling", "Display Cases",
        "Stages, Risers and Accessories", "Learning Resources",
        "Physical Education Equipment", "Partition & Panel Hardware Accessories",
        "Meeting & Collaborative Spaces", "Ergonomic Accessories",
        "Early Education", "Desk Parts", "Buffet Accessories",
    }
    if "level_1" in valid_df.columns:
        before = len(valid_df)
        valid_df = valid_df[~valid_df["level_1"].isin(JUNK_DEPARTMENTS)]
        filtered = before - len(valid_df)
        if filtered > 0:
            print(f"  Filtered {filtered} products from junk departments")

    # Clean product_class: take first if pipe-separated
    valid_df["class_clean"] = valid_df["product_class"].apply(
        lambda x: str(x).split("|")[0].strip() if pd.notna(x) else "Unknown")

    cat_counts = valid_df["class_clean"].value_counts()
    large_cats = set(cat_counts[cat_counts >= 100].index)
    medium_cats = set(cat_counts[(cat_counts >= 20) &
                                  (cat_counts < 100)].index)
    small_cats = set(cat_counts[(cat_counts >= 5) &
                                 (cat_counts < 20)].index)
    tiny_cats = set(cat_counts[cat_counts < 5].index)

    print(f"  Categories: {len(cat_counts)} total")
    print(f"    Large (100+):   {len(large_cats)}")
    print(f"    Medium (20-99): {len(medium_cats)}")
    print(f"    Small (5-19):   {len(small_cats)}")
    print(f"    Tiny (<5):      {len(tiny_cats)}")

    # Sort by attr_score within each category (richest labels first)
    valid_df = valid_df.sort_values("attr_score", ascending=False)

    queue = []
    for cat_set, max_n, tier in [
        (large_cats, 30, "large"),
        (medium_cats, 50, "medium"),
        (small_cats, 999, "small"),
        (tiny_cats, 999, "tiny"),
    ]:
        for cat in sorted(cat_set):
            cat_products = valid_df[valid_df["class_clean"] == cat]
            sampled = cat_products.head(max_n)

            for _, row in sampled.iterrows():
                levels = parse_hierarchy(row.get(hier_col, ""))

                # Safely extract attribute values (NaN → None)
                def safe_val(col):
                    v = row.get(col)
                    if v is None or pd.isna(v):
                        return None
                    return str(v)

                queue.append({
                    "product_id": str(row["product_id"]),
                    "product_name": str(row["product_name"]),
                    "product_class": str(row.get("product_class", "")),
                    "taxonomy": levels,
                    "taxonomy_depth": len(levels),
                    "color": safe_val("color_norm"),
                    "material": safe_val("material_norm"),
                    "style": safe_val("style_norm"),
                    "shape": safe_val("shape_norm"),
                    "assembly": safe_val("assembly_norm"),
                    "attr_score": float(row.get("attr_score", 0)),
                    "tier": tier,
                })

    # Queue stats
    total_q = len(queue)
    with_color = sum(1 for q in queue if q["color"] is not None)
    with_material = sum(1 for q in queue if q["material"] is not None)
    with_style = sum(1 for q in queue if q["style"] is not None)
    with_shape = sum(1 for q in queue if q["shape"] is not None)
    with_assembly = sum(1 for q in queue if q["assembly"] is not None)
    high_pri = sum(1 for q in queue if q["attr_score"] >= 4)

    tier_counts = Counter(q["tier"] for q in queue)
    dept_counts = Counter(
        q["taxonomy"][0] if q["taxonomy"] else "Unknown"
        for q in queue)
    depth_q = Counter(q["taxonomy_depth"] for q in queue)

    print(f"\n  QUEUE SUMMARY:")
    print(f"    Total products:      {total_q}")
    print(f"    With color labels:   {with_color} "
          f"({with_color/total_q*100:.0f}%)")
    print(f"    With material:       {with_material} "
          f"({with_material/total_q*100:.0f}%)")
    print(f"    With style:          {with_style} "
          f"({with_style/total_q*100:.0f}%)")
    print(f"    With shape:          {with_shape} "
          f"({with_shape/total_q*100:.0f}%)")
    print(f"    With assembly:       {with_assembly} "
          f"({with_assembly/total_q*100:.0f}%)")
    print(f"    High priority (≥4):  {high_pri} "
          f"({high_pri/total_q*100:.0f}%)")
    print(f"    Est. images (×5):    ~{total_q * 5}")

    print(f"\n    By tier:")
    for tier, count in tier_counts.most_common():
        print(f"      {tier:10s}: {count:5d}")

    print(f"\n    By department:")
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"      {dept:40s}: {count:5d}")

    print(f"\n    By taxonomy depth:")
    for d, count in sorted(depth_q.items()):
        print(f"      Depth {d}: {count:5d}")

    # ── Step 5: Save outputs ──
    print(f"\n{'='*70}")
    print(f"Step 5: Saving outputs...")

    # 5a: Cleaned products
    out_tsv = os.path.join(output_dir, "classifier_products.tsv")
    df.to_csv(out_tsv, index=False, sep="\t")
    print(f"  {out_tsv}")

    # 5b: Taxonomy tree (all unique paths)
    taxonomy_data = {
        "max_depth": max_depth,
        "level_counts": {
            f"level_{d+1}": int(df[f"level_{d+1}"].dropna().nunique())
            for d in range(max_depth) if f"level_{d+1}" in df.columns
        },
        "level_values": {
            f"level_{d+1}": sorted(
                df[f"level_{d+1}"].dropna().unique().tolist())
            for d in range(max_depth) if f"level_{d+1}" in df.columns
        },
        "all_paths": sorted(list(set(
            " / ".join(levels)
            for levels in all_levels if levels
        ))),
    }
    tax_path = os.path.join(output_dir, "taxonomy_tree.json")
    with open(tax_path, "w") as f:
        json.dump(taxonomy_data, f, indent=2)
    print(f"  {tax_path} (max_depth={max_depth})")

    # 5c: Attribute vocabulary
    attr_vocab = {}
    for attr_name in ["color", "material", "style", "shape", "assembly"]:
        col = f"{attr_name}_norm"
        vals = sorted(df[col].dropna().unique().tolist())
        attr_vocab[attr_name] = vals
    vocab_path = os.path.join(output_dir, "attribute_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(attr_vocab, f, indent=2)
    print(f"  {vocab_path}")
    for attr, vals in attr_vocab.items():
        print(f"    {attr}: {vals}")

    # 5d: Image queue
    queue_path = os.path.join(output_dir, "image_queue.json")
    with open(queue_path, "w") as f:
        json.dump(queue, f, indent=2, default=str)
    print(f"  {queue_path} ({len(queue)} products)")

    # 5e: Summary
    summary = {
        "total_products": len(df),
        "valid_products": len(valid_df),
        "invalid_products": len(invalid_df),
        "taxonomy": {
            "max_depth": max_depth,
            "depth_distribution": {str(k): v for k, v in
                                    sorted(Counter(depths).items())},
        },
        "attributes": {
            attr: {
                "source_key": target_map.get(attr),
                "coverage": int(df[f"{attr}_norm"].notna().sum()),
                "coverage_pct": round(
                    df[f"{attr}_norm"].notna().sum() / len(df) * 100, 1),
                "unique_values": int(df[f"{attr}_norm"].dropna().nunique()),
                "values": df[f"{attr}_norm"].value_counts().to_dict(),
            }
            for attr in ["color", "material", "style", "shape", "assembly"]
        },
        "queue": {
            "size": total_q,
            "with_color": with_color,
            "with_material": with_material,
            "with_style": with_style,
            "with_shape": with_shape,
            "with_assembly": with_assembly,
            "high_priority": high_pri,
            "tiers": dict(tier_counts),
            "departments": dict(dept_counts),
        },
    }
    summary_path = os.path.join(output_dir, "data_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  {summary_path}")

    print(f"\n{'='*70}")
    print(f"DONE! Next steps:")
    print(f"  1. Review outputs in {output_dir}/")
    print(f"  2. Run image sourcing:")
    print(f"     python scripts/source_wayfair_images.py \\")
    print(f"       --queue data/processed/image_queue.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()