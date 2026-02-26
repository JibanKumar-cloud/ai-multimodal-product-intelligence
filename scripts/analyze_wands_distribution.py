#!/usr/bin/env python3
"""Analyze WANDS dataset distribution for image sourcing strategy.

Run BEFORE sourcing images to understand:
  1. How many products per leaf category?
  2. How deep is the taxonomy? How many classes per level?
  3. What's the attribute coverage per category?
  4. Will 5000 balanced samples cover everything?
  5. Which categories/attributes are underrepresented?

Usage:
    python scripts/analyze_wands_distribution.py
"""
import pandas as pd
import json
import re
from collections import defaultdict, Counter
from pathlib import Path


def parse_features(features_str):
    """Parse product_features string into dict."""
    if pd.isna(features_str) or not features_str:
        return {}
    attrs = {}
    for pair in str(features_str).split("|"):
        pair = pair.strip()
        if ":" in pair:
            key, val = pair.split(":", 1)
            attrs[key.strip().lower()] = val.strip().lower()
    return attrs


def main():
    products_csv = "data/raw/WANDS/dataset/product.csv"
    df = pd.read_csv(products_csv, sep="\t")
    print(f"{'='*70}")
    print(f"WANDS DATASET DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Total products: {len(df)}\n")

    # ════════════════════════════════════════
    # 1. TAXONOMY ANALYSIS
    # ════════════════════════════════════════
    print(f"{'='*70}")
    print(f"1. TAXONOMY HIERARCHY")
    print(f"{'='*70}")

    # Parse hierarchy into levels
    level_values = defaultdict(Counter)
    depths = []
    leaf_counts = Counter()

    for _, row in df.iterrows():
        hierarchy = str(row.get("category_hierarchy", ""))
        levels = [l.strip() for l in hierarchy.split("/") if l.strip()]
        depths.append(len(levels))

        for i, level in enumerate(levels):
            level_values[f"level_{i+1}"].update([level])

        leaf = str(row.get("product_class", "unknown"))
        leaf_counts[leaf] += 1

    max_depth = max(depths) if depths else 0
    print(f"Max taxonomy depth: {max_depth}")
    print(f"Depth distribution:")
    depth_dist = Counter(depths)
    for d in sorted(depth_dist.keys()):
        print(f"  Depth {d}: {depth_dist[d]:6d} products "
              f"({depth_dist[d]/len(df)*100:.1f}%)")

    print(f"\nClasses per level:")
    for level_name in sorted(level_values.keys()):
        n_classes = len(level_values[level_name])
        print(f"  {level_name}: {n_classes} unique classes")

    # Leaf category distribution
    print(f"\nLeaf categories (product_class): {len(leaf_counts)}")
    sorted_leaves = leaf_counts.most_common()

    print(f"\n  Top 20 categories (most products):")
    for cat, count in sorted_leaves[:20]:
        print(f"    {cat:45s} {count:5d}")

    print(f"\n  Bottom 20 categories (fewest products):")
    for cat, count in sorted_leaves[-20:]:
        print(f"    {cat:45s} {count:5d}")

    # Size buckets
    tiny = sum(1 for _, c in sorted_leaves if c < 5)
    small = sum(1 for _, c in sorted_leaves if 5 <= c < 20)
    medium = sum(1 for _, c in sorted_leaves if 20 <= c < 100)
    large = sum(1 for _, c in sorted_leaves if c >= 100)

    print(f"\n  Category size distribution:")
    print(f"    <5 products:     {tiny:4d} categories (TINY - hard to learn)")
    print(f"    5-19 products:   {small:4d} categories")
    print(f"    20-99 products:  {medium:4d} categories")
    print(f"    100+ products:   {large:4d} categories")

    # ════════════════════════════════════════
    # 2. ATTRIBUTE ANALYSIS
    # ════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"2. ATTRIBUTE COVERAGE")
    print(f"{'='*70}")

    # Parse all attributes
    attr_counts = defaultdict(Counter)
    attr_present = defaultdict(int)

    for _, row in df.iterrows():
        features = parse_features(row.get("product_features", ""))
        for key, val in features.items():
            attr_counts[key][val] += 1
            attr_present[key] += 1

    # Key attributes for our model
    target_attrs = ["color", "color family", "material",
                    "primary material", "style", "room",
                    "room type", "assembly required",
                    "assembly", "shape"]

    print(f"\nAll attributes found: {len(attr_counts)}")
    print(f"\nAttribute coverage (how many products have it):")
    for attr in sorted(attr_present.keys(),
                       key=lambda x: -attr_present[x]):
        count = attr_present[attr]
        pct = count / len(df) * 100
        is_target = "  ← TARGET" if attr in target_attrs else ""
        print(f"  {attr:35s} {count:6d} ({pct:5.1f}%){is_target}")

    # Detailed breakdown for target attributes
    print(f"\n{'='*70}")
    print(f"3. TARGET ATTRIBUTE VALUE DISTRIBUTIONS")
    print(f"{'='*70}")

    for attr in target_attrs:
        if attr not in attr_counts:
            # Try fuzzy match
            matches = [k for k in attr_counts if attr in k]
            if matches:
                attr = matches[0]
            else:
                continue

        values = attr_counts[attr]
        total = sum(values.values())
        n_unique = len(values)

        print(f"\n  {attr.upper()} ({total} products, "
              f"{n_unique} unique values):")

        for val, count in values.most_common(15):
            bar = "█" * min(int(count / total * 50), 50)
            print(f"    {val:30s} {count:5d} "
                  f"({count/total*100:5.1f}%) {bar}")
        if n_unique > 15:
            print(f"    ... and {n_unique - 15} more values")

        # Check for rare values
        rare = sum(1 for _, c in values.items() if c < 5)
        if rare:
            print(f"    ⚠️  {rare} values with <5 examples "
                  f"(hard to learn)")

    # ════════════════════════════════════════
    # 4. SAMPLING STRATEGY ANALYSIS
    # ════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"4. SAMPLING STRATEGY: WILL 5000 BE ENOUGH?")
    print(f"{'='*70}")

    for max_per_cat in [5, 10, 15, 25, 50]:
        total_sampled = 0
        fully_covered = 0
        partially_covered = 0
        cats_covered = 0

        for cat, count in sorted_leaves:
            sampled = min(count, max_per_cat)
            total_sampled += sampled
            cats_covered += 1
            if sampled >= 10:
                fully_covered += 1
            elif sampled >= 3:
                partially_covered += 1

        print(f"\n  max_per_category={max_per_cat}:")
        print(f"    Total products sampled: {total_sampled}")
        print(f"    Categories covered:     {cats_covered}/{len(leaf_counts)}")
        print(f"    Well-covered (≥10):     {fully_covered}")
        print(f"    Partially (3-9):        {partially_covered}")
        print(f"    Barely (<3):            "
              f"{cats_covered - fully_covered - partially_covered}")
        print(f"    Est. images (×5/prod):  ~{total_sampled * 5}")

    # ════════════════════════════════════════
    # 5. CROSS-ANALYSIS: ATTRIBUTES × CATEGORIES
    # ════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"5. ATTRIBUTE COVERAGE BY TOP CATEGORIES")
    print(f"{'='*70}")

    # For top 30 categories, show attribute coverage
    print(f"\n  {'CATEGORY':<35s} {'COUNT':>5s} "
          f"{'COLOR':>6s} {'MATER':>6s} {'STYLE':>6s} "
          f"{'ROOM':>6s} {'ASSEM':>6s}")
    print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    attr_key_map = {}
    for target in ["color", "material", "style", "room", "assembly"]:
        for key in attr_counts.keys():
            if target in key:
                attr_key_map[target] = key
                break

    for cat, count in sorted_leaves[:30]:
        cat_df = df[df["product_class"] == cat]
        coverages = {}
        for target, key in attr_key_map.items():
            has_attr = 0
            for _, row in cat_df.iterrows():
                features = parse_features(row.get("product_features", ""))
                if key in features:
                    has_attr += 1
            coverages[target] = (has_attr / len(cat_df) * 100
                                 if len(cat_df) > 0 else 0)

        print(f"  {cat:<35s} {count:5d} "
              f"{coverages.get('color', 0):5.0f}% "
              f"{coverages.get('material', 0):5.0f}% "
              f"{coverages.get('style', 0):5.0f}% "
              f"{coverages.get('room', 0):5.0f}% "
              f"{coverages.get('assembly', 0):5.0f}%")

    # ════════════════════════════════════════
    # 6. RECOMMENDATION
    # ════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"6. RECOMMENDATION")
    print(f"{'='*70}")

    tiny_cats = [cat for cat, c in sorted_leaves if c < 5]
    small_cats = [cat for cat, c in sorted_leaves if 5 <= c < 20]

    print(f"""
    Total categories: {len(leaf_counts)}
    Tiny (<5 products): {len(tiny_cats)} categories
    
    For these tiny categories, even max_per_category=25 
    won't help — there just aren't enough products.
    
    SUGGESTION:
    1. Start with max_per_category=25 (~5000 products)
    2. Check how many categories got <5 images
    3. For those, increase to max_per_category=50
    4. If still not enough, consider merging tiny categories
       with their parent level in taxonomy
    
    For attributes: if an attribute value (e.g., color=teal)
    has <10 examples, the model may struggle. Consider 
    grouping rare values (teal→blue, mahogany→brown).
    """)

    # Save analysis for reference
    analysis = {
        "total_products": len(df),
        "total_categories": len(leaf_counts),
        "max_depth": max_depth,
        "category_sizes": dict(sorted_leaves),
        "tiny_categories": tiny_cats,
        "attribute_coverage": {k: v for k, v in attr_present.items()},
    }
    out_path = "data/wands_distribution_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
