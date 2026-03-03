"""Preview new attribute-enriched queries vs old queries.

Run this BEFORE patching to see what changes:
    python scripts/preview_queries.py --queue data/processed/image_queue_with_images.json
"""
import json
import argparse


def build_search_query(product):
    parts = []
    color = product.get("primary_color")
    if color and color not in ("other", "multi"):
        color_map = {"gold_metal": "gold", "natural_fiber": "natural"}
        parts.append(color_map.get(color, color))

    material = product.get("primary_material")
    if material and material not in ("other", "mixed"):
        mat_map = {"natural_fiber": "natural fiber", "synthetics": "synthetic"}
        parts.append(mat_map.get(material, material))

    shape = product.get("shape")
    if shape and shape not in ("rectangular", "other", None):
        parts.append(shape)

    parts.append(product["product_name"])

    style = product.get("style")
    if style and style not in ("other",):
        parts.append(style)

    return " ".join(parts) + " wayfair"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    with open(args.queue) as f:
        products = json.load(f)

    print(f"{'='*80}")
    print(f"QUERY COMPARISON (old vs new)")
    print(f"{'='*80}\n")

    for p in products[:args.limit]:
        old_query = f"{p['product_name']} wayfair product"
        new_query = build_search_query(p)

        color = p.get('primary_color', '-')
        mat = p.get('primary_material', '-')
        shape = p.get('shape', '-')
        style = p.get('style', '-')

        print(f"  Product: {p['product_name'][:50]}")
        print(f"  Attrs:   color={color}, mat={mat}, shape={shape}, style={style}")
        print(f"  OLD:     {old_query}")
        print(f"  NEW:     {new_query}")
        print()


if __name__ == "__main__":
    main()