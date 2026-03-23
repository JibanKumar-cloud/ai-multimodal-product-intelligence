"""Run Claude extraction on queue products not yet processed.

Reads image_queue.json and claude_progress.json.
Processes only products missing from progress. Fully resumable.
Updates queue in-place with Claude results.

Usage:
    # Re-expand first if prepare_classifier_data overwrote your 8K queue
    python scripts/expand_queue.py

    # Then run Claude on the new products
    python scripts/run_claude_on_new.py --api-key sk-ant-...

    # Ctrl+C safe — resume anytime
    python scripts/run_claude_on_new.py --api-key sk-ant-...
"""
import json
import os
import sys
import time
import signal
import argparse


ATTR_KEYS = [
    "primary_color", "secondary_color",
    "primary_material", "secondary_material",
    "style", "shape", "assembly",
]

# ── Attribute families ──

COLOR_FAMILIES = {
    "white": ["white", "snow", "alabaster", "eggshell"],
    "cream": ["cream", "ivory", "off-white", "off white", "linen", "pearl", "bone", "ecru"],
    "black": ["black", "jet", "onyx", "ebony"],
    "dark_gray": ["charcoal", "graphite", "slate", "dark gray", "dark grey"],
    "gray": ["gray", "grey", "pewter"],
    "light_gray": ["light gray", "light grey", "ash", "heather", "smoke", "silver gray"],
    "dark_brown": ["espresso", "chocolate", "cocoa", "coffee", "umber", "dark brown"],
    "brown": ["brown", "walnut", "chestnut", "sienna", "saddle", "russet"],
    "light_brown": ["tan", "khaki", "camel", "fawn", "light brown"],
    "mahogany": ["mahogany", "cherry", "auburn"],
    "cognac": ["cognac", "amber", "cinnamon"],
    "beige": ["beige", "sand", "taupe", "mushroom", "parchment", "champagne", "buff", "latte"],
    "natural": ["natural", "oatmeal", "wheat"],
    "navy": ["navy", "midnight blue", "indigo", "dark blue"],
    "blue": ["blue", "royal blue", "cobalt", "sapphire", "denim", "cornflower"],
    "light_blue": ["sky blue", "baby blue", "powder blue", "cerulean", "periwinkle", "light blue"],
    "teal": ["teal", "turquoise", "aqua"],
    "green": ["green", "emerald", "forest", "hunter", "kelly", "jade", "fern"],
    "sage": ["sage", "olive", "moss", "pistachio", "mint", "seafoam", "light green"],
    "red": ["red", "scarlet", "crimson", "ruby", "brick", "cranberry", "garnet"],
    "burgundy": ["burgundy", "maroon", "wine", "oxblood"],
    "pink": ["pink", "blush", "rose", "salmon", "dusty rose", "hot pink", "raspberry"],
    "coral": ["coral", "fuchsia", "magenta", "mauve"],
    "orange": ["orange", "tangerine", "peach", "apricot"],
    "rust": ["rust", "terracotta", "burnt orange", "copper"],
    "yellow": ["yellow", "lemon", "canary", "sunflower"],
    "gold": ["gold", "mustard", "honey", "golden"],
    "purple": ["purple", "violet", "eggplant", "amethyst", "grape", "plum"],
    "lavender": ["lavender", "lilac", "orchid"],
    "silver": ["silver", "chrome", "platinum", "metallic silver", "brushed silver", "stainless", "nickel"],
    "gold_metal": ["brass", "bronze", "antique gold", "rose gold", "brushed gold"],
    "clear": ["clear", "transparent", "glass", "crystal"],
    "multi": ["multi", "multicolor", "multicolored", "rainbow", "assorted", "various"],
}

MATERIAL_GROUPS = {
    "light_wood": ["oak", "pine", "maple", "birch", "bamboo", "cedar", "rubberwood", "acacia", "ash wood", "beech", "poplar", "light wood"],
    "dark_wood": ["walnut", "mahogany", "teak", "cherry wood", "rosewood", "ebony wood", "dark wood", "espresso wood"],
    "wood": ["wood", "solid wood", "hardwood", "solid + manufactured wood"],
    "manufactured_wood": ["manufactured wood", "mdf", "particle board", "plywood", "engineered wood", "laminate", "wood veneer"],
    "metal": ["metal", "steel", "aluminum", "stainless steel", "zinc", "chrome", "tin"],
    "iron": ["iron", "wrought iron", "cast iron"],
    "brass_metal": ["brass", "bronze", "copper"],
    "velvet": ["velvet"],
    "linen": ["linen", "cotton", "100 % cotton", "canvas", "tweed", "burlap", "muslin"],
    "microfiber": ["polyester", "microfiber / polyester", "microfiber", "chenille", "satin", "silk", "nylon fabric"],
    "leather": ["leather", "genuine leather", "top grain", "full grain"],
    "faux_leather": ["faux leather", "bonded leather", "pu leather", "vegan leather", "leatherette"],
    "plastic": ["plastic", "resin/plastic", "resin", "acrylic", "polycarbonate", "abs", "polypropylene", "pvc", "vinyl"],
    "glass": ["glass", "tempered glass", "frosted glass", "mirror glass", "crystal glass"],
    "ceramic": ["ceramic", "porcelain", "terracotta", "earthenware", "stoneware"],
    "stone": ["marble", "granite", "quartz", "stone", "concrete", "slate", "travertine", "limestone", "terrazzo"],
    "natural_fiber": ["wool", "jute", "sisal", "seagrass", "rattan", "wicker", "cane", "hemp", "raffia"],
    "foam": ["foam", "memory foam", "gel foam", "polyurethane foam"],
    "synthetics": ["synthetics", "synthetic", "olefin", "polypropylene fiber"],
}

STYLE_GROUPS = {
    "modern": ["modern & contemporary", "modern", "contemporary", "minimalist"],
    "traditional": ["traditional", "classic"],
    "mid-century modern": ["mid-century modern", "mid century", "midcentury"],
    "farmhouse": ["farmhouse / country", "farmhouse", "country/cottage", "modern farmhouse", "country"],
    "rustic": ["rustic", "lodge", "cabin"],
    "industrial": ["industrial", "urban"],
    "coastal": ["coastal", "nautical", "beach", "tropical"],
    "glam": ["glam", "glamorous", "hollywood regency"],
    "bohemian": ["bohemian", "boho", "eclectic", "global inspired"],
    "transitional": ["transitional"],
    "scandinavian": ["scandinavian", "nordic"],
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


def _families_to_desc(families):
    parts = []
    for group, keywords in sorted(families.items()):
        kws = ", ".join(keywords[:6])
        if len(keywords) > 6:
            kws += ", ..."
        parts.append(f"{group}: {kws}")
    return "; ".join(parts)


VALID_COLORS = sorted(COLOR_FAMILIES.keys())
VALID_MATERIALS = sorted(MATERIAL_GROUPS.keys())
VALID_STYLES = sorted(STYLE_GROUPS.keys())
VALID_SHAPES = sorted(SHAPE_GROUPS.keys())
VALID_ASSEMBLIES = sorted(ASSEMBLY_GROUPS.keys())

EXTRACT_TOOL = {
    "name": "extract_product_attributes",
    "description": (
        "Extract visual and physical attributes for a product. "
        "For upholstered furniture, primary_material = visible surface "
        "(fabric/leather), NOT internal frame. "
        "Product name color WINS over feature data. "
        "Shape = product's actual form (round chair, L-shaped sectional). "
        "Omit shape for storage furniture, lighting, and plumbing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "primary_color": {
                "type": "string", "enum": VALID_COLORS,
                "description": "Dominant visible color. " + _families_to_desc(COLOR_FAMILIES),
            },
            "secondary_color": {
                "type": "string", "enum": VALID_COLORS,
                "description": "Secondary/accent color if present",
            },
            "primary_material": {
                "type": "string", "enum": VALID_MATERIALS,
                "description": "Dominant VISIBLE material (surface, not frame). " + _families_to_desc(MATERIAL_GROUPS),
            },
            "secondary_material": {
                "type": "string", "enum": VALID_MATERIALS,
                "description": "Frame/internal material if different from primary",
            },
            "style": {
                "type": "string", "enum": VALID_STYLES,
                "description": "Design style. " + _families_to_desc(STYLE_GROUPS),
            },
            "shape": {
                "type": "string", "enum": VALID_SHAPES,
                "description": "Product's physical form. " + _families_to_desc(SHAPE_GROUPS),
            },
            "assembly": {
                "type": "string", "enum": VALID_ASSEMBLIES,
                "description": "Assembly required. " + _families_to_desc(ASSEMBLY_GROUPS),
            },
        },
        "required": [],
    },
}

SYSTEM_PROMPT = (
    "You are a product attribute extractor. Use the extract_product_attributes "
    "tool to return attributes. Only include attributes you can confidently "
    "determine. For upholstered furniture (sofas, chairs), primary_material "
    "is the visible fabric/leather surface, not the wood frame inside."
)


def call_claude(client, product, model):
    """Single product extraction with tool use."""
    name = product.get("product_name", "")
    pclass = product.get("product_class", "")
    taxonomy = product.get("taxonomy", [])
    cat = " > ".join(taxonomy) if taxonomy else ""

    user_msg = f"Name: {name}\nClass: {pclass}\n"
    if cat:
        user_msg += f"Category: {cat}\n"

    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "extract_product_attributes"},
        )

        attrs = {k: None for k in ATTR_KEYS}
        for block in response.content:
            if block.type == "tool_use":
                for k in attrs:
                    if k in block.input and block.input[k]:
                        attrs[k] = block.input[k]
                break
        return attrs

    except Exception as e:
        print(f"    API error: {e}")
        return {k: None for k in ATTR_KEYS}


def main():
    parser = argparse.ArgumentParser(
        description="Run Claude on new queue products not yet processed")
    parser.add_argument("--queue", default="data/processed/image_queue.json")
    parser.add_argument("--progress", default="data/processed/claude_progress.json")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    client = Anthropic(api_key=args.api_key)

    with open(args.queue) as f:
        queue = json.load(f)

    progress = {}
    if os.path.exists(args.progress):
        with open(args.progress) as f:
            progress = json.load(f)

    remaining = [p for p in queue if str(p["product_id"]) not in progress]
    total = len(remaining)

    print(f"Queue: {len(queue)} products")
    print(f"Already done: {len(progress)}")
    print(f"Remaining: {total}")

    if total == 0:
        print("All done!")
        return

    print(f"Est. cost: ~${total * 0.0008:.2f} (Haiku) / ~${total * 0.0023:.2f} (Sonnet)")
    print(f"Model: {args.model}")
    print(f"Press Ctrl+C to stop (progress saved)\n")

    shutdown = False
    def handler(sig, frame):
        nonlocal shutdown
        if shutdown:
            sys.exit(1)
        shutdown = True
        print("\n  Stopping gracefully...")
    signal.signal(signal.SIGINT, handler)

    done = 0
    errors = 0
    t0 = time.time()

    for product in remaining:
        if shutdown:
            break

        pid = str(product["product_id"])
        attrs = call_claude(client, product, args.model)
        progress[pid] = attrs

        if all(v is None for v in attrs.values()):
            errors += 1
        done += 1

        # Save every 10 products
        if done % 10 == 0:
            with open(args.progress, "w") as f:
                json.dump(progress, f)

        time.sleep(0.3)

        if done % 50 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1) * 3600
            eta = (total - done) / max(rate / 3600, 0.001)
            print(f"  [{len(progress)}/{len(queue)}] "
                  f"{rate:.0f}/hr, ETA {eta:.1f}min, errors={errors}")

    # Final save progress
    with open(args.progress, "w") as f:
        json.dump(progress, f)
    print(f"\nDone: {done} this run, {len(progress)} total, {errors} errors")

    # Update queue in-place
    updated = 0
    for q in queue:
        pid = str(q["product_id"])
        if pid in progress:
            for attr in ATTR_KEYS:
                cv = progress[pid].get(attr)
                if cv is not None:
                    q[attr] = cv
                    if "_attr_sources" not in q:
                        q["_attr_sources"] = {}
                    q["_attr_sources"][attr] = "claude"
            updated += 1

    with open(args.queue, "w") as f:
        json.dump(queue, f, indent=2, default=str)
    print(f"Updated queue: {updated}/{len(queue)} products")
    print(f"Saved: {args.queue}")


if __name__ == "__main__":
    main()