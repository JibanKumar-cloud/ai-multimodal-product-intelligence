#!/usr/bin/env python3
"""Create training data where some text deliberately CONFLICTS with image.

For visual attributes (color, material, style), we randomly replace
the correct word with a WRONG one. Ground truth stays aligned with IMAGE.
This forces the model to prioritize image over text.

Mix: 50% vague (stripped), 30% adversarial (wrong text), 20% original (correct text)
"""

import json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
random.seed(42)

COLORS = ["red", "blue", "green", "yellow", "black", "white", "gray",
          "brown", "beige", "navy", "teal", "orange", "purple", "pink"]

MATERIALS = ["wood", "metal", "leather", "velvet", "fabric", "glass",
             "ceramic", "marble", "plastic", "cotton", "polyester", "steel"]

STYLES = ["modern", "traditional", "rustic", "industrial", "farmhouse",
          "coastal", "bohemian", "scandinavian", "minimalist", "vintage"]

ROOM_WORDS = {
    "bedroom", "living", "dining", "outdoor", "kitchen", "bathroom",
    "office", "patio", "nursery", "entryway", "hallway", "lounge",
}

# Same strip words as create_vague_text.py
COLOR_WORDS = {
    "red","blue","green","yellow","orange","purple","pink","black",
    "white","gray","grey","brown","beige","ivory","cream","navy",
    "teal","turquoise","gold","silver","bronze","copper","walnut",
    "espresso","mahogany","oak","cherry","maple","tan","charcoal",
    "coral","burgundy","maroon","olive","aqua","emerald","ruby",
    "sapphire","amber","caramel","mocha","slate","pewter","nickel",
    "chrome","brushed","matte","glossy","satin",
}
MATERIAL_WORDS = {
    "wood","wooden","metal","steel","iron","aluminum","brass",
    "velvet","leather","fabric","cotton","polyester","linen",
    "silk","microfiber","suede","vinyl","acrylic","glass",
    "ceramic","porcelain","marble","granite","stone","concrete",
    "bamboo","rattan","wicker","plastic","resin","foam",
    "plywood","particleboard","mdf","hardwood","softwood",
    "teak","pine","birch","ash","rubber","stainless",
    "wrought","cast","forged","woven","knitted","upholstered",
    "tufted","nailhead","bonded","faux","genuine","solid",
    "manufactured","engineered","reclaimed",
}
STYLE_WORDS = {
    "modern","contemporary","traditional","rustic","industrial",
    "farmhouse","coastal","bohemian","scandinavian","mid-century",
    "minimalist","vintage","retro","classic","transitional",
    "glam","artisan","craftsman","cottage","french","country",
    "shabby","chic","elegant","luxurious","ornate","sleek",
}
ALL_STRIP = COLOR_WORDS | MATERIAL_WORDS | STYLE_WORDS | ROOM_WORDS


def make_vague(text):
    if not text or not isinstance(text, str):
        return text or ""
    words = text.split()
    filtered = [w for w in words if re.sub(r'[^a-zA-Z]','',w).lower() not in ALL_STRIP]
    result = re.sub(r'\s+', ' ', " ".join(filtered)).strip()
    result = re.sub(r',\s*,', ',', result)
    return re.sub(r'^\s*,|,\s*$', '', result)


def make_adversarial(text, target):
    """Replace correct visual words with WRONG ones."""
    if not text or not isinstance(text, str):
        return text or ""

    result = text

    # Replace color with wrong color
    gt_color = str(target.get("color_family", "") or "").lower()
    if gt_color:
        wrong_colors = [c for c in COLORS if c != gt_color and c not in gt_color]
        if wrong_colors:
            wrong = random.choice(wrong_colors)
            for c in COLORS:
                if c.lower() in result.lower():
                    result = re.sub(re.escape(c), wrong, result, flags=re.IGNORECASE)
                    break

    # Replace material with wrong material
    gt_mat = str(target.get("primary_material", "") or "").lower()
    if gt_mat:
        wrong_mats = [m for m in MATERIALS if m != gt_mat and m not in gt_mat]
        if wrong_mats:
            wrong = random.choice(wrong_mats)
            for m in MATERIALS:
                if m.lower() in result.lower():
                    result = re.sub(re.escape(m), wrong, result, flags=re.IGNORECASE)
                    break

    # Replace style with wrong style
    gt_style = str(target.get("style", "") or "").lower()
    if gt_style:
        wrong_styles = [s for s in STYLES if s != gt_style and s not in gt_style]
        if wrong_styles:
            wrong = random.choice(wrong_styles)
            for s in STYLES:
                if s.lower() in result.lower():
                    result = re.sub(re.escape(s), wrong, result, flags=re.IGNORECASE)
                    break

    return result


for split in ["train", "val", "test"]:
    src = Path(f"data/processed/{split}_multimodal.jsonl")
    dst = Path(f"data/processed/{split}_vague.jsonl")
    if not src.exists():
        print(f"Skipping {split}: {src} not found")
        continue

    examples = []
    with open(src) as f:
        for line in f:
            if not line.strip(): continue
            examples.append(json.loads(line))

    output = []
    stats = {"vague": 0, "adversarial": 0, "original": 0}

    for ex in examples:
        original_text = ex.get("input_text", "")
        target = ex.get("target_attributes", {})
        roll = random.random()

        if roll < 0.50:
            # 50% vague — stripped text
            ex["input_text_original"] = original_text
            ex["input_text"] = make_vague(original_text)
            stats["vague"] += 1
        elif roll < 0.80:
            # 30% adversarial — wrong visual words, correct GT from image
            ex["input_text_original"] = original_text
            ex["input_text"] = make_adversarial(original_text, target)
            stats["adversarial"] += 1
        else:
            # 20% original — correct text (model also sees agreement)
            ex["input_text_original"] = original_text
            stats["original"] += 1

        output.append(json.dumps(ex, default=str))

    with open(dst, "w") as f:
        f.write("\n".join(output) + "\n")

    print(f"\n{split}: {len(output)} examples -> {dst}")
    print(f"  Vague:       {stats['vague']} ({stats['vague']*100/len(output):.0f}%)")
    print(f"  Adversarial: {stats['adversarial']} ({stats['adversarial']*100/len(output):.0f}%)")
    print(f"  Original:    {stats['original']} ({stats['original']*100/len(output):.0f}%)")

    # Show one adversarial example
    for ex_str in output:
        ex = json.loads(ex_str)
        if ex.get("input_text") != ex.get("input_text_original") and \
           ex.get("input_text") != make_vague(ex.get("input_text_original","")):
            print(f"\n  ADVERSARIAL EXAMPLE:")
            print(f"  Original: {ex['input_text_original'][:120]}")
            print(f"  Modified: {ex['input_text'][:120]}")
            print(f"  GT stays: {json.dumps(ex.get('target_attributes',{}))[:120]}")
            break
