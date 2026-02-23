#!/usr/bin/env python3
"""Strip color/material/style words from text so model must learn from images."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
ALL_STRIP = COLOR_WORDS | MATERIAL_WORDS | STYLE_WORDS

def make_vague(text):
    if not text or not isinstance(text, str):
        return text or ""
    words = text.split()
    filtered = [w for w in words if re.sub(r'[^a-zA-Z]','',w).lower() not in ALL_STRIP]
    result = " ".join(filtered)
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r'^\s*,|,\s*$', '', result)
    return result

for split in ["train", "val", "test"]:
    src = Path(f"data/processed/{split}_multimodal.jsonl")
    dst = Path(f"data/processed/{split}_vague.jsonl")
    if not src.exists():
        print(f"Skipping {split}: {src} not found")
        continue
    lines = []
    with open(src) as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            ex["input_text_original"] = ex.get("input_text", "")
            ex["input_text"] = make_vague(ex.get("input_text", ""))
            lines.append(json.dumps(ex, default=str))
    with open(dst, "w") as f:
        f.write("\n".join(lines) + "\n")
    # Show example
    orig = json.loads(open(src).readline())
    vague = json.loads(open(dst).readline())
    print(f"\n{split}: {len(lines)} examples -> {dst}")
    print(f"  ORIGINAL: {orig.get('input_text','')[:120]}")
    print(f"  VAGUE:    {vague.get('input_text','')[:120]}")
