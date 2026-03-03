"""Product Classifier Dataset with Smart Per-Attribute Masking.

Training strategy:
  IMAGE: always present (user always provides)
  TEXT:  varies to simulate production reality

Text augmentation levels:
  15% full     (name + class + description)
  15% partial  (name + class)
  30% name     (product name only)
  20% minimal  (1-2 words from name)
  20% empty    (image only — but assembly/material appended back)

Per-attribute masking (applied on top):
  COLOR words:    50% masked  → force image learning
  SHAPE words:    40% masked  → force image learning
  MATERIAL words: 15% masked  → image primary, text backup
  STYLE words:    20% masked  → both modalities
  ASSEMBLY words: NEVER masked → always in text

Key: when text is degraded, assembly + material phrases are
preserved/appended back from description. Model learns:
  color/shape    → from IMAGE (text rarely has it)
  material       → from IMAGE primarily, TEXT helps
  assembly       → from TEXT always
  style          → from BOTH
  taxonomy       → from BOTH
"""
import json
import os
import re
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# ════════════════════════════════════════════════════════════════
# Attribute word sets
# ════════════════════════════════════════════════════════════════

COLOR_WORDS = {
    "white", "ivory", "cream", "snow", "pearl", "alabaster", "eggshell",
    "black", "jet", "onyx", "ebony",
    "gray", "grey", "charcoal", "graphite", "pewter", "smoke", "heather",
    "brown", "chocolate", "espresso", "mocha", "cocoa", "coffee", "umber",
    "beige", "tan", "khaki", "sand", "taupe", "camel", "natural", "oatmeal",
    "wheat", "buff", "mushroom", "fawn", "parchment", "champagne", "bone", "ecru",
    "blue", "navy", "teal", "turquoise", "aqua", "cobalt", "sapphire",
    "indigo", "denim", "cerulean", "periwinkle", "cornflower",
    "green", "sage", "olive", "emerald", "mint", "lime", "seafoam",
    "jade", "moss", "fern", "pistachio",
    "red", "burgundy", "crimson", "maroon", "scarlet", "wine", "ruby",
    "brick", "cranberry", "garnet", "oxblood",
    "pink", "blush", "rose", "coral", "salmon", "fuchsia", "magenta", "mauve",
    "orange", "rust", "terracotta", "peach", "apricot", "tangerine", "amber",
    "copper", "cinnamon",
    "yellow", "gold", "mustard", "lemon", "honey", "golden", "sunflower",
    "purple", "lavender", "violet", "plum", "lilac", "eggplant", "amethyst",
    "orchid", "grape",
    "silver", "chrome", "platinum", "nickel",
    "brass", "bronze",
    "clear", "transparent",
    "multi", "multicolor", "multicolored", "rainbow",
}

MATERIAL_WORDS = {
    "wood", "wooden", "hardwood", "plywood", "mdf", "oak", "pine",
    "bamboo", "teak", "cedar", "birch", "maple", "acacia", "rubberwood",
    "metal", "steel", "iron", "aluminum", "stainless", "chrome", "zinc",
    "brass", "bronze", "wrought",
    "fabric", "polyester", "cotton", "linen", "microfiber", "chenille",
    "tweed", "canvas", "satin", "silk", "velvet", "upholstered",
    "leather", "faux", "bonded", "vegan",
    "plastic", "resin", "acrylic", "polycarbonate", "vinyl", "pvc",
    "marble", "granite", "quartz", "stone", "concrete", "slate",
    "travertine", "limestone", "terrazzo",
    "ceramic", "porcelain", "terracotta", "earthenware", "stoneware",
    "glass", "tempered", "frosted", "mirror", "crystal",
    "wool", "jute", "sisal", "seagrass", "rattan", "wicker", "cane", "hemp",
    "foam", "memory", "gel",
    "synthetic", "nylon", "olefin",
}

SHAPE_WORDS = {
    "rectangular", "rectangle", "square", "round", "circular", "circle",
    "oval", "oblong", "l-shaped", "l-shape", "u-shaped", "u-shape",
    "runner", "irregular", "hexagon", "hexagonal", "freeform",
}

STYLE_WORDS = {
    "modern", "contemporary", "traditional", "transitional", "rustic",
    "industrial", "coastal", "farmhouse", "mid-century", "bohemian",
    "boho", "glam", "scandinavian", "cottage", "tropical", "minimalist",
    "retro", "vintage", "classic", "eclectic",
}

# Assembly phrases to extract and preserve
ASSEMBLY_PATTERNS = [
    r'(?:assembly|assemble)\s*(?:is\s+)?(?:required|needed|necessary)',
    r'(?:no|not?)\s*(?:assembly|assemble)\s*(?:required|needed)?',
    r'(?:fully|pre[\-\s]?)assembled',
    r'(?:partial|some)\s*assembly',
    r'(?:easy|simple|minimal)\s*assembly',
    r'(?:tools?\s*(?:included|required|needed))',
    r'assembly\s*(?:instructions?\s*)?included',
    r'arrives?\s*(?:fully\s*)?assembled',
    r'ready\s*to\s*(?:use|assemble)',
]

# Material phrases to extract and preserve
MATERIAL_PATTERNS = [
    r'(?:made|crafted|constructed|built)\s+(?:of|from|with)\s+(\w+(?:\s+\w+)?)',
    r'(?:solid|real|genuine|faux|bonded)\s+(?:wood|leather|marble|stone|metal)',
    r'(?:stainless\s+steel|wrought\s+iron|cast\s+iron|solid\s+wood)',
    r'(?:memory\s+foam|tempered\s+glass)',
]


def get_image_transforms(train=True, size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]),
        ])


def mask_words(text, word_set, mask_token="[MASK]"):
    """Replace attribute words in text with [MASK]."""
    words = text.split()
    masked = []
    for w in words:
        clean = re.sub(r'[^a-z-]', '', w.lower())
        if clean in word_set:
            masked.append(mask_token)
        else:
            masked.append(w)
    return " ".join(masked)


def extract_preserved_phrases(description):
    """Extract assembly + material phrases that should always be kept."""
    if not description:
        return ""

    desc_lower = description.lower()
    preserved = []

    # Extract assembly phrases
    for pattern in ASSEMBLY_PATTERNS:
        matches = re.findall(pattern, desc_lower)
        preserved.extend(matches)

    # Extract material phrases
    for pattern in MATERIAL_PATTERNS:
        matches = re.findall(pattern, desc_lower)
        if matches:
            for m in matches:
                if isinstance(m, tuple):
                    preserved.extend(m)
                else:
                    preserved.append(m)

    # Deduplicate and clean
    seen = set()
    clean = []
    for phrase in preserved:
        phrase = phrase.strip()
        if phrase and phrase not in seen:
            seen.add(phrase)
            clean.append(phrase)

    return ", ".join(clean)


class ProductClassifierDataset(Dataset):
    """Dataset with smart per-attribute masking."""

    ATTR_KEYS = [
        "primary_color", "secondary_color",
        "primary_material", "secondary_material",
        "style", "shape", "assembly",
    ]

    def __init__(
        self,
        queue_path,
        image_dir,
        vocab_path,
        taxonomy_path,
        description_map=None,
        train=True,
        max_images=2,
        image_size=224,
        # Text level probabilities
        text_full_prob=0.15,
        text_partial_prob=0.15,
        text_name_prob=0.30,
        text_minimal_prob=0.20,
        text_empty_prob=0.20,
        # Per-attribute mask probabilities
        color_mask_prob=0.50,
        shape_mask_prob=0.40,
        style_mask_prob=0.20,
        material_mask_prob=0.15,
        # assembly_mask_prob = 0.0 (NEVER)
    ):
        super().__init__()
        self.train = train
        self.max_images = max_images
        self.image_dir = Path(image_dir)

        self.text_probs = {
            "full": text_full_prob,
            "partial": text_partial_prob,
            "name": text_name_prob,
            "minimal": text_minimal_prob,
            "empty": text_empty_prob,
        }
        self.color_mask_prob = color_mask_prob
        self.shape_mask_prob = shape_mask_prob
        self.style_mask_prob = style_mask_prob
        self.material_mask_prob = material_mask_prob

        with open(queue_path) as f:
            self.products = json.load(f)
        self.desc_map = description_map or {}

        # Attribute vocab
        with open(vocab_path) as f:
            self.attr_vocab = json.load(f)
        self.attr_v2i, self.attr_i2v = {}, {}
        for attr_name, values in self.attr_vocab.items():
            v2i = {"<UNK>": 0}
            i2v = {0: "<UNK>"}
            for i, val in enumerate(values):
                v2i[val] = i + 1
                i2v[i + 1] = val
            self.attr_v2i[attr_name] = v2i
            self.attr_i2v[attr_name] = i2v

        # Taxonomy
        with open(taxonomy_path) as f:
            self.taxonomy = json.load(f)
        self.level_v2i, self.level_i2v = {}, {}
        for lk, values in self.taxonomy.get("level_values", {}).items():
            v2i = {"<UNK>": 0}
            i2v = {0: "<UNK>"}
            for i, val in enumerate(values):
                v2i[val] = i + 1
                i2v[i + 1] = val
            self.level_v2i[lk] = v2i
            self.level_i2v[lk] = i2v
        self.max_depth = self.taxonomy.get("max_depth", 5)

        # Product class
        self.class_v2i, self.class_i2v = {}, {}
        class_values = self.taxonomy.get("product_classes", [])
        if class_values:
            self.class_v2i = {"<UNK>": 0}
            self.class_i2v = {0: "<UNK>"}
            for i, val in enumerate(class_values):
                self.class_v2i[val] = i + 1
                self.class_i2v[i + 1] = val

        self.img_transform = get_image_transforms(train=train, size=image_size)
        self._index_images()
        print(f"Dataset: {len(self.products)} products, "
              f"{self.n_with_images} with images, train={train}")

    def _index_images(self):
        self.n_with_images = 0
        for p in self.products:
            pid = p["product_id"]
            pdir = self.image_dir / pid
            hero = next(
                (pdir / f"hero.{e}" for e in ("jpg", "png", "webp")
                 if (pdir / f"hero.{e}").exists()), None)
            crop = next(
                (pdir / f"material_crop.{e}" for e in ("jpg", "png", "webp")
                 if (pdir / f"material_crop.{e}").exists()), None)
            p["_hero"] = str(hero) if hero else None
            p["_crop"] = str(crop) if crop else None
            p["_has_img"] = hero is not None
            if hero:
                self.n_with_images += 1

    def __len__(self):
        return len(self.products)

    def _load_image(self, path):
        try:
            return self.img_transform(Image.open(path).convert("RGB"))
        except Exception:
            return None

    def _get_description(self, product):
        """Get description, handling multiple ID formats."""
        pid = str(product["product_id"])
        desc = self.desc_map.get(pid, "")
        if not desc or str(desc) == "nan":
            return ""
        return str(desc)[:400]

    def _build_text(self, product):
        """Build text with smart per-attribute masking."""
        name = product["product_name"]
        cls = product.get("product_class", "")
        desc = self._get_description(product)

        if not self.train:
            # Validation: full text, no masking
            parts = [name]
            if cls:
                parts.append(cls)
            if desc:
                parts.append(desc)
            return " [SEP] ".join(parts)

        # === Extract preserved phrases (assembly + material) ===
        preserved = extract_preserved_phrases(desc)

        # === Sample text richness level ===
        r = random.random()
        cumulative = 0
        level = "full"
        for lv, prob in self.text_probs.items():
            cumulative += prob
            if r < cumulative:
                level = lv
                break

        if level == "full":
            parts = [name]
            if cls:
                parts.append(cls)
            if desc:
                parts.append(desc)
            text = " [SEP] ".join(parts)

        elif level == "partial":
            text = f"{name} [SEP] {cls}" if cls else name
            # Append preserved phrases
            if preserved:
                text = f"{text} [SEP] {preserved}"

        elif level == "name":
            text = name
            # Append preserved phrases
            if preserved:
                text = f"{text} [SEP] {preserved}"

        elif level == "minimal":
            words = name.split()
            text = " ".join(words[:max(1, len(words) // 3)])
            # Append preserved phrases
            if preserved:
                text = f"{text} [SEP] {preserved}"

        else:  # empty
            # Even with "empty" text, keep assembly + material phrases
            text = preserved if preserved else ""

        # === Targeted attribute masking ===
        if text:
            # Color: mask aggressively (visual attribute)
            if random.random() < self.color_mask_prob:
                text = mask_words(text, COLOR_WORDS)

            # Shape: mask aggressively (visual attribute)
            if random.random() < self.shape_mask_prob:
                text = mask_words(text, SHAPE_WORDS)

            # Style: mask moderately (both modalities)
            if random.random() < self.style_mask_prob:
                text = mask_words(text, STYLE_WORDS)

            # Material: mask lightly (image primary, text backup)
            if random.random() < self.material_mask_prob:
                text = mask_words(text, MATERIAL_WORDS)

            # Assembly: NEVER masked

        return text

    def _build_images(self, product):
        """Load images — always present in production."""
        images = []
        image_mask = []

        if product.get("_has_img"):
            hero = self._load_image(product["_hero"])
            if hero is not None:
                images.append(hero)
                image_mask.append(True)

            if self.max_images > 1 and product.get("_crop"):
                crop = self._load_image(product["_crop"])
                if crop is not None:
                    images.append(crop)
                    image_mask.append(True)

        # Pad to max_images
        while len(images) < self.max_images:
            images.append(torch.zeros(3, 224, 224))
            image_mask.append(False)

        return (torch.stack(images[:self.max_images]),
                torch.tensor(image_mask[:self.max_images], dtype=torch.bool))

    def __getitem__(self, idx):
        product = self.products[idx]
        text = self._build_text(product)
        images, image_mask = self._build_images(product)

        # Taxonomy labels
        taxonomy = product.get("taxonomy", [])
        tax_labels = {}
        for d in range(1, self.max_depth + 1):
            lk = f"level_{d}"
            if d <= len(taxonomy):
                tax_labels[lk] = self.level_v2i.get(lk, {}).get(
                    taxonomy[d - 1], 0)
            else:
                tax_labels[lk] = -1

        # Product class label
        pc = product.get("product_class")
        pc_label = self.class_v2i.get(pc, 0) if pc and self.class_v2i else -1

        # Attribute labels
        attr_labels = {}
        for attr in self.ATTR_KEYS:
            val = product.get(attr)
            if val is None:
                attr_labels[attr] = -1
            else:
                attr_labels[attr] = self.attr_v2i.get(attr, {}).get(val, 0)

        result = {
            "product_id": product["product_id"],
            "text_input": text,
            "images": images,
            "image_mask": image_mask,
            "product_class": torch.tensor(pc_label, dtype=torch.long),
        }
        for k, v in tax_labels.items():
            result[f"tax_{k}"] = torch.tensor(v, dtype=torch.long)
        for k, v in attr_labels.items():
            result[f"attr_{k}"] = torch.tensor(v, dtype=torch.long)

        return result


def collate_fn(batch):
    c = {}
    c["product_id"] = [b["product_id"] for b in batch]
    c["text_input"] = [b["text_input"] for b in batch]
    c["images"] = torch.stack([b["images"] for b in batch])
    c["image_mask"] = torch.stack([b["image_mask"] for b in batch])
    c["product_class"] = torch.stack([b["product_class"] for b in batch])
    for key in batch[0]:
        if key.startswith("tax_") or key.startswith("attr_"):
            c[key] = torch.stack([b[key] for b in batch])
    return c


def load_descriptions(tsv_path):
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False,
                     usecols=["product_id", "product_description"])
    desc_map = {}
    for _, row in df.iterrows():
        pid = str(row["product_id"])
        desc = row.get("product_description", "")
        if pd.notna(desc) and str(desc) != "nan":
            desc_map[pid] = str(desc)
    print(f"Loaded descriptions for {len(desc_map)} products")
    return desc_map


def build_dataloaders(
    queue_path, image_dir, vocab_path, taxonomy_path,
    tsv_path=None, batch_size=32, val_split=0.1, num_workers=4,
    **kwargs,
):
    from torch.utils.data import DataLoader, Subset

    desc_map = {}
    if tsv_path and os.path.exists(tsv_path):
        desc_map = load_descriptions(tsv_path)

    full = ProductClassifierDataset(
        queue_path, image_dir, vocab_path, taxonomy_path,
        description_map=desc_map, train=True, **kwargs)

    n = len(full)
    idx = list(range(n))
    random.shuffle(idx)
    vs = int(n * val_split)

    train_loader = DataLoader(
        Subset(full, idx[vs:]), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True)

    val_ds = ProductClassifierDataset(
        queue_path, image_dir, vocab_path, taxonomy_path,
        description_map=desc_map, train=False,
        max_images=kwargs.get("max_images", 2),
        image_size=kwargs.get("image_size", 224))

    val_loader = DataLoader(
        Subset(val_ds, idx[:vs]), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    print(f"Train: {n - vs}, Val: {vs}, Batch: {batch_size}")
    return train_loader, val_loader