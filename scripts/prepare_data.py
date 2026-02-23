#!/usr/bin/env python3
"""Prepare WANDS data + sourced images into train/val/test splits.

If images have been sourced (via scripts/source_images.py), they are
automatically paired with their WANDS products. Otherwise, builds
a text-only dataset.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --images-dir data/images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger

from src.data.wands_loader import WANDSLoader
from src.data.feature_parser import process_products_dataframe, create_training_example
from src.utils.logger import setup_logger


def prepare_data(
    wands_dir: str | Path = "data/raw/WANDS/dataset",
    images_dir: str | Path = "data/images",
    output_dir: str | Path = "data/processed",
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load WANDS ──
    logger.info("Loading WANDS dataset...")
    loader = WANDSLoader(wands_dir)
    products = loader.get_products_with_features()
    processed = process_products_dataframe(products)

    # ── Load image metadata if available ──
    image_meta_path = Path(images_dir) / "image_metadata.jsonl"
    image_lookup = {}
    if image_meta_path.exists():
        logger.info("Loading sourced image metadata...")
        with open(image_meta_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    pid = rec["product_id"]
                    ip = rec["image_path"]
                    if Path(ip).exists():
                        image_lookup[pid] = ip
        logger.info(f"Found {len(image_lookup)} products with images")
    else:
        logger.info(
            "No image metadata found. Building text-only dataset.\n"
            "  To add images: python scripts/source_images.py --num-products 3000"
        )

    # ── Create examples ──
    logger.info("Creating training examples...")
    examples = []
    for _, row in processed.iterrows():
        example = create_training_example(row)
        non_null = sum(1 for v in example["target_attributes"].values() if v is not None)
        if non_null < 2:
            continue
        pid = row["product_id"]
        example["image_path"] = image_lookup.get(pid, None)
        example["has_image"] = example["image_path"] is not None
        examples.append(example)

    total = len(examples)
    n_img = sum(1 for e in examples if e["has_image"])
    logger.info(f"Valid examples: {total} ({n_img} with images, {total-n_img} text-only)")

    # ── Stratified split (images distributed proportionally) ──
    rng = np.random.RandomState(seed)
    img_ex = [e for e in examples if e["has_image"]]
    txt_ex = [e for e in examples if not e["has_image"]]
    rng.shuffle(img_ex)
    rng.shuffle(txt_ex)

    def split3(lst):
        n = len(lst)
        t1, t2 = int(n * train_split), int(n * (train_split + val_split))
        return lst[:t1], lst[t1:t2], lst[t2:]

    img_tr, img_va, img_te = split3(img_ex)
    txt_tr, txt_va, txt_te = split3(txt_ex)

    splits = {
        "train": img_tr + txt_tr,
        "val": img_va + txt_va,
        "test": img_te + txt_te,
    }
    for s in splits.values():
        rng.shuffle(s)

    # ── Save ──
    for name, data in splits.items():
        fp = output_dir / f"{name}.jsonl"
        with open(fp, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, default=str) + "\n")
        n_i = sum(1 for e in data if e["has_image"])
        logger.info(f"  {name}: {len(data)} examples ({n_i} with images) → {fp}")

    # ── Attribute stats ──
    from collections import Counter
    stats = {}
    for attr in ["style", "primary_material", "color_family", "product_type"]:
        vals = [e["target_attributes"].get(attr) for e in splits["train"]
                if e["target_attributes"].get(attr) is not None]
        c = Counter(vals)
        stats[attr] = {"unique": len(c), "total": len(vals), "top_10": c.most_common(10)}
    with open(output_dir / "attribute_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    summary = {
        "total_examples": total,
        "with_images": n_img,
        "text_only": total - n_img,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }

    with open(output_dir / "dataset_card.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    logger.info("DATA PREPARATION COMPLETE")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)
    return summary


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Prepare WANDS + images dataset")
    parser.add_argument("--wands-dir", type=str, default="data/raw/WANDS/dataset")
    parser.add_argument("--images-dir", type=str, default="data/images")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_data(args.wands_dir, args.images_dir, args.output_dir, args.train_split, seed=args.seed)


if __name__ == "__main__":
    main()
