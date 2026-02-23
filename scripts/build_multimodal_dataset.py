#!/usr/bin/env python3
"""Build the merged multimodal dataset: WANDS text + sourced images.

Creates train/val/test splits with both text-only and image+text examples.

Usage:
    python scripts/build_multimodal_dataset.py

This should be run AFTER:
    1. python scripts/download_wands.py
    2. python scripts/prepare_data.py
    3. python scripts/source_images.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from src.data.wands_loader import WANDSLoader
from src.data.feature_parser import process_products_dataframe, create_training_example
from src.utils.logger import setup_logger


def build_multimodal_dataset(
    wands_dir: str | Path = "data/raw/WANDS/dataset",
    images_dir: str | Path = "data/images",
    output_dir: str | Path = "data/processed",
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> dict:
    """Build merged dataset with text-only and multimodal examples.

    Products with matched images get an image_path field.
    Products without images have image_path = null.
    The model handles both cases — this is a feature, not a bug.

    Args:
        wands_dir: Path to WANDS dataset.
        images_dir: Path to sourced images directory.
        output_dir: Output directory for JSONL files.
        train_split: Training fraction.
        val_split: Validation fraction.
        seed: Random seed.

    Returns:
        Dataset statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load WANDS products with parsed features ──
    logger.info("Step 1: Loading and parsing WANDS products...")
    loader = WANDSLoader(wands_dir)
    products = loader.get_products_with_features()
    processed = process_products_dataframe(products)

    # ── Step 2: Load image metadata (if available) ──
    image_meta_path = Path(images_dir) / "image_metadata.jsonl"
    image_lookup = {}

    if image_meta_path.exists():
        logger.info("Step 2: Loading image metadata...")
        with open(image_meta_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    pid = record["product_id"]
                    img_path = record["image_path"]
                    # Verify image file actually exists
                    if Path(img_path).exists():
                        image_lookup[pid] = img_path
        logger.info(f"  Found {len(image_lookup)} products with valid images")
    else:
        logger.info("Step 2: No image metadata found — building text-only dataset")
        logger.info(f"  Run 'python scripts/source_images.py' first to add images")

    # ── Step 3: Create training examples ──
    logger.info("Step 3: Creating training examples...")
    examples = []

    for _, row in processed.iterrows():
        example = create_training_example(row)

        # Filter: need at least 2 non-null target attributes
        non_null = sum(
            1 for v in example["target_attributes"].values()
            if v is not None
        )
        if non_null < 2:
            continue

        # Add image path if available
        pid = row["product_id"]
        example["image_path"] = image_lookup.get(pid, None)
        example["has_image"] = example["image_path"] is not None

        examples.append(example)

    total = len(examples)
    with_images = sum(1 for e in examples if e["has_image"])
    text_only = total - with_images

    logger.info(f"  Total valid examples: {total}")
    logger.info(f"  With images:          {with_images} ({with_images/total*100:.1f}%)")
    logger.info(f"  Text-only:            {text_only} ({text_only/total*100:.1f}%)")

    # ── Step 4: Stratified split ──
    # Ensure image examples are distributed across splits proportionally
    logger.info("Step 4: Splitting into train/val/test...")
    rng = np.random.RandomState(seed)

    # Separate image and text-only, shuffle each, then merge
    img_examples = [e for e in examples if e["has_image"]]
    txt_examples = [e for e in examples if not e["has_image"]]
    rng.shuffle(img_examples)
    rng.shuffle(txt_examples)

    def split_list(lst, train_frac, val_frac):
        n = len(lst)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        return lst[:t1], lst[t1:t2], lst[t2:]

    img_train, img_val, img_test = split_list(img_examples, train_split, val_split)
    txt_train, txt_val, txt_test = split_list(txt_examples, train_split, val_split)

    # Merge and shuffle within each split
    train = img_train + txt_train
    val = img_val + txt_val
    test = img_test + txt_test
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # ── Step 5: Save ──
    logger.info("Step 5: Saving splits...")
    splits = {"train": train, "val": val, "test": test}

    for split_name, split_data in splits.items():
        filepath = output_dir / f"{split_name}.jsonl"
        with open(filepath, "w") as f:
            for example in split_data:
                f.write(json.dumps(example, default=str) + "\n")

        n_img = sum(1 for e in split_data if e["has_image"])
        logger.info(
            f"  {split_name}: {len(split_data)} examples "
            f"({n_img} with images, {len(split_data) - n_img} text-only) → {filepath}"
        )

    # ── Step 6: Save dataset card ──
    stats = {
        "total_examples": total,
        "with_images": with_images,
        "text_only": text_only,
        "image_coverage": round(with_images / total * 100, 1),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_images": sum(1 for e in train if e["has_image"]),
        "val_images": sum(1 for e in val if e["has_image"]),
        "test_images": sum(1 for e in test if e["has_image"]),
        "wands_source": str(wands_dir),
        "images_source": str(images_dir),
        "seed": seed,
    }

    card_path = output_dir / "dataset_card.json"
    with open(card_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("MULTIMODAL DATASET BUILD COMPLETE")
    logger.info(f"  Total:    {total} examples")
    logger.info(f"  Images:   {with_images} ({stats['image_coverage']}% coverage)")
    logger.info(f"  Train:    {len(train)} ({stats['train_images']} with images)")
    logger.info(f"  Val:      {len(val)} ({stats['val_images']} with images)")
    logger.info(f"  Test:     {len(test)} ({stats['test_images']} with images)")
    logger.info("=" * 60)

    return stats


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Build multimodal dataset")
    parser.add_argument("--wands-dir", type=str, default="data/raw/WANDS/dataset")
    parser.add_argument("--images-dir", type=str, default="data/images")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_multimodal_dataset(
        wands_dir=args.wands_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
