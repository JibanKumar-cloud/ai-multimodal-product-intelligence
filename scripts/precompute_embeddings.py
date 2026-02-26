"""Precompute frozen encoder embeddings for all WANDS products.

Run this ONCE before training. Caches:
  - Image embeddings: {product_id}_img.npy  [N_images, 768]
  - Text embeddings:  {product_id}_txt.npy  [768]
  - Attr indices:     {product_id}_attr.npy [num_keys]

After caching, training only runs the small trainable layers (~5 min).

Usage:
    python scripts/precompute_embeddings.py
    python scripts/precompute_embeddings.py --text-only  # no images
    python scripts/precompute_embeddings.py --batch-size 64
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.image_encoder import ImageEncoder
from src.classifier.text_encoder import TextEncoder
from src.classifier.attribute_encoder import AttributeEncoder
from src.classifier.dataset import build_taxonomy


def precompute_text(products_df, output_dir, batch_size=64):
    """Precompute all text embeddings."""
    encoder = TextEncoder(freeze=True)
    os.makedirs(output_dir, exist_ok=True)

    # Check what's already done
    done = set()
    for f in os.listdir(output_dir):
        if f.endswith("_txt.npy"):
            done.add(f.replace("_txt.npy", ""))

    remaining = products_df[
        ~products_df["product_id"].astype(str).isin(done)
    ]
    logger.info(f"Text: {len(done)} cached, {len(remaining)} remaining")

    if len(remaining) == 0:
        return

    # Process in batches
    texts = []
    pids = []
    for _, row in remaining.iterrows():
        text = TextEncoder.build_text(
            product_name=str(row.get("product_name", "")),
            product_description=str(row.get("product_description", "")),
            product_class=str(row.get("product_class", "")),
        )
        texts.append(text)
        pids.append(str(row["product_id"]))

    for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_pids = pids[i:i + batch_size]
        embeddings = encoder.encode_batch(batch_texts)

        for pid, emb in zip(batch_pids, embeddings):
            np.save(os.path.join(output_dir, f"{pid}_txt.npy"), emb)

    logger.info(f"Text embeddings done: {len(texts)} products")


def precompute_images(products_df, output_dir, image_manifest_path,
                      batch_size=32):
    """Precompute all image embeddings."""
    encoder = ImageEncoder(freeze=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load image manifest
    if not os.path.exists(image_manifest_path):
        logger.warning(f"No image manifest at {image_manifest_path}")
        logger.info("Creating empty image embeddings for all products")
        for _, row in products_df.iterrows():
            pid = str(row["product_id"])
            save_path = os.path.join(output_dir, f"{pid}_img.npy")
            if not os.path.exists(save_path):
                np.save(save_path, np.zeros((0, 768), dtype=np.float32))
        return

    with open(image_manifest_path) as f:
        manifest = json.load(f)

    done = set()
    for f in os.listdir(output_dir):
        if f.endswith("_img.npy"):
            done.add(f.replace("_img.npy", ""))

    remaining_pids = [
        str(row["product_id"]) for _, row in products_df.iterrows()
        if str(row["product_id"]) not in done
    ]
    logger.info(f"Images: {len(done)} cached, {len(remaining_pids)} remaining")

    from PIL import Image

    for pid in tqdm(remaining_pids, desc="Image embeddings"):
        save_path = os.path.join(output_dir, f"{pid}_img.npy")

        entry = manifest.get(pid, {})
        image_paths = entry.get("image_paths", [])

        if not image_paths:
            np.save(save_path, np.zeros((0, 768), dtype=np.float32))
            continue

        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                continue

        if images:
            embeddings = encoder.encode_images(images)
        else:
            embeddings = np.zeros((0, 768), dtype=np.float32)

        np.save(save_path, embeddings)

    logger.info(f"Image embeddings done")


def precompute_attributes(products_df, output_dir):
    """Precompute attribute indices."""
    encoder = AttributeEncoder()
    os.makedirs(output_dir, exist_ok=True)

    done = set()
    for f in os.listdir(output_dir):
        if f.endswith("_attr.npy"):
            done.add(f.replace("_attr.npy", ""))

    remaining = products_df[
        ~products_df["product_id"].astype(str).isin(done)
    ]
    logger.info(f"Attrs: {len(done)} cached, {len(remaining)} remaining")

    for _, row in tqdm(remaining.iterrows(), total=len(remaining),
                       desc="Attribute encoding"):
        pid = str(row["product_id"])
        feature_str = str(row.get("product_features", ""))
        parsed = encoder.parse_features(feature_str)
        indices = encoder.encode_features(parsed)
        np.save(os.path.join(output_dir, f"{pid}_attr.npy"), indices)

    logger.info(f"Attribute encoding done")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for classifier training")
    parser.add_argument("--data-path",
                        default="data/raw/WANDS/dataset/product.csv")
    parser.add_argument("--output-dir", default="data/embeddings")
    parser.add_argument("--image-manifest",
                        default="data/images/manifest.json")
    parser.add_argument("--text-only", action="store_true",
                        help="Skip image encoding")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Load products
    logger.info(f"Loading products from {args.data_path}")
    df = pd.read_csv(args.data_path, sep="\t")
    logger.info(f"Loaded {len(df)} products")

    # Build and save taxonomy
    taxonomy = build_taxonomy(df)
    taxonomy_path = os.path.join(args.output_dir, "taxonomy.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    logger.info(f"Taxonomy saved to {taxonomy_path}")

    # Precompute text embeddings
    precompute_text(df, args.output_dir, args.batch_size)

    # Precompute image embeddings
    if not args.text_only:
        precompute_images(df, args.output_dir, args.image_manifest,
                          args.batch_size)
    else:
        logger.info("Skipping images (--text-only mode)")
        # Create empty image embeddings
        for _, row in df.iterrows():
            pid = str(row["product_id"])
            save_path = os.path.join(args.output_dir, f"{pid}_img.npy")
            if not os.path.exists(save_path):
                np.save(save_path, np.zeros((0, 768), dtype=np.float32))

    # Precompute attribute indices
    precompute_attributes(df, args.output_dir)

    logger.info(f"\nAll embeddings cached in {args.output_dir}")
    logger.info("Run: python scripts/train_classifier.py")


if __name__ == "__main__":
    main()
