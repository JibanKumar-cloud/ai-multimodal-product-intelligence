#!/usr/bin/env python3
"""Source furniture images from free APIs to pair with WANDS products.

Searches Pexels, Unsplash, and Pixabay for product images matching
WANDS product names/categories. Creates a multimodal dataset by
pairing downloaded images with WANDS text and ground truth labels.

Usage:
    # Using Pexels (recommended — best furniture coverage)
    export PEXELS_API_KEY="your-key-here"
    python scripts/source_images.py --api pexels --num-products 3000

    # Using Unsplash
    export UNSPLASH_ACCESS_KEY="your-key-here"
    python scripts/source_images.py --api unsplash --num-products 3000

    # Using local/manual images (no API key needed)
    python scripts/source_images.py --api pixabay --num-products 3000

API Keys (all free):
    Pexels:   https://www.pexels.com/api/ (free, 200 req/hr)
    Unsplash: https://unsplash.com/developers (free, 50 req/hr)
    Pixabay:  https://pixabay.com/api/docs/ (free, 100 req/min)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from loguru import logger

from src.data.wands_loader import WANDSLoader
from src.data.feature_parser import process_products_dataframe
from src.utils.logger import setup_logger


# ─── Image Search API Clients ───────────────────────────────────────────────

class PexelsClient:
    """Search Pexels for product images."""

    BASE_URL = "https://api.pexels.com/v1/search"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": api_key}
        self.requests_made = 0

    def search(self, query: str, per_page: int = 1) -> Optional[dict]:
        """Search for an image. Returns {url, width, height, photographer} or None."""
        if self.requests_made > 0 and self.requests_made % 180 == 0:
            logger.info("Rate limit pause (Pexels: 200/hr)...")
            time.sleep(20)

        try:
            resp = requests.get(
                self.BASE_URL,
                headers=self.headers,
                params={"query": query, "per_page": per_page, "orientation": "landscape"},
                timeout=10,
            )
            self.requests_made += 1

            if resp.status_code == 429:
                logger.warning("Rate limited. Waiting 60s...")
                time.sleep(60)
                return self.search(query, per_page)

            if resp.status_code != 200:
                return None

            data = resp.json()
            photos = data.get("photos", [])
            if not photos:
                return None

            photo = photos[0]
            return {
                "url": photo["src"]["large"],  # 940px wide
                "thumbnail_url": photo["src"]["medium"],
                "width": photo["width"],
                "height": photo["height"],
                "photographer": photo["photographer"],
                "source": "pexels",
                "source_id": str(photo["id"]),
                "license": "Pexels License (free for commercial use)",
            }
        except Exception as e:
            logger.debug(f"Pexels search failed for '{query}': {e}")
            return None


class UnsplashClient:
    """Search Unsplash for product images."""

    BASE_URL = "https://api.unsplash.com/search/photos"

    def __init__(self, access_key: str):
        self.access_key = access_key
        self.requests_made = 0

    def search(self, query: str, per_page: int = 1) -> Optional[dict]:
        if self.requests_made > 0 and self.requests_made % 45 == 0:
            logger.info("Rate limit pause (Unsplash: 50/hr)...")
            time.sleep(75)

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "query": query,
                    "per_page": per_page,
                    "client_id": self.access_key,
                    "orientation": "landscape",
                },
                timeout=10,
            )
            self.requests_made += 1

            if resp.status_code != 200:
                return None

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None

            photo = results[0]
            return {
                "url": photo["urls"]["regular"],
                "thumbnail_url": photo["urls"]["small"],
                "width": photo["width"],
                "height": photo["height"],
                "photographer": photo["user"]["name"],
                "source": "unsplash",
                "source_id": photo["id"],
                "license": "Unsplash License (free for commercial use)",
            }
        except Exception as e:
            logger.debug(f"Unsplash search failed for '{query}': {e}")
            return None


class PixabayClient:
    """Search Pixabay for product images."""

    BASE_URL = "https://pixabay.com/api/"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.requests_made = 0

    def search(self, query: str, per_page: int = 1) -> Optional[dict]:
        if self.requests_made > 0 and self.requests_made % 90 == 0:
            time.sleep(10)

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "key": self.api_key,
                    "q": query,
                    "per_page": min(per_page, 3),
                    "image_type": "photo",
                    "orientation": "horizontal",
                    "category": "backgrounds",
                },
                timeout=10,
            )
            self.requests_made += 1

            if resp.status_code != 200:
                return None

            data = resp.json()
            hits = data.get("hits", [])
            if not hits:
                return None

            photo = hits[0]
            return {
                "url": photo["largeImageURL"],
                "thumbnail_url": photo["webformatURL"],
                "width": photo["imageWidth"],
                "height": photo["imageHeight"],
                "photographer": photo["user"],
                "source": "pixabay",
                "source_id": str(photo["id"]),
                "license": "Pixabay License (free for commercial use)",
            }
        except Exception as e:
            logger.debug(f"Pixabay search failed for '{query}': {e}")
            return None


# ─── Query Building ──────────────────────────────────────────────────────────

def build_search_query(product_name: str, product_class: Optional[str] = None) -> str:
    """Build an effective image search query from product info.

    Strategy: Use product class + key descriptors rather than full product name.
    "Wayfair Trenton 72'' W Bathroom Vanity" → "bathroom vanity"
    "Mercury Row® Borum Velvet Accent Chair" → "velvet accent chair"
    """
    # Remove brand names, model numbers, dimensions
    import re

    name = product_name or ""

    # Remove brand-like patterns (word ending in ® or ™, or "by BrandName")
    name = re.sub(r"\S+[®™]", "", name)
    name = re.sub(r"\bby\s+\w+\b", "", name, flags=re.IGNORECASE)

    # Remove dimensions (72'', 36" x 24", etc.)
    name = re.sub(r"\d+[''\"]\s*[xX×]\s*\d+[''\"]*", "", name)
    name = re.sub(r"\d+[''\"]+", "", name)
    name = re.sub(r"\d+\s*(?:inch|in|cm|mm|ft)\b", "", name, flags=re.IGNORECASE)

    # Remove model numbers (alphanumeric codes)
    name = re.sub(r"\b[A-Z]{2,}\d{2,}\b", "", name)

    # Clean up
    name = re.sub(r"\s+", " ", name).strip()

    # If product_class is available, use it as primary + adjectives from name
    if product_class and isinstance(product_class, str):
        # Extract useful adjectives from name
        adjectives = []
        useful_words = {
            "modern", "contemporary", "traditional", "farmhouse", "industrial",
            "rustic", "scandinavian", "bohemian", "coastal", "mid-century",
            "velvet", "leather", "wood", "wooden", "metal", "glass", "fabric",
            "upholstered", "tufted", "woven", "marble", "oak", "walnut",
            "white", "black", "gray", "grey", "brown", "blue", "green", "red",
            "beige", "cream", "navy", "gold", "brass", "chrome",
            "round", "square", "rectangular", "oval", "tall", "small", "large",
        }
        for word in name.lower().split():
            if word in useful_words and word not in product_class.lower():
                adjectives.append(word)
                if len(adjectives) >= 2:
                    break

        query = " ".join(adjectives[:2] + [product_class.lower()])
    else:
        # Use cleaned product name, truncated
        words = name.split()[:5]
        query = " ".join(words)

    # Append "furniture" if query is too generic
    if len(query.split()) <= 2 and "furniture" not in query.lower():
        query += " furniture"

    return query.strip()


# ─── Image Downloading ───────────────────────────────────────────────────────

def download_image(url: str, save_path: Path, timeout: int = 15) -> bool:
    """Download an image from URL to local path.

    Returns True if successful.
    """
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return False

        # Verify it's actually an image
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return False

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify file is valid
        file_size = save_path.stat().st_size
        if file_size < 1000:  # Less than 1KB is suspicious
            save_path.unlink()
            return False

        return True

    except Exception as e:
        logger.debug(f"Download failed for {url}: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def source_images(
    wands_dir: str | Path = "data/raw/WANDS/dataset",
    output_dir: str | Path = "data/images",
    api: str = "pexels",
    num_products: int = 3000,
    seed: int = 42,
) -> dict:
    """Source images for WANDS products from free image APIs.

    Args:
        wands_dir: Path to WANDS dataset.
        output_dir: Directory to save images and metadata.
        api: Which API to use (pexels, unsplash, pixabay).
        num_products: Number of products to find images for.
        seed: Random seed for product selection.

    Returns:
        Summary statistics.
    """
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "raw"
    images_dir.mkdir(exist_ok=True)

    # Load WANDS products with features
    logger.info("Loading WANDS products...")
    loader = WANDSLoader(wands_dir)
    products = loader.get_products_with_features()
    processed = process_products_dataframe(products)

    # Filter to products with at least 2 parsed attributes
    attr_cols = ["style", "primary_material", "color_family", "product_type"]
    attr_count = processed[attr_cols].notna().sum(axis=1)
    rich_products = processed[attr_count >= 2].copy()
    logger.info(f"Products with ≥2 parsed attributes: {len(rich_products)}")

    # Sample products
    rng = np.random.RandomState(seed)
    if num_products < len(rich_products):
        sampled = rich_products.sample(n=num_products, random_state=rng)
    else:
        sampled = rich_products
        num_products = len(sampled)

    logger.info(f"Sourcing images for {num_products} products using {api} API")

    # Initialize API client
    client = _get_client(api)
    if client is None:
        return {"error": f"No API key found for {api}"}

    # Search and download
    metadata = []
    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, (_, row) in enumerate(sampled.iterrows()):
        product_id = row["product_id"]
        product_name = row["product_name"]
        product_class = row.get("product_class")

        # Build search query
        query = build_search_query(product_name, product_class)

        # Check if already downloaded
        image_path = images_dir / f"{product_id}.jpg"
        if image_path.exists():
            skip_count += 1
            metadata.append(_build_metadata_record(row, image_path, None, query, "cached"))
            success_count += 1
            continue

        # Search for image
        result = client.search(query)

        if result is None:
            # Try broader query
            if product_class and isinstance(product_class, str):
                result = client.search(product_class.lower())

        if result is None:
            fail_count += 1
            if (idx + 1) % 100 == 0:
                logger.info(
                    f"Progress: {idx + 1}/{num_products} "
                    f"(✓ {success_count} / ✗ {fail_count} / ⊘ {skip_count})"
                )
            continue

        # Download image
        if download_image(result["url"], image_path):
            record = _build_metadata_record(row, image_path, result, query, "downloaded")
            metadata.append(record)
            success_count += 1
        else:
            fail_count += 1

        if (idx + 1) % 100 == 0:
            logger.info(
                f"Progress: {idx + 1}/{num_products} "
                f"(✓ {success_count} / ✗ {fail_count} / ⊘ {skip_count})"
            )

        # Small delay to be respectful of rate limits
        time.sleep(0.5)

    # Save metadata
    meta_path = output_dir / "image_metadata.jsonl"
    with open(meta_path, "w") as f:
        for record in metadata:
            f.write(json.dumps(record, default=str) + "\n")

    # Save summary
    summary = {
        "total_attempted": num_products,
        "images_downloaded": success_count - skip_count,
        "images_cached": skip_count,
        "images_failed": fail_count,
        "success_rate": round(success_count / num_products * 100, 1),
        "api_used": api,
        "metadata_path": str(meta_path),
        "images_dir": str(images_dir),
    }

    summary_path = output_dir / "sourcing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    logger.info("IMAGE SOURCING COMPLETE")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)

    return summary


def _get_client(api: str):
    """Get the appropriate API client."""
    if api == "pexels":
        key = os.getenv("PEXELS_API_KEY")
        if not key:
            logger.error("PEXELS_API_KEY not set. Get one free at https://www.pexels.com/api/")
            return None
        return PexelsClient(key)
    elif api == "unsplash":
        key = os.getenv("UNSPLASH_ACCESS_KEY")
        if not key:
            logger.error("UNSPLASH_ACCESS_KEY not set. Get one free at https://unsplash.com/developers")
            return None
        return UnsplashClient(key)
    elif api == "pixabay":
        key = os.getenv("PIXABAY_API_KEY")
        if not key:
            logger.error("PIXABAY_API_KEY not set. Get one free at https://pixabay.com/api/docs/")
            return None
        return PixabayClient(key)
    else:
        raise ValueError(f"Unknown API: {api}. Choose from: pexels, unsplash, pixabay")


def _build_metadata_record(row, image_path, api_result, query, status):
    """Build a metadata record for one product-image pair."""
    record = {
        "product_id": int(row["product_id"]),
        "product_name": row["product_name"],
        "product_class": row.get("product_class"),
        "image_path": str(image_path),
        "search_query": query,
        "status": status,
        # Ground truth attributes
        "ground_truth": {
            "style": row.get("style"),
            "primary_material": row.get("primary_material"),
            "color_family": row.get("color_family"),
            "product_type": row.get("product_type"),
            "room_type": row.get("room_type"),
            "assembly_required": row.get("assembly_required"),
        },
    }
    if api_result:
        record["image_source"] = {
            "url": api_result["url"],
            "photographer": api_result.get("photographer"),
            "source": api_result.get("source"),
            "source_id": api_result.get("source_id"),
            "license": api_result.get("license"),
        }
    return record


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Source furniture images for WANDS products")
    parser.add_argument(
        "--api",
        type=str,
        default="pexels",
        choices=["pexels", "unsplash", "pixabay"],
        help="Image API to use (default: pexels)",
    )
    parser.add_argument("--num-products", type=int, default=3000)
    parser.add_argument("--wands-dir", type=str, default="data/raw/WANDS/dataset")
    parser.add_argument("--output-dir", type=str, default="data/images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_images(
        wands_dir=args.wands_dir,
        output_dir=args.output_dir,
        api=args.api,
        num_products=args.num_products,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
