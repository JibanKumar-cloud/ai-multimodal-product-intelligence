#!/usr/bin/env python3
"""Source real Wayfair product images via DuckDuckGo image search.

Strategy:
  1. Search DDG images: "product_name wayfair"
  2. Take FIRST wfcdn.com result (real Wayfair hero image)
  3. Verify source URL contains product-relevant terms
  4. Download hero image
  5. Generate center crop (material/texture detail) from hero

Result: 2 images per product (hero + material crop), both same product.

Usage:
    python scripts/source_wayfair_images.py \
        --queue data/processed/image_queue.json

    # Resume (rerun same command)
    python scripts/source_wayfair_images.py \
        --queue data/processed/image_queue.json

    # Retry failed products
    python scripts/source_wayfair_images.py \
        --queue data/processed/image_queue.json --retry-failed
"""
import argparse
import json
import os
import re
import sys
import time
import random
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import requests
import pandas as pd
from PIL import Image

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


# ── Constants ──
WAYFAIR_CDN = "wfcdn.com"
MIN_IMAGE_BYTES = 3000
MAX_IMAGE_BYTES = 15_000_000
SEARCH_DELAY_MIN = 2.0
SEARCH_DELAY_MAX = 4.0
DOWNLOAD_TIMEOUT = 25
MIN_IMAGE_DIM = 200
CROP_RATIO = 0.5
CROP_MIN_DIM = 150

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/121.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Firefox/121.0",
]


class ProgressTracker:
    """Track completed products. Survives restarts."""

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.manifest = {}
        self._load()

    def _load(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            done = sum(1 for v in self.manifest.values()
                       if v.get("image_count", 0) > 0)
            total_imgs = sum(v.get("image_count", 0)
                             for v in self.manifest.values())
            print(f"Resuming: {done} products done, "
                  f"{total_imgs} images collected")

    def save(self):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(tmp, self.manifest_path)

    def is_done(self, product_id: str) -> bool:
        entry = self.manifest.get(str(product_id), {})
        return entry.get("image_count", 0) > 0

    def is_failed(self, product_id: str) -> bool:
        entry = self.manifest.get(str(product_id), {})
        return (entry.get("image_count", 0) == 0
                and entry.get("status") in (
                    "no_results", "no_wayfair_images",
                    "download_failed", "image_too_small",
                    "search_error"))

    def record_success(self, product_id: str, product_name: str,
                       source_url: str, image_url: str,
                       image_paths: list, image_labels: list):
        self.manifest[str(product_id)] = {
            "product_name": product_name,
            "source_url": source_url,
            "image_url": image_url,
            "image_count": len(image_paths),
            "image_paths": image_paths,
            "image_labels": image_labels,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

    def record_failure(self, product_id: str, product_name: str,
                       reason: str):
        self.manifest[str(product_id)] = {
            "product_name": product_name,
            "image_count": 0,
            "status": reason,
            "timestamp": datetime.now().isoformat(),
        }

    def summary(self):
        total = len(self.manifest)
        with_imgs = sum(1 for v in self.manifest.values()
                        if v.get("image_count", 0) > 0)
        total_imgs = sum(v.get("image_count", 0)
                         for v in self.manifest.values())
        multi = sum(1 for v in self.manifest.values()
                    if v.get("image_count", 0) > 1)

        status_counts = Counter(
            v.get("status", "unknown") for v in self.manifest.values())

        label_counts = Counter()
        for v in self.manifest.values():
            for lbl in v.get("image_labels", []):
                label_counts[lbl] += 1

        print(f"\n{'='*60}")
        print(f"IMAGE SOURCING SUMMARY")
        print(f"{'='*60}")
        print(f"Products processed:      {total}")
        print(f"Products WITH images:    {with_imgs}")
        print(f"  With hero+crop (2):    {multi}")
        print(f"Total images:            {total_imgs}")
        if with_imgs:
            print(f"Avg images/product:      {total_imgs / with_imgs:.1f}")
        print(f"\nStatus breakdown:")
        for status, count in status_counts.most_common():
            print(f"  {status:25s}: {count}")
        if label_counts:
            print(f"\nImage types:")
            for lbl, count in label_counts.most_common():
                print(f"  {lbl:25s}: {count}")
        print(f"{'='*60}")


def build_balanced_queue(products_csv: str, tracker: ProgressTracker,
                         max_per_category: int, retry_failed: bool):
    """Build balanced queue from raw products CSV."""
    df = pd.read_csv(products_csv, sep="\t")
    print(f"WANDS products: {len(df)}")

    category_groups = defaultdict(list)
    for _, row in df.iterrows():
        pid = str(row["product_id"])
        if tracker.is_done(pid):
            continue
        if not retry_failed and tracker.is_failed(pid):
            continue
        cat = str(row.get("product_class", "unknown"))
        cat = cat.split("|")[0].strip()
        category_groups[cat].append({
            "product_id": pid,
            "product_name": str(row["product_name"]),
        })

    queue = []
    for round_num in range(max_per_category):
        added = 0
        for cat in sorted(category_groups.keys()):
            products = category_groups[cat]
            if round_num < len(products):
                queue.append(products[round_num])
                added += 1
        if added == 0:
            break

    random.shuffle(queue)
    print(f"Queue: {len(queue)} products")
    return queue


def search_product_image(ddgs_client: DDGS, product_name: str) -> tuple:
    """Search DDG for first wfcdn.com image of this product.

    Returns: (image_url, source_url) or (None, None)
    """
    query = f"{product_name} wayfair"

    try:
        results = list(ddgs_client.images(query, max_results=20))
    except Exception:
        return None, None

    if not results:
        return None, None

    # Extract key words from product name for verification
    name_words = set(
        w.lower() for w in re.findall(r'[a-zA-Z]{3,}', product_name)
    )
    generic = {"the", "and", "with", "for", "from", "inch", "wide",
               "tall", "deep", "set", "piece", "pack"}
    name_words -= generic

    # Score each wfcdn result by relevance
    best_match = None
    best_score = -1

    for r in results:
        img_url = r.get("image", "")
        source_url = r.get("url", "")
        title = r.get("title", "").lower()

        if WAYFAIR_CDN not in img_url:
            continue

        combined = (source_url + " " + title).lower()
        score = sum(1 for w in name_words if w in combined)

        if "wayfair.com" in source_url:
            score += 2
        if "/pdp/" in source_url:
            score += 3

        if score > best_score:
            best_score = score
            best_match = (img_url, source_url)

    if best_match:
        return best_match

    # Fallback: any wfcdn.com image
    for r in results:
        img_url = r.get("image", "")
        if WAYFAIR_CDN in img_url:
            return img_url, r.get("url", "")

    return None, None


def download_image(url: str, save_path: str) -> bool:
    """Download image. Returns True on success."""
    try:
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://www.wayfair.com/",
        }
        resp = requests.get(url, headers=headers,
                            timeout=DOWNLOAD_TIMEOUT,
                            allow_redirects=True)
        if resp.status_code != 200:
            return False

        data = resp.content
        if len(data) < MIN_IMAGE_BYTES or len(data) > MAX_IMAGE_BYTES:
            return False

        if not (data[:3] == b'\xff\xd8\xff' or
                data[:8] == b'\x89PNG\r\n\x1a\n' or
                data[:4] == b'RIFF' or
                data[:4] == b'GIF8'):
            return False

        with open(save_path, "wb") as f:
            f.write(data)
        return True

    except Exception:
        return False


def generate_material_crop(hero_path: str, crop_path: str,
                           crop_ratio: float = CROP_RATIO) -> bool:
    """Generate center crop from hero image for material/texture detail.

    Crops the center 50% of the image, simulating a zoom-in
    to see material, texture, and finish details.
    """
    try:
        img = Image.open(hero_path)
        w, h = img.size

        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            return False

        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)

        if crop_w < CROP_MIN_DIM or crop_h < CROP_MIN_DIM:
            return False

        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        cropped = img.crop((left, top, right, bottom))

        if cropped.mode in ("RGBA", "P"):
            cropped = cropped.convert("RGB")
        cropped.save(crop_path, "JPEG", quality=90)
        return True

    except Exception:
        return False


def process_product(product: dict, ddgs_client: DDGS,
                    tracker: ProgressTracker, output_dir: str):
    """Process one product: search → download → crop."""
    pid = product["product_id"]
    name = product["product_name"]

    if tracker.is_done(pid):
        return

    image_url, source_url = search_product_image(ddgs_client, name)

    if not image_url:
        tracker.record_failure(pid, name, "no_wayfair_images")
        return

    product_dir = Path(output_dir) / pid
    product_dir.mkdir(parents=True, exist_ok=True)

    ext = ".jpg"
    if ".png" in image_url.lower():
        ext = ".png"
    elif ".webp" in image_url.lower():
        ext = ".webp"

    hero_path = str(product_dir / f"hero{ext}")
    crop_path = str(product_dir / "material_crop.jpg")

    if not os.path.exists(hero_path):
        success = download_image(image_url, hero_path)
        if not success:
            tracker.record_failure(pid, name, "download_failed")
            return

    # Verify image dimensions
    try:
        img = Image.open(hero_path)
        w, h = img.size
        img.close()
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            os.remove(hero_path)
            tracker.record_failure(pid, name, "image_too_small")
            return
    except Exception:
        if os.path.exists(hero_path):
            os.remove(hero_path)
        tracker.record_failure(pid, name, "download_failed")
        return

    image_paths = [hero_path]
    image_labels = ["hero"]

    if generate_material_crop(hero_path, crop_path):
        image_paths.append(crop_path)
        image_labels.append("material_crop")

    tracker.record_success(
        pid, name, source_url or "", image_url,
        image_paths, image_labels)


def run_pipeline(queue: list, tracker: ProgressTracker,
                 output_dir: str, save_every: int):
    """Main pipeline — sequential with delays."""
    ddgs_client = DDGS()

    total = len(queue)
    completed = 0
    start_time = time.time()
    consecutive_failures = 0

    for i, product in enumerate(queue):
        pid = product["product_id"]

        if tracker.is_done(pid):
            completed += 1
            continue

        # Random delay between searches
        delay = random.uniform(SEARCH_DELAY_MIN, SEARCH_DELAY_MAX)
        time.sleep(delay)

        try:
            process_product(product, ddgs_client, tracker, output_dir)
        except Exception as e:
            tracker.record_failure(
                pid, product["product_name"], "search_error")
            consecutive_failures += 1

            if consecutive_failures >= 10:
                print(f"\n⚠️  {consecutive_failures} consecutive failures. "
                      f"Backing off 60s...")
                time.sleep(60)
                consecutive_failures = 0
                ddgs_client = DDGS()
            continue

        if tracker.is_done(pid):
            consecutive_failures = 0

        completed += 1

        if completed % save_every == 0:
            tracker.save()

            elapsed = time.time() - start_time
            rate = completed / max(elapsed, 1) * 3600
            done_count = sum(
                1 for v in tracker.manifest.values()
                if v.get("image_count", 0) > 0)
            total_imgs = sum(
                v.get("image_count", 0)
                for v in tracker.manifest.values())
            failed = sum(
                1 for v in tracker.manifest.values()
                if v.get("status") != "success"
                and "status" in v)

            print(
                f"[{completed}/{total}] "
                f"{done_count} with images, "
                f"{total_imgs} total imgs, "
                f"{failed} failed, "
                f"{rate:.0f}/hr, "
                f"{elapsed/60:.1f}min"
            )

    tracker.save()


def main():
    parser = argparse.ArgumentParser(
        description="Source Wayfair product images via DDG search")
    parser.add_argument(
        "--queue", type=str, default=None,
        help="Prioritized queue JSON from prepare_classifier_data.py")
    parser.add_argument(
        "--products-csv", type=str,
        default="data/raw/WANDS/dataset/product.csv")
    parser.add_argument(
        "--output-dir", type=str,
        default="data/images/wayfair")
    parser.add_argument(
        "--manifest", type=str,
        default="data/images/wayfair_manifest.json")
    parser.add_argument(
        "--max-per-category", type=int, default=25)
    parser.add_argument(
        "--save-every", type=int, default=15,
        help="Save progress every N products")
    parser.add_argument(
        "--retry-failed", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_dir = os.path.dirname(args.manifest)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    tracker = ProgressTracker(args.manifest)

    if args.queue:
        with open(args.queue) as f:
            raw_queue = json.load(f)
        queue = [p for p in raw_queue
                 if not tracker.is_done(p["product_id"])]
        if not args.retry_failed:
            queue = [p for p in queue
                     if not tracker.is_failed(p["product_id"])]
        print(f"Loaded queue: {len(queue)} remaining "
              f"(from {len(raw_queue)} total)")
    else:
        queue = build_balanced_queue(
            args.products_csv, tracker,
            args.max_per_category, args.retry_failed)

    if not queue:
        print("Nothing to do — all products processed!")
        tracker.summary()
        return

    print(f"\n{'='*60}")
    print(f"WAYFAIR IMAGE SOURCER")
    print(f"{'='*60}")
    print(f"Queue:             {len(queue)} products")
    print(f"Output:            {args.output_dir}")
    print(f"Strategy:          1 hero image + 1 material crop")
    print(f"Source:            DDG → wfcdn.com (Wayfair CDN)")
    print(f"Delay:             {SEARCH_DELAY_MIN}-{SEARCH_DELAY_MAX}s/query")
    print(f"Est. time:         ~{len(queue) * 3 / 3600:.1f} hours")
    print(f"{'='*60}\n")

    try:
        run_pipeline(queue, tracker, args.output_dir, args.save_every)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        tracker.save()
        print("Progress saved. Rerun to resume.")

    tracker.summary()


if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """
# Wayfair image sourcer (v3): 1 hero image + 1 synthetic material crop.

# Why v3:
# - v2 depends on ddgs/duckduckgo_search and can silently return zero results due to API/signature differences.
# - v3 uses the same DuckDuckGo "vqd" + i.js flow as your v1 (which you said returns results),
#   but keeps the v2 "single hero + crop" strategy.

# High-level:
# 1) DDG images search for: "<product_name> wayfair"
# 2) Pick best wfcdn.com image (prefer Wayfair PDP source URLs if present)
# 3) Download and convert hero to JPEG (avoids WEBP/format issues)
# 4) Create center crop JPEG as material proxy

# Usage:
#   python source_wayfair_images_v3.py --queue data/processed/image_queue.json
#   python source_wayfair_images_v3.py --queue data/processed/image_queue.json --retry-failed
# """

# import argparse
# import json
# import os
# import random
# import re
# import time
# from collections import Counter
# from datetime import datetime
# from io import BytesIO
# from pathlib import Path
# from urllib.parse import quote_plus

# import requests
# from PIL import Image
# from urllib.parse import urlparse, parse_qs, unquote


# # ── Constants ──
# WAYFAIR_CDN = "wfcdn.com"
# DDG_URL = "https://duckduckgo.com/"
# DDG_IMAGES_URL = "https://duckduckgo.com/i.js"

# SEARCH_TIMEOUT = 20
# DOWNLOAD_TIMEOUT = 25

# SEARCH_DELAY_MIN = 2.0
# SEARCH_DELAY_MAX = 4.0

# MIN_IMAGE_DIM = 200
# CROP_RATIO = 0.5
# CROP_MIN_DIM = 150

# # Prefer JPEG/PNG to avoid WEBP decode issues on some Pillow builds
# IMG_ACCEPT = "image/jpeg,image/png,image/*;q=0.8,*/*;q=0.5"

# USER_AGENTS = [
#     "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
# ]


# def _headers(referer: str | None = None) -> dict:
#     h = {
#         "User-Agent": random.choice(USER_AGENTS),
#         "Accept": "text/html,application/xhtml+xml,application/json",
#         "Accept-Language": "en-US,en;q=0.9",
#     }
#     if referer:
#         h["Referer"] = referer
#     return h


# def _img_headers() -> dict:
#     return {
#         "User-Agent": random.choice(USER_AGENTS),
#         "Accept": IMG_ACCEPT,
#         "Referer": "https://www.wayfair.com/",
#     }


# def unwrap_ddg_image_url(url: str) -> str:
#     """
#     DDG sometimes returns a proxy URL like:
#     https://external-content.duckduckgo.com/iu/?u=<encoded>&f=1&nofb=1
#     We extract and decode the real underlying image URL.
#     """
#     if not url:
#         return url
#     if "duckduckgo.com/iu/" in url:
#         try:
#             qs = parse_qs(urlparse(url).query)
#             if "u" in qs and qs["u"]:
#                 return unquote(qs["u"][0])
#         except Exception:
#             return url
#     return url

# class ProgressTracker:
#     def __init__(self, manifest_path: str):
#         self.manifest_path = manifest_path
#         self.manifest: dict = {}
#         self._load()

#     def _load(self):
#         if os.path.exists(self.manifest_path):
#             with open(self.manifest_path) as f:
#                 self.manifest = json.load(f)
#             done = sum(1 for v in self.manifest.values()
#                        if v.get("image_count", 0) > 0)
#             total_imgs = sum(v.get("image_count", 0)
#                              for v in self.manifest.values())
#             print(f"Resuming: {done} products done, {total_imgs} images collected")

#     def save(self):
#         tmp = self.manifest_path + ".tmp"
#         with open(tmp, "w") as f:
#             json.dump(self.manifest, f, indent=2)
#         os.replace(tmp, self.manifest_path)

#     def is_done(self, product_id: str) -> bool:
#         entry = self.manifest.get(str(product_id), {})
#         return entry.get("image_count", 0) > 0

#     def is_failed(self, product_id: str) -> bool:
#         entry = self.manifest.get(str(product_id), {})
#         return (entry.get("image_count", 0) == 0 and entry.get("status") not in (None, "success"))

#     def record_success(self, product_id: str, product_name: str,
#                        source_url: str, image_url: str,
#                        image_paths: list[str], image_labels: list[str]):
#         self.manifest[str(product_id)] = {
#             "product_name": product_name,
#             "source_url": source_url,
#             "image_url": image_url,
#             "image_count": len(image_paths),
#             "image_paths": image_paths,
#             "image_labels": image_labels,
#             "timestamp": datetime.now().isoformat(),
#             "status": "success",
#         }

#     def record_failure(self, product_id: str, product_name: str, reason: str, debug: str | None = None):
#         payload = {
#             "product_name": product_name,
#             "image_count": 0,
#             "status": reason,
#             "timestamp": datetime.now().isoformat(),
#         }
#         if debug:
#             payload["debug"] = debug[:500]
#         self.manifest[str(product_id)] = payload

#     def summary(self):
#         total = len(self.manifest)
#         with_imgs = sum(1 for v in self.manifest.values() if v.get("image_count", 0) > 0)
#         total_imgs = sum(v.get("image_count", 0) for v in self.manifest.values())

#         status_counts = Counter(v.get("status", "unknown") for v in self.manifest.values())
#         label_counts = Counter()
#         for v in self.manifest.values():
#             for lbl in v.get("image_labels", []):
#                 label_counts[lbl] += 1

#         print(f"\n{'='*60}\nIMAGE SOURCING SUMMARY\n{'='*60}")
#         print(f"Products processed:      {total}")
#         print(f"Products WITH images:    {with_imgs}")
#         print(f"Total images:            {total_imgs}")
#         if with_imgs:
#             print(f"Avg images/product:      {total_imgs / with_imgs:.2f}")

#         print("\nStatus breakdown:")
#         for status, count in status_counts.most_common():
#             print(f"  {status:25s}: {count}")

#         if label_counts:
#             print("\nImage types:")
#             for lbl, count in label_counts.most_common():
#                 print(f"  {lbl:25s}: {count}")
#         print(f"{'='*60}")


# def ddg_image_urls(session: requests.Session, query: str, max_results: int = 20) -> list[dict]:
#     """
#     Returns list of DDG image result dicts (each has 'image', 'url', 'title', etc).
#     Uses the vqd token method (same idea as v1).
#     """
#     search_url = f"{DDG_URL}?q={quote_plus(query)}&iax=images&ia=images"

#     r = session.get(search_url, headers=_headers(referer="https://duckduckgo.com/"), timeout=SEARCH_TIMEOUT, allow_redirects=True)
#     if r.status_code != 200:
#         return []

#     text = r.text
#     vqd = ""
#     m = re.search(r'vqd=["\']([^"\']+)', text)
#     if m:
#         vqd = m.group(1)
#     if not vqd:
#         m = re.search(r'vqd=([a-zA-Z0-9_-]+)', text)
#         if m:
#             vqd = m.group(1)
#     if not vqd:
#         return []

#     params = {
#         "l": "us-en",
#         "o": "json",
#         "q": query,
#         "vqd": vqd,
#         "f": ",,,,,",
#         "p": "1",
#     }
#     r2 = session.get(DDG_IMAGES_URL, headers=_headers(referer=search_url), params=params, timeout=SEARCH_TIMEOUT)
#     if r2.status_code != 200:
#         return []

#     # DDG sometimes returns JSON with wrong content-type; parse manually
#     try:
#         data = r2.json()
#     except Exception:
#         try:
#             data = json.loads(r2.text)
#         except Exception:
#             return []

#     results = data.get("results", []) or []
#     return results[:max_results]


# def pick_best_wayfair_image(results: list[dict], product_name: str) -> tuple[str | None, str | None]:
#     """
#     Pick a wfcdn image URL; prefer ones whose source URL looks like a Wayfair PDP and matches product name tokens.
#     Returns (image_url, source_url).
#     """
#     if not results:
#         return None, None

#     name_words = set(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", product_name))
#     generic = {"the", "and", "with", "for", "from", "inch", "wide", "tall", "deep", "set", "piece", "pack"}
#     name_words -= generic

#     best = None
#     best_score = -1

#     for r in results:
#         img_url = (r.get("image") or r.get("img") or r.get("thumbnail") or "").strip()
#         img_url = unwrap_ddg_image_url(img_url)
#         source_url = (r.get("url") or r.get("source") or "").strip()
#         title = (r.get("title") or "").lower()

#         if not img_url:
#             continue

#         is_wfcdn = "wfcdn.com" in img_url
#         is_wayfair_source = "wayfair.com" in (source_url or "")

#         # allow either:
#         #  - direct wfcdn image, OR
#         #  - any image whose SOURCE PAGE is wayfair.com
#         if not (is_wfcdn or is_wayfair_source):
#             continue

#         combined = (source_url + " " + title).lower()
#         score = sum(1 for w in name_words if w in combined)

#         if "wayfair.com" in source_url:
#             score += 2
#         if "/pdp/" in source_url:
#             score += 3

#         # small preference to direct assets links
#         if "assets.wfcdn.com" in img_url:
#             score += 1

#         if score > best_score:
#             best_score = score
#             best = (img_url, source_url)

#     if best:
#         return best

#     # fallback: first wfcdn image
#     for r in results:
#         img_url = (r.get("image") or r.get("thumbnail") or "").strip()
#         img_url = unwrap_ddg_image_url(img_url)
#         if img_url and WAYFAIR_CDN in img_url:
#             return img_url, (r.get("url") or "").strip()

#     return None, None


# def download_hero_as_jpeg(session: requests.Session, url: str, save_path: str) -> tuple[bool, str | None]:
#     """
#     Download image and save as JPEG (RGB). Returns (ok, debug_msg).
#     Avoids WEBP/format surprises and normalizes filetype.
#     """
#     try:
#         resp = session.get(url, headers=_img_headers(), timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
#         if resp.status_code != 200:
#             return False, f"HTTP {resp.status_code}"

#         data = resp.content
#         if not data or len(data) < 3000 or len(data) > 15_000_000:
#             return False, f"bad_size={len(data) if data else 0}"

#         try:
#             img = Image.open(BytesIO(data))
#             img.load()
#         except Exception as e:
#             return False, f"PIL_open_failed: {e}"

#         w, h = img.size
#         if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
#             return False, f"too_small={w}x{h}"

#         if img.mode in ("RGBA", "P"):
#             img = img.convert("RGB")
#         elif img.mode != "RGB":
#             img = img.convert("RGB")

#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         img.save(save_path, "JPEG", quality=92)
#         return True, None

#     except Exception as e:
#         return False, f"download_exc: {e}"


# def generate_material_crop(hero_path: str, crop_path: str, crop_ratio: float = CROP_RATIO) -> tuple[bool, str | None]:
#     try:
#         img = Image.open(hero_path)
#         w, h = img.size

#         crop_w = int(w * crop_ratio)
#         crop_h = int(h * crop_ratio)

#         if crop_w < CROP_MIN_DIM or crop_h < CROP_MIN_DIM:
#             return False, f"crop_too_small={crop_w}x{crop_h}"

#         left = (w - crop_w) // 2
#         top = (h - crop_h) // 2
#         right = left + crop_w
#         bottom = top + crop_h

#         cropped = img.crop((left, top, right, bottom))
#         if cropped.mode != "RGB":
#             cropped = cropped.convert("RGB")
#         cropped.save(crop_path, "JPEG", quality=92)
#         return True, None

#     except Exception as e:
#         return False, f"crop_exc: {e}"


# def process_product(session: requests.Session, product: dict, tracker: ProgressTracker, output_dir: str, verbose: bool = False):
#     pid = str(product.get("product_id"))
#     name = str(product.get("product_name", "")).strip()

#     if not pid or not name:
#         tracker.record_failure(pid or "unknown", name or "unknown", "bad_queue_row")
#         return

#     if tracker.is_done(pid):
#         return

#     clean_name = re.sub(r"\s*''\s*", " inch ", name)  # 25 '' -> 25 inch
#     clean_name = re.sub(r"\s+", " ", clean_name).strip()
#     query = f"{clean_name} site:wayfair.com"

#     results = ddg_image_urls(session, query, max_results=20)
#     if verbose:
#         print(f"DDG results={len(results)} for: {query}")
#         if results:
#             r0 = results[0]
#             print("  sample image:", (r0.get("image") or r0.get("thumbnail")))
#             print("  sample url  :", (r0.get("url") or r0.get("source")))
#     img_url, source_url = pick_best_wayfair_image(results, name)

#     if not img_url:
#         tracker.record_failure(pid, name, "no_wayfair_images")
#         if verbose:
#             print(f"✗ {pid}: no wfcdn image for '{name[:80]}'")
#         return

#     product_dir = Path(output_dir) / pid
#     hero_path = str(product_dir / "hero.jpg")
#     crop_path = str(product_dir / "material_crop.jpg")

#     # Download + normalize hero
#     if not os.path.exists(hero_path):
#         ok, dbg = download_hero_as_jpeg(session, img_url, hero_path)
#         if not ok:
#             tracker.record_failure(pid, name, "download_failed", debug=dbg)
#             if verbose:
#                 print(f"✗ {pid}: download_failed ({dbg})")
#             return

#     image_paths = [hero_path]
#     image_labels = ["hero"]

#     ok, dbg = generate_material_crop(hero_path, crop_path)
#     if ok:
#         image_paths.append(crop_path)
#         image_labels.append("material_crop")
#     else:
#         # still count hero as success; crop can be done later if needed
#         if verbose:
#             print(f"• {pid}: crop skipped ({dbg})")
        

#     tracker.record_success(pid, name, source_url or "", img_url, image_paths, image_labels)
#     if verbose:
#         print(f"✓ {pid}: {len(image_paths)} image(s)")


# def run_pipeline(queue: list, tracker: ProgressTracker, output_dir: str, save_every: int, verbose: bool = False):
#     session = requests.Session()

#     total = len(queue)
#     completed = 0
#     start_time = time.time()
#     consecutive_failures = 0

#     for i, product in enumerate(queue):
#         pid = str(product.get("product_id", ""))

#         if tracker.is_done(pid):
#             completed += 1
#             continue

#         time.sleep(random.uniform(SEARCH_DELAY_MIN, SEARCH_DELAY_MAX))

#         before = tracker.manifest.get(pid, {}).get("status")
#         try:
#             process_product(session, product, tracker, output_dir, verbose=verbose)
#         except Exception as e:
#             tracker.record_failure(pid, str(product.get("product_name", "")), "pipeline_error", debug=str(e))
#             consecutive_failures += 1
#         after = tracker.manifest.get(pid, {}).get("status")

#         if after == "success":
#             consecutive_failures = 0
#         else:
#             consecutive_failures += 1

#         if consecutive_failures >= 10:
#             print("\n⚠️  10 consecutive failures. Backing off 60s + new session...")
#             time.sleep(60)
#             session = requests.Session()
#             consecutive_failures = 0

#         completed += 1

#         if completed % save_every == 0:
#             tracker.save()
#             elapsed = time.time() - start_time
#             rate = completed / max(elapsed, 1) * 3600
#             done_count = sum(1 for v in tracker.manifest.values() if v.get("image_count", 0) > 0)
#             total_imgs = sum(v.get("image_count", 0) for v in tracker.manifest.values())
#             failed = sum(1 for v in tracker.manifest.values() if v.get("status") not in (None, "success"))
#             print(f"[{completed}/{total}] {done_count} with images, {total_imgs} imgs, {failed} failed, {rate:.0f}/hr, {elapsed/60:.1f}min")

#     tracker.save()


# def main():
#     parser = argparse.ArgumentParser(description="Source Wayfair product images via DuckDuckGo (v3)")
#     parser.add_argument("--queue", type=str, default=None, help="Queue JSON from prepare_classifier_data.py")
#     parser.add_argument("--products-csv", type=str, default="data/raw/WANDS/dataset/product.csv")
#     parser.add_argument("--output-dir", type=str, default="data/images/wayfair")
#     parser.add_argument("--manifest", type=str, default="data/images/wayfair_manifest.json")
#     parser.add_argument("--max-per-category", type=int, default=25)
#     parser.add_argument("--save-every", type=int, default=15)
#     parser.add_argument("--retry-failed", action="store_true")
#     parser.add_argument("--verbose", action="store_true")
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)
#     md = os.path.dirname(args.manifest)
#     if md:
#         os.makedirs(md, exist_ok=True)

#     tracker = ProgressTracker(args.manifest)

#     if args.queue:
#         with open(args.queue) as f:
#             raw_queue = json.load(f)

#         # filter remaining
#         queue = [p for p in raw_queue if not tracker.is_done(str(p.get("product_id")))]
#         if not args.retry_failed:
#             queue = [p for p in queue if not tracker.is_failed(str(p.get("product_id")))]
#         print(f"Loaded queue: {len(queue)} remaining (from {len(raw_queue)} total)")
#     else:
#         # lightweight fallback queue builder (only id + name)
#         import pandas as pd
#         df = pd.read_csv(args.products_csv, sep="\t")
#         queue = []
#         for _, row in df.iterrows():
#             pid = str(row.get("product_id"))
#             name = str(row.get("product_name"))
#             if not pid or not name:
#                 continue
#             if tracker.is_done(pid):
#                 continue
#             if not args.retry_failed and tracker.is_failed(pid):
#                 continue
#             queue.append({"product_id": pid, "product_name": name})
#         random.shuffle(queue)
#         queue = queue[: args.max_per_category * 50]  # crude cap

#     if not queue:
#         print("Nothing to do — all products processed!")
#         tracker.summary()
#         return

#     print(f"\n{'='*60}\nWAYFAIR IMAGE SOURCER (v3)\n{'='*60}")
#     print(f"Queue:       {len(queue)} products")
#     print(f"Output:      {args.output_dir}")
#     print(f"Strategy:    1 hero JPEG + 1 center crop JPEG")
#     print(f"Delay:       {SEARCH_DELAY_MIN}-{SEARCH_DELAY_MAX}s/query")
#     print(f"{'='*60}\n")

#     try:
#         run_pipeline(queue, tracker, args.output_dir, args.save_every, verbose=args.verbose)
#     except KeyboardInterrupt:
#         print("\n\nInterrupted! Saving progress...")
#         tracker.save()
#         print("Progress saved. Rerun to resume.")

#     tracker.summary()


# if __name__ == "__main__":
#     main()
