"""Source product images via DuckDuckGo Image Search.

V4: Attribute-enriched queries for better image-label matching.
    - Query includes color, material, shape, style from product attributes
    - Relevance scoring boosts results matching attribute words
    - Raw DDG image API (VQD tokens + i.js) — proven to work
    - Accepts images from ANY domain (not just wfcdn.com)
    - 1 hero image + 1 center crop for material detail
    - Resume-safe with manifest tracking

Query strategy:
    OLD: "holly aged wood 3 drawer chest wayfair product"
    NEW: "gray wood holly aged wood 3 drawer chest farmhouse wayfair"

    parts = [primary_color, primary_material, shape*, product_name, style]
    * shape only if non-default (not rectangular/other)

Usage:
    python scripts/source_wayfair_images.py \\
        --queue data/processed/image_queue.json

    # Resume (just rerun — skips completed products)
    python scripts/source_wayfair_images.py \\
        --queue data/processed/image_queue.json

    # Retry failed products
    python scripts/source_wayfair_images.py \\
        --queue data/processed/image_queue.json --retry-failed
"""
import argparse
import asyncio
import aiohttp
import aiofiles
import json
import os
import re
import sys
import time
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from urllib.parse import quote_plus, urlparse

try:
    from PIL import Image
except ImportError:
    print("pip install Pillow")
    sys.exit(1)


# ── Constants ──
MIN_IMAGE_BYTES = 3_000
MAX_IMAGE_BYTES = 15_000_000
SEARCH_DELAY_MIN = 8
SEARCH_DELAY_MAX = 12
DOWNLOAD_TIMEOUT = 25
SEARCH_TIMEOUT = 20
MAX_RETRIES = 2
CROP_RATIO = 0.5          # center crop = 50% of original

COOLDOWN_EVERY_CALLS = 50          # every N DDG searches
COOLDOWN_RANGE_SEC = (60, 120)     # 1–2 minutes

EMPTY_STREAK_THRESHOLD = 15        # N empty searches in a row => likely blocked
THROTTLE_RANGE_SEC = (300, 400)    # 5–7 minutes

_ddg_calls = 0
_empty_streak = 0
_ddg_lock = asyncio.Lock()

DDG_URL = "https://duckduckgo.com/"
DDG_IMAGES_URL = "https://duckduckgo.com/i.js"

# Domains to skip (low-quality or problematic)
BLOCKED_DOMAINS = {
    "pinterest.com", "pinimg.com",          # watermarked / login walls
    "facebook.com", "fbcdn.net",            # login walls
    "instagram.com",                        # login walls
    "tiktok.com",                           # video platform
    "youtube.com", "ytimg.com",             # video thumbnails
    "aliexpress.com",                       # often wrong products
    "wish.com",                             # often wrong products
    "shutterstock.com", "gettyimages.com",  # watermarked stock
    "istockphoto.com", "alamy.com",         # watermarked stock
    "depositphotos.com",                    # watermarked stock
}

# Preferred domains (score bonus)
PREFERRED_DOMAINS = {
    "wfcdn.com": 5,          # Wayfair CDN — exact product
    "wayfair.com": 4,        # Wayfair pages
    "perigold.com": 3,       # Wayfair luxury brand
    "birchlane.com": 3,      # Wayfair brand
    "allmodern.com": 3,      # Wayfair brand
    "jossandmain.com": 3,    # Wayfair brand
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com/",
}

IMG_HEADERS = {
    "User-Agent": HEADERS["User-Agent"],
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://duckduckgo.com/",
}

GENERIC_WORDS = {
    "the", "and", "with", "for", "from", "inch", "wide",
    "tall", "deep", "set", "piece", "pack", "home", "room",
    "living", "outdoor", "indoor", "new", "best", "top",
}


async def ddg_maybe_cooldown(got_results: bool, reason: str = "") -> None:
    """Global backoff to avoid DDG throttling. Safe even with concurrency>1."""
    global _ddg_calls, _empty_streak

    async with _ddg_lock:
        _ddg_calls += 1
        if got_results:
            _empty_streak = 0
        else:
            _empty_streak += 1

        # periodic cooldown
        if _ddg_calls % COOLDOWN_EVERY_CALLS == 0:
            t = random.uniform(*COOLDOWN_RANGE_SEC)
            print(f"🧊 DDG cooldown {t/60:.1f} min (calls={_ddg_calls}) {reason}")
            await asyncio.sleep(t)

        # throttling cooldown
        if _empty_streak >= EMPTY_STREAK_THRESHOLD:
            t = random.uniform(*THROTTLE_RANGE_SEC)
            print(f"🚫 DDG likely throttled ({_empty_streak} empty). Sleep {t/60:.1f} min. {reason}")
            _empty_streak = 0
            await asyncio.sleep(t)


# ════════════════════════════════════════════════════════════════
# Progress Tracker (resume-safe)
# ════════════════════════════════════════════════════════════════

class ProgressTracker:
    """Track which products are done. Survives restarts."""

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.manifest = {}
        self._dirty = False
        self._lock = asyncio.Lock()
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

    async def save(self):
        async with self._lock:
            if not self._dirty:
                return
            tmp = self.manifest_path + ".tmp"
            async with aiofiles.open(tmp, "w") as f:
                await f.write(json.dumps(self.manifest, indent=2))
            os.replace(tmp, self.manifest_path)
            self._dirty = False

    def save_sync(self):
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
                and entry.get("status") in ("no_results",
                                             "download_failed",
                                             "no_good_images"))

    async def record_success(self, product_id: str, product_name: str,
                             image_paths: list, source_url: str,
                             source_domain: str):
        async with self._lock:
            self.manifest[str(product_id)] = {
                "product_name": product_name,
                "image_count": len(image_paths),
                "image_paths": image_paths,
                "source_url": source_url,
                "source_domain": source_domain,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
            }
            self._dirty = True

    async def record_failure(self, product_id: str, product_name: str,
                             reason: str):
        async with self._lock:
            self.manifest[str(product_id)] = {
                "product_name": product_name,
                "image_count": 0,
                "status": reason,
                "timestamp": datetime.now().isoformat(),
            }
            self._dirty = True

    def summary(self):
        total = len(self.manifest)
        with_imgs = sum(1 for v in self.manifest.values()
                        if v.get("image_count", 0) > 0)
        total_imgs = sum(v.get("image_count", 0)
                         for v in self.manifest.values())
        no_results = sum(1 for v in self.manifest.values()
                         if v.get("status") in ("no_results",
                                                  "no_good_images"))
        download_fail = sum(1 for v in self.manifest.values()
                            if v.get("status") == "download_failed")

        # Domain distribution
        domain_counts = defaultdict(int)
        for v in self.manifest.values():
            d = v.get("source_domain", "")
            if d:
                domain_counts[d] += 1

        # Status breakdown
        status_counts = defaultdict(int)
        for v in self.manifest.values():
            status_counts[v.get("status", "unknown")] += 1

        print(f"\n{'='*60}")
        print(f"IMAGE SOURCING SUMMARY")
        print(f"{'='*60}")
        print(f"Products processed:      {total}")
        print(f"Products WITH images:    {with_imgs}")
        print(f"Products WITHOUT:        {no_results + download_fail}")
        print(f"  No search results:     {no_results}")
        print(f"  Download failed:       {download_fail}")
        print(f"Total images:            {total_imgs}")
        if with_imgs:
            print(f"Avg images/product:      "
                  f"{total_imgs / with_imgs:.1f}")
            print(f"Success rate:            "
                  f"{with_imgs/max(total,1)*100:.1f}%")

        if status_counts:
            print(f"\nStatus breakdown:")
            for s, c in sorted(status_counts.items(),
                                key=lambda x: -x[1]):
                print(f"  {s:25s}: {c}")

        if domain_counts:
            print(f"\nSource domains (top 15):")
            for d, c in sorted(domain_counts.items(),
                                key=lambda x: -x[1])[:15]:
                print(f"  {d:35s} {c:5d} "
                      f"({c/max(with_imgs,1)*100:.1f}%)")
        print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════
# DDG Image Search (raw API — proven to work)
# ════════════════════════════════════════════════════════════════

async def ddg_image_search(session: aiohttp.ClientSession,
                           query: str,
                           max_results: int = 30) -> list:
    """Search DuckDuckGo for images using raw API.

    Returns list of dicts: {image, url, title, source}
    """
    try:
        # Step 1: Get VQD token
        search_url = (f"{DDG_URL}?q={quote_plus(query)}"
                      f"&iax=images&ia=images")
        async with session.get(
            search_url,
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=SEARCH_TIMEOUT),
            allow_redirects=True,
        ) as resp:
            text = await resp.text()
            vqd = ""
            # Try multiple VQD extraction patterns
            for pattern in [
                r'vqd=["\']([^"\']+)',
                r'vqd=([a-zA-Z0-9_-]+)',
                r'"vqd":"([^"]+)"',
            ]:
                match = re.search(pattern, text)
                if match:
                    vqd = match.group(1)
                    break
            if not vqd:
                return []

        # Step 2: Fetch image results from i.js
        params = {
            "l": "us-en",
            "o": "json",
            "q": query,
            "vqd": vqd,
            "f": ",,,,,",
            "p": "1",
        }
        async with session.get(
            DDG_IMAGES_URL,
            headers={**HEADERS, "Referer": search_url},
            params=params,
            timeout=aiohttp.ClientTimeout(total=SEARCH_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json(content_type=None)
            results = data.get("results", [])

            output = []
            for r in results[:max_results]:
                img_url = r.get("image", "")
                if not img_url:
                    continue
                output.append({
                    "image": img_url,
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "source": r.get("source", ""),
                })
            return output

    except Exception:
        return []


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def is_blocked(url: str) -> bool:
    """Check if URL is from a blocked domain."""
    domain = extract_domain(url)
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


# ════════════════════════════════════════════════════════════════
# Query Building (V4 — attribute-enriched)
# ════════════════════════════════════════════════════════════════

def build_search_query(product: dict) -> str:
    """Build attribute-enriched search query for DDG.

    Strategy:
      - primary_color:    always include (visual match critical)
      - primary_material: always include (texture match critical)
      - shape:            include if non-default (round, oval, l-shaped, etc.)
      - product_name:     always include (identifies the product)
      - style:            include (narrows aesthetic)

    Examples:
      "gray wood holly aged wood 3 drawer chest farmhouse wayfair"
      "round brown wood coffee table modern wayfair"
      "black leather arm chair industrial wayfair"
      "blue fabric l-shaped sectional sofa coastal wayfair"
    """
    parts = []

    # Primary color (most important visual match)
    color = product.get("primary_color")
    if color and color not in ("other", "multi"):
        color_map = {"gold_metal": "gold", "natural_fiber": "natural"}
        parts.append(color_map.get(color, color))

    # Primary material
    material = product.get("primary_material")
    if material and material not in ("other", "mixed"):
        mat_map = {"natural_fiber": "natural fiber", "synthetics": "synthetic"}
        parts.append(mat_map.get(material, material))

    # Shape (only non-default — rectangular is assumed for most furniture)
    shape = product.get("shape")
    if shape and shape not in ("rectangular", "other", None):
        parts.append(shape)

    # Product name (core identifier)
    parts.append(product["product_name"])

    # Style (helps narrow aesthetic)
    style = product.get("style")
    if style and style not in ("other",):
        parts.append(style)

    query = " ".join(parts) + " wayfair"
    return query


def get_attribute_words(product: dict) -> set:
    """Extract attribute words for relevance scoring boost."""
    words = set()
    for attr in ("primary_color", "primary_material", "style"):
        val = product.get(attr)
        if val and val not in ("other", "multi", "mixed"):
            for w in val.replace("_", " ").split():
                if len(w) >= 3:
                    words.add(w.lower())

    shape = product.get("shape")
    if shape and shape not in ("rectangular", "other"):
        for w in shape.replace("-", " ").split():
            if len(w) >= 3:
                words.add(w.lower())

    return words


# ════════════════════════════════════════════════════════════════
# Relevance Scoring (V4 — attribute-aware)
# ════════════════════════════════════════════════════════════════

def score_result(result: dict, name_words: set,
                 attr_words: set = None) -> float:
    """Score a search result by relevance to product.

    Considers both product name match AND attribute word match.
    Higher = better match.
    """
    img_url = result.get("image", "")
    source_url = result.get("url", "")
    title = result.get("title", "").lower()

    # Block bad domains
    if is_blocked(img_url) or is_blocked(source_url):
        return -1

    score = 0.0

    # Domain preference bonus
    img_domain = extract_domain(img_url)
    source_domain = extract_domain(source_url)
    for pref_domain, bonus in PREFERRED_DOMAINS.items():
        if pref_domain in img_domain or pref_domain in source_domain:
            score += bonus
            break

    # Product name word matching (most important signal)
    combined = (source_url + " " + title + " " +
                result.get("source", "")).lower()
    matched = sum(1 for w in name_words if w in combined)
    if name_words:
        score += (matched / len(name_words)) * 10

    # Attribute word matching (color, material, shape, style)
    if attr_words:
        attr_matched = sum(1 for w in attr_words if w in combined)
        score += (attr_matched / len(attr_words)) * 5

    # Image URL quality heuristics
    img_lower = img_url.lower()
    if any(dim in img_lower for dim in
           ["1000", "1200", "1500", "2000", "large", "full"]):
        score += 1
    if any(dim in img_lower for dim in
           ["thumb", "50x50", "75x75", "100x", "icon", "tiny"]):
        score -= 5
    if img_lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
        score += 0.5

    return score


def select_best_image(results: list, product_name: str,
                      attr_words: set = None) -> dict:
    """Pick the best image result for a product.

    Uses both product name and attribute words for scoring.
    Returns best result dict or None.
    """
    name_words = set(
        w.lower() for w in re.findall(r'[a-zA-Z]{3,}', product_name)
    )
    name_words -= GENERIC_WORDS

    scored = []
    for r in results:
        s = score_result(r, name_words, attr_words)
        if s >= 0:
            scored.append((s, r))

    if not scored:
        return None

    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


# ════════════════════════════════════════════════════════════════
# Image Download + Center Crop
# ════════════════════════════════════════════════════════════════

async def download_image(session: aiohttp.ClientSession,
                         url: str, save_path: str) -> bool:
    """Download a single image. Returns True on success."""
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(
                url,
                headers=IMG_HEADERS,
                timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT),
                allow_redirects=True,
            ) as resp:
                if resp.status != 200:
                    continue
                data = await resp.read()

                if len(data) < MIN_IMAGE_BYTES:
                    return False
                if len(data) > MAX_IMAGE_BYTES:
                    return False

                # Verify it's actually an image
                if not (data[:3] == b'\xff\xd8\xff' or      # JPEG
                        data[:8] == b'\x89PNG\r\n\x1a\n' or  # PNG
                        data[:4] == b'RIFF' or                # WebP
                        data[:4] == b'GIF8'):                 # GIF
                    return False

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                async with aiofiles.open(save_path, "wb") as f:
                    await f.write(data)
                return True

        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
    return False


def generate_center_crop(hero_path: str, crop_path: str,
                         crop_ratio: float = CROP_RATIO) -> bool:
    """Generate center crop from hero image for material detail."""
    try:
        img = Image.open(hero_path)
        w, h = img.size
        if w < 100 or h < 100:
            return False

        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2

        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        cropped.save(crop_path, quality=90)
        return True
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════
# Product Processing Pipeline
# ════════════════════════════════════════════════════════════════

async def process_product(product: dict,
                          session: aiohttp.ClientSession,
                          tracker: ProgressTracker,
                          output_dir: str,
                          search_semaphore: asyncio.Semaphore):
    """Process one product: build query → search → score → download → crop."""
    pid = product["product_id"]
    name = product["product_name"]

    if tracker.is_done(pid):
        return

    # Build attribute-enriched query
    query = build_search_query(product)
    attr_words = get_attribute_words(product)

    # Rate-limited search
    async with search_semaphore:
        delay = random.uniform(SEARCH_DELAY_MIN, SEARCH_DELAY_MAX)
        await asyncio.sleep(delay)
        results = await ddg_image_search(session, query, max_results=30)

    if not results:
        await tracker.record_failure(pid, name, "no_results")
        return

    # Pick best image (with attribute word scoring)
    best = select_best_image(results, name, attr_words)
    if not best:
        await tracker.record_failure(pid, name, "no_good_images")
        return

    # Download hero image
    product_dir = os.path.join(output_dir, pid)
    hero_path = os.path.join(product_dir, "hero.jpg")
    crop_path = os.path.join(product_dir, "material_crop.jpg")

    success = await download_image(session, best["image"], hero_path)
    if not success:
        await tracker.record_failure(pid, name, "download_failed")
        return

    # Generate center crop
    image_paths = [hero_path]
    crop_ok = generate_center_crop(hero_path, crop_path)
    if crop_ok:
        image_paths.append(crop_path)

    source_domain = extract_domain(best["image"])
    await tracker.record_success(
        pid, name, image_paths,
        best.get("url", ""), source_domain)


# ════════════════════════════════════════════════════════════════
# Main Pipeline
# ════════════════════════════════════════════════════════════════

async def run_pipeline(queue: list, tracker: ProgressTracker,
                       output_dir: str,
                       concurrent_searches: int,
                       save_every: int):
    """Main async pipeline."""
    search_semaphore = asyncio.Semaphore(concurrent_searches)

    connector = aiohttp.TCPConnector(
        limit=concurrent_searches + 20,
        limit_per_host=10,
        ttl_dns_cache=300,
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        total = len(queue)
        completed = 0
        start_time = time.time()

        for chunk_start in range(0, total, save_every):
            chunk = queue[chunk_start:chunk_start + save_every]
            tasks = [
                process_product(p, session, tracker,
                                output_dir, search_semaphore)
                for p in chunk
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            completed += len(chunk)
            await tracker.save()

            elapsed = time.time() - start_time
            rate = completed / max(elapsed, 1) * 3600
            done = sum(1 for v in tracker.manifest.values()
                       if v.get("image_count", 0) > 0)
            total_imgs = sum(v.get("image_count", 0)
                             for v in tracker.manifest.values())
            failed = completed - done
            pct = done / max(completed, 1) * 100
            eta_hrs = (total - completed) / max(rate, 1)

            print(
                f"[{completed}/{total}] "
                f"{done} success ({pct:.0f}%), "
                f"{failed} failed, "
                f"{total_imgs} imgs, "
                f"{rate:.0f}/hr, "
                f"ETA {eta_hrs:.1f}h"
            )

    await tracker.save()


def load_queue(queue_path: str, tracker: ProgressTracker,
               retry_failed: bool) -> list:
    """Load queue from image_queue.json, skip completed."""
    with open(queue_path) as f:
        all_products = json.load(f)

    queue = []
    skipped = 0
    for p in all_products:
        pid = p["product_id"]
        if tracker.is_done(pid):
            skipped += 1
            continue
        if not retry_failed and tracker.is_failed(pid):
            skipped += 1
            continue
        queue.append(p)

    total = len(all_products)
    remaining = len(queue)
    print(f"Loaded queue: {remaining} remaining "
          f"(from {total} total, {skipped} skipped)")
    return queue


def main():
    parser = argparse.ArgumentParser(
        description="Source product images via DuckDuckGo")
    parser.add_argument(
        "--queue", type=str, required=True,
        help="Path to image_queue.json")
    parser.add_argument(
        "--output-dir", type=str,
        default="data/images/wayfair")
    parser.add_argument(
        "--manifest", type=str,
        default="data/images/wayfair_manifest.json")
    parser.add_argument(
        "--concurrent-searches", type=int, default=1,
        help="Max concurrent DDG searches")
    parser.add_argument(
        "--save-every", type=int, default=15,
        help="Save manifest every N products")
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry previously failed products")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_dir = os.path.dirname(args.manifest)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    tracker = ProgressTracker(args.manifest)
    queue = load_queue(args.queue, tracker, args.retry_failed)

    if not queue:
        print("Nothing to do — all products processed!")
        tracker.summary()
        return

    est_hours = len(queue) * 3.0 / 3600  # ~3s per product
    print(f"\n{'='*60}")
    print(f"PRODUCT IMAGE SOURCER V4 (Attribute-Enriched Queries)")
    print(f"{'='*60}")
    print(f"Queue:             {len(queue)} products")
    print(f"Output:            {args.output_dir}")
    print(f"Strategy:          1 hero + 1 center crop")
    print(f"Query:             [color] [material] [shape] [name] [style] wayfair")
    print(f"Source:            DDG images (any domain)")
    print(f"Blocked:           pinterest, stock photos, etc.")
    print(f"Preferred:         wfcdn.com, wayfair.com + brands")
    print(f"Delay:             {SEARCH_DELAY_MIN}-{SEARCH_DELAY_MAX}s")
    print(f"Concurrency:       {args.concurrent_searches} searches")
    print(f"Est. time:         ~{est_hours:.1f} hours")
    print(f"{'='*60}\n")

    # Preview first 3 queries
    for p in queue[:3]:
        q = build_search_query(p)
        print(f"  Query: {q}")
    if len(queue) > 3:
        print(f"  ... and {len(queue) - 3} more")
    print()

    try:
        asyncio.run(run_pipeline(
            queue, tracker, args.output_dir,
            args.concurrent_searches,
            args.save_every))
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        tracker.save_sync()
        print("Progress saved. Rerun to resume.")

    tracker.summary()


if __name__ == "__main__":
    main()