"""Patch source_wayfair_images.py to use attribute-enriched search queries.

Before:  query = "holly aged wood 3 drawer chest wayfair product"
After:   query = "gray wood holly aged wood 3 drawer chest farmhouse wayfair"

Also updates relevance scoring to boost results matching attributes.

Usage:
    python scripts/patch_image_query.py
"""
import re

SCRIPT_PATH = "scripts/source_wayfair_images.py"

with open(SCRIPT_PATH) as f:
    code = f.read()

patches_applied = 0

# ════════════════════════════════════════════════════════════════
# PATCH 1: Add build_search_query() function
# Insert before process_product function
# ════════════════════════════════════════════════════════════════

NEW_QUERY_BUILDER = '''

def build_search_query(product: dict) -> str:
    """Build attribute-enriched search query for DDG.

    Strategy:
      - primary_color:    always include (visual match critical)
      - primary_material: always include (texture match critical)
      - shape:            include if non-default (round, oval, l-shaped, etc.)
      - product_name:     always include (identifies the product)
      - style:            include (narrows aesthetic)
      - secondary attrs:  skip (too noisy for search)

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
        # Convert internal names to search-friendly
        color_map = {"gold_metal": "gold", "natural_fiber": "natural"}
        parts.append(color_map.get(color, color))

    # Primary material
    material = product.get("primary_material")
    if material and material not in ("other", "mixed"):
        mat_map = {"natural_fiber": "natural fiber", "synthetics": "synthetic"}
        parts.append(mat_map.get(material, material))

    # Shape (only non-default — rectangular is assumed)
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
            # Split multi-word values
            for w in val.replace("_", " ").split():
                if len(w) >= 3:
                    words.add(w.lower())

    shape = product.get("shape")
    if shape and shape not in ("rectangular", "other"):
        for w in shape.replace("-", " ").split():
            if len(w) >= 3:
                words.add(w.lower())

    return words

'''

# Insert before process_product
ANCHOR = "async def process_product(product: dict,"
if ANCHOR in code:
    code = code.replace(ANCHOR, NEW_QUERY_BUILDER + ANCHOR)
    patches_applied += 1
    print("PATCH 1: Added build_search_query() + get_attribute_words()")
else:
    print("PATCH 1: FAILED - could not find process_product anchor")


# ════════════════════════════════════════════════════════════════
# PATCH 2: Update process_product to use new query builder
# Replace: query = f"{name} wayfair product"
# With:    query = build_search_query(product)
# And pass attribute words to select_best_image
# ════════════════════════════════════════════════════════════════

OLD_PROCESS = '''    # Rate-limited search
    async with search_semaphore:
        delay = random.uniform(SEARCH_DELAY_MIN, SEARCH_DELAY_MAX)
        await asyncio.sleep(delay)
        query = f"{name} wayfair product"
        results = await ddg_image_search(session, query, max_results=30)

    if not results:
        await tracker.record_failure(pid, name, "no_results")
        return

    # Pick best image
    best = select_best_image(results, name)'''

NEW_PROCESS = '''    # Build attribute-enriched query
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
    best = select_best_image(results, name, attr_words)'''

if OLD_PROCESS in code:
    code = code.replace(OLD_PROCESS, NEW_PROCESS)
    patches_applied += 1
    print("PATCH 2: Updated process_product query building")
else:
    print("PATCH 2: FAILED - could not find old process_product code")


# ════════════════════════════════════════════════════════════════
# PATCH 3: Update score_result to boost attribute matches
# ════════════════════════════════════════════════════════════════

OLD_SCORE = '''def score_result(result: dict, name_words: set) -> float:
    """Score a search result by relevance to product.

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

    return score'''

NEW_SCORE = '''def score_result(result: dict, name_words: set,
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
        if attr_words:
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

    return score'''

if OLD_SCORE in code:
    code = code.replace(OLD_SCORE, NEW_SCORE)
    patches_applied += 1
    print("PATCH 3: Updated score_result with attribute matching")
else:
    print("PATCH 3: FAILED - could not find old score_result code")


# ════════════════════════════════════════════════════════════════
# PATCH 4: Update select_best_image to accept and pass attr_words
# ════════════════════════════════════════════════════════════════

OLD_SELECT = '''def select_best_image(results: list, product_name: str) -> dict:
    """Pick the best image result for a product.

    Returns best result dict or None.
    """
    name_words = set(
        w.lower() for w in re.findall(r'[a-zA-Z]{3,}', product_name)
    )
    name_words -= GENERIC_WORDS

    scored = []
    for r in results:
        s = score_result(r, name_words)
        if s >= 0:
            scored.append((s, r))

    if not scored:
        return None

    scored.sort(key=lambda x: -x[0])
    return scored[0][1]'''

NEW_SELECT = '''def select_best_image(results: list, product_name: str,
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
    return scored[0][1]'''

if OLD_SELECT in code:
    code = code.replace(OLD_SELECT, NEW_SELECT)
    patches_applied += 1
    print("PATCH 4: Updated select_best_image with attr_words")
else:
    print("PATCH 4: FAILED - could not find old select_best_image code")


# ════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════

with open(SCRIPT_PATH, "w") as f:
    f.write(code)

print(f"\n{'='*60}")
print(f"Applied {patches_applied}/4 patches to {SCRIPT_PATH}")
if patches_applied == 4:
    print("All patches applied successfully!")
    print(f"\nQuery examples:")
    print(f"  Before: 'holly aged wood 3 drawer chest wayfair product'")
    print(f"  After:  'gray wood holly aged wood 3 drawer chest farmhouse wayfair'")
    print(f"\nRun image sourcing:")
    print(f"  python scripts/source_wayfair_images.py \\")
    print(f"    --queue data/processed/image_queue.json --retry-failed")
else:
    print("WARNING: Some patches failed. Check manually.")