#!/usr/bin/env python3
"""Pre-compute attributes for entire catalog via rule-based (~30 seconds)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loguru import logger
from src.utils.logger import setup_logger
from src.data.wands_loader import WANDSLoader

def enrich_catalog(wands_dir="data/raw/WANDS/dataset", output_path="data/enriched_catalog.jsonl"):
    setup_logger()
    from src.models.rule_based import RuleBasedExtractor
    extractor = RuleBasedExtractor()
    loader = WANDSLoader(wands_dir)
    products = loader.products
    logger.info(f"Enriching {len(products)} products...")
    start = time.time()
    results = []
    for i, (_, row) in enumerate(products.iterrows()):
        name = str(row.get("product_name", ""))
        desc = str(row.get("product_description", ""))
        cls = str(row.get("product_class", ""))
        features = str(row.get("product_features", ""))
        full_text = f"{name} {desc} {features}"
        attrs = extractor.extract(
            product_name=name if name != "nan" else "",
            product_description=full_text if full_text.strip() != "nan nan nan" else "",
            product_class=cls if cls != "nan" else "")
        results.append({"product_id": int(row["product_id"]), "product_name": name if name != "nan" else "",
                        "product_class": cls if cls != "nan" else "", "attributes": attrs})
        if (i + 1) % 5000 == 0:
            logger.info(f"  {i+1}/{len(products)} enriched")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    elapsed = time.time() - start
    logger.info(f"Done! {len(results)} products in {elapsed:.1f}s -> {output_path}")

if __name__ == "__main__":
    enrich_catalog()
