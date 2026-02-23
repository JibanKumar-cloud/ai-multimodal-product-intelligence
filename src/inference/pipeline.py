"""Batch inference pipeline for production-scale attribute extraction.

Orchestrates the full inference workflow:
1. Load products from input source
2. Run through the extractor model
3. Postprocess and validate outputs
4. Cache results
5. Generate inference report
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.models.attribute_extractor import BaseAttributeExtractor
from src.inference.postprocessor import PostProcessor
from src.inference.cache import InferenceCache


class InferencePipeline:
    """Orchestrate batch inference across products."""

    def __init__(
        self,
        extractor: BaseAttributeExtractor,
        postprocessor: Optional[PostProcessor] = None,
        cache: Optional[InferenceCache] = None,
        batch_size: int = 16,
    ):
        """Initialize inference pipeline.

        Args:
            extractor: The attribute extraction model.
            postprocessor: Output validator and normalizer.
            cache: Inference result cache.
            batch_size: Products per batch.
        """
        self.extractor = extractor
        self.postprocessor = postprocessor or PostProcessor()
        self.cache = cache
        self.batch_size = batch_size

        # Metrics
        self.total_processed = 0
        self.total_cached = 0
        self.total_errors = 0
        self.latencies: list[float] = []

    def run(
        self,
        products: list[dict] | pd.DataFrame,
        output_path: Optional[str | Path] = None,
    ) -> list[dict]:
        """Run inference on a list of products.

        Args:
            products: List of product dicts or DataFrame.
            output_path: Path to save results as JSONL.

        Returns:
            List of result dictionaries with predictions.
        """
        if isinstance(products, pd.DataFrame):
            products = products.to_dict("records")

        logger.info(f"Starting inference on {len(products)} products")
        logger.info(f"Model: {self.extractor.model_name}")
        logger.info(f"Batch size: {self.batch_size}")

        if not self.extractor.is_loaded:
            self.extractor.load()

        results = []
        start_time = time.time()

        for i in tqdm(range(0, len(products), self.batch_size), desc="Inference"):
            batch = products[i : i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        total_time = time.time() - start_time

        # Save results
        if output_path:
            self._save_results(results, output_path)

        # Log summary
        self._log_summary(total_time, len(products))

        return results

    def _process_batch(self, batch: list[dict]) -> list[dict]:
        """Process a single batch of products."""
        batch_results = []

        for product in batch:
            product_id = product.get("product_id", "unknown")

            # Check cache
            if self.cache:
                cached = self.cache.get(str(product_id))
                if cached is not None:
                    batch_results.append(cached)
                    self.total_cached += 1
                    continue

            # Run inference
            start = time.time()
            try:
                raw_prediction = self.extractor.extract(
                    product_name=product.get("product_name", ""),
                    product_description=product.get("product_description"),
                    product_class=product.get("product_class"),
                    image_path=product.get("image_path"),
                )

                # Postprocess
                prediction = self.postprocessor.process(raw_prediction)

                latency_ms = (time.time() - start) * 1000
                self.latencies.append(latency_ms)

                result = {
                    "product_id": product_id,
                    "prediction": prediction,
                    "latency_ms": round(latency_ms, 1),
                    "model": self.extractor.model_name,
                }

                # Cache result
                if self.cache:
                    self.cache.set(str(product_id), result)

                batch_results.append(result)
                self.total_processed += 1

            except Exception as e:
                logger.error(f"Error processing product {product_id}: {e}")
                batch_results.append({
                    "product_id": product_id,
                    "prediction": None,
                    "error": str(e),
                    "model": self.extractor.model_name,
                })
                self.total_errors += 1

        return batch_results

    def _save_results(self, results: list[dict], output_path: str | Path) -> None:
        """Save inference results to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result, default=str) + "\n")

        logger.info(f"Results saved to {output_path}")

    def _log_summary(self, total_time: float, total_products: int) -> None:
        """Log inference summary."""
        logger.info("=" * 50)
        logger.info("INFERENCE SUMMARY")
        logger.info(f"  Total products:  {total_products}")
        logger.info(f"  Processed:       {self.total_processed}")
        logger.info(f"  From cache:      {self.total_cached}")
        logger.info(f"  Errors:          {self.total_errors}")
        logger.info(f"  Total time:      {total_time:.1f}s")

        if self.latencies:
            sorted_lat = sorted(self.latencies)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
            logger.info(f"  Latency p50:     {p50:.0f}ms")
            logger.info(f"  Latency p99:     {p99:.0f}ms")
            logger.info(f"  Throughput:      {total_products / total_time:.1f} products/sec")

        logger.info("=" * 50)
