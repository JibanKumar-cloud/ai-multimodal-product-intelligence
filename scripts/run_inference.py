#!/usr/bin/env python3
"""Run batch inference with any model type. Supports multimodal.

Usage:
    python scripts/run_inference.py --model rule-based
    python scripts/run_inference.py --model llava --model-path outputs/checkpoints/best_model
    python scripts/run_inference.py --model gpt4o --sample-size 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.models.rule_based import RuleBasedExtractor
from src.models.llava_extractor import LLaVAExtractor
from src.models.gpt4o_extractor import GPT4oExtractor
from src.inference.pipeline import InferencePipeline
from src.inference.postprocessor import PostProcessor
from src.inference.cache import InferenceCache
from src.utils.logger import setup_logger


def build_model(model_type: str, model_path=None):
    if model_type == "rule-based":
        return RuleBasedExtractor()
    elif model_type == "llava-base":
        return LLaVAExtractor(adapter_path=None)
    elif model_type == "llava":
        if not model_path:
            raise ValueError("--model-path required for llava")
        return LLaVAExtractor(adapter_path=model_path)
    elif model_type == "gpt4o":
        return GPT4oExtractor()
    else:
        raise ValueError(f"Unknown model: {model_type}")


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model", type=str, required=True,
                        choices=["rule-based", "llava-base", "llava", "gpt4o"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--test-data", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    products, ground_truth = [], []
    with open(args.test_data) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            parts = ex["input_text"].split("\n")
            products.append({
                "product_id": ex.get("product_id"),
                "product_name": parts[0].replace("Product: ", "") if parts else "",
                "product_class": ex.get("product_class"),
                "product_description": "\n".join(parts[1:]) if len(parts) > 1 else "",
                "image_path": ex.get("image_path"),  # Pass image if available
            })
            ground_truth.append(ex["target_attributes"])

    if args.sample_size and args.sample_size < len(products):
        import random
        random.seed(42)
        indices = random.sample(range(len(products)), args.sample_size)
        products = [products[i] for i in indices]
        ground_truth = [ground_truth[i] for i in indices]

    n_img = sum(1 for p in products if p.get("image_path"))
    logger.info(f"Products: {len(products)} ({n_img} with images)")

    # Build model and pipeline
    extractor = build_model(args.model, args.model_path)
    pipeline = InferencePipeline(
        extractor=extractor,
        postprocessor=PostProcessor(),
        cache=InferenceCache(output_dir / "cache") if not args.no_cache else None,
        batch_size=args.batch_size,
    )

    results = pipeline.run(products, output_dir / f"{args.model}_predictions.jsonl")

    # Save eval-ready file with ground truth
    eval_path = output_dir / f"{args.model}_eval_ready.jsonl"
    with open(eval_path, "w") as f:
        for result, gt in zip(results, ground_truth):
            f.write(json.dumps({
                "product_id": result.get("product_id"),
                "prediction": result.get("prediction"),
                "ground_truth": gt,
                "model": args.model,
                "has_image": bool(result.get("has_image")),
            }, default=str) + "\n")

    logger.info(f"Eval-ready: {eval_path}")


if __name__ == "__main__":
    main()
