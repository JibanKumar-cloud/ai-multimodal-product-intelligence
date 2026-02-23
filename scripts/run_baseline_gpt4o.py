#!/usr/bin/env python3
"""Run GPT-4o baseline evaluation on WANDS products.

Establishes the quality upper-bound and cost reference point.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/run_baseline_gpt4o.py --sample-size 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.models.gpt4o_extractor import GPT4oExtractor
from src.utils.config import load_config
from src.utils.cost_tracker import CostTracker
from src.utils.logger import setup_logger


def run_gpt4o_baseline(
    test_data_path: str | Path = "data/processed/test.jsonl",
    config_path: str | Path = "configs/baseline_gpt4o.yaml",
    sample_size: int = 500,
    output_dir: str | Path = "outputs/predictions",
) -> dict:
    """Run GPT-4o baseline and save predictions.

    Args:
        test_data_path: Path to test JSONL file.
        config_path: GPT-4o configuration file.
        sample_size: Number of products to evaluate.
        output_dir: Directory to save predictions.

    Returns:
        Summary of the baseline run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(config_path)

    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    examples = []
    with open(test_data_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Sample if needed
    if sample_size and sample_size < len(examples):
        import random
        random.seed(42)
        examples = random.sample(examples, sample_size)

    logger.info(f"Evaluating on {len(examples)} products")

    # Initialize extractor
    cost_tracker = CostTracker(log_file=output_dir / "gpt4o_cost_log.jsonl")
    extractor = GPT4oExtractor(
        model_name=config["model"]["name"],
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["max_tokens"],
        system_prompt=config["prompting"]["system_prompt"],
        cost_tracker=cost_tracker,
    )
    extractor.load()

    # Run extraction
    predictions = []
    ground_truth = []

    for i, example in enumerate(examples):
        pred = extractor.extract(
            product_name=example["input_text"].split("\n")[0].replace("Product: ", ""),
            product_description=example["input_text"],
            product_class=example.get("product_class"),
        )
        predictions.append({
            "product_id": example.get("product_id"),
            "prediction": pred,
            "ground_truth": example["target_attributes"],
        })
        ground_truth.append(example["target_attributes"])

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(examples)} — Cost so far: ${cost_tracker.total_cost:.4f}")

    # Save predictions
    pred_path = output_dir / "gpt4o_predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p, default=str) + "\n")

    logger.info(f"Predictions saved to {pred_path}")

    # Print cost summary
    cost_tracker.print_summary()

    # Estimate full catalog cost
    full_est = cost_tracker.estimate_full_catalog_cost(
        cost_per_product=cost_tracker.total_cost / max(len(examples), 1),
    )
    logger.info(f"Estimated full catalog cost (40M products): {full_est['total_cost_formatted']}")

    return {
        "num_evaluated": len(examples),
        "cost_summary": cost_tracker.summary(),
        "predictions_path": str(pred_path),
    }


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Run GPT-4o baseline")
    parser.add_argument("--test-data", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--config", type=str, default="configs/baseline_gpt4o.yaml")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    run_gpt4o_baseline(
        test_data_path=args.test_data,
        config_path=args.config,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
