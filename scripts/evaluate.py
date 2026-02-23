#!/usr/bin/env python3
"""Evaluate and compare all model predictions.

Reads prediction files from outputs/predictions/ and generates
a comprehensive comparison report.

Usage:
    python scripts/evaluate.py --results-dir outputs/predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.evaluation.metrics import compute_full_evaluation, format_results_table
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.report_generator import generate_report
from src.utils.logger import setup_logger


# Cost estimates per 1K products for each model type
COST_ESTIMATES = {
    "rule-based": 0.0,
    "bert": 0.02,
    "llava-base": 0.20,
    "llava": 0.20,
    "gpt4o": 10.00,
}

LATENCY_ESTIMATES = {
    "rule-based": 2,
    "bert": 15,
    "llava-base": 380,
    "llava": 380,
    "gpt4o": 1200,
}


def evaluate_all(
    results_dir: str | Path = "outputs/predictions",
    output_dir: str | Path = "outputs/evaluation",
) -> None:
    """Evaluate all model predictions and generate comparison report.

    Args:
        results_dir: Directory containing *_eval_ready.jsonl files.
        output_dir: Directory to save evaluation reports.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all eval-ready files
    eval_files = sorted(results_dir.glob("*_eval_ready.jsonl"))

    if not eval_files:
        logger.error(f"No evaluation files found in {results_dir}")
        logger.info("Run inference first: python scripts/run_inference.py --model rule-based")
        return

    logger.info(f"Found {len(eval_files)} model result files")

    all_results = []

    for eval_file in eval_files:
        model_name = eval_file.stem.replace("_eval_ready", "")
        logger.info(f"\nEvaluating: {model_name}")

        # Load predictions and ground truth
        predictions = []
        ground_truth = []
        metadata = []

        with open(eval_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    pred = record.get("prediction")
                    gt = record.get("ground_truth")
                    if pred and gt:
                        predictions.append(pred)
                        ground_truth.append(gt)
                        metadata.append({"product_id": record.get("product_id")})

        if not predictions:
            logger.warning(f"No valid predictions in {eval_file}")
            continue

        # Compute metrics
        result = compute_full_evaluation(
            predictions=predictions,
            ground_truth=ground_truth,
            model_name=model_name,
            cost_per_1k=COST_ESTIMATES.get(model_name),
            avg_latency_ms=LATENCY_ESTIMATES.get(model_name),
        )

        all_results.append(result)
        logger.info(f"  Avg F1: {result['average_f1']}")
        logger.info(f"  Exact Match: {result['exact_match_rate']}")

    if not all_results:
        logger.error("No results to compare")
        return

    # Print comparison table
    table = format_results_table(all_results)
    logger.info("\n" + table)

    # Run error analysis on the best model
    best_model = max(all_results, key=lambda r: r["average_f1"])
    logger.info(f"\nRunning error analysis on best model: {best_model['model_name']}")

    # Reload best model's predictions for error analysis
    best_file = results_dir / f"{best_model['model_name']}_eval_ready.jsonl"
    best_preds, best_gt, best_meta = [], [], []
    with open(best_file) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("prediction") and record.get("ground_truth"):
                    best_preds.append(record["prediction"])
                    best_gt.append(record["ground_truth"])
                    best_meta.append({"product_id": record.get("product_id")})

    analyzer = ErrorAnalyzer()
    error_report = analyzer.analyze(best_preds, best_gt, best_meta)
    error_md = analyzer.format_report(error_report)
    logger.info("\n" + error_md)

    # Generate full report
    report_path = generate_report(
        evaluation_results=all_results,
        error_report=error_report,
        output_dir=output_dir,
    )

    logger.info(f"\nFull report: {report_path}")


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--results-dir", type=str, default="outputs/predictions")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    evaluate_all(results_dir=args.results_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
