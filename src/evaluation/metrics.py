"""Evaluation metrics for attribute extraction quality.

Provides per-attribute F1, exact match, and aggregate metrics.
Designed to produce the comparison tables shown in the README.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from loguru import logger


def compute_attribute_f1(
    predictions: list[dict],
    ground_truth: list[dict],
    attributes: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    """Compute per-attribute precision, recall, and F1.

    Args:
        predictions: List of predicted attribute dicts.
        ground_truth: List of ground truth attribute dicts.
        attributes: Which attributes to evaluate. None = all.

    Returns:
        Nested dict: {attribute_name: {precision, recall, f1, support}}.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) "
            "must have the same length"
        )

    if attributes is None:
        # Auto-detect from ground truth
        all_keys = set()
        for gt in ground_truth:
            all_keys.update(gt.keys())
        attributes = sorted(all_keys)

    results = {}
    for attr in attributes:
        tp, fp, fn = 0, 0, 0

        for pred, gt in zip(predictions, ground_truth):
            pred_val = _normalize_for_comparison(pred.get(attr))
            gt_val = _normalize_for_comparison(gt.get(attr))

            if gt_val is None:
                continue  # Skip if no ground truth for this attribute

            if isinstance(gt_val, list):
                # Multi-label comparison (e.g., room_type)
                pred_set = set(pred_val) if isinstance(pred_val, list) else set()
                gt_set = set(gt_val)
                tp += len(pred_set & gt_set)
                fp += len(pred_set - gt_set)
                fn += len(gt_set - pred_set)
            else:
                # Single-label comparison
                if pred_val == gt_val:
                    tp += 1
                elif pred_val is not None:
                    fp += 1
                    fn += 1
                else:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[attr] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return results


def compute_exact_match(
    predictions: list[dict],
    ground_truth: list[dict],
    attributes: Optional[list[str]] = None,
) -> float:
    """Compute exact match rate — all attributes must match.

    Args:
        predictions: List of predicted attribute dicts.
        ground_truth: List of ground truth attribute dicts.
        attributes: Which attributes to check.

    Returns:
        Fraction of examples where ALL attributes match exactly.
    """
    if not predictions:
        return 0.0

    if attributes is None:
        attributes = list(ground_truth[0].keys())

    matches = 0
    total = 0

    for pred, gt in zip(predictions, ground_truth):
        total += 1
        all_match = True
        for attr in attributes:
            pred_val = _normalize_for_comparison(pred.get(attr))
            gt_val = _normalize_for_comparison(gt.get(attr))
            if gt_val is not None and pred_val != gt_val:
                all_match = False
                break
        if all_match:
            matches += 1

    return round(matches / total, 4) if total > 0 else 0.0


def compute_weighted_f1(
    per_attribute_results: dict[str, dict[str, float]],
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute weighted average F1 across attributes.

    Args:
        per_attribute_results: Output from compute_attribute_f1.
        weights: Attribute importance weights.

    Returns:
        Weighted average F1 score.
    """
    default_weights = {
        "style": 1.5,
        "primary_material": 1.2,
        "color_family": 1.0,
        "room_type": 1.0,
        "product_type": 1.3,
    }
    weights = weights or default_weights

    total_weight = 0.0
    weighted_sum = 0.0

    for attr, metrics in per_attribute_results.items():
        w = weights.get(attr, 1.0)
        weighted_sum += w * metrics["f1"]
        total_weight += w

    return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0


def compute_full_evaluation(
    predictions: list[dict],
    ground_truth: list[dict],
    model_name: str = "unknown",
    cost_per_1k: Optional[float] = None,
    avg_latency_ms: Optional[float] = None,
) -> dict:
    """Run complete evaluation and return structured results.

    This is the main entry point for evaluation.
    """
    attributes = ["style", "primary_material", "color_family", "room_type", "product_type"]

    per_attr = compute_attribute_f1(predictions, ground_truth, attributes)
    exact_match = compute_exact_match(predictions, ground_truth, attributes)
    weighted_f1 = compute_weighted_f1(per_attr)

    # Compute average F1
    f1_scores = [per_attr[a]["f1"] for a in per_attr]
    avg_f1 = round(np.mean(f1_scores), 4) if f1_scores else 0.0

    return {
        "model_name": model_name,
        "num_examples": len(predictions),
        "per_attribute": per_attr,
        "average_f1": avg_f1,
        "weighted_f1": weighted_f1,
        "exact_match_rate": exact_match,
        "cost_per_1k": cost_per_1k,
        "avg_latency_ms": avg_latency_ms,
    }


def format_results_table(results: list[dict]) -> str:
    """Format evaluation results as a markdown table.

    Args:
        results: List of evaluation result dicts from compute_full_evaluation.

    Returns:
        Markdown-formatted comparison table.
    """
    # Header
    lines = [
        "| Model | Style F1 | Material F1 | Color F1 | Avg F1 | "
        "Exact Match | Cost/1K | Latency |",
        "|-------|----------|-------------|----------|--------|"
        "-------------|---------|---------|",
    ]

    for r in results:
        pa = r["per_attribute"]
        style_f1 = pa.get("style", {}).get("f1", "-")
        mat_f1 = pa.get("primary_material", {}).get("f1", "-")
        color_f1 = pa.get("color_family", {}).get("f1", "-")
        cost = f"${r['cost_per_1k']:.2f}" if r.get("cost_per_1k") else "-"
        latency = f"{r['avg_latency_ms']:.0f}ms" if r.get("avg_latency_ms") else "-"

        lines.append(
            f"| {r['model_name']} | {style_f1} | {mat_f1} | {color_f1} | "
            f"{r['average_f1']} | {r['exact_match_rate']} | {cost} | {latency} |"
        )

    return "\n".join(lines)


def _normalize_for_comparison(value) -> Optional[str | list[str] | bool]:
    """Normalize a value for comparison."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, list):
        return sorted([str(v).lower().strip() for v in value])
    return str(value).lower().strip()
