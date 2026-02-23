"""Error analysis for attribute extraction failures.

Identifies and categorizes failure modes to guide model improvement.
This is what makes the project research-grade — not just reporting numbers,
but understanding WHY the model fails.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import pandas as pd
from loguru import logger


class ErrorAnalyzer:
    """Analyze and categorize extraction failures."""

    # Known failure mode categories
    FAILURE_MODES = {
        "ambiguous_style": "Predicted a plausible but incorrect style",
        "multi_material": "Product has multiple materials, model picked wrong one",
        "color_lighting": "Color prediction affected by image lighting/context",
        "missing_attribute": "Model returned null for a non-null ground truth",
        "hallucinated": "Model predicted an attribute not present in ground truth",
        "wrong_category": "Completely wrong category (e.g., 'sofa' vs 'rug')",
        "granularity_mismatch": "Right general area but wrong specificity level",
    }

    def __init__(self):
        self.errors: list[dict] = []
        self.failure_counts = Counter()

    def analyze(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
        product_metadata: Optional[list[dict]] = None,
    ) -> dict:
        """Run full error analysis.

        Args:
            predictions: Predicted attributes.
            ground_truth: Ground truth attributes.
            product_metadata: Optional product info for context.

        Returns:
            Error analysis report.
        """
        self.errors = []
        self.failure_counts = Counter()

        attributes = ["style", "primary_material", "color_family", "product_type", "room_type"]

        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            metadata = product_metadata[i] if product_metadata else {}

            for attr in attributes:
                pred_val = self._normalize(pred.get(attr))
                gt_val = self._normalize(gt.get(attr))

                if gt_val is None:
                    continue  # No ground truth

                if pred_val != gt_val:
                    failure_mode = self._classify_failure(attr, pred_val, gt_val)
                    self.failure_counts[failure_mode] += 1

                    self.errors.append({
                        "index": i,
                        "attribute": attr,
                        "predicted": pred_val,
                        "ground_truth": gt_val,
                        "failure_mode": failure_mode,
                        "product_name": metadata.get("product_name", ""),
                        "product_class": metadata.get("product_class", ""),
                    })

        logger.info(f"Found {len(self.errors)} attribute-level errors")

        return self._build_report()

    def _classify_failure(self, attr: str, pred, gt) -> str:
        """Classify a single failure into a failure mode."""
        if pred is None:
            return "missing_attribute"

        if attr == "style":
            # Check if it's a close style (ambiguous boundary)
            similar_pairs = {
                frozenset({"contemporary", "modern"}),
                frozenset({"traditional", "transitional"}),
                frozenset({"farmhouse", "rustic"}),
                frozenset({"scandinavian", "minimalist"}),
                frozenset({"industrial", "modern"}),
            }
            if frozenset({str(pred), str(gt)}) in similar_pairs:
                return "ambiguous_style"

        if attr == "primary_material":
            return "multi_material"

        if attr == "color_family":
            return "color_lighting"

        if attr == "product_type":
            return "wrong_category"

        return "granularity_mismatch"

    def _build_report(self) -> dict:
        """Build structured error analysis report."""
        # Error distribution by attribute
        attr_errors = defaultdict(int)
        for err in self.errors:
            attr_errors[err["attribute"]] += 1

        # Error distribution by failure mode
        mode_dist = dict(self.failure_counts.most_common())

        # Top confused pairs per attribute
        confusion_pairs = defaultdict(lambda: Counter())
        for err in self.errors:
            pair = f"{err['ground_truth']} → {err['predicted']}"
            confusion_pairs[err["attribute"]][pair] += 1

        top_confusions = {}
        for attr, counter in confusion_pairs.items():
            top_confusions[attr] = counter.most_common(5)

        # Hardest product classes
        class_errors = Counter()
        for err in self.errors:
            if err.get("product_class"):
                class_errors[err["product_class"]] += 1

        return {
            "total_errors": len(self.errors),
            "errors_by_attribute": dict(attr_errors),
            "errors_by_failure_mode": mode_dist,
            "top_confusions": top_confusions,
            "hardest_product_classes": class_errors.most_common(10),
            "sample_errors": self.errors[:20],  # First 20 for inspection
        }

    def get_worst_examples(self, n: int = 20) -> list[dict]:
        """Get the N worst examples (most attribute errors per product)."""
        error_counts = Counter()
        for err in self.errors:
            error_counts[err["index"]] += 1

        return [
            {"index": idx, "num_errors": count}
            for idx, count in error_counts.most_common(n)
        ]

    def format_report(self, report: dict) -> str:
        """Format error analysis as readable markdown."""
        lines = ["## Error Analysis\n"]

        lines.append(f"**Total attribute-level errors:** {report['total_errors']}\n")

        lines.append("### Errors by Attribute")
        for attr, count in sorted(report["errors_by_attribute"].items(), key=lambda x: -x[1]):
            lines.append(f"- **{attr}**: {count}")

        lines.append("\n### Errors by Failure Mode")
        for mode, count in report["errors_by_failure_mode"].items():
            desc = self.FAILURE_MODES.get(mode, mode)
            lines.append(f"- **{mode}** ({count}): {desc}")

        lines.append("\n### Top Confusion Pairs")
        for attr, pairs in report["top_confusions"].items():
            lines.append(f"\n**{attr}:**")
            for pair, count in pairs:
                lines.append(f"  - {pair} ({count}x)")

        return "\n".join(lines)

    @staticmethod
    def _normalize(value):
        if value is None:
            return None
        if isinstance(value, list):
            return sorted([str(v).lower().strip() for v in value])
        return str(value).lower().strip()
