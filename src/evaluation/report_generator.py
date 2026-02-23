"""Generate evaluation reports with comparison tables and visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger


def generate_report(
    evaluation_results: list[dict],
    error_report: Optional[dict] = None,
    output_dir: str | Path = "outputs/evaluation",
) -> Path:
    """Generate a complete evaluation report.

    Args:
        evaluation_results: List of results from compute_full_evaluation.
        error_report: Optional error analysis from ErrorAnalyzer.
        output_dir: Directory to save reports.

    Returns:
        Path to the generated markdown report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Evaluation Report: Wayfair Catalog AI\n",
        f"Evaluated **{len(evaluation_results)} models** on WANDS product data.\n",
    ]

    # Comparison table
    report_lines.append("## Model Comparison\n")
    report_lines.append(_build_comparison_table(evaluation_results))
    report_lines.append("")

    # Per-model details
    for result in evaluation_results:
        report_lines.append(f"\n### {result['model_name']}\n")
        report_lines.append(f"- Examples evaluated: {result['num_examples']}")
        report_lines.append(f"- Average F1: {result['average_f1']}")
        report_lines.append(f"- Weighted F1: {result['weighted_f1']}")
        report_lines.append(f"- Exact Match: {result['exact_match_rate']}")

        if result.get("cost_per_1k"):
            report_lines.append(f"- Cost per 1K products: ${result['cost_per_1k']:.2f}")
        if result.get("avg_latency_ms"):
            report_lines.append(f"- Avg latency: {result['avg_latency_ms']:.0f}ms")

        report_lines.append("\n**Per-attribute breakdown:**\n")
        report_lines.append("| Attribute | Precision | Recall | F1 | Support |")
        report_lines.append("|-----------|-----------|--------|-----|---------|")

        for attr, metrics in result["per_attribute"].items():
            report_lines.append(
                f"| {attr} | {metrics['precision']} | {metrics['recall']} | "
                f"{metrics['f1']} | {metrics['support']} |"
            )

    # Cost analysis
    report_lines.append("\n## Cost Analysis at Wayfair Scale (40M products)\n")
    report_lines.append("| Model | Cost per 1K | Full Catalog Cost | Feasible? |")
    report_lines.append("|-------|-------------|-------------------|-----------|")

    for result in evaluation_results:
        cost_1k = result.get("cost_per_1k", 0)
        full_cost = (cost_1k or 0) * 40_000
        feasible = "Yes" if full_cost < 50_000 else "Expensive" if full_cost < 500_000 else "No"
        report_lines.append(
            f"| {result['model_name']} | ${(cost_1k or 0):.2f} | "
            f"${full_cost:,.0f} | {feasible} |"
        )

    # Error analysis section
    if error_report:
        report_lines.append("\n## Error Analysis\n")
        report_lines.append(f"Total attribute-level errors: {error_report['total_errors']}\n")

        report_lines.append("### Errors by Failure Mode\n")
        report_lines.append("| Failure Mode | Count |")
        report_lines.append("|-------------|-------|")
        for mode, count in error_report.get("errors_by_failure_mode", {}).items():
            report_lines.append(f"| {mode} | {count} |")

    # Save
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    # Also save raw JSON
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Raw results saved to {json_path}")

    # Generate plots if matplotlib available
    try:
        _generate_plots(evaluation_results, output_dir)
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")

    return report_path


def _build_comparison_table(results: list[dict]) -> str:
    """Build markdown comparison table."""
    lines = [
        "| Model | Style F1 | Material F1 | Color F1 | Avg F1 | "
        "Exact Match | Cost/1K |",
        "|-------|----------|-------------|----------|--------|"
        "-------------|---------|",
    ]

    for r in results:
        pa = r["per_attribute"]
        style = pa.get("style", {}).get("f1", "-")
        mat = pa.get("primary_material", {}).get("f1", "-")
        color = pa.get("color_family", {}).get("f1", "-")
        cost = f"${r['cost_per_1k']:.2f}" if r.get("cost_per_1k") else "-"

        lines.append(
            f"| {r['model_name']} | {style} | {mat} | {color} | "
            f"{r['average_f1']} | {r['exact_match_rate']} | {cost} |"
        )

    return "\n".join(lines)


def _generate_plots(results: list[dict], output_dir: Path) -> None:
    """Generate comparison visualization plots."""
    import matplotlib.pyplot as plt
    import numpy as np

    # F1 comparison bar chart
    models = [r["model_name"] for r in results]
    attributes = ["style", "primary_material", "color_family", "product_type"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2

    for i, attr in enumerate(attributes):
        values = [r["per_attribute"].get(attr, {}).get("f1", 0) for r in results]
        ax.bar(x + i * width, values, width, label=attr)

    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Attribute F1 Comparison Across Models")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "f1_comparison.png", dpi=150)
    plt.close()

    # Cost vs Quality scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        cost = r.get("cost_per_1k", 0) or 0
        quality = r["average_f1"]
        ax.scatter(cost, quality, s=100, zorder=5)
        ax.annotate(r["model_name"], (cost, quality), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Cost per 1K Products ($)")
    ax.set_ylabel("Average F1 Score")
    ax.set_title("Cost vs Quality Tradeoff")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_vs_quality.png", dpi=150)
    plt.close()

    logger.info(f"Plots saved to {output_dir}")
