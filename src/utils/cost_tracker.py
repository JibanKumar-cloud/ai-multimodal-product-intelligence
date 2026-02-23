"""API cost monitoring and tracking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from loguru import logger


# Approximate pricing per 1K tokens (as of 2024)
PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}

# Approximate GPU costs per hour
GPU_PRICING = {
    "a100_40gb": 1.10,      # $/hr typical cloud
    "a100_80gb": 1.60,
    "l4": 0.35,
    "t4": 0.20,
    "v100": 0.74,
}


@dataclass
class APICallRecord:
    """Record of a single API call."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: Optional[str] = None

    @property
    def cost(self) -> float:
        """Calculate cost of this API call."""
        if self.model not in PRICING:
            return 0.0
        pricing = PRICING[self.model]
        return (
            self.input_tokens / 1000 * pricing["input"]
            + self.output_tokens / 1000 * pricing["output"]
        )


class CostTracker:
    """Track and report API costs across an experiment."""

    def __init__(self, log_file: Optional[str | Path] = None):
        self.records: list[APICallRecord] = []
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def record(self, call: APICallRecord) -> None:
        """Record an API call."""
        self.records.append(call)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(call)) + "\n")

    @property
    def total_cost(self) -> float:
        """Total cost across all recorded calls."""
        return sum(r.cost for r in self.records)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    @property
    def success_rate(self) -> float:
        """Fraction of successful calls."""
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.success) / len(self.records)

    def summary(self) -> dict:
        """Generate cost summary."""
        return {
            "total_calls": len(self.records),
            "successful_calls": sum(1 for r in self.records if r.success),
            "total_cost_usd": round(self.total_cost, 4),
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "cost_per_1k_products": round(self.total_cost / max(len(self.records), 1) * 1000, 2),
        }

    def print_summary(self) -> None:
        """Print a formatted cost summary."""
        s = self.summary()
        logger.info("=" * 50)
        logger.info("COST SUMMARY")
        logger.info(f"  Total calls:       {s['total_calls']}")
        logger.info(f"  Successful:        {s['successful_calls']}")
        logger.info(f"  Total cost:        ${s['total_cost_usd']:.4f}")
        logger.info(f"  Cost per 1K:       ${s['cost_per_1k_products']:.2f}")
        logger.info(f"  Total tokens:      {s['total_tokens']:,}")
        logger.info(f"  Avg latency:       {s['avg_latency_ms']:.0f}ms")
        logger.info("=" * 50)

    @staticmethod
    def estimate_full_catalog_cost(
        cost_per_product: float,
        catalog_size: int = 40_000_000,
    ) -> dict:
        """Estimate cost to process Wayfair's full catalog.

        Args:
            cost_per_product: Cost per single product inference.
            catalog_size: Total products in catalog.

        Returns:
            Cost breakdown dictionary.
        """
        total = cost_per_product * catalog_size
        return {
            "catalog_size": catalog_size,
            "cost_per_product": cost_per_product,
            "total_cost_usd": round(total, 2),
            "total_cost_formatted": f"${total:,.2f}",
        }
