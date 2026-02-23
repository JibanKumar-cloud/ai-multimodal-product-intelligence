"""WANDS (Wayfair ANnotation DataSet) loader and parser.

Loads the three WANDS CSV files:
- product.csv: 42,994 products with features
- query.csv: 480 search queries
- label.csv: 233,448 query-product relevance labels

Reference: https://github.com/wayfaireng/WANDS
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class WANDSLoader:
    """Load and manage the WANDS dataset."""

    PRODUCT_COLS = [
        "product_id",
        "product_name",
        "product_class",
        "category_hierarchy",
        "product_description",
        "product_features",
    ]

    QUERY_COLS = ["query_id", "query", "query_class"]

    LABEL_COLS = ["id", "query_id", "product_id", "label"]

    # Relevance label mapping
    LABEL_MAP = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

    def __init__(self, data_dir: str | Path):
        """Initialize the WANDS loader.

        Args:
            data_dir: Path to directory containing WANDS CSV files.
        """
        self.data_dir = Path(data_dir)
        self._products: Optional[pd.DataFrame] = None
        self._queries: Optional[pd.DataFrame] = None
        self._labels: Optional[pd.DataFrame] = None

    def _find_csv(self, pattern: str) -> Path:
        """Find a CSV file matching a pattern in data_dir."""
        candidates = list(self.data_dir.rglob(f"*{pattern}*"))
        if not candidates:
            raise FileNotFoundError(
                f"No file matching '{pattern}' found in {self.data_dir}. "
                f"Run 'python scripts/download_wands.py' first."
            )
        return candidates[0]

    @property
    def products(self) -> pd.DataFrame:
        """Load products dataframe."""
        if self._products is None:
            path = self._find_csv("product.csv")
            self._products = pd.read_csv(path, sep="\t")
            logger.info(f"Loaded {len(self._products)} products from {path}")
        return self._products

    @property
    def queries(self) -> pd.DataFrame:
        """Load queries dataframe."""
        if self._queries is None:
            path = self._find_csv("query.csv")
            self._queries = pd.read_csv(path, sep="\t")
            logger.info(f"Loaded {len(self._queries)} queries from {path}")
        return self._queries

    @property
    def labels(self) -> pd.DataFrame:
        """Load labels dataframe."""
        if self._labels is None:
            path = self._find_csv("label.csv")
            self._labels = pd.read_csv(path, sep="\t")
            logger.info(f"Loaded {len(self._labels)} labels from {path}")
        return self._labels

    def get_product_by_id(self, product_id: int) -> dict:
        """Get a single product by ID."""
        row = self.products[self.products["product_id"] == product_id]
        if row.empty:
            raise ValueError(f"Product {product_id} not found")
        return row.iloc[0].to_dict()

    def get_products_with_features(self) -> pd.DataFrame:
        """Get products that have non-empty product_features."""
        df = self.products.copy()
        mask = df["product_features"].notna() & (df["product_features"].str.strip() != "")
        filtered = df[mask].copy()
        logger.info(
            f"Products with features: {len(filtered)} / {len(df)} "
            f"({len(filtered) / len(df) * 100:.1f}%)"
        )
        return filtered

    def get_product_classes(self) -> list[str]:
        """Get unique product classes."""
        return sorted(self.products["product_class"].dropna().unique().tolist())

    def summary(self) -> dict:
        """Get dataset summary statistics."""
        return {
            "num_products": len(self.products),
            "num_queries": len(self.queries),
            "num_labels": len(self.labels),
            "num_product_classes": self.products["product_class"].nunique(),
            "products_with_features": self.products["product_features"].notna().sum(),
            "label_distribution": self.labels["label"].value_counts().to_dict()
            if self._labels is not None
            else None,
        }
