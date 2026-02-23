"""Base interface for attribute extraction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseAttributeExtractor(ABC):
    """Abstract base class for all attribute extractors.

    All extractors (GPT-4o, LLaVA, BERT, rule-based) implement this interface
    so they can be evaluated uniformly.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights / initialize API client."""
        pass

    @abstractmethod
    def extract(
        self,
        product_name: str,
        product_description: Optional[str] = None,
        product_class: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> dict:
        """Extract structured attributes from a single product.

        Args:
            product_name: Product title.
            product_description: Optional description text.
            product_class: Optional product category.
            image_path: Optional path to product image.

        Returns:
            Dictionary of extracted attributes matching ATTRIBUTE_SCHEMA.
        """
        pass

    def extract_batch(
        self,
        products: list[dict],
        batch_size: int = 16,
    ) -> list[dict]:
        """Extract attributes for a batch of products.

        Default implementation processes sequentially.
        Subclasses can override for parallel/batched processing.

        Args:
            products: List of product dictionaries.
            batch_size: Batch size for processing.

        Returns:
            List of attribute dictionaries.
        """
        results = []
        for product in products:
            result = self.extract(
                product_name=product.get("product_name", ""),
                product_description=product.get("product_description"),
                product_class=product.get("product_class"),
                image_path=product.get("image_path"),
            )
            results.append(result)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, loaded={self._is_loaded})"
