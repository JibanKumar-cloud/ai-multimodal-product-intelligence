"""GPT-4o baseline for product attribute extraction.

This serves as the upper-bound quality benchmark.
Uses structured prompting to extract attributes from product text.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.attribute_extractor import BaseAttributeExtractor
from src.utils.cost_tracker import CostTracker, APICallRecord


class GPT4oExtractor(BaseAttributeExtractor):
    """Extract product attributes using GPT-4o API."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        super().__init__(model_name=model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_tracker = cost_tracker or CostTracker()
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.client = None

    def _default_system_prompt(self) -> str:
        return (
            "You are a product catalog specialist for a home goods/furniture retailer. "
            "Your task is to extract structured product attributes from a product listing.\n\n"
            "Return ONLY valid JSON (no markdown, no explanation) with these fields:\n"
            '- style: Design style (e.g., "mid-century modern", "scandinavian", "farmhouse")\n'
            '- primary_material: Main material (e.g., "solid wood", "metal", "fabric")\n'
            "- secondary_material: Secondary material or null\n"
            '- color_family: Dominant color (e.g., "brown", "gray", "white")\n'
            '- room_type: List of suitable rooms (e.g., ["living room", "bedroom"])\n'
            '- product_type: Specific category (e.g., "sofa", "dining table")\n'
            "- assembly_required: true/false/null"
        )

    def load(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Export it or add to .env file."
                )
            self.client = OpenAI(api_key=api_key)
            self._is_loaded = True
            logger.info(f"GPT-4o extractor initialized (model={self.model_name})")
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _call_api(self, user_message: str) -> tuple[str, int, int]:
        """Make a single API call with retry logic.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens).
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return text, usage.prompt_tokens, usage.completion_tokens

    def extract(
        self,
        product_name: str,
        product_description: Optional[str] = None,
        product_class: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> dict:
        """Extract attributes using GPT-4o.

        Args:
            product_name: Product title.
            product_description: Product description text.
            product_class: Product category.
            image_path: Not used (text-only baseline).

        Returns:
            Extracted attribute dictionary.
        """
        if not self._is_loaded:
            self.load()

        # Build user message
        user_msg = f"Product Name: {product_name}"
        if product_class:
            user_msg += f"\nProduct Class: {product_class}"
        if product_description:
            desc = str(product_description)[:500]
            user_msg += f"\nDescription: {desc}"
        user_msg += "\n\nExtract structured attributes as JSON:"

        # Call API with timing
        start = time.time()
        try:
            response_text, input_tokens, output_tokens = self._call_api(user_msg)
            latency_ms = (time.time() - start) * 1000

            # Track cost
            self.cost_tracker.record(
                APICallRecord(
                    model=self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    success=True,
                )
            )

            # Parse JSON response
            return self._parse_response(response_text)

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            self.cost_tracker.record(
                APICallRecord(
                    model=self.model_name,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )
            )
            logger.error(f"GPT-4o extraction failed: {e}")
            return self._empty_result()

    def _parse_response(self, text: str) -> dict:
        """Parse GPT-4o JSON response with fallbacks."""
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse GPT-4o response as JSON: {text[:100]}...")
            return self._empty_result()

    @staticmethod
    def _empty_result() -> dict:
        """Return empty attribute dictionary."""
        return {
            "style": None,
            "primary_material": None,
            "secondary_material": None,
            "color_family": None,
            "room_type": None,
            "product_type": None,
            "assembly_required": None,
        }

    def extract_batch(
        self,
        products: list[dict],
        batch_size: int = 10,
    ) -> list[dict]:
        """Extract attributes for a batch of products.

        Uses sequential API calls with rate limiting.
        For true parallel processing, use async version.
        """
        results = []
        for i, product in enumerate(products):
            result = self.extract(
                product_name=product.get("product_name", ""),
                product_description=product.get("product_description"),
                product_class=product.get("product_class"),
            )
            results.append(result)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(products)} products")

        return results
