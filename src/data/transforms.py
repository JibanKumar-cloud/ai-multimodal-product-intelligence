"""Image and text preprocessing transforms."""

from __future__ import annotations

import re
from typing import Optional

from PIL import Image


def preprocess_image(
    image: Image.Image,
    target_size: int = 336,
) -> Image.Image:
    """Preprocess a product image for VLM input.

    Args:
        image: PIL Image.
        target_size: Target dimension (LLaVA expects 336x336).

    Returns:
        Preprocessed PIL Image.
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize maintaining aspect ratio, then center crop
    width, height = image.size
    scale = target_size / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Center crop to target_size x target_size
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    image = image.crop((left, top, left + target_size, top + target_size))

    return image


def clean_product_text(text: str) -> str:
    """Clean and normalize product text.

    Args:
        text: Raw product text (name, description, etc.)

    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep useful punctuation
    text = re.sub(r"[^\w\s.,;:'\"-/()]+", " ", text)
    # Strip
    text = text.strip()

    return text


def truncate_description(
    description: str,
    max_chars: int = 500,
    preserve_sentences: bool = True,
) -> str:
    """Intelligently truncate a product description.

    Args:
        description: Full product description.
        max_chars: Maximum character length.
        preserve_sentences: If True, truncate at sentence boundary.

    Returns:
        Truncated description.
    """
    description = clean_product_text(description)

    if len(description) <= max_chars:
        return description

    if preserve_sentences:
        # Find the last sentence boundary before max_chars
        truncated = description[:max_chars]
        last_period = truncated.rfind(".")
        last_excl = truncated.rfind("!")
        last_q = truncated.rfind("?")
        boundary = max(last_period, last_excl, last_q)

        if boundary > max_chars * 0.5:  # Only use boundary if we keep >50%
            return description[: boundary + 1]

    return description[:max_chars].rsplit(" ", 1)[0] + "..."


def format_product_input(
    product_name: str,
    product_class: Optional[str] = None,
    description: Optional[str] = None,
    max_desc_chars: int = 500,
) -> str:
    """Format product information into model input text.

    Args:
        product_name: Product title/name.
        product_class: Product category.
        description: Product description.
        max_desc_chars: Max chars for description.

    Returns:
        Formatted input string.
    """
    parts = [f"Product: {clean_product_text(product_name)}"]

    if product_class:
        parts.append(f"Category: {clean_product_text(product_class)}")

    if description:
        desc = truncate_description(description, max_desc_chars)
        if desc:
            parts.append(f"Description: {desc}")

    return "\n".join(parts)
