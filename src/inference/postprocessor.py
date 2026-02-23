"""Post-process model outputs into structured attribute dicts.

Handles multiple output formats:
1. Valid JSON: {"style": "modern", "color_family": "white"}
2. Key-value pairs: style: modern; color_family: white
3. Key-value with newlines: style: modern\ncolor_family: white
"""

from __future__ import annotations

import json
import re
from loguru import logger

# Standard attribute keys we expect
VALID_KEYS = {
    "style", "primary_material", "secondary_material",
    "color_family", "room_type", "product_type", "assembly_required",
}

# Default empty result
EMPTY_RESULT = {k: None for k in VALID_KEYS}


class PostProcessor:
    def __init__(self):
        self.valid_keys = VALID_KEYS

    def process(self, raw_output: str) -> dict:
        """Parse raw model output into structured attributes dict."""
        if not raw_output or not raw_output.strip():
            return dict(EMPTY_RESULT)

        # Clean up: take only first meaningful chunk (stop at repetitions)
        raw_output = self._truncate_repetitions(raw_output)

        # Try JSON first
        result = self._try_json(raw_output)
        if result:
            return self._normalize(result)

        # Try key-value parsing
        result = self._try_key_value(raw_output)
        if result:
            return self._normalize(result)

        # Last resort: regex extraction
        result = self._try_regex(raw_output)
        return self._normalize(result)

    def _truncate_repetitions(self, text: str) -> str:
        """Stop at first repetition or excessive output."""
        # Cut at double newline (usually marks end of first response)
        if "\n\n" in text:
            text = text.split("\n\n")[0]

        # Cut repeating color lists like "blue/white/yellow/pink/..."
        # If a slash-separated list is longer than 50 chars, truncate to first value
        text = re.sub(r'(\w+(?:/\w+){5,})', lambda m: m.group(0).split('/')[0], text)

        # Limit total length
        return text[:500]

    def _try_json(self, text: str) -> dict | None:
        """Try to extract JSON from output."""
        # Direct parse
        try:
            d = json.loads(text.strip())
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, ValueError):
            pass

        # Find JSON object in text
        match = re.search(r'\{[^{}]+\}', text)
        if match:
            try:
                d = json.loads(match.group())
                if isinstance(d, dict):
                    return d
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def _try_key_value(self, text: str) -> dict | None:
        """Parse 'key: value; key: value' or 'key: value\nkey: value' format."""
        result = {}

        # Split by newline first, then by semicolons within each line
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("Product:"):
                continue

            # Split by semicolons to get individual key:value pairs
            parts = re.split(r';\s*', line)
            for part in parts:
                part = part.strip()
                if ":" not in part:
                    continue

                # Split on FIRST colon only
                key, _, value = part.partition(":")
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                # Only accept known keys
                if key in self.valid_keys and value:
                    # Don't overwrite if we already have this key
                    if key not in result:
                        result[key] = value

        return result if result else None

    def _try_regex(self, text: str) -> dict:
        """Last resort: regex extraction for known patterns."""
        result = {}
        text_lower = text.lower()

        for key in self.valid_keys:
            # Look for "key: value" or "key = value"
            pattern = rf'{key}\s*[:=]\s*([^;\n]+)'
            match = re.search(pattern, text_lower)
            if match:
                result[key] = match.group(1).strip()

        return result

    def _normalize(self, result: dict) -> dict:
        """Normalize result: ensure all expected keys, clean values."""
        normalized = dict(EMPTY_RESULT)

        for key, value in result.items():
            clean_key = key.strip().lower().replace(" ", "_")
            if clean_key in self.valid_keys:
                if value is None:
                    normalized[clean_key] = None
                elif isinstance(value, str):
                    value = value.strip()
                    # Remove trailing semicolons and extra key-value bleed
                    if ";" in value:
                        value = value.split(";")[0].strip()
                    # Remove key bleed (e.g., "cotton; color_family: black")
                    for k in self.valid_keys:
                        if k + ":" in value:
                            value = value.split(k + ":")[0].strip().rstrip(";").strip()
                    # Truncate very long values
                    if len(value) > 100:
                        value = value[:50]
                    # Handle "none", "null", "n/a"
                    if value.lower() in ("none", "null", "n/a", ""):
                        normalized[clean_key] = None
                    else:
                        normalized[clean_key] = value
                elif isinstance(value, bool):
                    normalized[clean_key] = value
                else:
                    normalized[clean_key] = str(value)

        return normalized
