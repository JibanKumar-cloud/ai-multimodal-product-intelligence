"""Simple file-based inference cache.

Avoids re-processing products that have already been extracted.
Uses JSONL files for persistence across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class InferenceCache:
    """File-backed cache for inference results."""

    def __init__(self, cache_dir: str | Path, cache_by: str = "product_id"):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files.
            cache_by: Key field for cache lookups.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_by = cache_by
        self._memory: dict[str, dict] = {}
        self._load_existing()

    def _cache_file(self) -> Path:
        return self.cache_dir / "inference_cache.jsonl"

    def _load_existing(self) -> None:
        """Load existing cache from disk."""
        cache_file = self._cache_file()
        if cache_file.exists():
            with open(cache_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            key = str(record.get(self.cache_by, ""))
                            if key:
                                self._memory[key] = record
                        except json.JSONDecodeError:
                            continue
            logger.info(f"Loaded {len(self._memory)} cached results from {cache_file}")

    def get(self, key: str) -> Optional[dict]:
        """Get a cached result.

        Args:
            key: Cache key (typically product_id).

        Returns:
            Cached result dict or None if not found.
        """
        return self._memory.get(key)

    def set(self, key: str, value: dict) -> None:
        """Cache a result.

        Args:
            key: Cache key.
            value: Result to cache.
        """
        self._memory[key] = value
        # Append to disk
        with open(self._cache_file(), "a") as f:
            f.write(json.dumps(value, default=str) + "\n")

    def clear(self) -> None:
        """Clear all cached results."""
        self._memory.clear()
        cache_file = self._cache_file()
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Cache cleared")

    def __len__(self) -> int:
        return len(self._memory)

    def __contains__(self, key: str) -> bool:
        return key in self._memory
