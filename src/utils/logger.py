"""Structured logging configuration."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_dir: str | Path | None = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure structured logging with loguru.

    Args:
        log_dir: Directory for log files. None = console only.
        level: Minimum log level.
        rotation: When to rotate log files.
        retention: How long to keep old log files.
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    # File handler if log_dir specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path / "wayfair_catalog_ai_{time}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="gz",
        )

    logger.info(f"Logger initialized at level={level}")
