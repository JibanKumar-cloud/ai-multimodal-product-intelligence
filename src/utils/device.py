"""GPU/CPU device management utilities."""

from __future__ import annotations

import torch
from loguru import logger


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available compute device.

    Args:
        preference: "auto", "cuda", "cpu", or specific "cuda:0".

    Returns:
        torch.device for computation.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")
    else:
        device = torch.device(preference)
        logger.info(f"Using device: {device}")

    return device


def print_gpu_stats() -> None:
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        logger.info("No GPU available")
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.info(
            f"GPU {i} ({torch.cuda.get_device_name(i)}): "
            f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Total={total:.1f}GB"
        )


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")
