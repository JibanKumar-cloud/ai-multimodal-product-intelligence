#!/usr/bin/env python3
"""Train LLaVA-7B with QLoRA on multimodal Wayfair catalog data.

Usage:
    python scripts/train.py --config configs/qlora_llava_7b.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.qlora_trainer import train
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.device import get_device, print_gpu_stats
from loguru import logger


def main():
    setup_logger(log_dir="outputs/logs")

    parser = argparse.ArgumentParser(description="Fine-tune LLaVA with QLoRA")
    parser.add_argument("--config", type=str, default="configs/qlora_llava_7b.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config: {args.config}")

    device = get_device()
    print_gpu_stats()

    best_path = train(config)

    logger.info(f"\nBest model: {best_path}")
    logger.info("Next steps:")
    logger.info(f"  python scripts/run_inference.py --model llava --model-path {best_path}")
    logger.info("  python scripts/evaluate.py")


if __name__ == "__main__":
    main()
