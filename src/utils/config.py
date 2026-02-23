"""Configuration loader and validator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing configuration values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def load_env(env_path: str | Path = ".env") -> None:
    """Load environment variables from .env file."""
    load_dotenv(env_path)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory, respecting DATA_DIR env var."""
    data_dir = os.getenv("DATA_DIR", str(get_project_root() / "data"))
    return Path(data_dir)


def get_output_dir() -> Path:
    """Get the output directory, respecting OUTPUT_DIR env var."""
    output_dir = os.getenv("OUTPUT_DIR", str(get_project_root() / "outputs"))
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge two config dictionaries, with override taking precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
