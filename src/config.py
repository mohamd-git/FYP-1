"""
src/config.py
=============
Single place to load configuration and resolve project-relative paths.

- ``load_config()`` reads ``config.yaml`` and overlays secrets from a local
  ``.env`` (secrets never live in config.yaml).
- ``resolve_path()`` turns a config path like ``"data/sample_run.mp4"`` into an
  absolute path anchored at the project root, so the app behaves the same no
  matter which directory you launch it from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Project root = the folder that contains this `src/` package (i.e. where
# config.yaml, run.py and try_vision.py live).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML config and overlay secrets from a local .env (if present)."""
    import yaml  # imported lazily so this module imports even before deps install
    from dotenv import load_dotenv

    cfg_path = Path(path) if path is not None else CONFIG_PATH
    load_dotenv(PROJECT_ROOT / ".env")  # no-op if .env is absent

    with cfg_path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    # Secrets come from the environment, never from config.yaml.
    config.setdefault("mqtt", {})
    config["mqtt"]["username"] = os.getenv("MQTT_USERNAME") or None
    config["mqtt"]["password"] = os.getenv("MQTT_PASSWORD") or None

    # Runtime/container overrides (e.g. Docker points the app at the broker
    # service name and binds the dashboard to all interfaces).
    if os.getenv("AGV_MQTT_HOST"):
        config["mqtt"]["host"] = os.getenv("AGV_MQTT_HOST")
    if os.getenv("AGV_MQTT_PORT"):
        config["mqtt"]["port"] = int(os.getenv("AGV_MQTT_PORT"))
    if os.getenv("AGV_DASHBOARD_HOST"):
        config.setdefault("dashboard", {})["host"] = os.getenv("AGV_DASHBOARD_HOST")
    return config


def resolve_path(rel: str | Path) -> Path:
    """Resolve a (possibly relative) config path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def seed_all(seed: int = 0) -> None:
    """Seed Python / NumPy / torch RNGs for reproducible runs."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
