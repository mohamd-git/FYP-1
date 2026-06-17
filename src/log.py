"""
src/log.py
==========
Central structured-logging configuration. Modules log via
``logging.getLogger(__name__)``; ``setup_logging()`` (called once at app start,
e.g. from run.py / the dashboard) installs a consistent, timestamped format.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once with a structured, timestamped format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Quieten very chatty third-party loggers.
    for noisy in ("werkzeug", "ultralytics"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
