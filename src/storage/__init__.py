"""Storage: SQLite persistence of the consolidated defect register, with query
helpers and CSV/JSON export. See db.py."""

from src.storage.db import Database

__all__ = ["Database"]
