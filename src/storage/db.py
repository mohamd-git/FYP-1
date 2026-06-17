"""
src/storage/db.py
=================
Tiny SQLite layer for the consolidated **defect register** -- one row per
physical defect (already de-duplicated by track in src/dedup.py).

Each row carries every contract field of the representative Detection, plus the
aggregation metadata (first/last-seen timestamps, frame range, frame count).
Provides query helpers and CSV / JSON export so the register is analytics-ready.

The detection contract is untouched: this is downstream persistence only.
"""

from __future__ import annotations

import csv as _csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.schema import Detection

_SCHEMA = """
CREATE TABLE IF NOT EXISTS defects (
    defect_key         TEXT    PRIMARY KEY,   -- "track-7" / "untracked-3"
    detection_id       TEXT    NOT NULL,
    track_id           INTEGER NOT NULL,
    defect_class       TEXT    NOT NULL,
    severity           TEXT    NOT NULL,
    urgency_score      INTEGER NOT NULL,
    confidence         REAL    NOT NULL,      -- of the representative (best) frame
    recommended_action TEXT    NOT NULL,
    bbox_xywh          TEXT    NOT NULL,      -- JSON [x, y, w, h]
    lat                REAL    NOT NULL,
    lng                REAL    NOT NULL,
    chainage_m         REAL    NOT NULL,
    image_ref          TEXT    NOT NULL,      -- representative crop
    model              TEXT    NOT NULL,
    timestamp          TEXT    NOT NULL,      -- representative frame time (ISO-8601)
    first_seen         TEXT    NOT NULL,
    last_seen          TEXT    NOT NULL,
    first_frame_id     INTEGER NOT NULL,
    last_frame_id      INTEGER NOT NULL,
    frame_count        INTEGER NOT NULL
);
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_defects_track ON defects(track_id);",
    "CREATE INDEX IF NOT EXISTS idx_defects_severity ON defects(severity);",
    "CREATE INDEX IF NOT EXISTS idx_defects_urgency ON defects(urgency_score DESC);",
)

# Column order for inserts / exports.
_COLUMNS = [
    "defect_key", "detection_id", "track_id", "defect_class", "severity",
    "urgency_score", "confidence", "recommended_action", "bbox_xywh", "lat",
    "lng", "chainage_m", "image_ref", "model", "timestamp", "first_seen",
    "last_seen", "first_frame_id", "last_frame_id", "frame_count",
]


def _iso(value: Any) -> str:
    """Render a datetime as ISO-8601; pass strings through unchanged."""
    return value.isoformat() if isinstance(value, datetime) else str(value)


class Database:
    """SQLite-backed defect register."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    @classmethod
    def from_config(cls, config: dict) -> "Database":
        from src.config import resolve_path

        storage = config.get("storage", {}) or {}
        paths = config.get("paths", {}) or {}
        db_rel = storage.get("sqlite_path") or paths.get("database") or "data/agv.db"
        return cls(resolve_path(db_rel))

    # ---- lifecycle ------------------------------------------------------ #
    def connect(self) -> "Database":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")   # concurrent dashboard reads + pipeline writes
        self._conn.execute("PRAGMA busy_timeout=3000")  # wait up to 3s on a locked DB
        self._conn.execute(_SCHEMA)
        for stmt in _INDEXES:
            self._conn.execute(stmt)
        self._conn.commit()
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def reset(self) -> None:
        """Clear the register (start a fresh inspection run)."""
        self._require().execute("DELETE FROM defects;")
        self._conn.commit()  # type: ignore[union-attr]

    # ---- writes --------------------------------------------------------- #
    def upsert_defect(
        self,
        detection: Detection,
        *,
        defect_key: str,
        first_seen: datetime,
        last_seen: datetime,
        frame_count: int,
        first_frame_id: int,
        last_frame_id: int,
    ) -> None:
        """Insert or replace one consolidated defect row (keyed by defect_key)."""
        d = detection
        row = {
            "defect_key": defect_key,
            "detection_id": d.detection_id,
            "track_id": d.track_id,
            "defect_class": d.defect_class.value,
            "severity": d.severity.value,
            "urgency_score": d.urgency_score,
            "confidence": d.confidence,
            "recommended_action": d.recommended_action,
            "bbox_xywh": json.dumps(list(d.bbox_xywh)),
            "lat": d.location.lat,
            "lng": d.location.lng,
            "chainage_m": d.location.chainage_m,
            "image_ref": d.image_ref,
            "model": d.model,
            "timestamp": _iso(d.timestamp),
            "first_seen": _iso(first_seen),
            "last_seen": _iso(last_seen),
            "first_frame_id": first_frame_id,
            "last_frame_id": last_frame_id,
            "frame_count": frame_count,
        }
        placeholders = ", ".join(["?"] * len(_COLUMNS))
        sql = f"INSERT OR REPLACE INTO defects ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        self._require().execute(sql, [row[c] for c in _COLUMNS])
        self._conn.commit()  # type: ignore[union-attr]

    # ---- queries -------------------------------------------------------- #
    def all_defects(self) -> list[dict[str, Any]]:
        """All defects, highest urgency first."""
        cur = self._require().execute(
            "SELECT * FROM defects ORDER BY urgency_score DESC, last_seen DESC"
        )
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def get_by_track(self, track_id: int) -> Optional[dict[str, Any]]:
        cur = self._require().execute(
            "SELECT * FROM defects WHERE track_id = ? LIMIT 1", (track_id,)
        )
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def count(self) -> int:
        return int(self._require().execute("SELECT COUNT(*) FROM defects").fetchone()[0])

    def severity_counts(self) -> dict[str, int]:
        cur = self._require().execute(
            "SELECT severity, COUNT(*) AS n FROM defects GROUP BY severity"
        )
        return {row["severity"]: int(row["n"]) for row in cur.fetchall()}

    # ---- export --------------------------------------------------------- #
    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.all_defects()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=_COLUMNS)
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["bbox_xywh"] = json.dumps(out["bbox_xywh"])  # list -> JSON text
                writer.writerow(out)
        return path

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.all_defects(), fh, indent=2)
        return path

    # ---- internals ------------------------------------------------------ #
    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database.connect() must be called first.")
        return self._conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        try:
            d["bbox_xywh"] = json.loads(d["bbox_xywh"])  # JSON text -> list
        except (TypeError, json.JSONDecodeError):
            pass
        return d

    # ---- context manager ------------------------------------------------ #
    def __enter__(self) -> "Database":
        return self.connect()

    def __exit__(self, *exc_info: object) -> None:
        self.close()
