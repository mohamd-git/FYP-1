"""
src/localisation/sim_track.py
=============================
Simulated GPS-track LocalisationSource (PoC implementation of seam #3).

Reads a corridor from ``data/track.csv`` (columns: ``lat, lng, chainage_m``) and
advances along it at the configured inspection speed, so every detection gets a
plausible ``(lat, lng, chainage_m)``. It exposes exactly the interface a real
GPS + wheel-odometry + IMU feed will expose in Phase 2 (``position_at`` /
``open`` / ``close``), so the rest of the pipeline never changes.

Model:
    chainage(t) = start_chainage_m + inspection_speed_mps * (t - t_open)
    (lat, lng)  = linear interpolation of the track polyline at that chainage

Also ships a tiny generator (:func:`generate_sample_track`) that writes a
realistic ~150 m straight-then-curved corridor near a Malaysian coordinate.
Run ``python -m src.localisation.sim_track`` to (re)generate ``data/track.csv``.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.localisation.base import GeoPosition, LocalisationSource

# Metres per degree of latitude (good enough for a short corridor).
_M_PER_DEG_LAT = 111_320.0


@dataclass
class TrackPoint:
    """One surveyed point on the corridor polyline."""

    chainage_m: float
    lat: float
    lng: float


def load_track(csv_path: str | Path) -> list[TrackPoint]:
    """Load track points from a CSV with columns lat, lng, chainage_m."""
    points: list[TrackPoint] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                points.append(
                    TrackPoint(
                        chainage_m=float(row["chainage_m"]),
                        lat=float(row["lat"]),
                        lng=float(row["lng"]),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue  # skip malformed / header-only rows
    points.sort(key=lambda p: p.chainage_m)
    return points


class SimTrack(LocalisationSource):
    """Replays a CSV corridor at a constant inspection speed."""

    def __init__(
        self,
        *,
        csv_path: str | Path = "data/track.csv",
        inspection_speed_mps: float = 0.4,
        start_chainage_m: float = 0.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.inspection_speed_mps = float(inspection_speed_mps)
        self.start_chainage_m = float(start_chainage_m)
        self._points: list[TrackPoint] = []
        self._t0: Optional[float] = None

    @classmethod
    def from_config(cls, config: dict) -> "SimTrack":
        """Build a SimTrack from a parsed config dict."""
        from src.config import resolve_path

        loc = config.get("localisation", {}) or {}
        paths = config.get("paths", {}) or {}
        csv_rel = loc.get("gps_csv") or paths.get("gps_csv") or "data/track.csv"
        return cls(
            csv_path=resolve_path(csv_rel),
            inspection_speed_mps=float(loc.get("inspection_speed_mps", 0.4) or 0.4),
            start_chainage_m=float(loc.get("start_chainage_m", 0.0) or 0.0),
        )

    # ---- LocalisationSource interface ----------------------------------- #
    def open(self) -> None:
        self._points = load_track(self.csv_path)
        if not self._points:
            raise ValueError(f"Track CSV has no usable points: {self.csv_path}")
        self._t0 = time.time()

    def position_at(self, timestamp: float) -> GeoPosition:
        """Position for a capture time (epoch seconds), relative to open()."""
        if self._t0 is None:
            raise RuntimeError("SimTrack.open() must be called before position_at().")
        return self.position_at_elapsed(timestamp - self._t0)

    def close(self) -> None:
        self._t0 = None

    # ---- deterministic helpers (handy for tests / the demo) ------------- #
    def position_at_elapsed(self, elapsed_s: float) -> GeoPosition:
        """Position after ``elapsed_s`` seconds of travel from the start."""
        chainage = self.start_chainage_m + self.inspection_speed_mps * max(0.0, elapsed_s)
        return self.position_at_chainage(chainage)

    def position_at_chainage(self, chainage_m: float) -> GeoPosition:
        """Interpolate (lat, lng) at a given chainage along the corridor."""
        pts = self._points
        if not pts:
            raise RuntimeError("No track loaded; call open() first.")

        if chainage_m <= pts[0].chainage_m:
            p = pts[0]
            return GeoPosition(lat=p.lat, lng=p.lng, chainage_m=p.chainage_m,
                               speed_mps=self.inspection_speed_mps)
        if chainage_m >= pts[-1].chainage_m:
            p = pts[-1]  # reached the end of the corridor -> stopped
            return GeoPosition(lat=p.lat, lng=p.lng, chainage_m=p.chainage_m, speed_mps=0.0)

        for a, b in zip(pts, pts[1:]):
            if a.chainage_m <= chainage_m <= b.chainage_m:
                span = b.chainage_m - a.chainage_m
                f = (chainage_m - a.chainage_m) / span if span > 0 else 0.0
                return GeoPosition(
                    lat=a.lat + f * (b.lat - a.lat),
                    lng=a.lng + f * (b.lng - a.lng),
                    chainage_m=chainage_m,
                    speed_mps=self.inspection_speed_mps,
                )

        p = pts[-1]
        return GeoPosition(lat=p.lat, lng=p.lng, chainage_m=p.chainage_m, speed_mps=0.0)

    @property
    def length_m(self) -> float:
        return (self._points[-1].chainage_m - self._points[0].chainage_m) if self._points else 0.0

    @property
    def num_points(self) -> int:
        return len(self._points)


def generate_sample_track(
    path: str | Path | None = None,
    *,
    start_lat: float = 3.13900,     # near Kuala Lumpur, Malaysia
    start_lng: float = 101.68690,
    length_m: float = 150.0,
    step_m: float = 5.0,
    start_bearing_deg: float = 90.0,        # head east
    curve_deg_per_100m: float = 20.0,       # gentle curve -> straight-then-curved
) -> Path:
    """Write a realistic ~length_m corridor to ``path`` (default data/track.csv).

    Returns the path written.
    """
    if path is None:
        from src.config import resolve_path

        path = resolve_path("data/track.csv")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lat, lng, chainage, bearing = start_lat, start_lng, 0.0, start_bearing_deg
    rows: list[tuple[float, float, float]] = []
    steps = int(round(length_m / step_m))
    for _ in range(steps + 1):
        rows.append((round(lat, 7), round(lng, 7), round(chainage, 2)))
        # advance one step along the current bearing
        dlat = (step_m * math.cos(math.radians(bearing))) / _M_PER_DEG_LAT
        dlng = (step_m * math.sin(math.radians(bearing))) / (
            _M_PER_DEG_LAT * math.cos(math.radians(lat))
        )
        lat += dlat
        lng += dlng
        chainage += step_m
        bearing += curve_deg_per_100m * (step_m / 100.0)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lat", "lng", "chainage_m"])
        writer.writerows(rows)
    return path


if __name__ == "__main__":  # python -m src.localisation.sim_track
    out = generate_sample_track()
    pts = load_track(out)
    print(f"Wrote {out} ({len(pts)} points, {pts[-1].chainage_m:.0f} m corridor)")
