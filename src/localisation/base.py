"""
src/localisation/base.py
========================
Abstract localisation source -- swappable seam #3.

PoC implementation (later step): replays a simulated GPS track from a CSV,
advancing chainage by the configured inspection speed
(e.g. ``src/localisation/sim_gps.py``).
Hardware implementation (Phase 2): fuses real GPS + wheel odometry + IMU.
Both answer the same question: "where was the AGV at time t?".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GeoPosition:
    """Position of the AGV at a moment in time."""

    lat: float
    lng: float
    chainage_m: float  # metres along the track from the start of the run
    speed_mps: float


class LocalisationSource(ABC):
    """Maps a timestamp to a geo-position so detections can be geo-tagged."""

    @abstractmethod
    def open(self) -> None:
        """Load the track / start the receiver."""

    @abstractmethod
    def position_at(self, timestamp: float) -> GeoPosition:
        """Return the AGV position for the given capture time (epoch seconds)."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources."""
