"""Prescriptive engine: turns detections into a severity band, an urgency score
(0-100) and a recommended maintenance action using rules.yaml -- the project's
core novelty (detect -> act). See engine.py for the rubric; tune via rules.yaml."""

from src.prescriptive.engine import (
    Prescriber,
    Prescription,
    band_label_for,
    load_rules,
    prescribe,
    resolve_severity,
    urgency_score,
)

__all__ = [
    "Prescriber",
    "Prescription",
    "prescribe",
    "load_rules",
    "resolve_severity",
    "urgency_score",
    "band_label_for",
]
