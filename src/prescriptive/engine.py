"""
src/prescriptive/engine.py
==========================
Rule-based PRESCRIPTIVE engine -- the project's core novelty.

It converts a raw detection into a *maintenance decision*:

    (defect_class, confidence, bbox area, frame area)
        -> severity            ("High" | "Medium" | "Low")
        -> urgency_score        (int, 0-100)
        -> recommended_action   (str, per class + action band)

Everything is read from ``rules.yaml`` so behaviour is fully tunable without
touching code. ``prescribe(...)`` is a PURE function (rules are passed in) and is
the unit-testable core; ``Prescriber`` is a thin wrapper that loads rules.yaml
once and keeps it.

------------------------------------------------------------------------------
SEVERITY RUBRIC  (illustrative defaults -- to be validated by the student)
------------------------------------------------------------------------------
    missing_fastener  High      (safety-critical)
    broken_fastener   High      (safety-critical)
    crack             High  if  bbox is "large" OR confidence is "high",
                      else Medium
    spalling          Medium
    squat             Medium
    loose_fastener    Medium
    corrugation       Low

URGENCY SCORE (0-100) blends three normalised factors:

    score = 100 * ( w_sev  * (severity_weight / max_weight)   # High=3,Med=2,Low=1
                  + w_conf * confidence                         # 0..1
                  + w_size * size_factor )                      # 0..1
    size_factor = min(1, (bbox_area / frame_area) / size_norm_frac)

The component weights, the severity weights and ``size_norm_frac`` all live in
rules.yaml -> easy to re-tune against real maintenance data.

ACTION BANDS (from the score):
    >= 75   Immediate (within 24 h)
    50-74   Schedule within 7 days
    25-49   Routine (next maintenance cycle)
    <  25   Monitor

DESIGNED TO BE TUNED: these numbers are a defensible starting point, not ground
truth. Adjust rules.yaml as the project is validated against railway standards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Defaults used only if a key is missing from rules.yaml, so the engine stays
# robust with a partial file. rules.yaml remains the real source of truth.
_DEFAULT_SEVERITY_WEIGHTS: dict[str, int] = {"High": 3, "Medium": 2, "Low": 1}
_DEFAULT_WEIGHTS: dict[str, float] = {"severity": 0.55, "confidence": 0.30, "size": 0.15}
_DEFAULT_SIZE_NORM_FRAC: float = 0.10
_DEFAULT_BANDS: list[dict[str, Any]] = [
    {"min": 75, "label": "Immediate (within 24 h)"},
    {"min": 50, "label": "Schedule within 7 days"},
    {"min": 25, "label": "Routine (next maintenance cycle)"},
    {"min": 0, "label": "Monitor"},
]


@dataclass(frozen=True)
class Prescription:
    """The maintenance decision produced for one detection."""

    severity: str            # "High" | "Medium" | "Low"
    urgency_score: int       # 0..100
    recommended_action: str  # per class + band, e.g. "...restrict speed. Immediate (within 24 h)."
    band: str                # action-band label, e.g. "Immediate (within 24 h)"
    breakdown: dict[str, float] = field(default_factory=dict)  # component values, for tuning


def load_rules(path: str | Path) -> dict[str, Any]:
    """Load and return the rules mapping from a YAML file."""
    import yaml  # local import: engine.py stays importable without PyYAML present

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def resolve_severity(
    defect_class: str, confidence: float, area_frac: float, rules: Mapping[str, Any]
) -> str:
    """Return the severity band for a detection per the rubric in ``rules``."""
    sev_rules = rules.get("severity", {}) or {}
    entry = sev_rules.get(defect_class)
    if entry is None:
        return (rules.get("defaults", {}) or {}).get("severity", "Low")
    if isinstance(entry, str):
        return entry
    # Conditional escalation block (e.g. crack).
    default = entry.get("default", "Medium")
    escalate_to = entry.get("escalate_to", "High")
    area_gte = entry.get("if_area_frac_gte")
    conf_gte = entry.get("if_confidence_gte")
    escalate = False
    if area_gte is not None and area_frac >= float(area_gte):
        escalate = True
    if conf_gte is not None and confidence >= float(conf_gte):
        escalate = True
    return escalate_to if escalate else default


def urgency_score(
    severity: str, confidence: float, size_factor: float, rules: Mapping[str, Any]
) -> tuple[int, dict[str, float]]:
    """Compute the 0-100 urgency score and return it with its component breakdown."""
    urg = rules.get("urgency", {}) or {}
    sev_weights = urg.get("severity_weight") or _DEFAULT_SEVERITY_WEIGHTS
    weights = urg.get("weights") or _DEFAULT_WEIGHTS

    max_weight = max(sev_weights.values()) if sev_weights else 1
    sev_norm = (sev_weights.get(severity, 1) / max_weight) if max_weight else 0.0
    conf_norm = _clamp01(confidence)
    size_norm = _clamp01(size_factor)

    w_sev = float(weights.get("severity", 0.0))
    w_conf = float(weights.get("confidence", 0.0))
    w_size = float(weights.get("size", 0.0))

    raw = (w_sev * sev_norm + w_conf * conf_norm + w_size * size_norm) * 100.0
    score = int(round(max(0.0, min(100.0, raw))))
    breakdown = {
        "severity_norm": round(sev_norm, 3),
        "confidence": round(conf_norm, 3),
        "size_factor": round(size_norm, 3),
        "score_raw": round(raw, 2),
    }
    return score, breakdown


def band_label_for(score: int, rules: Mapping[str, Any]) -> str:
    """Return the action-band label whose ``min`` the score reaches (highest wins)."""
    bands = rules.get("bands") or _DEFAULT_BANDS
    for band in sorted(bands, key=lambda b: b.get("min", 0), reverse=True):
        if score >= int(band.get("min", 0)):
            return str(band.get("label", "Monitor"))
    return "Monitor"


def prescribe(
    defect_class: str,
    confidence: float,
    bbox_area: float,
    frame_area: float,
    rules: Mapping[str, Any],
) -> Prescription:
    """Turn one detection into a maintenance decision (pure; rules passed in).

    Args:
        defect_class: one of the seven contract defect classes.
        confidence: detector confidence, 0..1.
        bbox_area: bounding-box area in pixels (w * h).
        frame_area: frame area in pixels (width * height); used to size-normalise.
        rules: the parsed rules.yaml mapping.

    Returns:
        A :class:`Prescription` (severity, urgency_score, recommended_action, band).
    """
    area_frac = (bbox_area / frame_area) if frame_area and frame_area > 0 else 0.0
    size_norm_frac = float(
        (rules.get("urgency", {}) or {}).get("size_norm_frac", _DEFAULT_SIZE_NORM_FRAC)
        or _DEFAULT_SIZE_NORM_FRAC
    )
    size_factor = min(1.0, area_frac / size_norm_frac) if size_norm_frac > 0 else 0.0

    severity = resolve_severity(defect_class, confidence, area_frac, rules)
    score, breakdown = urgency_score(severity, confidence, size_factor, rules)
    band = band_label_for(score, rules)

    actions = rules.get("actions", {}) or {}
    phrase = actions.get(defect_class) or (rules.get("defaults", {}) or {}).get(
        "recommended_action", "Manual review required"
    )
    recommended_action = f"{phrase}. {band}."

    breakdown["area_frac"] = round(area_frac, 5)
    return Prescription(
        severity=severity,
        urgency_score=score,
        recommended_action=recommended_action,
        band=band,
        breakdown=breakdown,
    )


class Prescriber:
    """Loads rules.yaml once, then prescribes maintenance decisions."""

    def __init__(self, rules: Mapping[str, Any]) -> None:
        self.rules: Mapping[str, Any] = rules

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Prescriber":
        """Build from rules.yaml (defaults to the project-root rules.yaml)."""
        from src.config import resolve_path

        rules_path = resolve_path(path) if path else resolve_path("rules.yaml")
        return cls(load_rules(rules_path))

    def prescribe(
        self, defect_class: str, confidence: float, bbox_area: float, frame_area: float
    ) -> Prescription:
        """Prescribe from raw areas (see module-level :func:`prescribe`)."""
        return prescribe(defect_class, confidence, bbox_area, frame_area, self.rules)

    def prescribe_bbox(
        self,
        defect_class: str,
        confidence: float,
        bbox_xywh: tuple[int, int, int, int],
        frame_size: tuple[int, int],
    ) -> Prescription:
        """Convenience: accept a bbox ``[x, y, w, h]`` and frame ``(width, height)``."""
        _, _, w, h = bbox_xywh
        frame_w, frame_h = frame_size
        return self.prescribe(defect_class, confidence, float(w * h), float(frame_w * frame_h))
