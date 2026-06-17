"""
try_prescriptive.py
===================
Step 3 demo for the prescriptive engine.

Feeds several sample detections through the engine and prints
severity / urgency / recommended action, then runs inline tests that
demonstrate each of the four action bands.

Run:
    python try_prescriptive.py
"""

from __future__ import annotations

from src.prescriptive.engine import Prescriber

# A representative frame size (Full HD) used to size-normalise the bboxes.
FRAME_W, FRAME_H = 1920, 1080
FRAME_AREA = FRAME_W * FRAME_H


def main() -> None:
    pres = Prescriber.from_yaml()

    print("=" * 94)
    print(f" Prescriptive engine -- sample detections (frame {FRAME_W}x{FRAME_H})")
    print("=" * 94)

    # (defect_class, confidence, bbox_w, bbox_h)
    samples: list[tuple[str, float, int, int]] = [
        ("missing_fastener", 0.91, 80, 60),
        ("broken_fastener", 0.66, 70, 50),
        ("loose_fastener", 0.72, 70, 55),
        ("crack", 0.82, 60, 40),       # high confidence  -> escalates to High
        ("crack", 0.48, 50, 30),       # low conf + small -> stays Medium
        ("crack", 0.62, 300, 160),     # large bbox       -> escalates to High
        ("spalling", 0.63, 200, 150),
        ("squat", 0.40, 80, 60),
        ("corrugation", 0.55, 120, 60),
        ("corrugation", 0.18, 30, 20),  # weak + low severity -> Monitor
    ]
    for cls, conf, w, h in samples:
        p = pres.prescribe(cls, conf, float(w * h), float(FRAME_AREA))
        frac_pct = (w * h) / FRAME_AREA * 100.0
        print(
            f"{cls:<16} conf={conf:0.2f}  bbox={w}x{h} ({frac_pct:4.1f}% frame)"
            f"  ->  {p.severity:<6} urgency={p.urgency_score:3d}  [{p.band}]"
        )
        print(f"    action: {p.recommended_action}")

    print()
    _band_tests(pres)


def _band_tests(pres: Prescriber) -> None:
    """Inline tests: one crafted detection per action band."""
    print("-" * 94)
    print(" Inline band tests (one detection per action band)")
    print("-" * 94)

    # (label, defect_class, confidence, bbox_w, bbox_h, expected_band)
    cases: list[tuple[str, str, float, int, int, str]] = [
        ("Immediate", "missing_fastener", 0.90, 90, 70, "Immediate (within 24 h)"),
        ("Schedule", "spalling", 0.65, 200, 150, "Schedule within 7 days"),
        ("Routine", "corrugation", 0.55, 120, 60, "Routine (next maintenance cycle)"),
        ("Monitor", "corrugation", 0.18, 30, 20, "Monitor"),
    ]
    all_ok = True
    for label, cls, conf, w, h, expected in cases:
        p = pres.prescribe(cls, conf, float(w * h), float(FRAME_AREA))
        ok = p.band == expected
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label:<9} -> urgency {p.urgency_score:3d}  band {p.band!r}")
        if not ok:
            print(f"           expected {expected!r}")

    assert all_ok, "One or more band tests failed."
    print("\nAll four action bands demonstrated. OK")


if __name__ == "__main__":
    main()
