"""
manual_review.py
================
Minimal helper for a human to log a MANUAL inspection of the same footage, so it
can be compared against the automated AGV (see baseline_compare.py).

It times the review session and writes a ``human_log.csv`` in the format
baseline_compare.py expects. Log one defect per line as::

    class,chainage_m[,severity]      e.g.   crack,12.5,High

Finish with ``q`` (or an empty line, or Ctrl-Z then Enter on Windows).

Usage:
    python manual_review.py --distance-m 62.4
    python manual_review.py --out evaluation/baseline/human_log.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from src.config import resolve_path

DEFECT_CLASSES = ["crack", "spalling", "corrugation", "squat",
                  "missing_fastener", "broken_fastener", "loose_fastener"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Log a manual inspection for the baseline comparison.")
    ap.add_argument("--out", default="evaluation/baseline/human_log.csv")
    ap.add_argument("--distance-m", type=float, default=None, help="corridor length inspected (m)")
    args = ap.parse_args()

    print("Manual review logger -- log each defect as you spot it in the footage.")
    print("Format:  class,chainage_m[,severity]   e.g.  crack,12.5,High")
    print("Classes:", ", ".join(DEFECT_CLASSES))
    print("Finish with 'q' or an empty line.\n")

    distance = args.distance_m
    if distance is None:
        try:
            distance = float((input("corridor distance inspected (m): ").strip() or "0"))
        except (ValueError, EOFError):
            distance = 0.0

    print("\nTimer started -- log defects now:\n")
    t0 = time.time()
    rows: list[tuple[str, str, str]] = []
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if line.lower() in ("q", "quit", "done", ""):
                break
            parts = [p.strip() for p in line.split(",")]
            cls = parts[0]
            chain = parts[1] if len(parts) > 1 else "0"
            sev = parts[2] if len(parts) > 2 else ""
            rows.append((cls, chain, sev))
            print(f"  logged: {cls} @ {chain} m {sev}")
    except KeyboardInterrupt:
        pass
    elapsed = time.time() - t0

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# time_s={elapsed:.1f} distance_m={distance}\n")
        w = csv.writer(fh)
        w.writerow(["defect_class", "chainage_m", "severity"])
        w.writerows(rows)

    print(f"\nWrote {len(rows)} defects + review time {elapsed:.1f}s -> {out}")
    print(f"Next: python baseline_compare.py --system <register.csv> --human {args.out}")


if __name__ == "__main__":
    main()
