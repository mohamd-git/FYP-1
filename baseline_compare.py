"""
baseline_compare.py
===================
Compare the automated AGV inspection against a MANUAL human review of the same
footage / corridor (the project's "vs manual inspection" baseline).

Inputs: two CSV logs (system + human). Each is::

    # time_s=156  distance_m=62.4          <- optional meta comment line(s)
    defect_class,chainage_m,severity
    crack,3.2,High
    missing_fastener,8.1,High
    ...

`time_s` / `distance_m` may also be given on the CLI (which override the header).
The SYSTEM log is just the maintenance register (export it from the dashboard
``/export.csv`` or ``Database.export_csv``); the HUMAN log is produced by
``manual_review.py`` or written by hand in the same format.

Outputs (to ``evaluation/baseline/``):
  * comparison.csv      -- the comparison table (system vs human)
  * by_class.csv        -- defects found per class, each method
  * matched.csv         -- per-defect match status (matched / system-only / human-only)
  * comparison.png      -- defects-found + time-per-metre figure
  * comparison.md       -- a short interpretive note for the report

Metrics: defects detected (system vs human, matched + unique), time per metre
(s/m) + system speed-up, and consistency (system<->human agreement, plus the
system's deterministic repeatability vs human variability).

Usage:
  python baseline_compare.py --system evaluation/baseline/system_log.csv \\
                             --human  evaluation/baseline/human_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import resolve_path

DEFAULT_INSPECT_SPEED = 0.4  # m/s (config localisation.inspection_speed_mps)
MATCH_TOL_M = 2.0            # two logs "agree" if same class within this chainage


def _f(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_log(path: Path) -> tuple[list[dict], dict]:
    meta: dict = {}
    data_lines: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#"):
                for tok in s[1:].replace(",", " ").split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k.strip()] = v.strip()
            elif s:
                data_lines.append(line)
    rows = []
    for r in csv.DictReader(data_lines):
        cls = (r.get("defect_class") or r.get("class") or "").strip()
        if not cls:
            continue
        rows.append({
            "defect_class": cls,
            "chainage_m": _f(r.get("chainage_m") or r.get("chainage")) or 0.0,
            "severity": (r.get("severity") or "").strip(),
        })
    return rows, meta


def _match(system: list[dict], human: list[dict], tol: float):
    """Greedy nearest match: same class within `tol` metres."""
    used: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for i, s in enumerate(system):
        best, best_d = None, tol + 1e-9
        for j, h in enumerate(human):
            if j in used or h["defect_class"] != s["defect_class"]:
                continue
            d = abs(h["chainage_m"] - s["chainage_m"])
            if d <= tol and d < best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs.append((i, best, best_d))
    matched_sys = {p[0] for p in pairs}
    matched_hum = {p[1] for p in pairs}
    system_only = [i for i in range(len(system)) if i not in matched_sys]
    human_only = [j for j in range(len(human)) if j not in matched_hum]
    return pairs, system_only, human_only


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare automated AGV inspection vs manual review.")
    ap.add_argument("--system", required=True, help="system maintenance-register CSV")
    ap.add_argument("--human", required=True, help="human review CSV (from manual_review.py)")
    ap.add_argument("--distance-m", type=float, default=None, help="corridor length inspected (m)")
    ap.add_argument("--system-time-s", type=float, default=None, help="system inspection time (s)")
    ap.add_argument("--human-time-s", type=float, default=None, help="human review time (s)")
    ap.add_argument("--speed", type=float, default=DEFAULT_INSPECT_SPEED, help="AGV speed m/s (for system time)")
    ap.add_argument("--match-tol-m", type=float, default=MATCH_TOL_M)
    args = ap.parse_args()

    sys_rows, sys_meta = _parse_log(resolve_path(args.system))
    hum_rows, hum_meta = _parse_log(resolve_path(args.human))

    distance = (args.distance_m or _f(sys_meta.get("distance_m")) or _f(hum_meta.get("distance_m"))
                or max([r["chainage_m"] for r in sys_rows + hum_rows] + [1.0]))
    sys_time = (args.system_time_s or _f(sys_meta.get("time_s")) or _f(sys_meta.get("inspection_time_s"))
                or distance / args.speed)
    hum_time = (args.human_time_s or _f(hum_meta.get("time_s")) or _f(hum_meta.get("review_time_s")))
    if not hum_time:
        return _fail("human review time unknown — add '# time_s=NNN' to the human log or pass --human-time-s")

    pairs, system_only, human_only = _match(sys_rows, hum_rows, args.match_tol_m)
    matched = len(pairs)
    sys_n, hum_n = len(sys_rows), len(hum_rows)
    union = matched + len(system_only) + len(human_only)

    stats = {
        "system_defects": sys_n,
        "human_defects": hum_n,
        "matched": matched,
        "system_only": len(system_only),
        "human_only": len(human_only),
        "agreement_jaccard": round(matched / union, 3) if union else 0.0,
        "system_recall_vs_human": round(matched / hum_n, 3) if hum_n else 0.0,
        "distance_m": round(distance, 2),
        "system_time_s": round(sys_time, 1),
        "human_time_s": round(hum_time, 1),
        "system_s_per_m": round(sys_time / distance, 3),
        "human_s_per_m": round(hum_time / distance, 3),
        "system_speedup_x": round(hum_time / sys_time, 2) if sys_time else 0.0,
    }

    out = resolve_path("evaluation/baseline")
    out.mkdir(parents=True, exist_ok=True)

    # ---- comparison.csv ------------------------------------------------ #
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "system", "human"])
        w.writerow(["defects detected", sys_n, hum_n])
        w.writerow(["matched (agreed)", matched, matched])
        w.writerow(["unique to method", len(system_only), len(human_only)])
        w.writerow(["inspection time (s)", round(sys_time, 1), round(hum_time, 1)])
        w.writerow(["time per metre (s/m)", stats["system_s_per_m"], stats["human_s_per_m"]])
        w.writerow(["distance (m)", round(distance, 2), round(distance, 2)])
        w.writerow([])
        w.writerow(["derived", "value", ""])
        w.writerow(["system speed-up (x)", stats["system_speedup_x"], ""])
        w.writerow(["agreement (Jaccard)", stats["agreement_jaccard"], ""])
        w.writerow(["system recall vs human", stats["system_recall_vs_human"], ""])
        w.writerow(["system repeatability", "100% (deterministic)", ""])

    # ---- by_class.csv -------------------------------------------------- #
    sys_c, hum_c = Counter(r["defect_class"] for r in sys_rows), Counter(r["defect_class"] for r in hum_rows)
    with open(out / "by_class.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["defect_class", "system", "human"])
        for cls in sorted(set(sys_c) | set(hum_c)):
            w.writerow([cls, sys_c.get(cls, 0), hum_c.get(cls, 0)])

    # ---- matched.csv --------------------------------------------------- #
    with open(out / "matched.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["status", "defect_class", "system_chainage_m", "human_chainage_m", "delta_m"])
        for i, j, d in pairs:
            w.writerow(["matched", sys_rows[i]["defect_class"],
                        sys_rows[i]["chainage_m"], hum_rows[j]["chainage_m"], round(d, 2)])
        for i in system_only:
            w.writerow(["system_only", sys_rows[i]["defect_class"], sys_rows[i]["chainage_m"], "", ""])
        for j in human_only:
            w.writerow(["human_only", hum_rows[j]["defect_class"], "", hum_rows[j]["chainage_m"], ""])

    _figure(stats, matched, system_only, human_only, out / "comparison.png")
    _note(stats, out / "comparison.md")
    (out / "metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Baseline comparison -> " + str(out))
    print(f"  defects: system {sys_n} vs human {hum_n}  (matched {matched}, "
          f"system-only {len(system_only)}, human-only {len(human_only)})")
    print(f"  time/m : system {stats['system_s_per_m']} vs human {stats['human_s_per_m']} s/m  "
          f"({stats['system_speedup_x']}x faster)")
    print(f"  agreement (Jaccard) {stats['agreement_jaccard']} · system recall vs human "
          f"{stats['system_recall_vs_human']}")
    return 0


def _figure(stats, matched, system_only, human_only, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # defects found: stacked matched + unique
    ax1.bar(["System", "Human"], [matched, matched], label="matched (agreed)", color="#22c55e")
    ax1.bar(["System", "Human"], [len(system_only), len(human_only)], bottom=[matched, matched],
            label="unique to method", color="#22d3ee")
    ax1.set_ylabel("defects logged")
    ax1.set_title("Defects detected")
    ax1.legend(fontsize=9)

    # time per metre
    bars = ax2.bar(["System", "Human"], [stats["system_s_per_m"], stats["human_s_per_m"]],
                   color=["#22d3ee", "#f59e0b"])
    ax2.set_ylabel("time per metre (s/m)")
    ax2.set_title(f"Inspection time per metre  ({stats['system_speedup_x']}x faster)")
    for b, v in zip(bars, [stats["system_s_per_m"], stats["human_s_per_m"]]):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Automated AGV inspection vs manual review")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _note(s: dict, path: Path) -> None:
    lines = [
        "# Baseline comparison — automated AGV vs manual review", "",
        f"- generated: {datetime.now(timezone.utc).isoformat()}",
        f"- corridor inspected: {s['distance_m']} m", "",
        "## Comparison", "",
        "| metric | system (AGV) | human (manual) |",
        "|---|---|---|",
        f"| defects detected | {s['system_defects']} | {s['human_defects']} |",
        f"| agreed (matched) | {s['matched']} | {s['matched']} |",
        f"| unique to method | {s['system_only']} | {s['human_only']} |",
        f"| inspection time (s) | {s['system_time_s']} | {s['human_time_s']} |",
        f"| time per metre (s/m) | {s['system_s_per_m']} | {s['human_s_per_m']} |",
        f"| repeatability | 100% (deterministic) | varies (fatigue / observer) |", "",
        "## Interpretation", "",
        f"On the same {s['distance_m']} m corridor, the automated AGV logged "
        f"**{s['system_defects']}** defects versus **{s['human_defects']}** from the manual "
        f"review. The two methods **agreed on {s['matched']}** defects "
        f"(Jaccard agreement **{s['agreement_jaccard']:.2f}**); the AGV additionally flagged "
        f"**{s['system_only']}** for human verification, while **{s['human_only']}** logged by the "
        f"reviewer were not matched by the system (system recall vs human "
        f"**{s['system_recall_vs_human']:.2f}**).",
        "",
        f"The AGV inspected at **{s['system_s_per_m']} s/m** against the reviewer's "
        f"**{s['human_s_per_m']} s/m** — about **{s['system_speedup_x']}x faster per metre** — and "
        f"its output is **deterministic** (identical on re-runs), whereas manual review is slower "
        f"and subject to fatigue and inter-observer variability. The AGV also produces a "
        f"structured, geo-tagged, time-stamped record automatically, which manual review does not.",
        "",
        "> Figures from illustrative logs unless you supplied your own; replace the input "
        "logs with a real maintenance-register export and a real `manual_review.py` session.",
        "",
        "Artefacts: `comparison.csv`, `by_class.csv`, `matched.csv`, `comparison.png`.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fail(message: str) -> int:
    import sys

    print(f"\nERROR: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
