"""
evaluate.py
===========
Evaluation / evidence pack for the trained rail-defect detector.

Runs a trained model on the dataset's TEST split (falls back to val) and writes,
into an ``evaluation/`` folder, the figures + tables an examiner expects:

  1. confusion_matrix.png (+ normalised) and PR / F1 curves          (images)
  2. per_class_map50.csv -- per-class mAP@0.5 with a PASS/FAIL flag
     against the project's success criterion (>= 0.70)
  3. latency.csv + latency_hist.png -- per-image CPU inference latency
     with p50 and p95 (ms)
  4. summary.md -- a one-page collation of the headline numbers for the report

CPU-only by default.

Usage:
  python evaluate.py --model data/output/training/fastener/train/weights/best.pt \\
                     --data  data/output/training/fastener/dataset/data.yaml
  # (or point --model at models/yolo_rail.pt and --data at any YOLO data.yaml)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import resolve_path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SUCCESS_MAP50 = 0.70  # project success criterion: per-class mAP@0.5 >= 0.70


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _load_data_yaml(data_yaml: Path) -> dict:
    import yaml

    return yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}


def _names_list(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names or []


def _split_image_dir(data: dict, data_yaml: Path, split: str) -> Path:
    base = Path(data.get("path") or data_yaml.parent)
    rel = data.get(split)
    p = Path(rel)
    return p if p.is_absolute() else (base / p)


def _test_images(data: dict, data_yaml: Path, split: str) -> list[Path]:
    img_dir = _split_image_dir(data, data_yaml, split)
    return sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained rail-defect detector (CPU).")
    ap.add_argument("--model", default="models/yolo_rail.pt", help="path to trained weights (best.pt)")
    ap.add_argument("--data", required=True, help="YOLO data.yaml with a test (or val) split")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default=None, help="optional subfolder under evaluation/")
    args = ap.parse_args()

    model_path = resolve_path(args.model)
    data_yaml = resolve_path(args.data)
    if not model_path.is_file():
        return _fail(f"model not found: {model_path}  (train one with train.py first)")
    if not data_yaml.is_file():
        return _fail(f"data.yaml not found: {data_yaml}")

    out = resolve_path("evaluation")
    if args.name:
        out = out / args.name
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    data = _load_data_yaml(data_yaml)
    names = _names_list(data)
    split = "test" if data.get("test") else "val"
    print(f"Evaluating {model_path.name} on the '{split}' split of {data_yaml} ...")

    model = YOLO(str(model_path))

    # ---- 1. validation metrics + Ultralytics plots --------------------- #
    metrics = model.val(data=str(data_yaml), split=split, device=args.device, workers=0,
                        imgsz=args.imgsz, plots=True, project=str(out), name="_val", exist_ok=True,
                        verbose=False)
    val_dir = out / "_val"
    # Ultralytics names the curve plots differently across versions (e.g.
    # PR_curve.png vs BoxPR_curve.png) -- try each candidate, save a stable name.
    plot_map = {
        "confusion_matrix.png": ["confusion_matrix.png"],
        "confusion_matrix_normalized.png": ["confusion_matrix_normalized.png"],
        "pr_curve.png": ["BoxPR_curve.png", "PR_curve.png"],
        "f1_curve.png": ["BoxF1_curve.png", "F1_curve.png"],
        "p_curve.png": ["BoxP_curve.png", "P_curve.png"],
        "r_curve.png": ["BoxR_curve.png", "R_curve.png"],
    }
    copied = []
    for dst, candidates in plot_map.items():
        for src in candidates:
            if (val_dir / src).is_file():
                shutil.copy2(val_dir / src, out / dst)
                copied.append(dst)
                break

    box = metrics.box
    overall = {
        "precision": float(box.mp), "recall": float(box.mr),
        "map50": float(box.map50), "map50_95": float(box.map),
    }
    overall["f1"] = (2 * overall["precision"] * overall["recall"]
                     / (overall["precision"] + overall["recall"])
                     if (overall["precision"] + overall["recall"]) > 0 else 0.0)

    # ---- 2. per-class mAP@0.5 CSV + below-threshold flags -------------- #
    per_class = []
    try:
        for i, c in enumerate(box.ap_class_index):
            ci = int(c)
            per_class.append({
                "class": names[ci] if ci < len(names) else str(ci),
                "map50": round(float(box.ap50[i]), 4),
                "precision": round(float(box.p[i]), 4),
                "recall": round(float(box.r[i]), 4),
                "f1": round(float(box.f1[i]), 4),
            })
    except Exception as exc:
        print(f"(per-class extraction issue: {exc!r})")
    for row in per_class:
        row["pass_0.70"] = "PASS" if row["map50"] >= SUCCESS_MAP50 else "FAIL"
    below = [r["class"] for r in per_class if r["map50"] < SUCCESS_MAP50]

    pc_csv = out / "per_class_map50.csv"
    with open(pc_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "map50", "precision", "recall", "f1", "pass_0.70"])
        w.writeheader()
        w.writerows(per_class)
        w.writerow({"class": "ALL", "map50": round(overall["map50"], 4),
                    "precision": round(overall["precision"], 4), "recall": round(overall["recall"], 4),
                    "f1": round(overall["f1"], 4), "pass_0.70": ""})

    # ---- 3. inference-latency distribution (CPU) ----------------------- #
    images = _test_images(data, data_yaml, split)
    latencies: list[float] = []
    if images:
        model.predict(str(images[0]), device=args.device, imgsz=args.imgsz, verbose=False)  # warmup
        for img in images:
            r = model.predict(str(img), device=args.device, imgsz=args.imgsz, verbose=False)
            latencies.append(float(r[0].speed.get("inference", 0.0)))

    lat_stats = {
        "n_images": len(latencies),
        "p50_ms": round(_percentile(latencies, 0.50), 2),
        "p95_ms": round(_percentile(latencies, 0.95), 2),
        "mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "min_ms": round(min(latencies), 2) if latencies else 0.0,
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
        "fps_at_p50": round(1000.0 / _percentile(latencies, 0.50), 1) if latencies else 0.0,
    }
    with open(out / "latency.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "inference_ms"])
        for img, ms in zip(images, latencies):
            w.writerow([img.name, round(ms, 3)])
    with open(out / "latency_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in lat_stats.items():
            w.writerow([k, v])

    if latencies:
        _latency_hist(latencies, lat_stats, out / "latency_hist.png", args.device)

    # ---- 4. one-page summary.md ---------------------------------------- #
    result = {
        "model": str(model_path), "dataset": str(data_yaml), "split": split,
        "classes": names, "overall": {k: round(v, 4) for k, v in overall.items()},
        "per_class": per_class, "classes_below_0.70": below, "latency_ms": lat_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_summary(result, copied, out)

    print("\nEvaluation complete -> " + str(out))
    print(f"  overall  mAP@0.5={overall['map50']:.3f}  P={overall['precision']:.3f}  "
          f"R={overall['recall']:.3f}  F1={overall['f1']:.3f}")
    print(f"  latency  p50={lat_stats['p50_ms']} ms  p95={lat_stats['p95_ms']} ms  "
          f"(n={lat_stats['n_images']}, {args.device})")
    if below:
        print(f"  BELOW 0.70 mAP@0.5: {', '.join(below)}")
    return 0


def _latency_hist(latencies: list[float], stats: dict, path: Path, device: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(latencies, bins=min(30, max(5, len(latencies) // 2)), color="#22d3ee",
            edgecolor="#0e7490", alpha=0.85)
    ax.axvline(stats["p50_ms"], color="#16a34a", linestyle="--", label=f"p50 = {stats['p50_ms']} ms")
    ax.axvline(stats["p95_ms"], color="#ef4444", linestyle="--", label=f"p95 = {stats['p95_ms']} ms")
    ax.set_xlabel("inference latency (ms)")
    ax.set_ylabel("test images")
    ax.set_title(f"CPU inference latency ({device}, n={stats['n_images']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_summary(result: dict, figures: list[str], out: Path) -> None:
    o, lat = result["overall"], result["latency_ms"]
    lines = [
        "# Evaluation summary — rail-defect detector", "",
        f"- generated: {result['timestamp']}",
        f"- model: `{result['model']}`",
        f"- dataset: `{result['dataset']}`  ·  split: **{result['split']}**",
        f"- classes ({len(result['classes'])}): {', '.join(result['classes'])}", "",
        "## Headline metrics", "",
        "| metric | value |", "|---|---|",
        f"| precision | {o['precision']:.3f} |",
        f"| recall | {o['recall']:.3f} |",
        f"| F1 | {o['f1']:.3f} |",
        f"| mAP@0.5 | {o['map50']:.3f} |",
        f"| mAP@0.5:0.95 | {o['map50_95']:.3f} |", "",
        "## Inference latency (CPU)", "",
        "| stat | ms |", "|---|---|",
        f"| p50 (median) | {lat['p50_ms']} |",
        f"| p95 | {lat['p95_ms']} |",
        f"| mean | {lat['mean_ms']} |",
        f"| min / max | {lat['min_ms']} / {lat['max_ms']} |",
        f"| throughput @ p50 | {lat['fps_at_p50']} fps |",
        f"| images measured | {lat['n_images']} |", "",
        "## Per-class mAP@0.5 (success criterion: >= 0.70)", "",
        "| class | mAP@0.5 | precision | recall | F1 | result |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["per_class"]:
        lines.append(f"| {r['class']} | {r['map50']:.3f} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} | {r['pass_0.70']} |")
    lines.append("")
    if result["classes_below_0.70"]:
        lines += [f"> ⚠ **{len(result['classes_below_0.70'])} class(es) below the 0.70 mAP@0.5 "
                  f"criterion:** {', '.join(result['classes_below_0.70'])}.", ""]
    else:
        lines += ["> ✅ All classes meet the 0.70 mAP@0.5 success criterion.", ""]
    lines += ["## Figures", ""]
    for fig in ["confusion_matrix.png", "pr_curve.png", "f1_curve.png", "latency_hist.png"]:
        if (out / fig).is_file():
            lines.append(f"- `{fig}`")
    lines += ["", "## Tables (CSV)", "",
              "- `per_class_map50.csv` — per-class mAP@0.5 + PASS/FAIL",
              "- `latency.csv` — per-image inference latency",
              "- `latency_stats.csv` — p50 / p95 / mean / min / max", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"summary -> {out / 'summary.md'}")


def _fail(message: str) -> int:
    import sys

    print(f"\nERROR: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
