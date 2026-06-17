"""
train.py
========
Train a YOLO (v8 / v11) detector on the rail-defect datasets (project objective:
train a YOLO-family detector for rail-surface + fastener defects, benchmark on
public datasets with a 70/15/15 split + augmentation).

- Input: a dataset in standard YOLO format -- either a Roboflow export folder
  (data.yaml + train/valid[/test] with images/ + labels/) or a folder you point
  --data at. By default the splits are re-pooled into a fresh **70/15/15**
  train/val/test split (use --keep-splits to keep the dataset's own splits).
- Augmentation: Ultralytics' standard training augmentation is on (mosaic, HSV,
  flips, scale, translate) -- helpful for these class-imbalanced defect sets.
- Model: configurable via --model. Default **yolov8n.pt** (the safe target for
  the later Coral / TFLite-INT8 export path). yolo11n.pt etc. also work.
- After training it reports precision, recall, F1, mAP@0.5 and per-class
  mAP@0.5, saves the Ultralytics run, and (unless --no-save-weights) copies the
  best weights to models/yolo_rail.pt so the live pipeline (Step 2 inference
  engine) picks them up automatically.

CPU-only by default. See the README "Training" section for dataset layout and
how to point this script at each public dataset.

Examples:
  # quick run on the small surface-fault set:
  python train.py --data "...\\Dataset\\railway fault.v1i.yolov11" --epochs 50 --imgsz 640

  # fastener set, CPU-feasible subset, custom model:
  python train.py --data "...\\Dataset\\railway-track_fastener_defcts1.v1i.yolov11" --max-images 800 --epochs 30 --imgsz 512 --model yolov8n.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import resolve_path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _find_yaml(path: Path) -> Optional[Path]:
    if path.is_file() and path.suffix in (".yaml", ".yml"):
        return path
    for candidate in ("data.yaml", "data.yml"):
        if (path / candidate).is_file():
            return path / candidate
    return None


def _read_names(yaml_path: Path) -> list[str]:
    import yaml

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names or []


def _pairs_in(split_dir: Path) -> list[tuple[Path, Path]]:
    """(image, label) pairs for a Roboflow-style split dir (images/ + labels/)."""
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    pairs: list[tuple[Path, Path]] = []
    if img_dir.is_dir():
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                pairs.append((img, lbl_dir / (img.stem + ".txt")))
    return pairs


def prepare(
    data_arg: Path, out_dir: Path, *, resplit: bool,
    ratios=(0.7, 0.15, 0.15), max_images: Optional[int] = None, seed: int = 0,
) -> tuple[Path, dict]:
    """Build a clean train/val/test dataset (absolute-path data.yaml) for YOLO."""
    import yaml

    yaml_path = _find_yaml(data_arg)
    if not yaml_path:
        raise SystemExit(f"No data.yaml found in {data_arg}")
    root = yaml_path.parent
    names = _read_names(yaml_path)
    if not names:
        raise SystemExit(f"No class names in {yaml_path}")

    if resplit:
        # Pool all images across the existing splits, then re-split 70/15/15.
        pooled: list[tuple[Path, Path]] = []
        for sp in ("train", "valid", "val", "test"):
            pooled += _pairs_in(root / sp)
        seen, uniq = set(), []
        for img, lbl in pooled:
            if img.name not in seen:
                seen.add(img.name)
                uniq.append((img, lbl))
        if not uniq:
            raise SystemExit(f"No images found under {root}")
        random.Random(seed).shuffle(uniq)
        n, n_tr, n_va = len(uniq), int(len(uniq) * ratios[0]), int(len(uniq) * ratios[1])
        parts = {"train": uniq[:n_tr], "val": uniq[n_tr:n_tr + n_va], "test": uniq[n_tr + n_va:]}
    else:
        parts = {}
        for out_sp, in_sps in (("train", ["train"]), ("val", ["valid", "val"]), ("test", ["test"])):
            got: list[tuple[Path, Path]] = []
            for s in in_sps:
                got += _pairs_in(root / s)
            parts[out_sp] = got
        if not parts["test"]:
            parts["test"] = list(parts["val"])  # evaluate on val if there is no test split

    if max_images and parts["train"]:
        random.Random(seed).shuffle(parts["train"])
        parts["train"] = parts["train"][:max_images]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts = {}
    for sp, items in parts.items():
        (out_dir / sp / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / sp / "labels").mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, out_dir / sp / "images" / img.name)
            dst = out_dir / sp / "labels" / (img.stem + ".txt")
            if lbl and lbl.is_file():
                shutil.copy2(lbl, dst)
            else:
                dst.write_text("", encoding="utf-8")  # background-only image
        counts[sp] = len(items)

    data = {"path": str(out_dir.resolve()), "train": "train/images", "val": "val/images",
            "test": "test/images", "nc": len(names), "names": names}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    meta = {"source": str(root), "names": names,
            "split_mode": "re-split 70/15/15" if resplit else "dataset's own splits", "counts": counts}
    print(f"prepared: {counts}  classes={names}  ({meta['split_mode']})")
    return out_dir / "data.yaml", meta


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def train_and_eval(
    data_yaml: Path, meta: dict, *, epochs: int, imgsz: int, batch: int,
    model_name: str, out_dir: Path, device: str = "cpu", save_weights: bool = True,
) -> dict:
    import yaml as _yaml
    from ultralytics import YOLO

    try:
        import torch

        torch.set_num_threads(max(1, os.cpu_count() or 1))
    except Exception:
        pass

    model = YOLO(model_name)
    t0 = time.perf_counter()
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                workers=0, project=str(out_dir), name="train", exist_ok=True, plots=True,
                verbose=False, seed=0)
    train_s = time.perf_counter() - t0

    data = _yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    split = "test" if data.get("test") else "val"
    m = model.val(data=str(data_yaml), split=split, device=device, workers=0,
                  project=str(out_dir), name="eval", exist_ok=True, plots=True, verbose=False)
    box, names = m.box, meta["names"]

    precision, recall = float(box.mp), float(box.mr)
    per_class = {}
    try:
        for i, c in enumerate(box.ap_class_index):
            ci = int(c)
            per_class[names[ci]] = {
                "map50": round(float(box.ap50[i]), 4),
                "precision": round(float(box.p[i]), 4),
                "recall": round(float(box.r[i]), 4),
                "f1": round(float(box.f1[i]), 4),
            }
    except Exception:
        pass

    result = {
        "dataset": meta["source"], "split_mode": meta["split_mode"], "counts": meta["counts"],
        "classes": names, "model": model_name, "epochs": epochs, "imgsz": imgsz, "batch": batch,
        "device": device, "train_seconds": round(train_s, 1), "split_evaluated": split,
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_f1(precision, recall), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
        },
        "per_class_map50": {k: v["map50"] for k, v in per_class.items()},
        "per_class": per_class,
        "weights": str((out_dir / "train" / "weights" / "best.pt").resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if save_weights:
        best = out_dir / "train" / "weights" / "best.pt"
        if best.is_file():
            dest = resolve_path("models/yolo_rail.pt")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, dest)
            result["exported_to"] = str(dest)
            print(f"exported best.pt -> {dest}  (the live pipeline now loads this)")
    return result


def write_report(result: dict, out_dir: Path) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    m = result["metrics"]
    lines = [
        "# Training report — rail-defect detector", "",
        f"- generated: {result['timestamp']}",
        f"- dataset: `{result['dataset']}`",
        f"- split: {result['split_mode']} — {result['counts']}",
        f"- classes ({len(result['classes'])}): {', '.join(result['classes'])}",
        f"- model: {result['model']} · epochs {result['epochs']} · imgsz {result['imgsz']} "
        f"· batch {result['batch']} · {result['device']}",
        f"- training time: {result['train_seconds']} s · evaluated on the {result['split_evaluated']} split",
        "", "## Overall", "", "| metric | value |", "|---|---|",
        f"| precision | {m['precision']:.3f} |",
        f"| recall | {m['recall']:.3f} |",
        f"| F1 | {m['f1']:.3f} |",
        f"| mAP@0.5 | {m['map50']:.3f} |",
        f"| mAP@0.5:0.95 | {m['map50_95']:.3f} |", "",
    ]
    if result.get("per_class"):
        lines += ["## Per class", "", "| class | mAP@0.5 | precision | recall | F1 |",
                  "|---|---|---|---|---|"]
        for c, v in result["per_class"].items():
            lines.append(f"| {c} | {v['map50']:.3f} | {v['precision']:.3f} | "
                         f"{v['recall']:.3f} | {v['f1']:.3f} |")
        lines.append("")
    if result.get("note"):
        lines += ["> " + result["note"], ""]
    lines += ["## Run artefacts", "",
              f"- weights: `{result['weights']}`"
              + (f"  (copied to `{result['exported_to']}`)" if result.get("exported_to") else ""),
              f"- Ultralytics plots: `{out_dir}/train/` and `{out_dir}/eval/` "
              "(confusion_matrix.png, PR_curve.png, results.png).", ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {out_dir / 'report.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train/benchmark a YOLO rail-defect detector (CPU-friendly).")
    ap.add_argument("--data", required=True, help="dataset folder (Roboflow YOLO export) or a data.yaml")
    ap.add_argument("--keep-splits", action="store_true",
                    help="use the dataset's own train/valid/test instead of re-splitting 70/15/15")
    ap.add_argument("--max-images", type=int, default=None, help="cap TRAIN images (CPU feasibility)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolov8n.pt",
                    help="YOLO weights/config; default yolov8n.pt (safe for the Coral export path)")
    ap.add_argument("--name", default=None, help="run name under data/output/training/")
    ap.add_argument("--no-save-weights", action="store_true",
                    help="do not copy best.pt to models/yolo_rail.pt")
    ap.add_argument("--device", default=None,
                    help="training device: '0' = first GPU, 'cpu', etc. Default auto-detects CUDA.")
    args = ap.parse_args()

    if args.device is not None:
        device = args.device
    else:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"[train] device: {device}")

    run_name = args.name or Path(args.data).name.replace(" ", "_")[:40]
    out = resolve_path("data/output/training") / run_name
    data_yaml, meta = prepare(Path(args.data), out / "dataset", resplit=not args.keep_splits,
                              max_images=args.max_images)
    result = train_and_eval(data_yaml, meta, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                            model_name=args.model, out_dir=out, save_weights=not args.no_save_weights,
                            device=device)
    result["note"] = ("Trained on CPU; raise --epochs / --max-images (or use a GPU) for "
                      "production-grade metrics. Class set follows the dataset, not necessarily "
                      "the 7 contract classes.")
    write_report(result, out)
    print("\nTraining complete. Metrics:")
    print(json.dumps(result["metrics"], indent=2))
    if result.get("per_class_map50"):
        print("per-class mAP@0.5:", json.dumps(result["per_class_map50"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
