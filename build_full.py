"""
build_full.py  (comprehensive rail-defect dataset: fastener + crack + Chula)
============================================================================
Merges THREE sources into one clean, leakage-free YOLO-detection dataset:

  * fastener  (railway-track_fastener_defcts1, YOLO boxes)  -> classes 0..5 as-is
  * crack     (rail surface defects v3, YOLO-OBB)            -> Cracks+breaks -> `crack`
  * Chula     (01_FullDevelopment_Railway, TF CSV boxes)     -> Spalling/Corrugation/Squat

Every file is grouped by its BASE photo and split 70/15/15 by base (TRAIN keeps all
augmented copies; VAL/TEST keep one clean copy per photo) so no photo lands in two
splits -> honest scores. (Wheel Burn dropped; it's not a contract class.)

Unified classes (10):
  0 fastener 1 fastener-2 2 fastener2_broken 3 fastener_broken 4 missing
  5 trackbed_stuff 6 crack 7 spalling 8 corrugation 9 squat

Run:  python build_full.py
"""
from __future__ import annotations

import csv
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from random import Random

FASTENER = Path(r"C:\Users\mohammed sharafuddin\Desktop\FYP\Dataset\railway-track_fastener_defcts1.v1i.yolov11")
CRACK    = Path(r"C:\Users\mohammed sharafuddin\Desktop\FYP\Dataset\rail surface defects.v3i.yolov8-obb")
CHULA    = Path(r"C:\Users\mohammed sharafuddin\Desktop\FYP\Dataset\01_FullDevelopment_RailwayDataset.v1i.tensorflow")
OUT      = Path(__file__).resolve().parent / "data" / "combined_full"

NAMES = ["fastener", "fastener-2", "fastener2_broken", "fastener_broken", "missing",
         "trackbed_stuff", "crack", "spalling", "corrugation", "squat"]
INCLUDE_SQUAT = True
CHULA_MAP = {"Spalling": 7, "Corrugation": 8, "Squat": 9}   # Wheel Burn -> dropped
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SEED = 0


def base_photo(stem: str) -> str:
    b = stem.split("_jpg.rf.")[0]
    b = re.sub(r"-Copy.*$", "", b)
    return b


# ---- gather (img_path, [yolo lines]) per source ------------------------------
def gather_fastener():
    out = []
    for sp in ("train", "valid", "test"):
        idir, ldir = FASTENER / sp / "images", FASTENER / sp / "labels"
        if not idir.is_dir():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lines = []
            lp = ldir / (img.stem + ".txt")
            if lp.is_file():
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    t = ln.split()
                    if len(t) == 5 and 0 <= int(float(t[0])) < 6:
                        lines.append(ln.strip())
            out.append((img, lines))
    return out


def gather_crack():
    out = []
    for sp in ("train", "valid", "test"):
        idir, ldir = CRACK / sp / "images", CRACK / sp / "labels"
        if not idir.is_dir():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lines = []
            lp = ldir / (img.stem + ".txt")
            if lp.is_file():
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    t = ln.split()
                    if len(t) < 9 or int(float(t[0])) not in (0, 3):
                        continue
                    c = list(map(float, t[1:9]))
                    xs, ys = c[0::2], c[1::2]
                    xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                    w, h = max(xs) - min(xs), max(ys) - min(ys)
                    if w > 0 and h > 0:
                        lines.append(f"6 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            out.append((img, lines))
    return out


def gather_chula():
    csvp = CHULA / "train" / "_annotations.csv"
    rows_by_img = defaultdict(list)
    for r in csv.DictReader(open(csvp, encoding="utf-8")):
        rows_by_img[r["filename"]].append(r)
    out = []
    for fn, rows in rows_by_img.items():
        img = CHULA / "train" / fn
        if not img.is_file():
            continue
        lines = []
        for r in rows:
            cls = r["class"]
            if cls not in CHULA_MAP:
                continue
            if cls == "Squat" and not INCLUDE_SQUAT:
                continue
            W, H = float(r["width"]), float(r["height"])
            x1, y1, x2, y2 = float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])
            xc, yc, w, h = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
            clip = lambda v: min(max(v, 0.0), 1.0)
            if w > 0 and h > 0:
                lines.append(f"{CHULA_MAP[cls]} {clip(xc):.6f} {clip(yc):.6f} {clip(w):.6f} {clip(h):.6f}")
        out.append((img, lines))
    return out


# ---- place into clean base-split ---------------------------------------------
def place(name: str, pairs, cls_counter: Counter):
    by_base = defaultdict(list)
    for img, lines in pairs:
        by_base[base_photo(img.stem)].append((img, lines))
    bases = sorted(by_base)
    Random(SEED).shuffle(bases)
    n = len(bases)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    counts = {"train": 0, "val": 0, "test": 0}
    placed = defaultdict(set)
    for i, b in enumerate(bases):
        sp = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        items = sorted(by_base[b], key=lambda x: x[0].name)
        chosen = items if sp == "train" else items[:1]
        for img, lines in chosen:
            shutil.copy2(img, OUT / sp / "images" / img.name)
            (OUT / sp / "labels" / (img.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            counts[sp] += 1
            for ln in lines:
                cls_counter[(sp, int(ln.split()[0]))] += 1
        placed[sp].add(b)
    ov = (placed["train"] & placed["val"]) | (placed["train"] & placed["test"]) | (placed["val"] & placed["test"])
    assert not ov, f"LEAKAGE in {name}: {len(ov)} bases in >1 split"
    print(f"  {name:9}: {n} photos -> train {counts['train']} / val {counts['val']} / test {counts['test']}  [leakage 0]")
    return counts


def main():
    print(f"Building -> {OUT}  (squat {'IN' if INCLUDE_SQUAT else 'OUT'})")
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT / sp / "labels").mkdir(parents=True, exist_ok=True)

    cc = Counter()
    place("fastener", gather_fastener(), cc)
    place("crack", gather_crack(), cc)
    place("chula", gather_chula(), cc)

    import yaml
    (OUT / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(OUT.resolve()), "train": "train/images", "val": "val/images",
         "test": "test/images", "nc": len(NAMES), "names": NAMES}, sort_keys=False), encoding="utf-8")

    print("\nper-class INSTANCE counts (train / val / test):")
    for ci, nm in enumerate(NAMES):
        print(f"  {ci} {nm:18} {cc[('train',ci)]:>5} / {cc[('val',ci)]:>4} / {cc[('test',ci)]:>4}")
    print(f"\nclasses ({len(NAMES)}): {NAMES}")
    print(f"data.yaml -> {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()
