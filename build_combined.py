"""
build_combined.py  (one-off data-prep for the crack+fastener candidate model)
=============================================================================
Builds ONE clean YOLO-detection dataset that merges:

  * the fastener dataset (already axis-aligned boxes, 6 classes), and
  * the new "rail surface defects" crack dataset (YOLO-OBB) -- converted to
    axis-aligned boxes, ONLY Cracks(0) + breaks(3) -> a single `crack` class
    (Rails/Scars/lightbands dropped).

Why we re-split BOTH ourselves (the careful part):
  * The fastener export's own split has ~no test set (all in train) -> unusable
    for a fair test.
  * The crack export is heavily augmented: the SAME photo appears as many files
    with different `.rf.<hash>` names, scattered across splits -> naive splitting
    leaks the test answers into training.
  Fix: group EVERY file by its BASE photo, split the BASE photos 70/15/15, and
  send all copies of a photo to the same split. TRAIN keeps all (augmented)
  copies; VAL/TEST keep one clean copy per photo so the score is honest.
  => no photo can appear in two splits (asserted).

Output: data/combined_crack_fastener/{train,val,test}/{images,labels} + data.yaml
Unified classes (7): fastener, fastener-2, fastener2_broken, fastener_broken,
                     missing, trackbed_stuff, crack   (crack = index 6)

Run:  python build_combined.py
"""
from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path
from random import Random

# --- sources -----------------------------------------------------------------
FASTENER = Path(r"C:\Users\mohammed sharafuddin\Desktop\FYP\Dataset\railway-track_fastener_defcts1.v1i.yolov11")
CRACK    = Path(r"C:\Users\mohammed sharafuddin\Downloads\rail surface defects.v3i.yolov8-obb")
OUT      = Path(__file__).resolve().parent / "data" / "combined_crack_fastener"

FASTENER_NAMES = ["fastener", "fastener-2", "fastener2_broken", "fastener_broken", "missing", "trackbed_stuff"]
CRACK_IDX      = len(FASTENER_NAMES)            # 6
UNIFIED_NAMES  = FASTENER_NAMES + ["crack"]
KEEP_CRACK_SRC = {0, 3}                          # Cracks, breaks  -> crack
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SEED = 0


def base_photo(stem: str) -> str:
    """Collapse Roboflow augmentation/duplicate variants to one source photo id."""
    b = stem.split("_jpg.rf.")[0]
    b = re.sub(r"-Copy.*$", "", b)
    return b


def collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for sp in ("train", "valid", "val", "test"):
        img_dir, lbl_dir = root / sp / "images", root / sp / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                pairs.append((img, lbl_dir / (img.stem + ".txt")))
    return pairs


def base_split(pairs: list[tuple[Path, Path]]) -> tuple[dict, dict]:
    by_base: dict[str, list] = defaultdict(list)
    for img, lbl in pairs:
        by_base[base_photo(img.stem)].append((img, lbl))
    bases = sorted(by_base)
    Random(SEED).shuffle(bases)
    n = len(bases)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    split_of = {b: ("train" if i < n_tr else ("val" if i < n_tr + n_va else "test"))
                for i, b in enumerate(bases)}
    return by_base, split_of


# --- label transforms --------------------------------------------------------
def fastener_label(src_lbl: Path) -> tuple[list[str], int]:
    out, bad = [], 0
    if src_lbl.is_file():
        for ln in src_lbl.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            t = ln.split()
            if len(t) == 5 and 0 <= int(float(t[0])) < len(FASTENER_NAMES):
                out.append(ln)
            else:
                bad += 1
    return out, bad


def crack_label(src_lbl: Path) -> tuple[list[str], int]:
    """OBB `cls x1 y1..x4 y4` -> `crack xc yc w h`, keeping only Cracks/breaks."""
    out = []
    if src_lbl.is_file():
        for ln in src_lbl.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            t = ln.split()
            if len(t) < 9 or int(float(t[0])) not in KEEP_CRACK_SRC:
                continue
            c = list(map(float, t[1:9]))
            xs, ys = c[0::2], c[1::2]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            clip = lambda v: min(max(v, 0.0), 1.0)
            xc, yc, w, h = clip((xmin + xmax) / 2), clip((ymin + ymax) / 2), clip(xmax - xmin), clip(ymax - ymin)
            if w > 0 and h > 0:
                out.append(f"{CRACK_IDX} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return out, 0


def add_dataset(root: Path, label_fn, name: str) -> tuple[dict, dict]:
    pairs = collect_pairs(root)
    by_base, split_of = base_split(pairs)
    counts = {"train": 0, "val": 0, "test": 0}
    defect_imgs = {"train": 0, "val": 0, "test": 0}      # imgs that kept >=1 box
    placed = defaultdict(set)
    total_bad = 0
    for b, items in by_base.items():
        sp = split_of[b]
        chosen = items if sp == "train" else items[:1]    # train: all copies; val/test: one clean copy
        for img, lbl in chosen:
            lines, bad = label_fn(lbl)
            total_bad += bad
            shutil.copy2(img, OUT / sp / "images" / img.name)
            (OUT / sp / "labels" / (img.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            counts[sp] += 1
            if lines:
                defect_imgs[sp] += 1
        placed[sp].add(b)
    overlap = (placed["train"] & placed["val"]) | (placed["train"] & placed["test"]) | (placed["val"] & placed["test"])
    assert not overlap, f"LEAKAGE in {name}: {len(overlap)} base photos in >1 split"
    print(f"  {name}: {len(by_base)} unique photos (from {len(pairs)} files) | "
          f"split train {counts['train']} / val {counts['val']} / test {counts['test']} | "
          f"with-defect {defect_imgs} | bad-lines {total_bad} | leakage 0 [OK]")
    return counts, defect_imgs


def main() -> None:
    print(f"Building combined dataset -> {OUT}")
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT / sp / "labels").mkdir(parents=True, exist_ok=True)

    fc, _ = add_dataset(FASTENER, fastener_label, "fastener")
    cc, cd = add_dataset(CRACK, crack_label, "crack   ")

    import yaml
    (OUT / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(OUT.resolve()), "train": "train/images", "val": "val/images",
         "test": "test/images", "nc": len(UNIFIED_NAMES), "names": UNIFIED_NAMES},
        sort_keys=False), encoding="utf-8")

    print("\nFINAL combined dataset (images per split):")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5}: {fc[sp] + cc[sp]:5}  (fastener {fc[sp]} + crack {cc[sp]}; crack-with-defect {cd[sp]})")
    print(f"classes ({len(UNIFIED_NAMES)}): {UNIFIED_NAMES}")
    print(f"\nTest set has BOTH fastener and crack images -> fair comparison. data.yaml -> {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()
