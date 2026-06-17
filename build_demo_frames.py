"""
build_demo_frames.py — curate CLEAN demo footage for the live dashboard.
=======================================================================
Sources from the in-project `data/combined_full` val+test splits (clean,
de-duplicated, one copy per photo, spanning all classes) rather than the
original download folders (which the user keeps relocating). Rejects any image
with black cutout/rotation augmentation patches, and balances across the new
defect classes so the live view shows variety.

Run:  python build_demo_frames.py
"""
from __future__ import annotations

import shutil
from pathlib import Path
from random import Random

import cv2

AGV = Path(__file__).resolve().parent
SRC = AGV / "data" / "combined_full"
OUT = AGV / "data" / "frames"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
RNG = Random(7)


def black_frac(p: Path) -> float:
    img = cv2.imread(str(p))
    if img is None:
        return 1.0
    return float((img.max(axis=2) < 6).mean())


def gather():
    items = []
    for split in ("train", "val", "test"):   # train has all aug copies -> more clean (flip-only) options
        ldir, idir = SRC / split / "labels", SRC / split / "images"
        if not ldir.is_dir():
            continue
        for lbl in ldir.glob("*.txt"):
            classes = {int(float(x.split()[0])) for x in lbl.read_text(encoding="utf-8").splitlines() if x.strip()}
            if not classes:
                continue
            imgs = [p for p in idir.glob(lbl.stem + ".*") if p.suffix.lower() in IMG_EXTS]
            if imgs:
                items.append((imgs[0], classes))
    RNG.shuffle(items)
    return items


def main():
    items = gather()
    # (label, class-ids that count, how many)  — emphasise the new classes
    targets = [
        ("corrugation", {8}, 8),
        ("spalling",    {7}, 8),
        ("squat",       {9}, 6),
        ("crack",       {6}, 9),
        ("fastener",    {2, 3, 4}, 9),
    ]
    used, groups = set(), []
    for label, cls, n in targets:
        picks = []
        for img, classes in items:
            if img in used or not (classes & cls):
                continue
            if black_frac(img) <= 0.012:
                picks.append(img)
                used.add(img)
            if len(picks) >= n:
                break
        groups.append((label, picks))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    seq, summary = 0, {}
    for label, paths in groups:
        summary[label] = len(paths)
        for p in paths:
            seq += 1
            shutil.copy2(p, OUT / f"{seq:02d}_{label}{p.suffix.lower()}")
    print("clean demo frames -> data/frames:")
    for k, v in summary.items():
        print(f"  {k:12} {v}")
    print(f"  TOTAL        {seq}")
    if seq == 0:
        print("WARNING: no frames selected — check data/combined_full exists.")


if __name__ == "__main__":
    main()
