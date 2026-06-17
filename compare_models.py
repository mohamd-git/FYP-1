"""
compare_models.py — fair, leakage-free OLD-vs-NEW comparison
============================================================
Compares the OLD baseline (fastener-only, `models/yolo_rail.backup-*.pt`) against
the NEW candidate (crack+fastener, trained by build_combined.py + train.py) on the
SAME clean, leakage-free test split.

Honesty notes baked in:
  * The OLD model has 6 classes (no crack) -> it is evaluated on a fastener-only
    view of the clean test, so it isn't unfairly penalised for a class it lacks.
  * The OLD model was originally TRAINED with a leaky re-split, so even on this
    clean test it has *seen* some of these photos -> its number is optimistic.
    The NEW candidate has seen NONE of the test photos -> its number is the honest
    one. (Stated in the printout so nobody is misled.)

Usage:
  python compare_models.py --build-only      # just build the fastener-only test
  python compare_models.py                    # build + eval baseline + compare
  python compare_models.py --device 0         # eval on GPU (only when free)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

AGV       = Path(__file__).resolve().parent
COMBINED  = AGV / "data" / "combined_crack_fastener"
BASELINE  = AGV / "models" / "yolo_rail.backup-2026-06-11.pt"
CAND_DIR  = AGV / "data" / "output" / "training" / "crack_fastener_combined"
EVAL      = AGV / "data" / "_eval_fastener_only"
FAST_NAMES = ["fastener", "fastener-2", "fastener2_broken", "fastener_broken", "missing", "trackbed_stuff"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_fastener_only_test() -> int:
    """FALLBACK only: the 102 true fastener test images (label has a class 0..5) + 6-class yaml.
    Used if Ultralytics refuses to val the 6-class baseline against the 7-class data.yaml."""
    if EVAL.exists():
        shutil.rmtree(EVAL)
    (EVAL / "test" / "images").mkdir(parents=True)
    (EVAL / "test" / "labels").mkdir(parents=True)
    src, n = COMBINED / "test", 0
    for lab in (src / "labels").glob("*.txt"):
        lines = [x for x in lab.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not any(0 <= int(x.split()[0]) <= 5 for x in lines):
            continue  # keep ONLY real fastener images
        imgs = [p for p in (src / "images").glob(lab.stem + ".*") if p.suffix.lower() in IMG_EXTS]
        if not imgs:
            continue
        shutil.copy2(imgs[0], EVAL / "test" / "images" / imgs[0].name)
        (EVAL / "test" / "labels" / lab.name).write_text("\n".join(lines), encoding="utf-8")
        n += 1
    import yaml
    (EVAL / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(EVAL.resolve()), "train": "test/images", "val": "test/images",
         "test": "test/images", "nc": 6, "names": FAST_NAMES}, sort_keys=False), encoding="utf-8")
    print(f"[fallback] fastener-only clean test built: {n} images")
    return n


def eval_baseline(device: str) -> tuple[float, dict]:
    """Evaluate the 6-class baseline on the SAME 182-image clean test as the candidate.
    Class order matches (0..5 identical), so indices align; crack(6) simply gets 0."""
    from ultralytics import YOLO
    model = YOLO(str(BASELINE))
    proj = str(AGV / "data" / "output" / "baseline_eval")
    try:
        r = model.val(data=str(COMBINED / "data.yaml"), split="test", device=device, workers=0,
                      verbose=False, project=proj, name="clean_test", exist_ok=True, plots=False)
    except Exception as e:
        print(f"[warn] 7-class val failed ({type(e).__name__}: {e}); using fastener-only fallback")
        build_fastener_only_test()
        r = model.val(data=str(EVAL / "data.yaml"), split="test", device=device, workers=0,
                      verbose=False, project=proj, name="clean_test_fb", exist_ok=True, plots=False)
    box = r.box
    per = {FAST_NAMES[int(c)]: round(float(box.ap50[i]), 4)
           for i, c in enumerate(box.ap_class_index) if int(c) < len(FAST_NAMES)}
    fast_mean = round(sum(per.values()) / len(per), 4) if per else 0.0
    return fast_mean, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu (default, safe during training) or 0 for GPU")
    args = ap.parse_args()

    if args.build_only:
        build_fastener_only_test()
        return

    base_map, base_per = eval_baseline(args.device)
    cand_metrics = CAND_DIR / "metrics.json"
    cand = json.loads(cand_metrics.read_text(encoding="utf-8")) if cand_metrics.is_file() else None
    cper = cand["per_class_map50"] if cand else {}

    print("\n" + "=" * 64)
    print("  mAP@0.5 on the CLEAN, leakage-free test set")
    print("  OLD = fastener-only baseline (optimistic: trained on some of these)")
    print("  NEW = crack+fastener candidate (honest: never saw any test photo)")
    print("=" * 64)
    print(f"  {'class':22}{'OLD baseline':>14}{'NEW candidate':>15}")
    for c in FAST_NAMES:
        print(f"  {c:22}{base_per.get(c, 0.0):>14.3f}{cper.get(c, float('nan')):>15.3f}")
    print(f"  {'crack':22}{'(no class)':>14}{cper.get('crack', float('nan')):>15.3f}")
    print("-" * 64)
    print(f"  {'fastener mean mAP':22}{base_map:>14.3f}"
          f"{(sum(cper.get(c,0) for c in FAST_NAMES)/len(FAST_NAMES)) if cper else float('nan'):>15.3f}")
    if cand:
        (CAND_DIR / "comparison.json").write_text(json.dumps(
            {"baseline_fastener_map50": base_map, "baseline_per_class": base_per,
             "candidate_per_class": cper, "candidate_overall": cand.get("metrics")}, indent=2),
            encoding="utf-8")
        print(f"\nsaved -> {CAND_DIR/'comparison.json'}")
    else:
        print("\n(candidate not trained yet — re-run after training to fill the NEW column)")


if __name__ == "__main__":
    main()
