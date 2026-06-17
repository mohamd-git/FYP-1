"""
resume_and_eval.py — resume the interrupted crack+fastener run, then evaluate.
Resumes training from the last saved checkpoint (no progress lost), trains out to
100 epochs, then evaluates best.pt on the clean TEST split and writes metrics.json
in the same shape train.py would (so compare_models.py can read it).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO

AGV  = Path(__file__).resolve().parent
RUN  = AGV / "data" / "output" / "training" / (sys.argv[1] if len(sys.argv) > 1 else "crack_fastener_combined")
LAST = RUN / "train" / "weights" / "last.pt"
BEST = RUN / "train" / "weights" / "best.pt"
DATA = RUN / "dataset" / "data.yaml"


def f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def main() -> None:
    print(f"[resume] from {LAST}", flush=True)
    YOLO(str(LAST)).train(resume=True)          # continues to the original 100 epochs
    print("[resume] training complete; evaluating best.pt on TEST split", flush=True)

    names = yaml.safe_load(DATA.read_text(encoding="utf-8")).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    res = YOLO(str(BEST)).val(data=str(DATA), split="test", device="0", workers=0,
                              plots=True, project=str(RUN), name="eval", exist_ok=True, verbose=False)
    box = res.box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        ci = int(c)
        per_class[names[ci]] = {
            "map50": round(float(box.ap50[i]), 4), "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4), "f1": round(float(box.f1[i]), 4),
        }
    result = {
        "dataset": "data/combined_crack_fastener", "classes": names,
        "metrics": {
            "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
            "f1": round(f1(float(box.mp), float(box.mr)), 4),
            "map50": round(float(box.map50), 4), "map50_95": round(float(box.map), 4),
        },
        "per_class_map50": {k: v["map50"] for k, v in per_class.items()},
        "per_class": per_class,
        "weights": str(BEST),
    }
    (RUN / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[resume] metrics.json written.")
    print("overall:", json.dumps(result["metrics"], indent=2))
    print("per-class mAP50:", json.dumps(result["per_class_map50"], indent=2))


if __name__ == "__main__":
    main()
