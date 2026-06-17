# AGV for Rail Defect Detection — Inspection & Smart Maintenance System

A made-in-Malaysia, rail-bound **Automated Guided Vehicle (AGV)** that inspects railway
track, detects defects with AI vision, geo-tags each finding to its position along the
track, and turns every detection into a **prioritised, actionable maintenance record**
streamed live to an operator dashboard.

> **Core novelty — the prescriptive layer.** Most systems stop at *detecting* a defect.
> This one closes the loop: every detection gets a **severity**, an **urgency score
> (0–100)**, and a **recommended maintenance action** — so a finding is immediately
> useful to a maintenance crew.

---

## FYP1 vs FYP2

This repository is the **FYP1 software prototype**: a laptop-runnable system that
demonstrates the full workflow — **detect → prescribe → localise → record → display** —
on sample footage and a simulated GPS track. It is built so the FYP2 hardware drops in
behind the same interfaces with **no downstream change**.

| Built — FYP1 (this repo) | Planned — FYP2 (hardware) |
|---|---|
| 6-class YOLOv8 detector (real trained weights) | Physical AGV: Raspberry Pi 5 + Coral Edge TPU |
| Prescriptive severity / urgency / action engine | 4K camera + real GPS / IMU / wheel odometry |
| Live dashboard + maintenance register + map | On-track deployment + Malaysian-rail fine-tuning |
| Evaluation evidence pack, Docker, automated tests | Predictive (remaining-life) analytics |

---

## The detector

Trained (GPU or CPU) on three public datasets — fastener defects, rail-surface defects,
and a rail-defect set covering corrugation/spalling/squat — over **6 defect classes**,
evaluated on a clean, **leakage-free** held-out split:

| Class | mAP@0.5 |
|---|---|
| Missing fastener | 0.99 |
| Crack | 0.80 |
| Broken fastener | 0.77 – 0.98 |
| Squat | 0.73 |
| Corrugation | 0.68 |
| Spalling | 0.60 |
| **Overall (6 classes)** | **0.818** |

`loose_fastener` is in the contract but has no public boxed data, so it is documented as
**future work**. Metrics are on public benchmark data — not Malaysian rail yet.

---

## Quick start (Windows / PowerShell — Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env        # placeholders; can stay blank for the PoC

python run.py --all                # pipeline + dashboard → http://127.0.0.1:5000
```

Modes: `python run.py` (pipeline only) · `--dashboard` (UI only) · `--all` (both).
macOS/Linux: `source .venv/bin/activate` and `cp .env.example .env`.

The live engine loads custom weights from **`models/yolo_rail.pt`** (produced by
`train.py`); if absent it falls back to a generic placeholder so the pipeline still runs.

---

## Dashboard

Open **http://127.0.0.1:5000** — a live operator dashboard (HUD dark theme, with a light
"Control Room" toggle), updating over WebSocket:

- **Stat cards + Health Score**, and **gauges** (battery, inspection progress, avg urgency)
- **Live map** (OpenStreetMap) placing each defect on a real rail line + the inspected track
- **Severity & by-class charts**, a **chainage strip**, and a **priority work queue**
  (top defects with crop thumbnails + recommended action)
- **Maintenance log** — sortable / filterable, with CSV / JSON export

Live data streams over **MQTT** — a local Mosquitto broker, or the public HiveMQ broker as
an automatic fallback (both anonymous; no keys needed).

---

## Train / evaluate

```powershell
python train.py --data "..\Dataset\<roboflow-export>" --epochs 100   # auto-uses GPU if present, else CPU
python evaluate.py --model <best.pt> --data <data.yaml>              # evidence pack
```

`train.py` consumes standard YOLO datasets, re-splits 70/15/15 (`--keep-splits` to keep the
export's own splits), reports precision/recall/F1/mAP@0.5 + per-class mAP, and copies the
best weights to `models/yolo_rail.pt`. `evaluate.py` writes an `evaluation/` pack: confusion
matrix, PR/F1 curves, per-class mAP with a **≥ 0.70 PASS/FAIL** flag, p50/p95 CPU latency,
and a one-page `summary.md`. `baseline_compare.py` compares the system against a manual review.

---

## Architecture — three swappable seams

The design hangs on three components behind abstract interfaces. In FYP2 each is replaced by
a hardware version with **nothing downstream changing**.

| Seam | Interface | FYP1 | FYP2 |
|---|---|---|---|
| Frame source | `sources.base.FrameSource` | video / image folder | live camera |
| Inference | `inference.base.InferenceEngine` | Ultralytics YOLO | TFLite-INT8 on Coral |
| Localisation | `localisation.base.LocalisationSource` | simulated GPS CSV | real GPS + odometry + IMU |

**The contract** — one source of truth in [`src/schema.py`](src/schema.py): `Detection`,
`Telemetry`, `Status` (Pydantic), carried on MQTT topics `agv/detections`, `agv/telemetry`,
`agv/status`. Every module imports these, so the wire format never drifts.

**Prescriptive rubric** ([`rules.yaml`](rules.yaml), tunable): missing/broken fastener = High;
crack = High if large or high-confidence else Medium; spalling/squat/loose = Medium;
corrugation = Low. `urgency = 100·(0.55·severity + 0.30·confidence + 0.15·size)`, banded into
**Immediate / Schedule / Routine / Monitor**.

---

## Docker & tests

```powershell
docker compose up --build     # Mosquitto broker + app → http://localhost:5000
pytest -q                     # prescriptive / schema / dedup suite (seeded for determinism)
```

---

## FYP2 migration

Add **three files** — one concrete class per seam — and change nothing else:
`src/sources/camera_source.py`, `src/inference/coral_engine.py`, `src/localisation/gps_imu.py`.
Point `config.yaml` at them; the schema, MQTT topics, prescriptive engine, persistence and
dashboard are identical in both phases because every one depends only on the abstract interfaces.

---

## Project structure

```
AGV code/
├── run.py              entry point  (--dashboard / --all)
├── train.py            train YOLO → models/yolo_rail.pt
├── evaluate.py         evaluation evidence pack
├── baseline_compare.py system-vs-manual comparison
├── config.yaml         all settings      ·   rules.yaml   prescriptive rules
├── data/track.csv      sample corridor (on a real KL rail line)
├── tests/              pytest suite
└── src/
    ├── schema.py       the Detection / Telemetry / Status contract
    ├── pipeline.py     detect → prescribe → localise → dedup → persist → publish
    ├── sources/  inference/  localisation/    the three swappable seams
    ├── prescriptive/engine.py    severity / urgency / action
    ├── messaging/      MQTT publisher + subscriber
    ├── storage/db.py   SQLite defect register + CSV/JSON export
    └── dashboard/      Flask + Socket.IO operator UI
```

---
*Final-year undergraduate project — iterative agile SDLC.*
