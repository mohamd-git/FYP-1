"""
run.py
======
Single entry point for the AGV PoC.

Modes:
    python run.py                # run the inspection pipeline (publishes MQTT + writes SQLite)
    python run.py --dashboard    # run the dashboard server only (subscribes + serves the UI)
    python run.py --all          # run the dashboard AND a looping pipeline together (one terminal)

Live messages go to MQTT (default local Mosquitto, public HiveMQ as fallback):
    agv/detections   agv/telemetry   agv/status

Watch the stream in a second terminal:
    mosquitto_sub -t 'agv/#' -v            # if Mosquitto's tools are installed
    python -m src.messaging.subscriber     # built-in, no extra tools
Or open the dashboard:  python run.py --dashboard  ->  http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import sys
import threading
import traceback

from src.config import CONFIG_PATH, load_config, seed_all
from src.log import setup_logging


def _fail(message: str, code: int = 1) -> int:
    print(f"\nERROR: {message}", file=sys.stderr)
    return code


def _load_config_or_exit() -> dict:
    try:
        return load_config()
    except FileNotFoundError:
        sys.exit(_fail(f"config.yaml not found at {CONFIG_PATH}. Run from the project root."))
    except Exception as exc:  # YAML parse error, etc.
        sys.exit(_fail(f"could not parse config.yaml ({exc}). Check the YAML syntax / indentation."))


def run_pipeline(config: dict) -> int:
    """Build and run the inspection pipeline once; print a summary."""
    try:
        from src.pipeline import Pipeline

        pipeline = Pipeline.from_config(config)
    except Exception as exc:
        return _fail(f"failed to build the pipeline: {exc}")

    try:
        summary = pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user -- shutting down cleanly.")
        return 0
    except FileNotFoundError as exc:
        return _fail(f"input not found: {exc}\n"
                     f"Add a clip at paths.video_input (config.yaml) or set source.type: images.")
    except ImportError as exc:
        return _fail(f"a dependency is missing ({exc}). Run: pip install -r requirements.txt")
    except Exception as exc:
        traceback.print_exc()
        return _fail(f"pipeline error: {exc}")

    print("\n" + "=" * 60)
    print(" Inspection complete")
    print("=" * 60)
    print(f"  frames processed       : {summary.get('frames')}")
    print(f"  raw detections         : {summary.get('raw_detections')}")
    print(f"  skipped (off-contract) : {summary.get('skipped_classes')}")
    print(f"  defects persisted      : {summary.get('persisted_defects')}")
    print(f"  detections published   : {summary.get('published_detections')}")
    print(f"  telemetry messages     : {summary.get('telemetry_sent')}")
    print("=" * 60)
    if summary.get("persisted_defects", 0) == 0 and summary.get("raw_detections", 0) > 0:
        print("Note: detections were found but none matched a contract defect class.")
        print("With the placeholder model, keep inference.class_map enabled in config.yaml")
        print("(it is on by default) or train rail weights at models/yolo_rail.pt.")
    return 0


def run_all(config: dict) -> int:
    """Run the dashboard (foreground) and a looping pipeline (background thread)."""
    from src.dashboard.app import run_dashboard

    # Loop the clip so the live feed keeps flowing while the dashboard is open.
    config.setdefault("source", {})["loop"] = True

    def _pipeline_thread() -> None:
        try:
            from src.pipeline import Pipeline

            Pipeline.from_config(config).run()
        except Exception as exc:  # keep the dashboard alive even if the pipeline stops
            print(f"[pipeline] stopped: {exc}", file=sys.stderr)

    threading.Thread(target=_pipeline_thread, name="pipeline", daemon=True).start()
    print("[run] launched pipeline (looping) + dashboard together.")
    try:
        run_dashboard(config)
    except KeyboardInterrupt:
        print("\nInterrupted by user -- shutting down cleanly.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AGV Rail Defect Detection PoC")
    parser.add_argument("--dashboard", action="store_true",
                        help="run the dashboard server only (no pipeline)")
    parser.add_argument("--all", action="store_true",
                        help="run the dashboard and a looping pipeline together")
    args = parser.parse_args()

    banner = "AGV for Rail Defect Detection -- Inspection & Smart Maintenance (PoC)"
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    config = _load_config_or_exit()
    setup_logging((config.get("logging") or {}).get("level", "INFO"))
    seed_all(int(config.get("random_seed", 0)))
    mqtt = config.get("mqtt", {}) or {}
    fb = mqtt.get("fallback", {}) or {}
    print(f"  config       : {CONFIG_PATH}")
    print(f"  MQTT primary : {mqtt.get('host')}:{mqtt.get('port')} "
          f"(fallback: {fb.get('host')}:{fb.get('port')})")
    print(f"  topics       : {mqtt.get('topics')}")
    print()

    if args.dashboard:
        try:
            from src.dashboard.app import run_dashboard

            run_dashboard(config)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
        except Exception as exc:
            traceback.print_exc()
            return _fail(f"dashboard error: {exc}")
        return 0

    if args.all:
        return run_all(config)

    return run_pipeline(config)


if __name__ == "__main__":
    sys.exit(main())
