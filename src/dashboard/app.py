"""
src/dashboard/app.py
====================
Flask + Flask-SocketIO backend for the operator dashboard (PoC).

- Subscribes to the MQTT contract topics and relays each message to connected
  browsers over WebSocket (Socket.IO) in real time.
- REST endpoints:
    GET /                  -> the operator dashboard page
    GET /api/defects       -> current maintenance log (consolidated, from SQLite)
    GET /api/state         -> latest telemetry/status snapshot + relay counters
    GET /export.csv        -> maintenance log as CSV (download)
    GET /export.json       -> maintenance log as JSON (download)
    GET /latest_frame.jpg  -> most recent defect image (representative crop)
    GET /crops/<file>      -> any saved crop by name (for image_ref lookups)

The styled operator UI (HUD dashboard, with a light "Control Room" theme) is
served from ``templates/index.html`` and ``static/`` (CSS, JS, vendored libs).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from flask import Flask, abort, jsonify, render_template, send_file, send_from_directory
from flask_socketio import SocketIO, emit

from src.config import load_config, resolve_path
from src.messaging.subscriber import MqttSubscriber
from src.storage.db import Database


def create_app(config: Optional[dict] = None):
    """Build the Flask app, Socket.IO server and MQTT relay subscriber.

    Returns ``(app, socketio, subscriber)``. The caller connects the subscriber
    and runs the server (see :func:`run_dashboard`).
    """
    config = config or load_config()
    here = Path(__file__).resolve().parent

    app = Flask(
        __name__,
        template_folder=str(here / "templates"),
        static_folder=str(here / "static"),
    )
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-agv-secret-change-me")
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

    paths = config.get("paths", {}) or {}
    storage = config.get("storage", {}) or {}
    db_path = resolve_path(storage.get("sqlite_path") or paths.get("database", "data/agv.db"))
    crops_dir = resolve_path(paths.get("crops_dir", "data/output/crops"))
    export_dir = resolve_path(paths.get("output_dir", "data/output"))
    mqtt_topics = (config.get("mqtt", {}) or {}).get("topics", {})

    # In-memory snapshot of the latest live state (sent to new browser clients).
    state: dict[str, Any] = {
        "latest_image_ref": None, "telemetry": None, "status": None, "detections": 0,
    }

    # ---- MQTT -> Socket.IO relay --------------------------------------- #
    def on_detection(det) -> None:
        state["latest_image_ref"] = det.image_ref
        state["detections"] += 1
        socketio.emit("detection", det.model_dump(mode="json"))

    def on_telemetry(tel) -> None:
        state["telemetry"] = tel.model_dump(mode="json")
        socketio.emit("telemetry", state["telemetry"])

    def on_status(st) -> None:
        state["status"] = st.model_dump(mode="json")
        socketio.emit("status", state["status"])

    subscriber = MqttSubscriber.from_config(
        config, on_detection=on_detection, on_telemetry=on_telemetry, on_status=on_status
    )

    def _read_defects() -> list[dict]:
        db = Database(db_path).connect()
        try:
            return db.all_defects()
        finally:
            db.close()

    # ---- REST ----------------------------------------------------------- #
    @app.route("/")
    def index():
        return render_template("index.html", topics=mqtt_topics)

    @app.route("/api/defects")
    def api_defects():
        return jsonify(_read_defects())

    @app.route("/api/state")
    def api_state():
        return jsonify({
            "telemetry": state["telemetry"],
            "status": state["status"],
            "detections_relayed": state["detections"],
        })

    @app.route("/export.csv")
    def export_csv():
        db = Database(db_path).connect()
        try:
            out = db.export_csv(export_dir / "maintenance_log.csv")
        finally:
            db.close()
        return send_file(out, mimetype="text/csv", as_attachment=True,
                         download_name="maintenance_log.csv")

    @app.route("/export.json")
    def export_json():
        db = Database(db_path).connect()
        try:
            out = db.export_json(export_dir / "maintenance_log.json")
        finally:
            db.close()
        return send_file(out, mimetype="application/json", as_attachment=True,
                         download_name="maintenance_log.json")

    @app.route("/latest_frame.jpg")
    def latest_frame():
        ref = state["latest_image_ref"]
        if ref:
            path = resolve_path(ref)
            if path.is_file():
                return send_file(path, mimetype="image/jpeg")
        # Fallback: the most recently modified crop on disk.
        crops = sorted(crops_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if crops:
            return send_file(crops[0], mimetype="image/jpeg")
        abort(404, "no frame available yet")

    @app.route("/crops/<path:filename>")
    def serve_crop(filename: str):
        return send_from_directory(crops_dir, filename)

    # ---- Socket.IO ------------------------------------------------------ #
    @socketio.on("connect")
    def _on_connect():
        # Send the current snapshot to the freshly-connected client only.
        if state["status"]:
            emit("status", state["status"])
        if state["telemetry"]:
            emit("telemetry", state["telemetry"])

    return app, socketio, subscriber


def run_dashboard(config: Optional[dict] = None) -> None:
    """Connect the MQTT relay and run the dashboard server (blocking)."""
    config = config or load_config()
    app, socketio, subscriber = create_app(config)

    if subscriber.connect():
        print(f"[dashboard] MQTT relay connected on {subscriber.active_broker}")
    else:
        print("[dashboard] WARNING: no MQTT broker reachable -- live feed idle, but the "
              "REST endpoints (/api/defects, /export.csv) still work.")

    dash = config.get("dashboard", {}) or {}
    host, port = dash.get("host", "127.0.0.1"), int(dash.get("port", 5000))
    print(f"[dashboard] serving on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)
    finally:
        subscriber.stop()


if __name__ == "__main__":
    run_dashboard()
