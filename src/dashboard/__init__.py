"""Dashboard: Flask + Socket.IO backend that relays MQTT to the browser and
serves the styled operator UI, the maintenance log, exports, and the latest
frame. See app.py."""

from src.dashboard.app import create_app, run_dashboard

__all__ = ["create_app", "run_dashboard"]
