"""
src/messaging/subscriber.py
===========================
Thin paho-mqtt (v2) subscriber for the AGV contract topics, QoS 1.

Subscribes to agv/detections, agv/telemetry, agv/status and dispatches each
message to a typed callback (parsed + validated against the schema). The
dashboard (next step) uses this to drive its live feed. Run it directly as a
quick CLI alternative to ``mosquitto_sub``::

    python -m src.messaging.subscriber

Broker selection (primary + HiveMQ fallback) mirrors the publisher.
"""

from __future__ import annotations

import socket
import sys
import time
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt

from src.schema import Detection, Status, Telemetry

_DEFAULT_TOPICS = {
    "detections": "agv/detections",
    "telemetry": "agv/telemetry",
    "status": "agv/status",
}


class MqttSubscriber:
    """Subscribes to the contract topics and dispatches typed messages."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 1883,
        topics: Optional[dict[str, str]] = None,
        client_id: str = "agv-subscriber",
        qos: int = 1,
        keepalive: int = 60,
        username: Optional[str] = None,
        password: Optional[str] = None,
        fallback: Optional[dict[str, Any]] = None,
        on_detection: Optional[Callable[[Detection], None]] = None,
        on_telemetry: Optional[Callable[[Telemetry], None]] = None,
        on_status: Optional[Callable[[Status], None]] = None,
        on_message: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.topics = {**_DEFAULT_TOPICS, **(topics or {})}
        self.client_id = client_id
        self.qos = int(qos)
        self.keepalive = int(keepalive)
        self.username = username
        self.password = password
        self.fallback = fallback or {}
        self.on_detection = on_detection
        self.on_telemetry = on_telemetry
        self.on_status = on_status
        self.on_message = on_message
        self._client: Optional[mqtt.Client] = None
        self.active_broker: Optional[str] = None

    @classmethod
    def from_config(cls, config: dict, **callbacks: Any) -> "MqttSubscriber":
        m = config.get("mqtt", {}) or {}
        return cls(
            host=m.get("host", "localhost"),
            port=m.get("port", 1883),
            topics=m.get("topics"),
            client_id=f"{m.get('client_id', 'agv')}-sub",
            qos=m.get("qos", 1),
            keepalive=m.get("keepalive", 60),
            username=m.get("username"),
            password=m.get("password"),
            fallback=m.get("fallback"),
            **callbacks,
        )

    # ---- callbacks ------------------------------------------------------ #
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        for topic in self.topics.values():
            client.subscribe(topic, qos=self.qos)

    def _on_message(self, client, userdata, message):
        payload = message.payload.decode("utf-8", errors="replace")
        topic = message.topic
        try:
            if topic == self.topics["detections"] and self.on_detection:
                self.on_detection(Detection.from_json(payload))
            elif topic == self.topics["telemetry"] and self.on_telemetry:
                self.on_telemetry(Telemetry.from_json(payload))
            elif topic == self.topics["status"] and self.on_status:
                self.on_status(Status.from_json(payload))
        except Exception as exc:  # malformed/off-contract message -> don't crash
            print(f"[MQTT] dropped malformed message on {topic}: {exc!r}", file=sys.stderr)
        if self.on_message:
            self.on_message(topic, payload)

    # ---- connection ----------------------------------------------------- #
    def _make_client(self) -> mqtt.Client:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{self.client_id}-{int(time.time())}",
        )
        if self.username:
            client.username_pw_set(self.username, self.password or "")
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        return client

    def connect(self, timeout: float = 10.0) -> bool:
        if self._connect_to(self.host, self.port, timeout):
            self.active_broker = f"{self.host}:{self.port}"
            return True
        fb_host, fb_port = self.fallback.get("host"), self.fallback.get("port")
        if fb_host:
            print(f"[MQTT] primary broker {self.host}:{self.port} unreachable -> "
                  f"trying public fallback {fb_host}:{fb_port}.")
            if self._connect_to(fb_host, int(fb_port or 1883), timeout):
                self.active_broker = f"{fb_host}:{fb_port}"
                return True
        return False

    def _connect_to(self, host: str, port: int, timeout: float) -> bool:
        client = self._make_client()
        try:
            client.connect(host, port, self.keepalive)
            client.loop_start()
        except (OSError, socket.error):
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if client.is_connected():
                self._client = client
                return True
            time.sleep(0.05)
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass
        return False

    def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> "MqttSubscriber":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def _main() -> None:
    """CLI: print every agv/* message (a quick mosquitto_sub stand-in)."""
    from src.config import load_config

    config = load_config()

    def show(topic: str, payload: str) -> None:
        print(f"[{topic}] {payload}")

    sub = MqttSubscriber.from_config(config, on_message=show)
    if not sub.connect():
        print("Could not connect to any MQTT broker (start Mosquitto or check the network).")
        sys.exit(1)
    print(f"Subscribed to agv/# on {sub.active_broker} -- Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping subscriber.")
        sub.stop()


if __name__ == "__main__":
    _main()
