"""
src/messaging/publisher.py
==========================
Thin paho-mqtt (v2) publisher for the AGV contract topics, QoS 1:

    agv/detections   <- Detection
    agv/telemetry    <- Telemetry
    agv/status       <- Status   (retained heartbeat; also the Last-Will)

Broker host / port / credentials come from config (``mqtt:`` section), defaulting
to a local Mosquitto broker. If the primary broker is unreachable it falls back
to the public **HiveMQ** test broker (``broker.hivemq.com:1883``) so the demo
runs even without a local broker. Both default brokers are anonymous -- no
credentials. Credentials, if ever needed, come from ``.env`` (never hard-coded).
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Any, Optional

import paho.mqtt.client as mqtt

from src.schema import Detection, Status, SystemState, Telemetry

logger = logging.getLogger(__name__)

_DEFAULT_TOPICS = {
    "detections": "agv/detections",
    "telemetry": "agv/telemetry",
    "status": "agv/status",
}


class MqttPublisher:
    """Publishes contract messages to MQTT (QoS 1), with broker fallback."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 1883,
        topics: Optional[dict[str, str]] = None,
        client_id: str = "agv-publisher",
        qos: int = 1,
        keepalive: int = 60,
        username: Optional[str] = None,
        password: Optional[str] = None,
        fallback: Optional[dict[str, Any]] = None,
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
        self._client: Optional[mqtt.Client] = None
        self.active_broker: Optional[str] = None

    @classmethod
    def from_config(cls, config: dict) -> "MqttPublisher":
        m = config.get("mqtt", {}) or {}
        return cls(
            host=m.get("host", "localhost"),
            port=m.get("port", 1883),
            topics=m.get("topics"),
            client_id=f"{m.get('client_id', 'agv')}-pub",
            qos=m.get("qos", 1),
            keepalive=m.get("keepalive", 60),
            username=m.get("username"),
            password=m.get("password"),
            fallback=m.get("fallback"),
        )

    # ---- connection ----------------------------------------------------- #
    def _make_client(self) -> mqtt.Client:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{self.client_id}-{int(time.time())}",
        )
        if self.username:
            client.username_pw_set(self.username, self.password or "")
        # Last Will: if we drop unexpectedly, subscribers see an OFFLINE status.
        client.will_set(
            self.topics["status"],
            Status(state=SystemState.OFFLINE, detail="unexpected disconnect").to_json(),
            qos=self.qos,
            retain=True,
        )
        return client

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to the primary broker, else the configured fallback.

        Returns True if connected, False if no broker could be reached.
        """
        if self._connect_to(self.host, self.port, timeout):
            self.active_broker = f"{self.host}:{self.port}"
            return True

        fb_host, fb_port = self.fallback.get("host"), self.fallback.get("port")
        if fb_host:
            logger.warning("Primary broker %s:%s unreachable -> trying public fallback %s:%s "
                           "(HiveMQ test broker; demo data only).", self.host, self.port, fb_host, fb_port)
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

    @property
    def connected(self) -> bool:
        return bool(self._client and self._client.is_connected())

    # ---- publish -------------------------------------------------------- #
    def _publish(self, topic: str, payload: str, retain: bool = False) -> bool:
        if self._client is None:
            return False
        info = self._client.publish(topic, payload, qos=self.qos, retain=retain)
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def publish_detection(self, detection: Detection) -> bool:
        return self._publish(self.topics["detections"], detection.to_json())

    def publish_telemetry(self, telemetry: Telemetry) -> bool:
        return self._publish(self.topics["telemetry"], telemetry.to_json())

    def publish_status(self, status: Status) -> bool:
        return self._publish(self.topics["status"], status.to_json(), retain=True)

    # ---- teardown ------------------------------------------------------- #
    def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            self.publish_status(Status(state=SystemState.OFFLINE, detail="pipeline stopped"))
            time.sleep(0.1)  # let the QoS-1 message flush
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass
        self._client = None

    def __enter__(self) -> "MqttPublisher":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disconnect()
