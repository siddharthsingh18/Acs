"""
tests.load.locustfile
Enterprise load testing suite for ACS API and WebSocket fan-out using Locust.

Simulates:
1. Control Room Operators:
   - Polling incident list and filtering by status
   - Querying live traffic analytics dashboard
   - Inspecting detailed incident dossiers
   - Triaging review queue decisions
2. WebSocket Fan-Out:
   - Multiple concurrent operator browsers subscribing to live camera & incident WebSockets
   - Validating message delivery without server event loop starvation
3. Edge Device Telemetry:
   - Edge boxes pushing heartbeats and incident events under burst conditions

Usage:
  locust -f tests/load/locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10 --run-time 1m --headless
"""
import json
import logging
import random
import time
from locust import HttpUser, TaskSet, between, task
import websocket

logger = logging.getLogger("acs.loadtest")


class OperatorUser(HttpUser):
    """Simulates municipal traffic control room operators interacting with dashboard."""
    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Authenticates or initializes operator session."""
        self.auth_headers = {
            "Accept": "application/json",
            "X-Forwarded-For": f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
        }
        # Attempt to obtain JWT access token
        try:
            resp = self.client.post(
                "/api/auth/login",
                json={"username": "operator", "password": "operator_password"},
                headers=self.auth_headers,
                name="[Auth] Operator Login",
            )
            if resp.status_code == 200:
                token = resp.json().get("access_token")
                if token:
                    self.auth_headers["Authorization"] = f"Bearer {token}"
        except Exception:
            pass

    @task(5)
    def get_dashboard_stats(self):
        """Polls traffic and detection analytics."""
        self.client.get(
            "/api/stats",
            headers=self.auth_headers,
            name="[API] GET /api/stats",
        )

    @task(6)
    def get_incidents_list(self):
        """Queries paginated incident list."""
        page = random.randint(1, 3)
        self.client.get(
            f"/api/incidents?page={page}&page_size=15",
            headers=self.auth_headers,
            name="[API] GET /api/incidents (list)",
        )

    @task(3)
    def get_review_queue(self):
        """Fetches pending operator review queue."""
        self.client.get(
            "/api/review?limit=10",
            headers=self.auth_headers,
            name="[API] GET /api/review",
        )

    @task(2)
    def get_cameras_list(self):
        """Fetches camera register and status."""
        self.client.get(
            "/api/cameras",
            headers=self.auth_headers,
            name="[API] GET /api/cameras",
        )

    @task(1)
    def get_health(self):
        """Liveness check."""
        self.client.get("/health", headers=self.auth_headers, name="[Health] GET /health")


class EdgeDeviceUser(HttpUser):
    """Simulates edge computing nodes transmitting telemetry and incident events."""
    wait_time = between(1.0, 3.0)

    def on_start(self):
        self.node_id = f"edge-node-{random.randint(100, 999)}"
        self.edge_headers = {
            "X-Edge-API-Key": "dev_edge_secret_key",
            "Accept": "application/json",
        }

    @task(4)
    def send_heartbeat(self):
        """Sends periodic edge node telemetry."""
        payload = {
            "node_id": self.node_id,
            "cpu_percent": random.uniform(5.0, 45.0),
            "ram_percent": random.uniform(15.0, 60.0),
            "active_cameras": random.randint(2, 8),
            "timestamp": time.time(),
        }
        self.client.post(
            "/api/edge/heartbeat",
            json=payload,
            headers=self.edge_headers,
            name="[Edge] POST /api/edge/heartbeat",
        )

    @task(1)
    def submit_edge_incident(self):
        """Submits detected incident event to central server."""
        inc_id = f"INC-LOAD-{self.node_id}-{int(time.time()*1000)}"
        meta = {
            "incident_id": inc_id,
            "camera_id": f"{self.node_id}-cam01",
            "edge_node_id": self.node_id,
            "class_name": random.choice(["car_car_accident", "car_bike_accident", "pedestrian_accident"]),
            "max_confidence": round(random.uniform(0.75, 0.98), 2),
            "severity": random.choice(["low", "medium", "high"]),
            "severity_score": random.randint(30, 90),
            "created_at": time.time(),
        }
        self.client.post(
            "/api/edge/incidents",
            data={"metadata": json.dumps(meta)},
            headers=self.edge_headers,
            name="[Edge] POST /api/edge/incidents",
        )


class WebSocketFanOutUser(HttpUser):
    """Simulates concurrent WebSocket clients receiving real-time fan-out broadcasts."""
    wait_time = between(3.0, 8.0)

    @task
    def test_websocket_connection(self):
        """Connects via WebSocket, measures connection handshake and message receipt."""
        ws_host = self.host.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_host}/api/incidents/ws"

        t0 = time.perf_counter()
        try:
            ws = websocket.create_connection(ws_url, timeout=3.0)
            connect_latency = (time.perf_counter() - t0) * 1000.0
            self.environment.events.request.fire(
                request_type="WebSocket",
                name="[WS] Connect /api/incidents/ws",
                response_time=connect_latency,
                response_length=0,
                exception=None,
                context={},
            )
            # Listen briefly for broadcast ping/message
            ws.settimeout(2.0)
            try:
                _ = ws.recv()
            except websocket.WebSocketTimeoutException:
                pass
            ws.close()
        except Exception as exc:
            connect_latency = (time.perf_counter() - t0) * 1000.0
            self.environment.events.request.fire(
                request_type="WebSocket",
                name="[WS] Connect /api/incidents/ws",
                response_time=connect_latency,
                response_length=0,
                exception=exc,
                context={},
            )
