import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';

// Custom Prometheus/k6 performance metrics
export const apiLatency = new Trend('acs_api_latency_ms');
export const wsConnectionTime = new Trend('acs_ws_connect_time_ms');
export const errorRate = new Rate('acs_error_rate');
export const wsMessagesReceived = new Counter('acs_ws_messages_total');

export const options = {
  stages: [
    { duration: '15s', target: 20 },  // Ramp-up to 20 operators
    { duration: '30s', target: 50 },  // Sustained load: 50 operators
    { duration: '15s', target: 100 }, // Peak burst: 100 concurrent users
    { duration: '10s', target: 0 },   // Graceful cool-down
  ],
  thresholds: {
    // 95% of API requests must complete under 150ms
    http_req_duration: ['p(95)<150', 'p(99)<300'],
    // HTTP failure rate must stay below 0.5%
    acs_error_rate: ['rate<0.005'],
    // WebSocket connection handshake under 250ms
    acs_ws_connect_time_ms: ['p(95)<250'],
  },
};

const BASE_URL = __ENV.TARGET_URL || 'http://localhost:8000';
const WS_URL = BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');

export default function () {
  const headers = {
    'Accept': 'application/json',
    'X-Forwarded-For': `192.168.1.${__VU % 250 + 1}`,
  };

  // 1. GET Dashboard Stats
  let resStats = http.get(`${BASE_URL}/api/stats`, { headers });
  let okStats = check(resStats, {
    'stats status is 200': (r) => r.status === 200,
  });
  errorRate.add(!okStats);
  apiLatency.add(resStats.timings.duration);

  sleep(0.5);

  // 2. GET Incidents List
  let resIncidents = http.get(`${BASE_URL}/api/incidents?page=1&page_size=20`, { headers });
  let okInc = check(resIncidents, {
    'incidents status is 200': (r) => r.status === 200,
    'incidents returns list': (r) => r.json('items') !== undefined,
  });
  errorRate.add(!okInc);
  apiLatency.add(resIncidents.timings.duration);

  sleep(0.5);

  // 3. GET Active Cameras
  let resCams = http.get(`${BASE_URL}/api/cameras`, { headers });
  check(resCams, { 'cameras status is 200': (r) => r.status === 200 });

  sleep(0.5);

  // 4. WebSocket Fan-Out Test (Every 5th virtual user opens WS telemetry stream)
  if (__VU % 5 === 0) {
    const wsStartTime = Date.now();
    const wsUrl = `${WS_URL}/api/incidents/ws`;

    ws.connect(wsUrl, {}, function (socket) {
      socket.on('open', function () {
        wsConnectionTime.add(Date.now() - wsStartTime);
        socket.setTimeout(function () {
          socket.close();
        }, 2000);
      });

      socket.on('message', function () {
        wsMessagesReceived.add(1);
      });

      socket.on('error', function (e) {
        errorRate.add(1);
      });
    });
  }

  sleep(1.0);
}
