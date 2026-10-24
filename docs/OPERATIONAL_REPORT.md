# Operational acceptance report

**Status: BLOCKED — no operational test results are claimed.**

## Requested scenario

The provisional scenario is 4 simulated cameras at 1280×720, based on the earlier project setup. Exact camera count, frame rate, hardware model, storage capacity, and alert-provider test credentials remain unconfirmed. No 24-hour soak or failure-injection run has been performed.

## Acceptance criteria and results

| Scenario | Required evidence | Result |
|---|---|---|
| 24-hour soak | Memory, file handles, dropped frames, worker restarts per camera | NOT RUN |
| Worker termination | Camera worker recovers; health reflects degraded/recovered state | NOT RUN |
| Database outage | Health reports database unavailable; queued/reviewed alerts recover exactly once | NOT RUN |
| VLM timeout | Operator receives a safe unverified state; no emergency-service call | NOT RUN |
| Disk full | Graceful degradation with explicit health/error state | NOT RUN |
| Alert-provider failure | Retry/failure surfaced; no silent success or duplicate delivery | NOT RUN |
| Operator concurrency | M operators review while cameras stream without lost or duplicated decisions | NOT RUN |
| Mid-incident restart | Incident is recovered exactly once after restart | NOT RUN |

No result may be marked PASS until the scenario, hardware, numeric acceptance thresholds, run logs, and relevant test artifacts are recorded. Decision support remains operator-only; an alert is not an emergency dispatch.
