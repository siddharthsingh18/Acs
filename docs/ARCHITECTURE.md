# Automated CCTV Accident Detection System (ACS)
# System Architecture Specification

This document details the architectural topology, data pipelines, state machines, and observability fabric of the **Automated CCTV Accident Detection System (ACS)**.

---

## 1. End-to-End Real-Time Detection & Verification Pipeline

```mermaid
flowchart TD
    subgraph Ingestion ["1. Video & CCTV Ingestion"]
        A1["Overhead CCTV RTSP Stream"] --> B1["OpenCV Stream Capture"]
        A2["Custom MP4 Upload / Job"] --> B1
        B1 --> C1["Ring Buffer (10s Pre-Roll)"]
    end

    subgraph Perception ["2. Real-Time Spatial Perception"]
        B1 --> D1["YOLOv11 Spatial Detector\n(640x640 @ 44+ FPS CPU)"]
        D1 --> E1["ByteTrack Multi-Object Tracker\n(Trajectories & Velocity Vectors)"]
        E1 --> F1{"Consecutive Confident\nFrames >= N ?\n(N=5, Conf >= 0.55)"}
    end

    subgraph StateMachine ["3. Temporal Incident State Machine"]
        F1 -- "No" --> G1["Normal Traffic Log / Background"]
        F1 -- "Yes" --> H1["Open Incident Session\n(Start Timestamp, Cooldown Timer)"]
        H1 --> I1["Extract Keyframe &\nBuffered Video Clip"]
        I1 --> J1["Multi-Factor Severity Scoring\n(Velocity Delta, Entities, Injury)"]
    end

    subgraph Verification ["4. Automated Two-Stage Verification"]
        J1 --> K1{"Physics & Motion Pre-Check\n(IoU >= 0.02 & Decel >= 15 px/f)"}
        K1 -- "Fails Physics" --> L1["Suppress (False Alarm Artifact)"]
        K1 -- "Passes Physics" --> M1["Google Gemini VLM Inference\n(Multi-Frame Chronological Reasoning)"]
        M1 --> N1{"VLM Decision Matrix"}
        N1 -- "YOLO + VLM Agree" --> O1["Status: auto_confirmed"]
        N1 -- "Disagreement" --> P1["Status: needs_review"]
        N1 -- "API Offline / Budget" --> Q1["Status: unverified"]
    end

    subgraph Dispatch ["5. Dispatch & Human-in-the-Loop Triage"]
        O1 --> R1["Review Queue Console\n(High Priority)"]
        P1 --> R1
        Q1 --> R1
        R1 --> S1{"Operator Action"}
        S1 -- "Approve" --> T1["Emergency Dispatch Channels\n(Telegram, SMTP, Web Push)"]
        S1 -- "Reject" --> U1["Mark False Alarm / Audit Log"]
        T1 --> V1["Escalation Timer (5 min)\n(Escalate to Secondary if unack'd)"]
    end
```

---

## 2. Automated Tri-State Verification Decision Matrix

The verification layer uses a multi-factor decision matrix to balance false alarm suppression with crash recall:

```mermaid
stateDiagram-v2
    [*] --> YOLO_Detection: Collision Confidence >= 0.55 (N >= 5 frames)
    
    YOLO_Detection --> Physics_Check: Extract 3 Chronological Keyframes
    
    Physics_Check --> Suppressed: Overlap < 0.02 AND Decel < 15 px/f (Zero Kinematic Signal)
    Physics_Check --> Gemini_VLM: Passes Spatial-Kinematic Pre-Check
    
    state Gemini_VLM {
        [*] --> Multimodal_Query: Structured Prompt (3 Frames)
        Multimodal_Query --> Parse_JSON: Schema {is_accident, confidence, reason}
    }
    
    Gemini_VLM --> Auto_Confirmed: Gemini: true AND Confidence >= 0.50
    Gemini_VLM --> Needs_Review: Gemini: false OR Low Confidence
    Gemini_VLM --> Unverified: API Error / Quota Limit Fallback
    
    Auto_Confirmed --> Emergency_Alert: Direct Push & High-Priority Queue
    Needs_Review --> Review_Queue: Withhold External Alerts, Require Operator Triage
    Unverified --> Review_Queue: Flag for Immediate Operator Inspection
    Suppressed --> Audit_Log: Record Suppressed Event (0.00 FA/hr)
```

---

## 3. Incident Lifecycle & Alert Escalation State Machine

Every incident transitions through an append-only, auditable lifecycle:

```mermaid
stateDiagram-v2
    [*] --> Pending: Incident Created
    
    Pending --> Approved: Operator Clicks 'Approve'
    Pending --> Rejected: Operator Clicks 'False Alarm'
    Pending --> Escalated: Unacknowledged after 5 Minutes
    
    Escalated --> Approved: Secondary Dispatcher Acknowledges
    Escalated --> Rejected: Operator Dismisses
    
    Approved --> Dispatched: Alert Payloads Sent (Telegram, SMTP, Push)
    Dispatched --> Resolved: On-Scene Emergency Clear
    Rejected --> Closed: Logged to Audit Trail
    Resolved --> Closed: Archived per Retention Policy
```

---

## 4. Observability & Telemetry Architecture

ACS instruments operational metrics, distributed logging, error capture, and model drift:

```mermaid
flowchart LR
    subgraph AppServer ["ACS FastAPI Application"]
        Core["Core Camera Pipeline\n& Inference Loops"]
        Mtr["Prometheus Metrics Collector\n(fps, latencies, queues, alerts)"]
        Log["Structured JSON Logger\n(with Request ID & User Context)"]
        Drift["Model Drift Engine\n(Weekly Conf & Detection Distributions)"]
        Sentry["Sentry Error SDK\n(PII & Secret Scrubbed)"]
    end

    subgraph ObservabilityStack ["Enterprise Monitoring Stack"]
        Prom["Prometheus Server\n(Scrapes GET /metrics every 15s)"]
        Graf["Grafana Dashboard\n(Real-Time Surveillance Telemetry)"]
        SentrySrv["Sentry Cloud / Self-Hosted\n(Stack Traces & Alerting)"]
        AuditDB["SQLite / PostgreSQL Audit Log\n(Append-Only Compliance)"]
    end

    Core --> Mtr
    Core --> Log
    Core --> Drift
    Core --> Sentry

    Mtr --> Prom
    Prom --> Graf
    Log --> AuditDB
    Sentry --> SentrySrv
```

---

## 5. Distributed Edge-to-Cloud Hybrid Topology

For massive CCTV deployments, ACS supports distributed edge inference with centralized control room coordination:

```mermaid
flowchart TD
    subgraph EdgeNode1 ["Edge Node A (NVIDIA Jetson / On-Site NVR)"]
        Cam1["CCTV Cameras 1-4"] --> RTSP1["Local RTSP Engine"]
        RTSP1 --> YoloEdge1["YOLOv11 TensorRT (FP16)"]
        YoloEdge1 --> Filter1["Local Debounce Filter"]
        Filter1 -- "HTTP POST /api/edge/incidents\n(X-Edge-Api-Key)" --> Gateway["Central ACS API Gateway"]
    end

    subgraph EdgeNode2 ["Edge Node B (Intel NUC / x86 On-Premise)"]
        Cam2["CCTV Cameras 5-8"] --> RTSP2["Local RTSP Engine"]
        RTSP2 --> YoloEdge2["YOLOv11 OpenVINO (INT8)"]
        YoloEdge2 --> Filter2["Local Debounce Filter"]
        Filter2 -- "HTTP POST /api/edge/incidents\n(X-Edge-Api-Key)" --> Gateway
    end

    subgraph CloudControlRoom ["Central Surveillance Hub (Cloud / Private VPS)"]
        Gateway --> FastAPICentral["Central ACS FastAPI Services"]
        FastAPICentral --> CentralDB["Central SQLite Database"]
        FastAPICentral --> VLM["Google Gemini VLM Cloud"]
        FastAPICentral --> WebConsole["React 19 Operator Console\n(Live Camera Wall, Review Queue, Map)"]
        FastAPICentral --> AlertHub["Emergency Dispatch Gateway\n(Telegram, SMTP, Web Push)"]
    end
```
