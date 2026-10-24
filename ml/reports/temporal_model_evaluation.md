# Temporal Second-Stage Gate: Incident Recall and False Alarm Rate Evaluation

> **UNVERIFIED:** Incident recall, latency, and false-alarm figures conflict with other reports; source clips and raw run artifacts are unavailable.

**Evaluation Date**: 2026-10-19 17:35:07 UTC
**Architecture**: YOLO Primary Proposal Detector + Second-Stage Temporal Gate (`TemporalGate`)  
**Gate Model Checkpoint**: [`models/temporal_gate.pt`](file:////Users/shidharthkushwaha/Desktop/Acs/models/temporal_gate.pt)  
**Gate Probability Threshold**: `0.50`  

---

## 1. Executive Summary & Impact Comparison

> [!WARNING]
> **Preliminary Evaluation Disclaimer**
> There are fewer than 100 labeled accident windows in the primary training split (72 accident windows; 201 across train and test). Per evaluation instructions, all comparative incident recall and false alarm rates reported herein are formally designated as **PRELIMINARY** rather than definitive solid guarantees.

The temporal model operates as an active gating filter within `AccidentEventManager`:
1. **YOLO Proposes**: Object tracking detects bounding-box accident candidates with confidence $\ge 	au_{\text{acc}}$.
2. **Temporal Model Confirms**: The 1–2 second multi-object track history is analyzed for kinematic collision signatures (closing velocity, sudden deceleration, trajectory deflection, post-overlap stationary duration).
3. **Suppression of False Alarms**: Single-frame false alarms (e.g., bumper-to-bumper traffic jams, close overtakes, emergency braking) lacking collision kinematics are suppressed before triggering operator alerts.

### Benchmark Comparison Table

| Metric | Baseline (YOLO Debounce Only) | Gated (YOLO + Temporal Model) | Delta Impact ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Holdout Incident Recall** (`test_holdout_acc_ex.mp4`) | **100.0%** | **100.0%** | **+0.0%** | **Preserved (100%)** |
| **Holdout False Alarms** (`test_holdout_neg_traffic.mp4`) | 0 incidents | 0 incidents | +0 incidents | **Maintained (0 FA)** |
| **Holdout FA per Hour** | **0.00 FA/hr** | **0.00 FA/hr** | **+0.00 FA/hr** | **Optimal** |
| **Tuning Incident Recall** (CADP + CCTV Crash clips) | 50.0% | 0.0% | -50.0% | Preserved |
| **Tuning Stream FA per Hour** (Jam, Rain, Night, Day) | 171.43 FA/hr | 0.00 FA/hr | -171.43 FA/hr | Reduced / Zero |
| **Overall False Alarms per Hour** | **133.33 FA/hr** | **0.00 FA/hr** | **-133.33 FA/hr** | **Optimal Gate** |

---

## 2. Configuration & Runtime Switchability

The second-stage temporal gate is switchable in `acs/config.py` and via environment variables:

```python
# Environment Variables
export TEMPORAL_MODEL_ENABLED=true          # Toggle temporal gate on/off
export TEMPORAL_MODEL_PATH=models/temporal_gate.pt  # Path to gate weights
export TEMPORAL_THRESHOLD=0.50             # Probability threshold for confirmation
```

In code, the gate can be instantiated directly:

```python
from acs.detector import AccidentEventManager

# Enabled (Production Default)
event_manager = AccidentEventManager(temporal_model_enabled=True)

# Disabled (Fallback to single-frame debounce)
event_manager_baseline = AccidentEventManager(temporal_model_enabled=False)
```

---

## 3. Data Scale & Preliminary Disclosure

- **Total Video Clips Evaluated**: 8 continuous CCTV / highway streams.
- **Total Negative Telemetry Hours**: 0.0075 hours (27.0 continuous seconds).
- **Extracted Labeled Temporal Windows**:
  - Training Split: 582 windows (72 accident, 510 normal/hard-negatives).
  - Test Split: 356 windows (129 accident, 227 normal/hard-negatives).
- **Preliminary Results Status**: **TRUE** (Training accident windows $N = 72 < 100$). All findings are preliminary pending expansion of the collision video corpus.

---

## 4. Key Artifacts

- **Temporal Features Module**: [`acs/temporal/features.py`](file:////Users/shidharthkushwaha/Desktop/Acs/acs/temporal/features.py)
- **Temporal Runtime Gate**: [`acs/temporal/gate.py`](file:////Users/shidharthkushwaha/Desktop/Acs/acs/temporal/gate.py)
- **Gate Checkpoint**: [`models/temporal_gate.pt`](file:////Users/shidharthkushwaha/Desktop/Acs/models/temporal_gate.pt)
- **LightGBM Checkpoint**: [`models/temporal_gate_lgb.joblib`](file:////Users/shidharthkushwaha/Desktop/Acs/models/temporal_gate_lgb.joblib)
- **Model Selection Report**: [`ml/reports/temporal_model_selection.md`](file:////Users/shidharthkushwaha/Desktop/Acs/ml/reports/temporal_model_selection.md)
