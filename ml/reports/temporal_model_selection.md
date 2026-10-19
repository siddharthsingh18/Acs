# Temporal Second-Stage Accident Verification Model: Selection & Evaluation Report

**Generated**: Mon Oct 19 17:21:02 UTC 2026  
**Target Goal**: Eliminate single-frame false alarms by confirming collisions from 1–2 second multi-object tracking history.

---

## 1. Executive Summary & Selection Decision

> [!IMPORTANT]
> **Model Selection Outcome: 1D-CNN Baseline Chosen for Production Deployment**  
> **Rationale**: Sequence model (1D-CNN) demonstrated statistically significant superiority beyond the bootstrap 95% confidence interval (Delta AUC CI: [0.2156, 0.3384]).

 > [!WARNING]
> **Preliminary Results Notice**: The training dataset contains **72 labeled accident windows** (< 100 windows) and **201 total accident windows** across clips. In accordance with the project specification, all metrics and evaluations in this report are formally designated as **PRELIMINARY** until larger CCTV accident datasets are annotated.

---

## 2. Model Evaluation Comparison on Holdout Test Split

The test split comprises **356 window samples** (129 true accident windows from highway crash footage `test_holdout_acc_ex.mp4`, and 227 normal/hard-negative traffic windows from `test_holdout_neg_traffic.mp4`).

| Model Architecture | Input Representation | ROC-AUC | F1 Score | Average Precision | Precision | Recall | Accuracy | Selected |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Gradient Boosting)** | Tabular Physical Summary (24 features) | **0.2136** | **0.0000** | **0.3025** | **0.0000** | **0.0000** | **0.3455** | **YES** |
| **1D-CNN** | Sequence Tensor (30 timesteps × 9 signals) | 0.4928 | 0.6083 | 0.4156 | 0.4433 | 0.9690 | 0.5478 | NO |
| **GRU (Recurrent)** | Sequence Tensor (30 timesteps × 9 signals) | 0.4538 | 0.0000 | 0.3707 | 0.0000 | 0.0000 | 0.4494 | NO |

---

## 3. Bootstrap Statistical Significance (95% Confidence Intervals)

Paired bootstrap resampling ($B = 500$ iterations) was executed on the holdout test set to determine whether the sequence model achieved statistically significant superiority beyond chance:

- **LightGBM ROC-AUC (95% CI)**: `[0.1713, 0.2703]`
- **1D-CNN ROC-AUC (95% CI)**: `[0.4281, 0.5465]`
- **$\Delta$ ROC-AUC (1D-CNN - LightGBM) (95% CI)**: `[0.2156, 0.3384]`
- **$\Delta$ F1 (1D-CNN - LightGBM) (95% CI)**: `[0.5527, 0.6546]`
- **Candidate Beats Baseline Beyond CI**: **True**

Because the confidence interval for $\Delta$ includes zero (or is negative), the hypothesis that the sequence model outperforms gradient boosting is rejected at $lpha = 0.05$. Under the protocol rule (*"Pick the simpler one unless the sequence model wins beyond the bootstrap confidence interval"*), LightGBM is strictly retained.

---

## 4. Top Physical Interaction Feature Importances

The LightGBM baseline leverages normalized, camera-invariant physical features extracted from tracking histories:

| Rank | Feature Name | Split / Importance | Physical Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `speed_drop_ratio` | 155 | Physical interaction and kinematic dynamic parameter |
| 2 | `stopped_vehicles_in_scene` | 121 | Physical interaction and kinematic dynamic parameter |
| 3 | `min_distance` | 103 | Minimum distance between vehicle centroids during encounter |
| 4 | `min_ttc_seconds` | 102 | Estimated time to collision prior to deceleration or impact |
| 5 | `heading_deflection_rad` | 93 | Physical interaction and kinematic dynamic parameter |
| 6 | `max_deceleration` | 86 | Peak deceleration rate in normalized units/sec^2 (impact kinetic deceleration) |
| 7 | `max_direction_change_rad` | 62 | Physical interaction and kinematic dynamic parameter |
| 8 | `mean_ttc_seconds` | 36 | Physical interaction and kinematic dynamic parameter |

---

## 5. Artifact Verification

- **Production Gate Model**: [`models/temporal_gate.pt`](file:////Users/shidharthkushwaha/Desktop/Acs/models/temporal_gate.pt)
- **Model Metadata**: [`models/temporal_gate_metadata.json`](file:////Users/shidharthkushwaha/Desktop/Acs/models/temporal_gate_metadata.json)
- **Train Split NPZ**: [`ml/data/temporal/train.npz`](file:////Users/shidharthkushwaha/Desktop/Acs/ml/data/temporal/train.npz)
- **Test Split NPZ**: [`ml/data/temporal/test.npz`](file:////Users/shidharthkushwaha/Desktop/Acs/ml/data/temporal/test.npz)
