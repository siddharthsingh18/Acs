# Operational Robustness Benchmark & Stress Test Report

**Date Generated**: 2026-10-20 20:11:27 UTC  
**Base Model Checkpoint**: `best.pt`  
**Corruptions Evaluated**: 11 environmental & streaming degradations (Severities 1 to 3)  
**Clean Baseline Performance**: Recall = `100.0%`, False Alarms = `0.0 FA/hr`  

---

## 1. Executive Summary & Worst Three Corruptions

The model was stressed across 11 corruptions representing CCTV hardware and weather faults.
Based on composite degradation (incident recall drop + false alarm rate surge), the **top 3 failure modes** are:

1. **`night_darkening`** — Avg Recall Drop: `33.33%`, Max Recall Drop: `100.0%`, Avg FA/hr Surge: `+201.68 FA/hr` (Degradation Score: `153.92`)
2. **`gaussian_noise`** — Avg Recall Drop: `100.0%`, Max Recall Drop: `100.0%`, Avg FA/hr Surge: `+0.0 FA/hr` (Degradation Score: `150.0`)
3. **`rain_streaks`** — Avg Recall Drop: `0.0%`, Max Recall Drop: `0.0%`, Avg FA/hr Surge: `+403.36 FA/hr` (Degradation Score: `141.18`)

---

## 2. Comprehensive Robustness Breakdown (All 11 Corruptions)

| Corruption | Severity | Recall (%) | Recall Drop (%p) | False Alarms (Total) | FA / Hour | Rise in FA/hr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `night_darkening` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `night_darkening` | Sev 2 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `night_darkening` | Sev 3 | 0.0% | -100.0%p | 1 | 605.0 | +605.0 |
| `gaussian_noise` | Sev 1 | 0.0% | -100.0%p | 0 | 0.0 | +0.0 |
| `gaussian_noise` | Sev 2 | 0.0% | -100.0%p | 0 | 0.0 | +0.0 |
| `gaussian_noise` | Sev 3 | 0.0% | -100.0%p | 0 | 0.0 | +0.0 |
| `rain_streaks` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `rain_streaks` | Sev 2 | 100.0% | -0.0%p | 1 | 605.0 | +605.0 |
| `rain_streaks` | Sev 3 | 100.0% | -0.0%p | 1 | 605.0 | +605.0 |
| `motion_blur` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `motion_blur` | Sev 2 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `motion_blur` | Sev 3 | 0.0% | -100.0%p | 0 | 0.0 | +0.0 |
| `glare` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `glare` | Sev 2 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `glare` | Sev 3 | 0.0% | -100.0%p | 0 | 0.0 | +0.0 |
| `defocus_blur` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `defocus_blur` | Sev 2 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `defocus_blur` | Sev 3 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `fog` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `fog` | Sev 2 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `fog` | Sev 3 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `low_bitrate` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `low_bitrate` | Sev 2 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `low_bitrate` | Sev 3 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `resolution_drop` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `resolution_drop` | Sev 2 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `resolution_drop` | Sev 3 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `camera_shake` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `camera_shake` | Sev 2 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `camera_shake` | Sev 3 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `lens_dirt` | Sev 1 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `lens_dirt` | Sev 2 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |
| `lens_dirt` | Sev 3 | 100.0% | -0.0%p | 0 | 0.0 | +0.0 |

---

## 3. Targeted Data Augmentation Retraining (Before vs After)

**Retrained Model Checkpoint**: `/Users/shidharthkushwaha/Desktop/Acs/models/robust_model.pt`  
**Augmentation Transforms Added**: `albumentations` matching the worst failure modes: 
(`RandomRain`, `RandomFog`, `ImageCompression`, `MotionBlur`, `RandomBrightnessContrast`, `Cutout`)  

| Metric | Before Retraining | After Retraining | Delta |
| :--- | :---: | :---: | :---: |
| Clean Incident Recall | 100.0% | 0.0% | -100.0%p |
| Clean False Alarms/hr | 0.0 | 0.0 | +0.0 |

### Worst 3 Corruptions: Before vs After

| Corruption | Severity | Before Recall | After Recall | Recall Gain | Before FA/hr | After FA/hr | FA/hr Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `night_darkening` | Sev 1 | 100.0% | 0.0% | **-100.0%p** | 0.0 | 0.0 | **+0.0** |
| `night_darkening` | Sev 2 | 100.0% | 0.0% | **-100.0%p** | 0.0 | 0.0 | **+0.0** |
| `night_darkening` | Sev 3 | 0.0% | 0.0% | **+0.0%p** | 605.0 | 0.0 | **-605.0** |
| `gaussian_noise` | Sev 1 | 0.0% | 0.0% | **+0.0%p** | 0.0 | 0.0 | **+0.0** |
| `gaussian_noise` | Sev 2 | 0.0% | 0.0% | **+0.0%p** | 0.0 | 0.0 | **+0.0** |
| `gaussian_noise` | Sev 3 | 0.0% | 0.0% | **+0.0%p** | 0.0 | 0.0 | **+0.0** |
| `rain_streaks` | Sev 1 | 100.0% | 100.0% | **+0.0%p** | 0.0 | 0.0 | **+0.0** |
| `rain_streaks` | Sev 2 | 100.0% | 100.0% | **+0.0%p** | 605.0 | 0.0 | **-605.0** |
| `rain_streaks` | Sev 3 | 100.0% | 0.0% | **-100.0%p** | 605.0 | 0.0 | **-605.0** |

---

## 4. Scope and Limitations (Untested Conditions Disclosure)

> [!IMPORTANT]
> **Strict Boundary of Robustness Claims:**
> In compliance with rigorous ML deployment standards, **robustness is NOT claimed for unbenchmarked conditions**.
> Robustness metrics documented above apply strictly and exclusively to the 11 explicitly tested failure modes at severities 1 to 3:
> 1. Gaussian noise (sigma 15-50)
> 2. Motion blur (kernel 9-31px)
> 3. Defocus blur (radius 3-11px)
> 4. Fog (atmospheric transmission factor 0.12-0.40)
> 5. Rain streaks (150-850 slanted streaks/frame)
> 6. Night darkening (gamma 1.35-2.40, scale 0.65-0.20)
> 7. Glare (1-3 specular sources with radial bloom)
> 8. Low bitrate re-encoding (H.264 at 500 kbps, 350 kbps, and 200 kbps)
> 9. Resolution drops (540p, 480p, 360p)
> 10. Camera shake (jitter up to 22px translation and 3.0° rotation)
> 11. Lens dirt overlay (semi-opaque smudges and droplets covering 5-25% lens area)
> 
> **Untested Operational Domains (Explicit Non-Claims):**
> - Blizzard / heavy snow accumulation & icing on housing
> - Direct floodwater specular mirror reflections
> - Structure fires / chemical smoke plume occlusion
> - Extreme fisheye optical barrel warping (>180° FOV)
> - Complete sensor blockage / vandalism spray-paint covering 100% of FOV
> - High-speed rolling shutter wobble artifacts from drone / UAV platforms
> - Thermal / Infrared (IR) sensor spectrum shift
> 
> Deployments operating in untested domains must execute independent field validation before activation.
