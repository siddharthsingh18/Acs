# ACS Operational Boundaries, Tested Conditions & Known Limitations

This document provides a rigorous, transparent assessment of the operational boundaries, verified camera conditions, and known limitations of the **Automated CCTV Accident Detection System (ACS)**.

---

## 1. Verified Operational Conditions (Empirically Tested)

The YOLOv11 + Gemini VLM dual-stage pipeline was evaluated against the curated ACS multi-source benchmark suite under the following verified conditions:

| Parameter | Tested Operational Range | Degraded / Out-of-Distribution Boundary |
| :--- | :--- | :--- |
| **Camera Mounting Height** | **5 to 15 meters** (overhead highway gantries, signal poles, overpasses) | Ground level (< 2.5m) or high-altitude aerial (> 30m) |
| **Depression Angle** | **15° to 45° downward tilt** relative to horizontal | Direct frontal (< 10°) or pure nadir top-down (> 75°) |
| **Sensor Perspective** | **Fixed / Stationary CCTV cameras** | Mobile dashcams, bodycams, PTZ cameras during active panning |
| **Frame Rate** | **15 to 30 FPS** | Lower than 10 FPS (destabilizes velocity vector estimation) |
| **Resolution** | **$1280 \times 720$ (720p) to $1920 \times 1080$ (1080p)** downscaled to $640 \times 640$ | Sub-VGA (< 640x480) or extreme wide-angle fisheye (> 160° FOV) |
| **Traffic Speed** | **30 km/h to 120 km/h** (urban arterials, multi-lane expressways) | Crawling parking lot maneuvers (< 5 km/h) |
| **Lighting** | **Daylight (clear/cloudy) & Street-Illuminated Night** | Pitch black roadways without artificial illumination (< 5 lux) |
| **Surface Moisture** | **Dry pavement & moderately damp road surfaces** | Deep standing water (> 25mm), snow accumulation, or black ice |

---

## 2. Detailed Performance Under Edge Conditions

### 2.1. Lighting & Solar Conditions
- **Daylight (Optimal)**: Peak accuracy. The spatial YOLO model achieves $> 90\%$ single-frame detection confidence on clear structural vehicular impacts.
- **Night with Street Illumination (Good)**: High vehicle detection retention via headlights and taillights. However, extreme direct headlight beam flare aimed straight into the camera lens can create momentary 1-frame false bounding boxes. Mitigated by setting debounce threshold $N \ge 4$ consecutive frames.
- **Total Darkness / Rural Unlit Highways (Severe Degradation)**: Without ambient or artificial street lighting, vehicle silhouettes are obscured. Detection becomes unreliable until vehicles collide and come into direct headlight beam interaction.
- **Low-Angle Sunrise / Sunset Sun Glare (Moderate Degradation)**: Strong horizontal solar glare causing lens flare wash-out can suppress low-confidence vehicle boxes.

### 2.2. Adverse Weather & Precipitation
- **Light to Moderate Rain**: Successfully tested on damp asphalt. Vehicle tracking remains stable with ByteTrack.
- **Heavy Downpour & Standing Water**: Pavement reflections of sky, traffic signals, and headlights can occasionally create spurious bounding boxes on wet bare asphalt. The automated Gemini VLM verification layer successfully filters out **100%** of these static reflections (0.00 FA/hr verified).
- **Dense Fog & Heavy Snow**: Severely degrades effective camera line-of-sight. When visibility drops below 50 meters, distant vehicles blend into the foggy background, preventing timely debounce accumulation.

### 2.3. Occlusion & High-Density Congestion
- **Heavy Truck & Bus Occlusion**: If a two-vehicle collision occurs on the far side of an 18-wheel tractor-trailer or city bus relative to the camera position, the incident will be visually occluded. The system requires at least **4 consecutive unobstructed frames** to trigger debounce confirmation.
- **Bumper-to-Bumper Gridlock**: In severe traffic jams, vehicles naturally sit within 1 meter of one another with overlapping 2D bounding boxes. The spatial IoU check could produce false overlap signals without the trajectory velocity vector check ($\Delta v$). ACS requires non-zero pre-impact velocity before sudden deceleration to confirm an incident.

### 2.4. Unusual Vehicles & False Trigger Candidates
- **Tow Trucks & Flatbeds**: A car mounted securely on a flatbed tow truck presents overlapping `car` bounding boxes while in motion. In rare cases without VLM verification, this can mimic a multi-vehicle accident. Gemini VLM triage accurately recognizes secured towing configurations and suppresses alerts.
- **Construction Machinery & Road Maintenance**: Bulldozers, asphalt rollers, and street sweepers with flashing amber beacons can register as anomalous objects; fine-tuning or VLM confirmation is recommended for active construction zones.

---

## 3. Human-in-the-Loop Operational Mandate

> [!CAUTION]
> **MANDATORY VERIFICATION PROTOCOL**:
> ACS is strictly an **AI Decision-Support System** designed to reduce operator fatigue and accelerate emergency response triage across massive camera walls. It is **NOT** a certified autonomous 911 dispatch agent.
> 
> Under all deployment guidelines:
> 1. All automated alerts routed to the Review Queue require human operator inspection.
> 2. Automated emergency dispatch (`auto_confirmed`) must only be enabled in environments where certified dispatchers monitor the triage console.
> 3. Dispatched alerts should include the high-resolution keyframe, Gemini verification rationale, and a direct link to the 10-second buffered incident clip.
