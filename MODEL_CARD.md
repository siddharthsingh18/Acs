# ACS Model Card

## Status and limitations

**Status: research prototype.**
Evaluated on 0 frozen test clips and 0.000 test camera-hours.
Accident-class mAP is currently unreliable unless the generated run artifacts show an evaluable confidence bound for every accident class.
Decision support only. A human must verify every alert; no automated emergency dispatch is permitted.

This card is generated from `run_artifacts.json`; missing metrics fail closed and cannot be interpreted as zero.

- Model: `/Users/shidharthkushwaha/Desktop/Acs-1/best.pt`
- Model SHA256: `5e1a582110d1f16ebc3b67c2a2348873af30a14c55297c06e5c377ce17a21a68`
- Data manifest SHA256: `35ae4d15c5d04831f71a8faa4f79d5ab5efbf63ce23685d567e83e7a5624ddd1`
- Class file SHA256: `de5ca1e40ec5cbd44c64ac7e52785d8ccd26ba4d8745772f5a52ea09cd0d0115`
- Run ID: `3eb11fd7-0014-4b02-ad04-c74db6908924`

## Promotion gates

| Gate | Result |
|---|---|
| promotion thresholds approved | FAIL |
| evaluation metrics present | FAIL |
| no leakage failures | FAIL |
| no unverified claims | FAIL |
| accident macro map50 lcb | FAIL |
| every accident class evaluable | FAIL |
| incident recall lcb | FAIL |
| false alarm upper bound | FAIL |
| minimum test events | FAIL |
| minimum test camera scene diversity | FAIL |
| minimum accident free camera hours | FAIL |
| robustness limits | FAIL |

## Evaluation evidence

No quantitative evaluation metrics are available for this run.

## Minimum evidence requirements

- Recall lower bound target 0.80: at least 17/17 detected events are needed for an exact two-sided 95% Clopper-Pearson lower bound to meet the target. At the 40-event minimum, at least 38/40 must be detected.
- False-alarm upper bound target 0.25/camera-hour: zero false alarms require at least 12 camera-hours by the exact one-sided 95% Poisson bound (2.996/T); the 20-hour minimum remains mandatory.

Generated: 2026-10-23T23:33:54.749346+00:00
