# Current checkpoint diagnosis — PRELIMINARY / BLOCKED

**Training is prohibited until this diagnosis can be regenerated on a new leakage-safe test split and label review is complete.**

## Evidence available in this checkout

- The canonical detector taxonomy is four classes: `bike`, `car`, `person`, `accident` (`ml/configs/classes.yaml`).
- The checked-in YOLO checkpoint is a legacy 10-output model: `bike`, `bike_bike_accident`, `bike_object_accident`, `bike_person_accident`, `car`, `car_bike_accident`, `car_car_accident`, `car_object_accident`, `car_person_accident`, `person` (`ml/configs/checkpoint_class_mappings.yaml`).
- The checkpoint class order differs from the canonical class order. Existing inference has an explicit decoder, but class-index-based metrics from `model.val()` are invalid unless the checkpoint is canonical; the evaluator now refuses that silent mismatch.
- `ml/data/` contains no images, labels, `data.yaml`, provenance file, or frozen test manifest. No old split is reused.
- Therefore current per-class AP, predictions, and GT box counts cannot be computed. **Unavailable is not zero.**

## Current-weight evaluation on the new test split

| Class | AP@0.50 | GT boxes | Predictions | Verdict |
|---|---:|---:|---:|---|
| bike | NOT COMPUTABLE | NOT AVAILABLE | NOT AVAILABLE | No rebuilt test set |
| car | NOT COMPUTABLE | NOT AVAILABLE | NOT AVAILABLE | No rebuilt test set |
| person | NOT COMPUTABLE | NOT AVAILABLE | NOT AVAILABLE | No rebuilt test set |
| accident | NOT COMPUTABLE | NOT AVAILABLE | NOT AVAILABLE | No rebuilt test set |

The runnable evaluator is `ml/eval/diagnose.py`. It decodes model IDs explicitly into canonical IDs, counts all predictions above its documented confidence floor, and reports AP only when labels and test images exist.

## Accident label coverage

| Split / phase | Accident GT boxes | Finding |
|---|---:|---|
| Previous train / val / test | NOT AVAILABLE | No old split or dataset is present; historical source-wide counts are not training-split counts. |
| Rebuilt train / val / test | NOT AVAILABLE | No approved source archive or provenance metadata is available; the rebuild has not run. |

## Root causes and fixes

1. **Taxonomy/index mismatch is confirmed.** The legacy detector emits 10 IDs while the current data taxonomy is four classes. Runtime inference must decode through the explicit mapping; raw index-based test metrics can report the wrong class or no GT for accident labels. The mapping fixes interpretation only; it does not retrain or improve the weights.
2. **Accident label coverage is not established.** The previous model card says most accident classes lacked test GT, but there are no split label files to verify those counts. A new grouped split must report counts; any accident class below 30 GT boxes in val or test is marked NOT EVALUABLE and blocks promotion.
3. **Label quality and leakage are unknown.** Images, source provenance, and camera/scene identities are missing. Perceptual-hash plus local CLIP/DINOv2 clustering, label review, and group-level splitting must run before evaluation.

## Why the historical slice and threshold reports disagree

The reports do not measure the same quantity, and neither historical result is reproducible from the current checkout:

| Report path | Numerator / denominator | Interpretation |
|---|---|---|
| `ml/reports/slice_analysis.md` | False-positive frames divided by accumulated video frame duration within each slice; the report also historically mixed image samples into its time denominator. | Frame-level error density, not false incident alerts per camera-hour. A long predicted burst contributes many frames. The current slice code separates video false-positive frames from static-image false-positive samples and reports the video-only metric as **false-positive frames/video-hour**. |
| `ml/reports/threshold_tuning.md` | Debounced event-manager incidents divided by total duration of selected negative video clips. | Incident-level false alarms/hour. It is not comparable to frame-level error density. The prior report's fixed clip assumptions and counts have no available clip or raw-output artifacts. |

The legacy slice implementation also checked raw checkpoint class IDs against canonical accident IDs, despite the checkpoint's 10-class legacy order. Predictions are now decoded through `decode_model_class` before canonical accident-class checks. This fixes the index interpretation, but cannot validate the historical totals without data.

The prior threshold report selected candidates from a point false-alarm rate and later ran a separate, fixed-file test evaluation. The helper now computes the exact one-sided 95% Poisson upper rate and uses that bound for its candidate target; zero observed incidents therefore report approximately `2.996 / camera-hours`, never a zero-rate guarantee. The old CLI entry point is redirected to `ml.eval.run_all`, which currently marks temporal and threshold metrics **BLOCKED** rather than importing or regenerating legacy claims. This avoids retuning from test data.

These are confirmed implementation-level differences, not a determination of which historic numeric claim was correct. No current run can establish either rate until permitted clips, validated event labels, camera identities, and raw predictions are present.

## Required next action

Clear source licenses and archive checksums, fetch the source data, add complete camera/scene/video provenance, provide a local CLIP/DINOv2 TorchScript encoder, and review the exported contact sheets. Then rebuild the dataset, freeze its test manifest, and run `python -m ml.eval.diagnose --model best.pt --data-yaml ml/data/merged/data.yaml`. No retraining or test threshold tuning is authorized before this report has real GT counts and per-class AP.
