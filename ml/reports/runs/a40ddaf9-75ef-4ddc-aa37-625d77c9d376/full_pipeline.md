---
run_id: a40ddaf9-75ef-4ddc-aa37-625d77c9d376
git_commit: be3c00774092c187ecefbc6b6febb359ebd23829
model_sha256: 5e1a582110d1f16ebc3b67c2a2348873af30a14c55297c06e5c377ce17a21a68
data_manifest_sha256: 463e43161280b853bf49bee59ed1731077b3be78e02cd13ee0933a5c1f2d890f
class_file_sha256: de5ca1e40ec5cbd44c64ac7e52785d8ccd26ba4d8745772f5a52ea09cd0d0115
date_utc: '2026-10-23T23:19:06.557461+00:00'
metric_definitions:
  true_positive: One-to-one greedy confidence-ranked match of same class at IoU >=
    0.50.
  image_ap: Interpolated precision-recall area; GT-free classes are NOT EVALUABLE,
    not zero AP.
  incident_recall: True detected labeled accident events divided by all labeled test
    events; exact two-sided Clopper-Pearson 95% interval over events.
  false_alarms: False incident events divided by accident-free camera-hours; zero
    false alarms report exact one-sided 95% Poisson upper bound, approximately 2.996/T
    per hour.
  bootstrap: Resample camera/scene clusters, never frames; report percentile 95% interval.
  test_protocol: Tune thresholds only with rows labeled tuning; inspect the frozen
    test split once.
video_scope:
  test_clips: 0
  test_events: 0
  test_cameras: 0
  test_scenes: 0
  test_camera_hours: 0.0
  test_accident_free_camera_hours: 0.0
  tuning_clips: 0
---

# Full Pipeline — PRELIMINARY / BLOCKED

- rebuilt image dataset is missing: /Users/shidharthkushwaha/Desktop/Acs-1/ml/data/merged/data.yaml
- validated tuning/test video labels are missing: /Users/shidharthkushwaha/Desktop/Acs-1/ml/data/video_eval/labels.csv
- full tracking/debounce/temporal/VLM incident metrics are not implemented in this shared runner; legacy reports are not imported

No metrics are inferred from historical reports or plots.
