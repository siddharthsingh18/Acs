# ACS Dataset Card

## Status

There are no dataset files in this checkout. No source is currently both rights-cleared and checksum-pinned; therefore `make data` stops before downloading or building a training dataset. This prevents the documentation from implying reproducibility that the available artifacts cannot support.

## Sources and collection

The candidate source inventory, immutable source identifiers, current use status, and missing acquisition metadata are in [`configs/data_sources.yaml`](./configs/data_sources.yaml); current license review is in [`data/LICENSES.md`](./data/LICENSES.md). All sources are disabled. Prior repository notes mention Roboflow Universe projects, CADP frame extractions, and a Hugging Face dataset, but do not provide verified archive SHA256 values or complete rights evidence.

When cleared, `ml/scripts/fetch_data.py` will retrieve only manifest-approved archives, verify their SHA256 before extraction, check the expected extracted file count, and write `ml/data/manifest.sha256`. Commit that hash manifest with the code revision that uses the data; never commit image/video assets. `make data` then rebuilds and audits derived data.

## Class mapping

The canonical detection taxonomy is [`configs/classes.yaml`](./configs/classes.yaml): `bike`, `car`, `person`, and `accident`. Legacy accident subtype is optional incident metadata; legacy checkpoint ID mapping is documented in [`configs/checkpoint_class_mappings.yaml`](./configs/checkpoint_class_mappings.yaml). Source labels are mapped or rejected in [`configs/class_map.yaml`](./configs/class_map.yaml). Generic or ambiguous accident labels must not be treated as a verified collision without review.

## Scene and camera counts

Verified source scenes: **unknown**. Verified unique cameras: **unknown**. The source archives and scene/camera identifiers are absent, so neither deduplication nor camera-level split independence can be recomputed. Counts must be populated from the rebuilt data audit before model evaluation.

## Known biases and limitations

- Source availability and licenses are not independently established for the exact versions listed.
- Prior audits describe duplicate images, cross-split leakage, coarse boxes, and heterogeneous accident labels in candidate sources; see the dataset audit, but these findings are not reproducible without the original data.
- Accident imagery may overrepresent severe, visible collisions and underrepresent near-misses, occlusion, low light, weather, and geography.
- Camera IDs, scene IDs, and collection locations are not available, so scene-level and camera-level generalization cannot be claimed.
- Class-specific training-box counts are unavailable. The fewer-than-100-box merge decision cannot be measured on this checkout; the approved four-class taxonomy is provisional, and training/promotion must wait for a verified dataset rebuild and class-count audit.
- These data do not establish operational emergency-response performance. The detector is decision support only and each alert requires human verification.

## Options for sparse accident labels

The prior approval selected four detection classes, but missing training-box counts mean the fewer-than-100-box rule cannot be adjudicated yet:

| Option | Tradeoff |
|---|---|
| Keep one `accident` class and store verified subtype as optional metadata (current provisional choice) | Pools scarce accident boxes and simplifies training; loses subtype-level detector metrics and depends on consistent event labels. |
| Restore separate accident subtype classes | Preserves subtype-specific outputs; requires adequate, independently sourced examples per subtype and risks severe imbalance/noisy labels when counts are low. |
| Exclude accident labels from detector training until adequate evidence exists | Avoids training from ambiguous/underrepresented labels; leaves no learned accident detector capability in the interim. |

Do not choose among these based on the historical source-wide count. First compute per-class training-box counts on the legally cleared, deduplicated training split, then review annotation quality and scene/camera diversity. Any taxonomy change after that requires an explicit decision and a new checkpoint.

## Promotion gates

`ml/promotion_gates.yaml` defines the current evidence-based gates: accident macro mAP50 lower confidence bound >0.50, exact incident-recall lower bound ≥0.80, false-alarm upper bound ≤0.25 per camera-hour, at least 40 test events from 15 cameras/scenes, 20 accident-free camera-hours, all accident classes evaluable, robustness drop ≤0.10 in fog/rain/night/low bitrate, and no leakage or unverified claims. These defaults are provisional until accepted. `ml/scripts/promote.py` fails closed and generates the top-level model card from a run artifact.

## Reproduction requirements

Clear each source in the manifest and license table; populate immutable version/file identifiers, expected file counts, exact license and attribution, and archive SHA256; then run `make data`. Record unique scenes, cameras, split counts, and class box counts in the regenerated audit and this card.
