# Legacy report archive notice

All top-level reports and JSON/plot summaries in this directory that predate `runs/<run_id>/` are archived historical artifacts. They retain their existing paths because the claim audit links to exact source lines; each historical metric remains **UNVERIFIED** and must not be used for model promotion.

The authoritative report set is the six header-stamped Markdown files and `run_artifacts.json` in the directory named by `latest_run.txt`. Generate a replacement with `python -m ml.eval.run_all`; the report-header CI check rejects stale input hashes. A blocked run is an honest result and does not supersede missing evidence with historical numbers.
