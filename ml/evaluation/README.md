# SoundPulse ML Evaluation

This directory is reserved for benchmark protocols, evaluation summaries, and comparison reports.

Planned outputs:

- validation metrics per backbone
- open-set and threshold calibration summaries
- device-ready benchmark inputs for downstream runtime evaluation

Current reports:

- `tinycnn_baseline_report.md` and `tinycnn_baseline_summary.json` for the recorded TinyCNN notebook run
- `yamnet_kaggle_report.md` and `yamnet_kaggle_summary.json` for the recorded YAMNet notebook run
- `model_comparison_report.md` for the canonical backbone recommendation, now based on the shared-cap rerun
- `same_cap_model_comparison_report.md` and `same_cap_model_comparison_summary.json` for the stricter shared-cap prototype comparison
- `tinycnn_same_cap_predictions.csv` and `yamnet_same_cap_predictions.csv` for clip-level predictions under the shared-cap protocol
- `yamnet_threshold_calibration_report.md`, `yamnet_threshold_calibration_summary.json`, and `yamnet_threshold_candidate.json` for the latest M7 YAMNet threshold calibration pass using out-of-scope FSD50K dev/eval clips as the unknown proxy pool
- `yamnet_threshold_sweep.csv` for the full threshold sweep across distance, ambiguity ratio, and loudness operating points
- `yamnet_threshold_operating_points.json` for the named operating-point comparison exported from the sweep (`selected_release_candidate`, `best_balanced_score`, `best_known_retention`)
- `yamnet_known_test_calibrated_predictions.csv` and `yamnet_unknown_holdout_calibrated_predictions.csv` for calibrated YAMNet routing behavior on the known test set and the FSD50K out-of-scope holdout set
- `release_artifact_validation_summary.json` for the machine-readable export integrity check output from `scripts/validate_export_artifacts.py`
- `release_gate_summary.json` and `release_gate_report.md` for the M9 release-gate status across leakage, artifact integrity, loadability, preprocessing parity status input, and benchmark baseline presence

Release-gate commands:

- `python scripts/build_export_package.py`
- `python scripts/validate_export_artifacts.py`
- `python scripts/run_release_gate.py --preprocessing-parity-status not_run`

Current calibration status:

- The FSD50K-based proxy pass is more realistic than the earlier ESC-50-only pass and should be treated as the current reference for M7 threshold tuning.
- The exported operating points make the tradeoff explicit: stronger unknown rejection currently comes with a large drop in known-set retention, so these artifacts are still evaluation outputs rather than ship-default thresholds.
