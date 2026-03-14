# SoundPulse M9 Release Gate Report

- generated_at_utc: 2026-03-14T22:14:20.664587+00:00
- release_dir: C:\Users\ZBOOK STUDIO G5\Documents\GitHub\SoundPulse\ml\export\artifacts\generated\release_v1.0.0
- overall_status: pass

## Gate Checks

| Check | Status | Details |
|---|---|---|
| leakage_check_state | pass | Dataset split-integrity validation passed for 21456 clips and 21386 source groups. |
| artifact_integrity | pass | Export artifact validation overall_status=pass. |
| artifact_loadability | pass | TinyCNN TFLite model loaded successfully. YAMNet SavedModel loaded successfully. |
| preprocessing_parity_status_input | pass | Preprocessing parity tests reported as passing by operator input. |
| benchmark_baseline_presence | pass | Benchmark summaries present. tinycnn_macro_f1=0.12, yamnet_macro_f1=0.52, winner=YAMNet. |

## Export Validation

- overall_status: pass
- file_presence: pass (no issues)
- manifest_schema_conformance: pass (no issues)
- model_files_and_manifest_checksums: pass (no issues)
- checksum_manifest_integrity: pass (no issues)
- threshold_payload_version_consistency: pass (no issues)
- metadata_consistency: pass (no issues)
