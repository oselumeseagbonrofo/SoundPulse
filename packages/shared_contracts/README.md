# SoundPulse Shared Contracts

This directory is the single source of truth for versioned contracts shared across the ML, runtime, app, and evaluation workstreams.

Rules:
- Additive changes belong in a new version directory.
- Runtime and app code should consume exported artifacts and these contracts, not training internals.
- ML code should validate exported artifacts against the schemas here before handoff.

Current version:
- `v1`

Contents:
- `class_labels.json`: canonical class order and urgency flags.
- `preprocessing_config.json`: training and runtime preprocessing settings.
- `threshold_defaults.json`: default operating thresholds for inference and modes.
- `thresholds.schema.json`: schema for threshold payloads.
- `event_chains.json`: context-manager chain defaults.
- `inference_result.schema.json`: runtime output contract.
- `export_manifest.schema.json`: handoff package manifest contract.
