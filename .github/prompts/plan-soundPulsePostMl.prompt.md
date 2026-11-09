## Plan: SoundPulse Post-ML Stages

This plan continues from `planSoundPulse.prompt.md` after M1-M9 (ML-first milestones) and contract freeze are complete. It covers only the remaining stages not yet implemented.

## Scope

- Included: runtime engine, Flutter app integration, storage/security implementation, personalization, context manager, device benchmarking, release hardening.
- Excluded: redoing completed ML milestones M1-M9.

## Stage A: Runtime Inference Engine (Critical Path)

### Objective
Build Android runtime components that consume exported artifacts and frozen v1 contracts.

### Work
1. Implement microphone capture lifecycle and continuous 1s segmentation.
2. Implement preprocessing path to match `packages/shared_contracts/v1/preprocessing_config.json`.
3. Integrate LiteRT model loading for TinyCNN and YAMNet artifact packages.
4. Implement embedding scoring against prototype store.
5. Implement open-set gating (`Unknown`) and `Background` suppression invariants.
6. Implement confidence-tier output and top-N candidates.
7. Add optional speech pitch cue pipeline invoked only for `speech` predictions.

### Deliverables
1. Runtime package at planned path `packages/runtime_android/`.
2. Stable runtime interface for app integration.
3. Runtime contract conformance tests.

### Exit Criteria
1. Runtime outputs conform to `packages/shared_contracts/v1/inference_result.schema.json`.
2. `Background` never triggers notify=true.
3. `Unknown` is policy-routed and never forced to known classes.
4. End-to-end offline inference path runs without network dependencies.

## Stage B: Runtime Verification Harness

### Objective
Guarantee correctness before broad app feature integration.

### Work
1. Add model-load tests for exported artifacts.
2. Add preprocessing parity fixtures against ML references.
3. Add schema checks for inference and threshold payloads.
4. Add policy invariant tests for unknown/background handling.

### Deliverables
1. Automated runtime verification suite.
2. Baseline performance traces for regression comparison.

### Exit Criteria
1. Parity checks pass within agreed numeric tolerance.
2. All schema and invariant tests pass in CI/local gate.

## Stage C: Flutter App Shell and Runtime Bridge

### Objective
Create Android-first Flutter shell and bind runtime APIs.

### Work
1. Scaffold Flutter app workspace (`pubspec.yaml`, app entry, Android integration).
2. Implement bridge APIs: start/stop monitoring, mode update, settings update, event stream.
3. Implement onboarding and permission-state UX for microphone/notifications/vibration/flashlight.
4. Implement monitoring dashboard and confidence-tier alert surfaces.
5. Implement no-context baseline toggle for A/B evaluation.

### Deliverables
1. Flutter app skeleton with runtime connectivity.
2. User-facing monitoring and alert flows for US1/US4 baseline.

### Exit Criteria
1. App can run continuous monitoring and display inference events from runtime.
2. Permission revocation states degrade safely and clearly.

## Stage D: Local Storage and Security

### Objective
Implement fixed storage architecture from PRD.

### Work
1. Implement `Isar` models for prototypes, thresholds, modes, class settings, event history.
2. Implement `hive` integration for keys/sensitive values.
3. Implement encrypted local audio file storage for personalization recordings.
4. Ensure metadata and audio files are separable for safe delete/reset flows.

### Deliverables
1. Storage layer adapters and migration/versioning strategy.
2. Delete/reset workflows verified for user-controlled samples.

### Exit Criteria
1. Entire storage path is local-only.
2. User can delete/re-record without metadata corruption.

## Stage E: Personalization End-to-End

### Objective
Deliver PRD-compliant few-shot personalization.

### Work
1. Implement guided recording for 3-5 samples per personalizable class.
2. Add waveform-first feedback, min-duration checks, overlap/noise warnings, playback-before-save.
3. Extract embeddings from local recordings and build class centroids.
4. Refresh background prototype for ambient adaptation.
5. Record personalization audit metadata.

### Deliverables
1. End-to-end personalization flow in app + runtime.
2. Prototype refresh service and persistence path.

### Exit Criteria
1. Prototype refresh completes locally within <= 6s per class update.
2. Re-record and delete actions are reliable and reversible.

## Stage F: Context Manager and Alert Policy

### Objective
Implement context-aware routing and suppression logic.

### Work
1. Implement modes: `Sleep`, `Focus`, `General`.
2. Implement context-aware on/off control.
3. Implement recent-event buffer and event-chain boosts from `packages/shared_contracts/v1/event_chains.json` within 15s windows.
4. Implement threshold/loudness/modality policy updates without app restart.
5. Keep urgent-sound handling prioritized under suppression policies.

### Deliverables
1. Context manager module with policy engine.
2. Mode-aware alert behavior integrated with runtime output.

### Exit Criteria
1. Mode changes apply immediately without restart.
2. Context-aware behavior can be disabled cleanly for baseline evaluation.

## Stage G: Device Benchmarking and Model Selection

### Objective
Collect release-decision evidence on target devices.

### Work
1. Run TinyCNN and YAMNet on Samsung Galaxy A14.
2. Measure accuracy, macro F1, latency (P50/P95), CPU, battery usage, model size, miss rate for urgent sounds, false alarms
3. Run context-on vs context-off matched scenarios.

### Deliverables
1. Device-level benchmark CSV/Markdown reports.
2. TinyCNN vs YAMNet release recommendation with evidence.

### Exit Criteria
1. PRD latency and context-improvement targets are evaluated with reproducible runs.
2. Default shipped model is selected and documented.

## Stage H: Release Hardening and Regression Gates

### Objective
Stabilize release candidate and prevent regressions.

### Work
1. Add regression checks for schema compatibility and artifact integrity.
2. Add preprocessing equivalence regression gates.
3. Add alert-policy regression checks for unknown/background/context behavior.
4. Add benchmark trend regression checks for latency and false alarms.
5. Produce final release report, model card updates, and artifact manifest version lock.

### Deliverables
1. Release gate checklist and pass/fail report.
2. Versioned release artifact package with traceable configs.

### Exit Criteria
1. Release gate passes with documented evidence and known limitations.
2. Artifact/config versions are immutable for release tagging.

## Dependency Order

1. Stage A -> Stage B (runtime first, then correctness gate).
2. Stage C starts after Stage A API shape stabilizes.
3. Stage D can proceed in parallel with late Stage C.
4. Stage E depends on Stage A + Stage C + Stage D.
5. Stage F depends on Stage A + Stage D and integrates with Stage C/E.
6. Stage G depends on integrated Stages A-F.
7. Stage H depends on Stage G outputs.

## Immediate Next Sprint Recommendation

1. Create `packages/runtime_android/` and implement audio capture + LiteRT inference pipeline.
2. Add runtime schema/parity tests before broad app UI work.
3. Scaffold Flutter shell and wire runtime bridge for monitoring + mode control.
4. Stand up storage layer (`Hive` + secure storage) before personalization implementation.
