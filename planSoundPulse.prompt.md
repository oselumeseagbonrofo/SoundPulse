## Plan: SoundPulse ML-First Build

Build SoundPulse as a single monorepo, but prioritize one lead workstream first: the ML model/export pipeline. The reason is straightforward: runtime inference, personalization, context policy, and benchmarking all depend on stable artifacts for preprocessing, class order, embedding shape, thresholds, and exported LiteRT models. Instead of starting every major area at once, front-load the ML and contract layer, then bring up runtime, app, and evaluation in a staged sequence.

**Steps**
1. Phase 0: Foundation and repo bootstrap. Create the monorepo layout with top-level workspaces for the Flutter app, Android-native inference/plugin code, Python ML pipeline, shared versioned artifacts, and evaluation harness.
2. Phase 0: Freeze shared contracts before any downstream build. Define the canonical class inventory, sample rate, 1-second window contract, preprocessing spec, embedding API, prototype-store schema, inference-result schema, alert-decision schema, threshold config schema, pitch-cue payload schema, event-chain config schema, and version manifest format.
3. Phase 1: Prioritize the ML/model-export workstream first using the milestone-by-milestone checklist below.
4. Phase 1 exit gate: Do not start full runtime or app integration until Milestones M1-M8 are complete and the ML track produces a usable artifact package: exported models, preprocessing spec, label map, thresholds, embedding metadata, and benchmark baselines.
5. Phase 2: Start the runtime inference engine against the exported artifacts. Build Android audio capture, streaming segmentation, DSP/log-mel extraction, VAD gate, LiteRT model loading, embedding extraction, prototype scoring, open-set rejection, background suppression, and the speech-pitch module.
6. Phase 2: Start the evaluation harness in parallel with runtime once the exported artifacts exist. Build dataset reports, runtime equivalence checks, latency tooling, open-set tests, personalization timing checks, and device benchmark harnesses.
7. Phase 3: Start the Flutter app workstream once runtime interfaces are concrete. Build the app shell, onboarding, permissions, monitoring controls, mode switching, alert rendering, local storage adapters, and app-facing bindings to the runtime engine.
8. Phase 4: Integrate personalization end to end. Implement guided recording for 3-5 samples per class, sample-quality validation, waveform-first UX, embedding extraction from recordings, centroid generation, background prototype refresh, encrypted sample storage, and deletion/reset flows.
9. Phase 4: Integrate context manager and alert policy. Implement Sleep, Focus, and General modes, context-aware on/off toggle, recent-event buffering, event-chain boosts within 15 seconds, urgency-aware modality routing, and confidence-tier rendering rules.
10. Phase 5: End-to-end integration and hardening. Replace any remaining stubs with real runtime/plugin implementations, validate preprocessing equivalence between training and runtime, harden background-monitoring lifecycle behavior, and verify that threshold/config updates apply without restart.
11. Phase 6: Release benchmarking and ship-default decision. Run Tiny CNN vs YAMNet on Samsung Galaxy A03 and Galaxy A14, compare context-aware on/off under matched scenarios, collect CSV/Markdown reports, and choose the default shipped model based on the PRD success metrics.
12. Phase 7: Stabilization and regression protection. Add automated checks for schema compatibility, dataset leakage, exported-artifact integrity, runtime preprocessing equivalence, alert-policy regressions, and device-performance regressions. Produce the final release recommendation, model card, and benchmark package.

**Detailed execution sequence**
1. Bootstrap monorepo structure and dependency management.
2. Author shared configuration and schema artifacts first. This blocks ML export and all runtime parsing.
3. Execute the ML/export pipeline as the first implementation workstream and keep focus there until artifact handoff is stable.
4. Start runtime core work only after Step 3 yields stable models and configs.
5. Start evaluation harness work as soon as the first export package exists so runtime correctness and performance can be validated immediately.
6. Start the Flutter app shell and integration layer after runtime interfaces are no longer speculative.
7. Add personalization once runtime embeddings and storage contracts are stable.
8. Add context manager once inference results and settings persistence are stable.
9. Run device benchmarking and use results to pick the default model.
10. Close with stabilization, regression coverage, and release documentation.

**ML-first milestone checklist**
- [x] M1. Set up the ML workspace structure under `ml/`.
- [x] M1 deliverables: workspace README, config directories, scripts directory, export directory, evaluation directory, dataset manifest template.
- [x] M1 exit criteria: the repo has a stable folder layout for data prep, training, calibration, export, and evaluation.
- [x] M2. Freeze ML-facing shared contracts.
- [x] M2 tasks: finalize class inventory, label IDs, preprocessing parameters, 1-second window contract, embedding output schema, threshold schema, and export manifest schema.
- [x] M2 deliverables: versioned `class_labels.json`, `preprocessing_config.json`, `thresholds.schema.json` or equivalent contract file, and export metadata schema.
- [x] M2 exit criteria: no unresolved ambiguity remains in class ordering, preprocessing settings, or exported artifact shape.
- [x] M3. Build dataset ingestion and split integrity tooling.
- [x] M3 tasks: ingest ESC-50, FSD50K, AudioSet subsets, and Nigerian background recordings; normalize metadata; generate train, validation, and test splits; add leakage checks across source clips and augmented derivatives.
- [x] M3 deliverables: dataset manifest, split manifests, class-to-source mapping, leakage-check script, background-data inventory.
- [x] M3 exit criteria: all clips are assigned to exactly one split and the class coverage matches the PRD inventory plus background.
- [ ] M4. Implement the preprocessing and feature pipeline.
- [ ] M4 tasks: build waveform loading, resampling, windowing, STFT, mel projection, log compression, and feature-shape validation.
- [ ] M4 deliverables: reusable preprocessing module, config-driven feature generation, parity tests against reference outputs.
- [ ] M4 exit criteria: preprocessing is deterministic and parameterized entirely by the shared config.
- [x] M5. Train and validate the Tiny CNN baseline.
- [x] M5 tasks: implement the Tiny CNN backbone, training loop, checkpointing, validation metrics, confusion matrix output, and early benchmark reporting.
- [x] M5 deliverables: first trained checkpoint, training config, validation report, per-class metrics, confusion matrix.
- [x] M5 exit criteria: the Tiny CNN produces stable embeddings and achieves a credible baseline worthy of downstream comparison.
- [x] M6. Integrate and benchmark the YAMNet path.
- [x] M6 tasks: wire in the frozen YAMNet embedding path, align its preprocessing assumptions with the shared contract or explicitly document differences, and run the same validation suite used for Tiny CNN.
- [x] M6 deliverables: YAMNet integration module, validation report, embedding metadata, side-by-side comparison with Tiny CNN.
- [x] M6 exit criteria: both model paths can be evaluated under the same dataset protocol and exported through the same handoff format.
- [x] M7. Calibrate thresholds, open-set behavior, and personalization augmentation.
- [x] M7 tasks: tune distance threshold `T`, loudness floor `L`, unknown-rejection behavior, background suppression behavior, and context-augmentation behavior with `alpha = 0.3`.
- [x] M7 deliverables: calibrated thresholds, open-set evaluation report, unknown-rejection metrics, personalization augmentation notes, baseline mode defaults.
- [x] M7 exit criteria: threshold behavior is versioned and good enough to unblock runtime scoring and policy work.
+ [x] M8. Export LiteRT artifacts and handoff package.
+ [x] M8 tasks: export Tiny CNN and YAMNet artifacts, generate label maps, metadata, preprocessing spec, threshold defaults, model cards, and artifact manifests.
+ [x] M8 deliverables: exported `.tflite` artifacts, label map, preprocessing spec, thresholds, model metadata, model card, checksums, benchmark-ready manifest.
+ [x] M8 exit criteria: runtime can consume the package without guessing any shapes, labels, thresholds, or preprocessing values.
+ [x] M9. Run the ML release gate before downstream expansion.
+ [x] M9 checks: leakage checks pass, exported artifacts load successfully, preprocessing parity tests pass, benchmark baselines are recorded, and the model package is versioned for runtime consumption.
+ [x] M9 outcome: unlock runtime implementation, evaluation harness acceleration, and app integration against real artifacts rather than placeholders.

**Dependencies and staging**
1. Primary blocking dependency: preprocessing spec, class order, embedding dimension contract, threshold schema, prototype schema, and inference-result schema.
2. ML-first output required before downstream acceleration: exported LiteRT models, label map, preprocessing config, thresholds, model metadata, and benchmark-ready manifests.
3. Runtime depends on ML artifact stability, not just conceptual contracts.
4. App depends on concrete runtime interfaces and storage schemas.
5. Device latency, battery, and context A/B benchmarks depend on exported artifacts plus integrated app/runtime builds.

**Workstream priority**
1. First: ML/export workstream.
2. Second: Android runtime inference workstream.
3. Third: Evaluation harness, starting as soon as export artifacts exist and continuing alongside runtime.
4. Fourth: Flutter app integration and UX layers.
5. Fifth: Personalization and context-policy integration.

**Why ML first**
1. It fixes the highest-risk ambiguity early: training/runtime preprocessing parity.
2. It gives every downstream layer stable contracts instead of placeholder assumptions.
3. It makes runtime and app work more deterministic because model shape, thresholds, class order, and artifact packaging are already fixed.
4. It enables benchmarking to start before the full UI exists, which reduces late surprises on latency, battery, and model quality.

**Workstream deliverables**
1. ML/export deliverables: curated manifests, train/val/test splits, Tiny CNN checkpoint/export, YAMNet export package, preprocessing spec, class label map, thresholds, model metadata, model card, benchmark-ready manifests.
2. Runtime deliverables: Android audio stream pipeline, DSP module, LiteRT wrapper, VAD gate, embedding/scoring engine, open-set/background policy, speech-pitch analyzer, runtime tests.
3. Evaluation deliverables: dataset reports, open-set reports, preprocessing-equivalence checks, personalization timing report, device benchmark CSV/Markdown, context on/off comparison tables, ship-default recommendation.
4. App deliverables: onboarding, permission recovery states, monitoring UI, mode selector, personalization recorder/player, settings, alert surfaces, secure local storage flows, event history/review UI where in scope.

**Relevant files**
- /Users/oselumese/Documents/GitHub/SoundPulse/PRD.md — source requirements and acceptance criteria.
- /Users/oselumese/Documents/GitHub/SoundPulse/.FYP.md — supporting research basis referenced by the PRD.
- /Users/oselumese/Documents/GitHub/SoundPulse/ml/README.md — planned ML workspace entrypoint to create first.
- /Users/oselumese/Documents/GitHub/SoundPulse/ml/config/ — planned versioned training/export/eval configs to create first.
- /Users/oselumese/Documents/GitHub/SoundPulse/ml/export/artifacts/ — planned exported LiteRT artifacts and manifests to create first.
- /Users/oselumese/Documents/GitHub/SoundPulse/packages/shared_contracts/ — planned shared config/schema package to create before runtime and app work.
- /Users/oselumese/Documents/GitHub/SoundPulse/packages/runtime_android/ — planned Android-native runtime/plugin package to create after ML artifacts stabilize.
- /Users/oselumese/Documents/GitHub/SoundPulse/tooling/evaluation/ — planned device and report harnesses to create once export artifacts exist.
- /Users/oselumese/Documents/GitHub/SoundPulse/pubspec.yaml — planned Flutter workspace manifest to create after runtime and storage contracts are defined.
- /Users/oselumese/Documents/GitHub/SoundPulse/lib/main.dart — planned Flutter app entrypoint to create after runtime and storage contracts are concrete.

**Verification**
1. Schema validation: shared artifact schemas parse consistently in ML export, Android runtime, and Flutter layers.
2. ML gate before downstream expansion: no clip leakage across train/val/test; macro F1 and urgent-class recall meet or approach PRD thresholds; export package is loadable and versioned.
3. Runtime validation: preprocessing matches training numerically; background never alerts; unknown routing follows policy; mode switches apply without restart.
4. Personalization validation: prototype refresh completes locally within 6 seconds per class; users can delete and re-record without corrupting metadata.
5. Device validation: P50 latency <= 0.2 seconds and P95 <= 0.5 seconds on supported targets where required by PRD; monitoring sustains 8 hours under the agreed protocol.
6. Context validation: non-urgent alerts drop by at least 25 percent versus baseline while urgent recall decreases by no more than 2 percentage points.
7. Release validation: Tiny CNN and YAMNet are both benchmarked on Galaxy A03 and Galaxy A14, with a documented default-model recommendation.

**Decisions**
- Solution structure: single monorepo.
- Delivery style: one prioritized lead workstream first, then staged downstream execution.
- Lead workstream: ML/model export.
- Speech pitch cue: included in v1, but isolated behind a clear runtime interface so it does not destabilize core detection.
- Recommended architecture: Flutter Android-first product app with native Android runtime/plugin code for audio capture, continuous listening, and LiteRT execution rather than forcing all low-level audio/DSP concerns into pure Dart.
- Recommended implementation priority: shared contracts first, then ML export first, then runtime and evaluation, then app integration, then personalization/context integration, then device benchmarking and release choice.

**Scope boundaries**
- Included: Android 10+ only, offline inference, offline personalization, local-only storage, Tiny CNN and YAMNet evaluation, context-aware alerting, pitch cue for speech segments, release benchmarking on Galaxy A03 and Galaxy A14.
- Excluded: iOS, cloud sync/storage, speech-to-text/captioning, smart-home integrations, unlimited class expansion, end-user study execution.

**Further considerations**
1. The highest-risk technical dependency remains exact preprocessing parity between training and runtime. Treat the preprocessing spec as a first-class artifact and verify it before any serious runtime or app benchmarking.
2. The highest-risk product/platform dependency remains Android background audio behavior and battery policy. Even with an ML-first plan, prototype the foreground-service monitoring architecture early in runtime Phase 2.
3. This revised plan reduces coordination risk, but it will slow visible app progress at the start. That tradeoff is justified because downstream rework is more expensive than front-loading the model contract.