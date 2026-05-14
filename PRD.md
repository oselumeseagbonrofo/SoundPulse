# Product Requirements Document

## Product

**Working title:** SoundPulse  
**Source basis:** Derived from Chapters 1-3 of `FYP.md`  
**Document intent:** Define the product, system boundaries, and delivery workstreams for an offline, context-aware environmental sound recognition app for Deaf and Hard of Hearing (DHH) users on Android.  
**Implementation note:** This PRD is formatted for AI-agent delivery. Each component should be treated as an isolated workstream with explicit inputs, outputs, interfaces, and non-goals so multiple agents can implement in parallel with minimal ambiguity.

## 1. Executive Summary

- **Problem Statement**: DHH users in Nigeria need reliable awareness of urgent home sounds, but existing sound recognition tools often degrade across environments, overload users with notifications, or depend on cloud processing and stable internet access.
- **Proposed Solution**: Build an Android-first, fully offline sound recognition system that combines lightweight on-device audio classification, few-shot personalization, open-set rejection, and a context-aware alert manager to deliver relevant visual and haptic notifications for selected home safety sounds.
- **Success Criteria**:
  - `100%` of core detection, personalization, and alerting flows must function without network connectivity after installation.
  - The shipped model configuration must achieve `>= 0.80` macro F1 on the curated 9-class evaluation set plus background, with `>= 0.90` recall on urgent classes (`fire alarm`, `clock alarm`, `door knock`, `doorbell`).
  - End-to-end alert latency must be `<= 0.2s` at P50 and `<= 0.5s` at P95 from sound onset on supported Android test devices.
  - Continuous monitoring must support at least `8 hours` of operation on target mid-range Android test devices under the agreed test protocol.
  - With context awareness enabled, non-urgent notifications must drop by `>= 25%` versus the no-context baseline without reducing urgent-sound recall by more than `2 percentage points`.

## 2. User Experience & Functionality

### User Personas

- **Primary Persona: DHH Android User in Nigeria**
  - Needs awareness of household sounds that affect safety and situational awareness.
  - Relies on visual and haptic cues rather than audio.
  - Needs reliable offline behavior because internet access may be inconsistent.
- **Secondary Persona: Hard-of-Hearing User with Selective Alert Needs**
  - Wants alerts for only the most relevant sounds depending on time, room, or activity.
  - Needs control over alert sensitivity, sound classes, and output modality.
- **Tertiary Persona: Researcher / Evaluator**
  - Needs to compare Tiny CNN and YAMNet on-device performance across accuracy, latency, battery use, model size, false alarms, and miss rate.
  - Needs reproducible evaluation artifacts and clear benchmark outputs.

### User Stories

- **US1**: As a DHH user, I want the app to detect urgent home sounds offline so that I can stay aware of important events without internet access.
- **US2**: As a DHH user, I want to personalize sound classes with a few guided recordings so that the system adapts to my home environment.
- **US3**: As a DHH user, I want to switch between context modes such as `Sleep`, `Focus`, and `General` so that I receive fewer unnecessary alerts.
- **US4**: As a DHH user, I want notifications to reflect confidence and urgency so that I can quickly judge whether to act.
- **US5**: As a DHH user, I want unknown and background sounds suppressed or downgraded so that I am not overwhelmed by noisy or irrelevant alerts.
- **US6**: As a researcher or product evaluator, I want the system to benchmark both model paths on real Android devices so that the final shipped configuration is evidence-based.

### Acceptance Criteria

- **US1 Done**
  - The app continuously captures mono audio in the foreground or approved background monitoring mode on Android.
  - Audio is segmented into `1 second` windows for inference.
  - The shipped runtime supports the nine target classes plus `Background`.
  - No network call is required for capture, inference, thresholding, or notification.
- **US2 Done**
  - Users can record `3-5` samples per personalizable class.
  - The recording flow includes waveform feedback, minimum-duration checks, overlap/noise warnings, and playback before save.
  - Personalization generates or refreshes locally stored class prototypes without retraining the backbone model on-device.
  - Users can delete and re-record samples for a class.
- **US3 Done**
  - The app exposes `Sleep`, `Focus`, and `General` modes.
  - Mode changes update alert thresholds, loudness floor, and output modality policy without requiring app restart.
  - The context manager can use recent event history to boost or suppress alert routing.
  - The app must support a no-context baseline mode for A/B evaluation.
- **US4 Done**
  - High-confidence alerts show the predicted label and confidence tier.
  - Medium-confidence alerts show the top `N` candidate labels, where `N` is user-configurable within a bounded range.
  - Low-confidence alerts are rendered as generic `Sound detected` events rather than overclaiming a class.
  - Alerts support vibration and visual notification; flashlight blinking is user-controlled and permission-aware.
- **US5 Done**
  - If a sound is classified as `Background`, the user is not alerted.
  - If open-set gating marks a sound as `Unknown`, the alert policy applies the configured generic or suppressed behavior.
  - Users can review and adjust sensitivity-related settings exposed by product design without editing raw model files.
- **US6 Done**
  - The evaluation harness runs Tiny CNN and YAMNet on at least two Android test devices.
  - Benchmark outputs include macro F1, per-class recall, urgent-sound miss rate, false alarm rate, latency, battery drain, and model artifact size.
  - Context manager on/off comparisons are recorded under the same scenario set.
  - The final release candidate includes a recommendation for which model path should ship by default.

### Non-Goals

- iOS, smartwatch, or multi-device sync in v1.
- Cloud inference, cloud storage, or always-online features in v1.
- General speech-to-text captioning, conversation transcription, or sign language features.
- Unlimited sound-class support beyond the curated 9 target classes plus background in v1.
- Smart-home hardware integration, external microphones, or multi-room sensor networks in v1.
- A full end-user study as part of the initial engineering delivery; this PRD covers product/system build and technical evaluation.

## 3. AI System Requirements

### Tool Requirements

- **Model Creation Workstream**
  - Python 3.x training environment.
  - TensorFlow / Keras preferred for alignment with YAMNet and LiteRT export.
  - `NumPy`, `Librosa`, and related DSP utilities for waveform processing and log-mel feature generation.
  - Curated data from `ESC-50`, `FSD50K`, and `AudioSet` for the nine selected classes.
  - Additional locally captured Nigerian ambient background recordings for the `Background` class and context augmentation.
- **Runtime AI Workstream**
  - LiteRT interpreter on Android.
  - On-device preprocessing for resampling, windowing, and log-mel generation unless embedded in model graphs.
  - Voice Activity Detection (VAD) gate to reduce unnecessary inference.
  - Distance-based open-set classifier over embedding vectors.
  - Pitch-analysis module invoked only when the upstream classifier predicts `speech`.
- **Evaluation Workstream**
  - Reproducible local scripts for model export, threshold calibration, and benchmark reporting.
  - Device-level test harness for latency, thermal behavior, CPU use, battery drain, and false alarms under fixed scenarios.
  - Structured reports in CSV or Markdown for comparison across models, devices, and context-manager settings.

### Evaluation Strategy

- **Dataset Evaluation**
  - Train, validation, and test splits must avoid leakage between source clips and augmented derivatives.
  - Report macro F1, weighted F1, per-class precision/recall, confusion matrix, and urgent-sound recall.
- **Open-Set Evaluation**
  - Measure unknown rejection precision/recall using out-of-scope environmental sounds.
  - Tune distance threshold `T` and loudness floor `L` on validation data before device testing.
- **Personalization Evaluation**
  - Compare baseline performance versus performance after `3-5` user samples per class using the context-augmentation workflow with `alpha = 0.3`.
  - Validate that prototype refresh completes locally within an acceptable UX window of `<= 6 seconds` per class update.
- **On-Device Evaluation**
  - Run both Tiny CNN and YAMNet on at least two Android phones representing low/mid-range hardware.
  - Measure end-to-end latency, memory footprint, CPU load, battery drain, and model bundle size.
- **Context-Manager Evaluation**
  - Compare `context-aware on` versus `context-aware off` using matched scenarios.
  - Report reduction in non-urgent alerts, urgent-sound recall delta, and event-chain boost success rate.
- **Speech Pitch Module Evaluation**
  - Validate only as an auxiliary speech cue on speech segments.
  - Product UI must avoid presenting pitch-based output as identity truth; default copy should describe it as an approximate voice-pitch cue.

## 4. Technical Specifications

### Architecture Overview

The system is an offline, on-device pipeline with a separate model-development track and product-application track. Agents should treat these as distinct delivery workstreams with explicit interfaces.

**AI-agent delivery rules**

- Each component owner must expose stable input/output contracts before downstream work begins.
- Model-training agents must not make Flutter or Android runtime assumptions beyond the documented model interface.
- Mobile-app agents must consume exported model artifacts and configuration contracts rather than modifying model-training code directly.
- Shared constants such as class labels, window size, embedding dimensions, thresholds, and mode names must live in versioned configuration artifacts rather than being duplicated across components.
- Evaluation agents must benchmark released artifacts exactly as exported by the model and app workstreams, not local experimental variants.

**High-level data flow**

`Microphone -> Audio Capture -> 1s Segmentation -> Feature Extraction -> Embedding Backbone -> Prototype / Open-set Scoring -> Context Manager -> Notification UI`

If the detected class is `speech`, the runtime invokes the pitch-analysis module before final alert rendering.

**Component A: Model Creation and Export Pipeline**

- **Scope**: Curate datasets, train the Tiny CNN embedding model, integrate frozen YAMNet embeddings, calibrate thresholds, and export LiteRT-ready artifacts.
- **Inputs**: Public dataset clips, background-noise recordings, class list, preprocessing configuration.
- **Target class inventory**:
  - `fire alarm`
  - `clock alarm`
  - `door knock`
  - `doorbell`
  - `crying baby`
  - `water running`
  - `dog barking`
  - `door wood creaking`
  - `speech`
  - `background`
- **Outputs**:
  - `tiny_cnn.tflite`
  - `yamnet.tflite`
  - label map
  - preprocessing specification
  - threshold defaults
  - model card and benchmark report
- **Excludes**: Flutter UI, Android permission handling, user recording flows.

**Component B: Personalization and Prototype Builder**

- **Scope**: Handle guided recording, sample validation, context augmentation, embedding extraction, and local prototype generation.
- **Inputs**: User-recorded samples, background prototype, selected backbone model.
- **Outputs**:
  - per-class centroid vectors
  - sample metadata
  - personalization audit log
- **Key rule**: Prototype creation averages embeddings from `3-5` validated user samples per class.
- **Background rule**: The local ambient environment is used to initialize or refresh the `background` prototype for context adaptation.
- **Key algorithm note**: Context augmentation uses the Jain et al. cross-setting formula with `alpha = 0.3`.

**Component C: Runtime Inference Engine**

- **Scope**: Live audio streaming, segmentation, feature extraction, embedding generation, nearest-prototype classification, open-set gating, background suppression, and optional speech pitch cueing.
- **Inputs**: PCM audio frames, prototype store, threshold config, active mode.
- **Outputs**:
  - predicted label
  - top-N candidates
  - confidence tier
  - unknown/background flags
  - optional speech-pitch cue
- **Key rules**:
  - `Background` never generates a user alert.
  - `Unknown` routes through policy rather than being forced into a known class.

**Component D: Context Manager and Alert Policy**

- **Scope**: Modify routing behavior based on mode, time of day, event history, and lightweight event-chain reasoning.
- **Inputs**: Inference result, mode, timestamp, recent event buffer.
- **Outputs**:
  - final notify / suppress decision
  - alert modality
  - adjusted thresholds for the next inference window where applicable
- **Behavioral requirements**:
  - Support `Sleep`, `Focus`, and `General` modes.
  - Support predefined event chains such as `door knock -> speech` and `doorbell -> speech/dog barking` within a `15s` window.
  - Allow disabling context behavior for evaluation.

**Component E: Flutter Mobile App**

- **Scope**: Onboarding, permissions, mode selection, sound-class management, personalization flows, notification rendering, settings, and accessible visualizations.
- **Inputs**: Runtime engine output, stored preferences, prototype metadata.
- **Outputs**:
  - user-facing screens
  - local configuration updates
  - recording actions
- **UX requirements**:
  - Android-first implementation in Flutter.
  - Clear waveform-first guidance during recording.
  - Visualizations should be toggleable between waveform, spectrogram, and correlogram, but waveform must be primary because it is the least expert-dependent option in the literature review.

**Component F: Local Storage and Configuration**

- **Scope**: Persist models, thresholds, modes, prototypes, event logs, and optional user-recorded samples.
- **Requirements**:
  - Entire storage design must be local-only in v1.
  - User-recorded samples must be deletable by the user at any time.
  - Sensitive local artifacts should be encrypted at rest where supported by the chosen mobile stack.
  - The recommended storage architecture is:
    - `Isar` for structured local metadata such as prototypes, thresholds, mode settings, class configuration, and event history.
    - `Hive` for keys and small sensitive configuration values.
    - Encrypted local file storage for user-recorded audio samples, with key management backed by Android Keystore through the secure-storage layer.
  - The storage layer must separate audio files from metadata so users can delete recordings without corrupting model configuration.

**Component G: Evaluation and Benchmark Harness**

- **Scope**: Automate repeatable model/device comparison and context-manager A/B testing.
- **Outputs**:
  - per-device benchmark reports
  - regression suite for latency and false alarms
  - release recommendation for default shipped model
- **Requirements**:
  - Must test with and without VAD gating where supported.
  - Must produce comparison tables for Tiny CNN versus YAMNet.

### Integration Points

- **Android OS**
  - Microphone permission and audio capture lifecycle.
  - Background execution rules for continuous listening.
  - Notification, vibration, and flashlight APIs.
- **LiteRT Runtime**
  - Model loading, tensor inference, CPU delegate configuration, and model versioning.
- **DSP / Feature Layer**
  - Resampling to the model-required sample rate.
  - Log-mel generation with shared configuration between training and runtime.
- **App-to-Model Interfaces**
  - `Embedding API`: `1s mono PCM -> float embedding vector`.
  - `Prototype Store API`: `class_id + centroid + sample_count + mode/context metadata`.
  - `Inference Result API`: `label + score/distance + confidence_tier + top_n + flags`.
  - `Alert Decision API`: `notify:boolean + modality + payload`.
- **Data / Auth**
  - No user account, remote database, or cloud auth is required in v1.
  - Any future sync or backup feature is out of scope for this document and must be treated as a new product phase.

### Security & Privacy

- Raw environmental audio must not be transmitted off-device in v1.
- The app must request only permissions required for microphone, notifications, vibration, and flashlight behavior used by enabled features.
- User-recorded personalization samples are user-controlled data and must support delete/reset flows.
- Event logs should store only the minimum metadata needed for debugging and evaluation unless the user explicitly enables richer logging.
- The system must degrade safely when permissions are revoked, including clear UI states for disabled monitoring.
- Model and threshold updates must be versioned so evaluation results can be tied to an exact release artifact.

## 5. Risks & Roadmap

### Phased Rollout

- **MVP**
  - Android Flutter app with offline inference for the curated class set.
  - Dual-model evaluation pipeline in place, but only one model ships as default runtime.
  - Guided personalization with prototype building.
  - `Sleep`, `Focus`, and `General` modes.
  - Visual and haptic alert delivery with confidence tiers.
- **v1.1**
  - Improved threshold calibration wizard.
  - Better false-alarm controls and richer context-manager policies.
  - Stronger event-history review for debugging and evaluation.
  - More robust sample-quality coaching during personalization.
- **v2.0**
  - Expanded class library and richer personalization controls.
  - Optional accessory integrations such as wearables or smart-home endpoints.
  - Advanced adaptive context learning if justified by evaluation and privacy review.

### Technical Risks

- **Domain mismatch risk**: Public datasets may not match Nigerian home acoustics closely enough, causing degraded real-world performance.
- **Battery and thermal risk**: Continuous listening and on-device inference may drain battery faster than acceptable on low-end phones.
- **Few-shot quality risk**: Poor user recordings can create unstable prototypes and misclassifications.
- **Threshold tuning risk**: Open-set rejection and loudness gates may trade false alarms against missed urgent events if not calibrated carefully.
- **Notification overload risk**: Even with context rules, excessive low-value alerts may reduce trust and app retention.
- **Android execution risk**: Background microphone behavior and vendor-specific battery restrictions may limit reliable always-on monitoring.
- **Pitch-labeling risk**: A simple pitch threshold is not a robust proxy for identity; product wording and use of this output must remain conservative.
- **Research-validation risk**: The document is informed by literature and technical evaluation goals, but no end-user study is included in the initial scope.

### Fixed Delivery Constraints

- The PRD does not assume a fixed number of human developers because implementation is intended to be carried out by AI agents.
- The PRD does not fix delivery by a day-count estimate; sequencing should instead be driven by dependency order between workstreams.
- Exact Android device support matrix is fixed as:
  - Support target: `Android 10 and above`
  - `Samsung Galaxy A03`
    - RAM: `4 GB`
    - Storage: `64 GB`
    - Android version for test plan: `Android 15`
    - Role in testing: low-end baseline device used to validate minimum acceptable inference speed, memory behavior, and battery drain under constrained hardware conditions.
  - `Samsung Galaxy A14`
    - RAM: `4 GB`
    - Storage: `64 GB`
    - Android version for test plan: `Android 15`
    - Role in testing: primary mid-range validation device used for release-candidate tuning, context-manager evaluation, and default-user experience verification.
- Final choice of encrypted local storage library is fixed as:
  - `Isar` for structured local metadata such as prototypes, thresholds, modes, class configuration, and event history.
  - `Hive` for keys and sensitive configuration values.
  - Encrypted local files for stored user-recorded audio samples, referenced from `Isar`.
