# SoundPulse

SoundPulse is an offline-first, on-device Android sound awareness application designed for Deaf and Hard-of-Hearing (DHH) users in Nigerian homes. It detects key environmental and safety-critical sounds, processes them locally, and delivers context-aware visual and haptic alerts.

---

## Key Features

- **Offline Sound Recognition**: On-device audio processing utilizing LiteRT (TensorFlow Lite) backbones (Tiny CNN & YAMNet embeddings).
- **Personalization**: Few-shot prototype centroid adaptation with user recordings to accommodate local acoustics without cloud storage.
- **Open-Set & Noise Rejection**: Euclidean distance-based gating rejects unknown sound events and suppresses background/generator noise.
- **Context-Aware Alert Routing**: Dynamically scales alert thresholds, priorities, and notification modalities across context modes (*General, Sleep, Focus, Cooking, Bathroom*).
- **Visual & Haptic Alerts**: Highly visible full-screen alerts, screen/camera flash patterns, and structured vibration outputs.

---

## Canonical Sound Classes (v1)

- `fire_alarm` (Urgent)
- `clock_alarm` (Urgent)
- `door_knock` (Urgent)
- `doorbell` (Urgent)
- `crying_baby` (Urgent, Personalizable)
- `water_running` (Personalizable)
- `dog_barking` (Personalizable)
- `door_wood_creaking` (Personalizable)
- `speech`
- `background` (Personalizable, Suppressed)

---

## Repository Structure

- [ml/](file:///C:/Users/ZBOOK%20STUDIO%20G5/Documents/GitHub/SoundPulse/ml) — Python workspace for data curation, preprocessing pipelines, model training, and LiteRT (TFLite) export.
- [mobile_app/](file:///C:/Users/ZBOOK%20STUDIO%20G5/Documents/GitHub/SoundPulse/mobile_app) — Flutter mobile application project with local Hive storage.
- [packages/shared_contracts/](file:///C:/Users/ZBOOK%20STUDIO%20G5/Documents/GitHub/SoundPulse/packages/shared_contracts) — Central source of truth for versioned configurations, label maps, and default thresholds.
- [packages/runtime_android/](file:///C:/Users/ZBOOK%20STUDIO%20G5/Documents/GitHub/SoundPulse/packages/runtime_android) — Kotlin runtime verification, continuous segmenting simulations, and contract validation tools.

---

## Steps for Running the System

### 1. Verification & Fast Checks (from Repository Root)
Validate contracts and verify the sliding window simulation/policies using the Kotlin/Python verification tools:
```powershell
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Run contract, policy, and preprocessing parity checks
python packages/runtime_android/tools/validate_inference_result.py
python packages/runtime_android/tools/validate_policy_invariants.py
python packages/runtime_android/tools/check_preprocessing_parity_fixture.py
python packages/runtime_android/tools/run_runtime_verification.py
```

### 2. ML Workspace Setup & Execution (within `ml/`)
1. **Set Up Workspace Dependencies**:
   ```powershell
   cd ml
   pip install -e .
   ```
2. **Build and Validate Dataset Manifests**:
   ```powershell
   python scripts/build_phase1_manifest.py
   python scripts/validate_dataset_manifest.py data/curated/soundpulse_manifest.v1.json
   ```
3. **Generate Features & Run Parity Checks**:
   ```powershell
   python scripts/generate_preprocessing_features.py --job-config config/preprocessing/feature_generation.default.json
   python scripts/validate_preprocessing_determinism.py --job-config config/preprocessing/determinism.default.json
   python scripts/run_preprocessing_parity.py --job-config config/preprocessing/parity.default.json
   ```
4. **Train and Export Models**:
   - Work through notebooks `1`, `2`, and `3` in `ml/notebooks/` to curate data and train/export models.
   - Build, validate, and release artifacts via the scripts:
     ```powershell
     python scripts/build_export_package.py
     python scripts/validate_export_artifacts.py
     python scripts/run_release_gate.py
     ```

### 3. Running the Mobile Application (within `mobile_app/`)
1. **Prerequisites**: Ensure you have Flutter SDK and Android SDK installed and configured.
2. **Build & Deploy**:
   ```powershell
   cd mobile_app
   flutter pub get
   flutter run
   ```
