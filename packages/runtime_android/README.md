# SoundPulse Runtime Android (Sprint 1)

This package now includes a concrete Stage-A runtime path plus Stage-B validation utilities.

## Implemented in this slice

- Runtime contracts and mode model.
- Runtime artifact loaders for shared contracts and prototype stores.
- Continuous PCM segmentation with start/stop lifecycle and sliding windows.
- Policy-aware inference shaping (background and unknown handling).
- Prototype-scoring runtime engine with mode-aware gating and top-N candidates.
- Optional speech-pitch cue generation invoked only on speech detections.
- LiteRT model-loader and embedding extractor interfaces for TinyCNN/YAMNet backbones.
- Contract validation tools for inference payload shape and policy invariants.
- Preprocessing parity fixture checker against shared contract values.

## Not implemented yet

- Android AudioRecord microphone capture integration.
- Production LiteRT tensor-shape bindings and model I/O adapters.
- Full on-device DSP (STFT + log-mel parity with ML pipeline).
- Production context-manager event-chain boosts.
- Android foreground service integration.

## Quick checks

Run from repository root:

```powershell
python packages/runtime_android/tools/validate_inference_result.py
python packages/runtime_android/tools/validate_policy_invariants.py
python packages/runtime_android/tools/check_preprocessing_parity_fixture.py
```
