# TinyCNN vs YAMNet Comparison Report

## Canonical conclusion

YAMNet is the stronger backbone under the current controlled comparison protocol and should be treated as the lead baseline for M7 threshold calibration and open-set work.

## Authoritative protocol

The authoritative comparison is the shared-cap rerun, not the earlier notebook-only snapshot.

- Shared caps: `train=128`, `validation=64`, `test=64`
- Test support per backbone: `528`
- Scoring method: nearest-prototype inference using train-split centroids
- Reproducible script: `ml/scripts/evaluate_same_cap_backbones.py`
- Detailed report: `ml/evaluation/same_cap_model_comparison_report.md`

## Shared-cap comparison

| Metric | TinyCNN | YAMNet | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.2936 | 0.5076 | +0.2140 |
| Macro precision | 0.2738 | 0.5673 | +0.2935 |
| Macro recall | 0.3200 | 0.5432 | +0.2232 |
| Macro F1 | 0.2618 | 0.5181 | +0.2563 |
| Weighted precision | 0.2930 | 0.5820 | +0.2890 |
| Weighted recall | 0.2936 | 0.5076 | +0.2140 |
| Weighted F1 | 0.2657 | 0.5108 | +0.2451 |

## Class-level readout under the shared-cap rerun

YAMNet improves materially on most classes that matter for SoundPulse:

- `fire_alarm`: recall `0.0312 -> 0.2969`
- `clock_alarm`: recall `0.5000 -> 0.8750`
- `doorbell`: recall `0.2941 -> 0.3824`
- `crying_baby`: recall `0.5000 -> 0.7368`
- `water_running`: recall `0.4375 -> 0.7969`
- `dog_barking`: recall `0.4219 -> 0.5625`
- `door_wood_creaking`: recall `0.1875 -> 0.4688`
- `speech`: recall `0.0938 -> 0.8281`
- `background`: recall `0.0781 -> 0.2344`

The main retained TinyCNN advantage is:

- `door_knock`: recall `0.6562 -> 0.2500`

## Historical note

The earlier notebook-output comparison remains useful as a historical run record, but it is no longer the canonical release comparison because the saved notebook runs used different test supports.

- Historical TinyCNN notebook support: `752`
- Historical YAMNet notebook support: `528`

Those historical summaries remain in `tinycnn_baseline_report.md` and `yamnet_kaggle_report.md`.

## Recommendation for the current project state

Use YAMNet as the leading baseline for threshold calibration, open-set behavior, and later export-handoff decisions. Keep TinyCNN as the lighter custom-model path for later tuning and device benchmarking.