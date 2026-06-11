# MobileNetV4 Threshold Calibration Report

## Protocol

Threshold calibration was rerun on the MobileNetV4 prototype path using out-of-scope FSD50K clips.

- Known caps: train=128, validation=64, test=64
- Unknown calibration proxy: FSD50K dev out-of-scope, support=1024, categories=86
- Unknown holdout proxy: FSD50K eval out-of-scope, support=1024, categories=93
- Unknown cap policy: max_total=1024/1024, max_per_category=12
- Urgent-release metric classes: fire_alarm, clock_alarm, door_knock, doorbell

## Selected thresholds

- Distance threshold: 0.7941
- Unknown ratio threshold: 0.9945
- Loudness floor dBFS: -55.0

## Operating-point comparison on holdout

| Operating point | Accuracy | Macro F1 | Urgent recall | Unknown non-alert | Unknown recall | Unknown precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected_release_candidate | 0.2177 | 0.1798 | 0.1804 | 0.1211 | 0.1025 | 0.7447 |
| best_balanced_score | 0.0647 | 0.0742 | 0.0425 | 0.8486 | 0.8428 | 0.6965 |
| best_unknown_rejection | 0.1918 | 0.1750 | 0.1574 | 0.3096 | 0.2109 | 0.7347 |

## Baseline vs selected on known test

| Metric | Baseline | Selected | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.2328 | 0.2177 | -0.0151 |
| Macro F1 | 0.1860 | 0.1798 | -0.0062 |
| Macro recall | 0.2086 | 0.1966 | -0.0120 |
| Urgent recall | 0.1960 | 0.1804 | -0.0156 |
| Background non-alert rate | 0.0000 | 0.0000 | +0.0000 |

## Selected open-set holdout results

- Unknown precision: 0.7447
- Unknown recall: 0.1025
- Unknown non-alert rate: 0.1211
- Unknown false-alert rate: 0.8789

## Top sweep candidates on calibration

| Candidate ID | Distance | Ratio | Loudness | Balanced | Val macro F1 | Val urgent recall | Unknown non-alert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 361 | 0.5909 | 0.8500 | -60.0 | 0.3358 | 0.0833 | 0.0762 | 0.8496 |
| 481 | 0.6268 | 0.8500 | -60.0 | 0.3355 | 0.0860 | 0.0813 | 0.8398 |
| 362 | 0.5909 | 0.8500 | -58.5 | 0.3348 | 0.0833 | 0.0762 | 0.8496 |
| 482 | 0.6268 | 0.8500 | -58.5 | 0.3345 | 0.0860 | 0.0813 | 0.8398 |
| 363 | 0.5909 | 0.8500 | -55.0 | 0.3342 | 0.0833 | 0.0762 | 0.8535 |
| 483 | 0.6268 | 0.8500 | -55.0 | 0.3340 | 0.0860 | 0.0813 | 0.8438 |
| 241 | 0.5610 | 0.8500 | -60.0 | 0.3332 | 0.0780 | 0.0663 | 0.8574 |
| 364 | 0.5909 | 0.8500 | -52.0 | 0.3330 | 0.0833 | 0.0762 | 0.8555 |
| 721 | 0.7419 | 0.8500 | -60.0 | 0.3329 | 0.0881 | 0.0813 | 0.8262 |
| 601 | 0.6747 | 0.8500 | -60.0 | 0.3329 | 0.0852 | 0.0813 | 0.8311 |

## Holdout routing breakdown

- Selected known-test routed labels: {"background": 3, "clock_alarm": 13, "crying_baby": 86, "dog_barking": 20, "door_knock": 36, "door_wood_creaking": 28, "doorbell": 31, "fire_alarm": 35, "speech": 33, "unknown": 36, "water_running": 143}
- Selected unknown-holdout routed labels: {"background": 19, "clock_alarm": 45, "crying_baby": 143, "dog_barking": 25, "door_knock": 116, "door_wood_creaking": 65, "doorbell": 38, "fire_alarm": 178, "speech": 52, "unknown": 105, "water_running": 238}

## Assessment

This is a MobileNetV4 calibration artifact. The sweep export makes the tradeoff between known retention and unknown rejection explicit.

Hard floors applied during selection: urgent recall ≥ 0.1, macro F1 ≥ 0.25.

The selected operating point does not yet meet the PRD urgent-recall target of 0.90. Current urgent recall is 0.1804.

## Top holdout false alerts

- Cowbell -> door_knock: 10
- Fill_(with_liquid) -> water_running: 10
- Car_passing_by -> water_running: 9
- Fixed-wing_aircraft_and_airplane -> water_running: 9
- Female_singing -> crying_baby: 8
- Electric_guitar -> fire_alarm: 8
- Hiss -> water_running: 7
- Bass_drum -> door_knock: 7
- Cheering -> crying_baby: 7
- Chirp_and_tweet -> water_running: 7
- Gull_and_seagull -> water_running: 7
- Gurgling -> water_running: 7
- Gong -> fire_alarm: 7
- Boiling -> water_running: 6
- Harp -> fire_alarm: 6
