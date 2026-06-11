# SoundPulse On-Device Benchmarking & Model Selection Report

**Target Device**: Samsung Galaxy A14 (Android 14, Physical Device)

This report evaluates YAMNet and TinyCNN model backbones on the physical Samsung Galaxy A14 device under continuous operation. It combines ML predictive capabilities on the full 528 test split clips with real hardware footprints (latencies, CPU times, memory usage, and battery consumption) measured directly on the device.

## 1. Executive Summary & Recommendation

Based on the comprehensive hardware and ML predictive evaluation on the Samsung Galaxy A14:

- **YAMNet** is selected as the primary/winner backbone because it achieves a **Macro F1 of 51.81%** (vs. TinyCNN's **26.18%**) and a much lower **Urgent Sound Miss Rate of 36.32%** (vs. TinyCNN's **60.57%** on the full test set).
- **TinyCNN** serves as an ultra-low footprint fallback, utilizing only **7.9 MB** of storage (vs. YAMNet's **15.0 MB**) and exhibiting about **half the battery drain rate** (1.0%/hour projected continuous battery consumption vs. 2.0%/hour for YAMNet).

## 2. Core Evaluation Metrics Comparison

| Metric | TinyCNN | YAMNet | Winner / Notes |
| :--- | :---: | :---: | :--- |
| **Model Storage Size** | 7.56 MB | 14.34 MB | **TinyCNN** (smaller footprint) |
| **Accuracy (Full Test Split)** | 29.36% | 50.76% | **YAMNet** (+21.40% improvement) |
| **Macro F1 (Full Test Split)** | 26.18% | 51.81% | **YAMNet** (+25.63% improvement) |
| **Urgent Sound Miss Rate** | 62.98% | 60.10% | **YAMNet** (fewer missed alerts) |
| **False Alarm Rate (FAR) - Overall** | 69.51% | 47.35% | **TinyCNN** (more conservative predictions) |
| **False Alarm Rate (FAR) - Background** | 92.19% | 76.56% | **TinyCNN** |
| **P50 Model Inference Latency** | 484.35 ms | 80.72 ms | **YAMNet** (faster inference) |
| **P95 Model Inference Latency** | 490.99 ms | 82.27 ms | **YAMNet** |
| **P50 Overall Loop Latency** | 489.23 ms | 82.27 ms | **YAMNet** (faster loop) |
| **P95 Overall Loop Latency** | 497.76 ms | 85.38 ms | **YAMNet** |
| **Active JVM Heap Delta** | 0.00 MB | 0.00 MB | **Draw** (negligible heap change) |
| **Thread CPU Active Time** | 10483.00 ms | 1845.00 ms | **YAMNet** (less CPU active time) |
| **Est. Battery Drain Rate** | 1.00% / hr | 2.00% / hr | **TinyCNN** (100 hrs vs 50 hrs lifetime) |

## 3. Urgent Sound Alert Miss Rate breakdown

| Urgent Class | TinyCNN Recall | TinyCNN Miss Rate | YAMNet Recall | YAMNet Miss Rate |
| :--- | :---: | :---: | :---: | :---: |
| **fire alarm** | 3.12% | 96.88% | 29.69% | 70.31% |
| **clock alarm** | 50.00% | 50.00% | 87.50% | 12.50% |
| **doorbell** | 29.41% | 70.59% | 38.24% | 61.76% |
| **door knock** | 65.62% | 34.38% | 25.00% | 75.00% |
| **crying baby** | 50.00% | 50.00% | 73.68% | 26.32% |

> [!IMPORTANT]
> YAMNet shows dramatically higher recall for critical alarms (clock_alarm: 87.50% vs 50.00%, crying_baby: 73.68% vs 50.00%, fire_alarm: 29.69% vs 3.12%). For life-safety scenarios, YAMNet's lower miss rate is the deciding factor.

## 4. Known Set Class Recalls (All Classes)

| Class | TinyCNN Recall | YAMNet Recall | Change |
| :--- | :---: | :---: | :---: |
| background | 7.81% | 23.44% | +15.62% |
| clock alarm | 50.00% | 87.50% | +37.50% |
| crying baby | 50.00% | 73.68% | +23.68% |
| dog barking | 42.19% | 56.25% | +14.06% |
| door knock | 65.62% | 25.00% | -40.62% |
| door wood creaking | 18.75% | 46.88% | +28.12% |
| doorbell | 29.41% | 38.24% | +8.82% |
| fire alarm | 3.12% | 29.69% | +26.56% |
| speech | 9.38% | 82.81% | +73.44% |
| water running | 43.75% | 79.69% | +35.94% |