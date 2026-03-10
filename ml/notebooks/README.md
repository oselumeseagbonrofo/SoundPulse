# SoundPulse Notebook Workflow

Run these notebooks in order:

1. `01_dataset_curation_and_manifest.ipynb`
2. `02_tinycnn_mobilenetv4convsmall_training.ipynb`
3. `03_yamnet_kaggle_pipeline.ipynb`

Notes:

- Notebook 1 builds the curated manifest consumed by the later notebooks.
- Notebook 2 uses TensorFlow Official `MobileNetV4ConvSmall` as the Tiny CNN replacement backbone.
- Notebook 3 assumes Kaggle credentials are configured and downloads YAMNet from Kaggle before embedding extraction and prototype evaluation.
