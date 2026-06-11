import shutil
from pathlib import Path

# Roots
project_root = Path(__file__).resolve().parent
contracts_dir = project_root / "packages" / "shared_contracts" / "v1"
ml_export_models_dir = project_root / "ml" / "export" / "artifacts" / "generated" / "release_v1.0.0" / "models"
tinycnn_baseline_dir = project_root / "ml" / "export" / "artifacts" / "generated" / "tinycnn_baseline"
yamnet_kaggle_dir = project_root / "ml" / "export" / "artifacts" / "generated" / "yamnet_kaggle"

mobile_assets_dir = project_root / "mobile_app" / "assets"

print("Creating assets folders in mobile_app...")
# Target directories
assets_config_dir = mobile_assets_dir / "config"
assets_tiny_cnn_dir = mobile_assets_dir / "models" / "tiny_cnn"
assets_yamnet_dir = mobile_assets_dir / "models" / "yamnet"

assets_config_dir.mkdir(parents=True, exist_ok=True)
assets_tiny_cnn_dir.mkdir(parents=True, exist_ok=True)
assets_yamnet_dir.mkdir(parents=True, exist_ok=True)

# 1. Copy Shared Contracts to assets/config/
print("Copying shared contracts to assets/config/...")
shutil.copy2(contracts_dir / "class_labels.json", assets_config_dir / "class_labels.json")
shutil.copy2(contracts_dir / "preprocessing_config.json", assets_config_dir / "preprocessing_config.json")
shutil.copy2(contracts_dir / "threshold_defaults.json", assets_config_dir / "threshold_defaults.json")

# 2. Copy TinyCNN assets
print("Copying TinyCNN assets...")
shutil.copy2(ml_export_models_dir / "tinycnn_embedding.tflite", assets_tiny_cnn_dir / "tinycnn_embedding.tflite")
shutil.copy2(tinycnn_baseline_dir / "prototype_store.json", assets_tiny_cnn_dir / "prototype_store.json")
shutil.copy2(contracts_dir / "tinycnn_threshold_defaults.json", assets_tiny_cnn_dir / "threshold_defaults.json")

# 3. Copy YAMNet assets
print("Copying YAMNet assets...")
shutil.copy2(ml_export_models_dir / "yamnet_embedding.tflite", assets_yamnet_dir / "yamnet_embedding.tflite")
shutil.copy2(yamnet_kaggle_dir / "prototype_store.json", assets_yamnet_dir / "prototype_store.json")
shutil.copy2(contracts_dir / "yamnet_threshold_defaults.json", assets_yamnet_dir / "threshold_defaults.json")

# Also copy yamnet_embedding.tflite to yamnet_kaggle for notebook exports consistency
shutil.copy2(ml_export_models_dir / "yamnet_embedding.tflite", yamnet_kaggle_dir / "yamnet_embedding.tflite")

print("All assets successfully copied to mobile_app/assets!")
