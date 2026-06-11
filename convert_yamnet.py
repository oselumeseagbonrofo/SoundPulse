import os
import tensorflow as tf
from pathlib import Path

# Paths
project_root = Path(__file__).resolve().parent
saved_model_dir = project_root / "ml" / "export" / "artifacts" / "generated" / "release_v1.0.0" / "models" / "yamnet_saved_model"
tflite_output_path = project_root / "ml" / "export" / "artifacts" / "generated" / "release_v1.0.0" / "models" / "yamnet_embedding.tflite"

print(f"SavedModel directory: {saved_model_dir}")
print(f"Target TFLite path: {tflite_output_path}")

if not saved_model_dir.exists():
    raise FileNotFoundError(f"SavedModel directory not found at {saved_model_dir}")

# Load and convert
print("Initializing TFLiteConverter from SavedModel...")
converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

# Enable SELECT_TF_OPS for YAMNet, as YAMNet uses complex operations
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

print("Converting...")
tflite_model = converter.convert()

# Save
tflite_output_path.parent.mkdir(parents=True, exist_ok=True)
tflite_output_path.write_bytes(tflite_model)
print("Conversion complete!")
