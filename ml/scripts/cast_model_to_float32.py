import tensorflow as tf
from pathlib import Path
import shutil

project_root = Path(__file__).resolve().parents[2]
model_dir = project_root / "ml" / "export" / "artifacts" / "generated" / "mobilenetv4_convsmall_10"
saved_model_path = model_dir / "saved_model"
target_dir = project_root / "ml" / "export" / "artifacts" / "generated" / "tinycnn_baseline"

print("Loading SavedModel...")
model = tf.saved_model.load(str(saved_model_path))

# Get the serving signature
serving_fn = model.signatures["serving_default"]
print("Original Signature structured inputs:", serving_fn.structured_input_signature)
print("Original Signature structured outputs:", serving_fn.structured_outputs)

# Create wrapper module
class Float32Model(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Track the original model variables
        self.serving_fn = model.signatures["serving_default"]

    @tf.function
    def __call__(self, log_mel):
        # Forward inputs using keyword arguments matching the original signature
        outputs = self.serving_fn(log_mel=log_mel)
        # Extract the output tensor
        output_tensor = list(outputs.values())[0]
        # Cast to float32
        return tf.cast(output_tensor, tf.float32)

# Instantiate wrapper with the full loaded model
wrapper = Float32Model(model)

# Get target spec from original input spec
original_spec = list(serving_fn.structured_input_signature[1].values())[0]
print(f"Original input spec: {original_spec}")

# Create concrete function with dynamic batch size matching original
concrete_fn = wrapper.__call__.get_concrete_function(
    tf.TensorSpec(shape=original_spec.shape, dtype=original_spec.dtype, name="log_mel")
)

# Export to temp SavedModel
temp_saved_model_dir = model_dir / "temp_saved_model_f32"
if temp_saved_model_dir.exists():
    shutil.rmtree(temp_saved_model_dir)

print("Saving float32 wrapper SavedModel...")
tf.saved_model.save(wrapper, str(temp_saved_model_dir), signatures={"serving_default": concrete_fn})

# Convert to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_saved_model_dir))
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

# Save TFLite model
target_dir.mkdir(parents=True, exist_ok=True)
target_tflite_path = target_dir / "tinycnn_embedding.tflite"
target_tflite_path.write_bytes(tflite_model)
print(f"Saved float32 TFLite model to {target_tflite_path}")

# Clean up
shutil.rmtree(temp_saved_model_dir)

# Verify
interpreter = tf.lite.Interpreter(model_path=str(target_tflite_path))
print("\nNew Input Details:")
for detail in interpreter.get_input_details():
    print(detail)
print("\nNew Output Details:")
for detail in interpreter.get_output_details():
    print(detail)
