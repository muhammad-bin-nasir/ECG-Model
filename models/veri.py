import onnxruntime as ort
import numpy as np

# 1. Load the ONNX model
print("Loading ONNX model...")
try:
    # Create an inference session
    session = ort.InferenceSession("ecg_model.onnx")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Check Input Requirements
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Model expects Input Name: '{input_name}'")
print(f"Model expects Input Shape: {input_shape}")

# 3. Create Dummy Data (Simulating 10 seconds of ECG)
# Shape: (Batch=1, Sequence=2500, Features=1)
dummy_input = np.random.randn(1, 2500, 1).astype(np.float32)

# 4. Run Inference (Prediction)
print("Running test prediction...")
outputs = session.run(None, {input_name: dummy_input})

# 5. Check Output
reconstructed_data = outputs[0]
print("Prediction successful!")
print(f"Output Shape: {reconstructed_data.shape}")

# Simple check: Does output shape match input shape?
if reconstructed_data.shape == (1, 2500, 1):
    print("\n✅ SUCCESS: The ONNX model is healthy and ready for Android.")
else:
    print("\n❌ WARNING: Output shape mismatch. Something is wrong.")