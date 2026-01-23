import onnxruntime as ort
import onnxruntime_ep_basic as basic_ep
import numpy as np

from pathlib import Path

script_dir = Path(__file__).parent

# Path to the plugin EP library
ep_lib_path = basic_ep.get_library_path()
# Registration name can be anything the application chooses
ep_registration_name = "basic_ep_registration"

# Register plugin EP library with ONNX Runtime
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Create ORT session with explicit OrtEpDevice(s)

# Get EP name(s) from the plugin EP library
ep_names = basic_ep.get_ep_names()
# For this example we'll use the first one
ep_name = ep_names[0]

# Select an OrtEpDevice
# For this example, we'll use any OrtEpDevices matching our EP name
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_name]

assert len(selected_ep_devices) > 0

sess_options = ort.SessionOptions()

# EP-specific options
ep_options = {}

# Equivalent to the C API's SessionOptionsAppendExecutionProvider_V2 that appends the plugin EP to the session options
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

assert sess_options.has_providers() == True

# Create ORT session with the plugin EP
model_path = str(script_dir / "mul.onnx")
sess = ort.InferenceSession(model_path, sess_options=sess_options)

# Run the model
input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
print(f"input:\n{input}")
output = sess.run([], {'x': input, 'y': input})
print(f"output:\n{output[0]}")

del sess

# Unregister the library using the same registration name specified earlier
# Must only unregister a library after all sessions that use the library have been released
ort.unregister_execution_provider_library(ep_registration_name)
