import onnxruntime as onnxrt
import plugin_trt_ep
import numpy as np

# Path to the plugin EP library
ep_lib_path = plugin_trt_ep.get_path()
# Registration name can be anything the application chooses
ep_registration_name = "TensorRTEp"
# EP name should match the name assigned by the EP factory when creating the EP (i.e., in the implementation of OrtEP::CreateEp)
ep_name = ep_registration_name

# Register plugin EP library with ONNX Runtime
onnxrt.register_execution_provider_library(ep_registration_name, ep_lib_path)

#
# Create ORT session with explicit OrtEpDevice(s)
#

# Find the OrtEpDevice for "TensorRTEp"
ep_devices = onnxrt.get_ep_devices()
trt_ep_device = None
for ep_device in ep_devices:
    if ep_device.ep_name == ep_name:
        trt_ep_device = ep_device

assert trt_ep_device != None

sess_options = onnxrt.SessionOptions()

# Equivalent to the C API's SessionOptionsAppendExecutionProvider_V2 that appends "TensorRTEp" to ORT session option
sess_options.add_provider_for_devices([trt_ep_device], {'trt_engine_cache_enable': '1'})

assert sess_options.has_providers() == True

# Create ORT session with "TensorRTEp" plugin EP
sess = onnxrt.InferenceSession("C:\\models\\mul_1.onnx", sess_options=sess_options)

# Run sample model and check output
x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
input_name = sess.get_inputs()[0].name
res = sess.run([], {input_name: x})
output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

# Unregister the library using the application-specified registration name.
# Must only unregister a library after all sessions that use the library have been released.
onnxrt.unregister_execution_provider_library(ep_registration_name)


# Note:
# The mul_1.onnx can be found here:
# https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/testdata/mul_1.onnx