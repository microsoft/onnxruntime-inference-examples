import onnxruntime as onnxrt
import numpy as np
                                                                                                                                                                                                                                                                      
ep_lib_path = "C:\\path\\to\\plugin_trt_ep\\TensorRTEp.dll"
ep_name = "TensorRTEp"
ep_registration_name = ep_name

onnxrt.register_execution_provider_library(ep_registration_name, ep_lib_path)

ep_devices = onnxrt.get_ep_devices()
trt_ep_device = None
for ep_device in ep_devices:
    if ep_device.ep_name == ep_name:
        trt_ep_device = ep_device

assert trt_ep_device != None                                                                                                                                                                                                                                                            
sess_options = onnxrt.SessionOptions()
sess_options.add_provider_for_devices([trt_ep_device], {'trt_engine_cache_enable': '1'})

assert sess_options.has_providers() == True

# Run sample model and check output
sess = onnxrt.InferenceSession("C:\\modles\\mul_1.onnx", sess_options=sess_options)

x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
input_name = sess.get_inputs()[0].name
res = sess.run([], {input_name: x})
output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

onnxrt.unregister_execution_provider_library(ep_registration_name)