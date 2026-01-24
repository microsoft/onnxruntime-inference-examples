import os
import importlib.resources
import ctypes
import onnxruntime as ort

ort_dir = os.path.dirname(os.path.abspath(ort.__file__))
dll_path = os.path.join(ort_dir, "capi", "onnxruntime.dll")

# When the application calls ort.register_execution_provider_library() with the path to the plugin EP DLL,
# ORT internally uses LoadLibraryExW() to load that DLL. Since the plugin EP depends on onnxruntime.dll,
# the operating system will attempt to locate and load onnxruntime.dll first.
#
# On Windows, LoadLibraryExW() searches the directory containing the plugin EP DLL before searching system directories.
# Because onnxruntime.dll is not located in the plugin EP’s directory, Windows ends up loading the copy from a 
# system directory instead — which is not the correct version.
#
# To ensure the plugin EP uses the correct onnxruntime.dll bundled with the ONNX Runtime package, 
# we load that DLL explicitly before loading the plugin EP DLL.
ctypes.WinDLL(dll_path)

def get_path(filename: str = "TensorRTEp.dll") -> str:
    """
    Returns the absolute filesystem path to a DLL (or any file)
    packaged inside plugin_trt_ep/libs.
    """
    package = __name__ + ".libs"
    with importlib.resources.as_file(importlib.resources.files(package) / filename) as path:
        return str(path)