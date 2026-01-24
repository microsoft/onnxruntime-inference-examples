from setuptools import setup, find_packages
from setuptools.dist import Distribution
import os
import shutil

ep_dll = "TensorRTEp.dll"
src_folder = r".\build\\Debug"
dst_folder  = r".\\plugin_trt_ep\\libs"

class BinaryDistribution(Distribution):
    # This ensures wheel is marked as "non-pure" (has binary files)
    def has_ext_modules(self):
        return True
    
def copy_ep_dll(src_folder: str, dst_folder: str, ep_dll: str = "TensorRTEp.dll"):
    """
    Copy EP DLL from src_folder to dst_folder.
    Create dst_folder if it doesn't exist.
    """
    src_dll_path = os.path.join(src_folder, ep_dll)

    # Validate source file
    if not os.path.isfile(src_dll_path):
        raise FileNotFoundError(f"Source DLL not found: {src_dll_path}")

    # Create destination folder if needed
    os.makedirs(dst_folder, exist_ok=True)

    dst_dll_path = os.path.join(dst_folder, ep_dll)

    # Copy file
    shutil.copy2(src_dll_path, dst_dll_path)

    print(f"Copied {ep_dll} to: {dst_dll_path}")

try:
        copy_ep_dll(src_folder, dst_folder, ep_dll)
except Exception as e:
        print(f"Error: {e}")
    
setup(
    name="plugin_trt_ep",
    version="0.1.0",
    packages=["plugin_trt_ep"],
    include_package_data=True,  # include MANIFEST.in contents
    package_data={
        "plugin_trt_ep": ["libs/*.dll"],  # include DLLs inside the wheel
    },
    distclass=BinaryDistribution,
    description="Example package including DLLs",
    author="ORT",
    python_requires=">=3.8",
)
