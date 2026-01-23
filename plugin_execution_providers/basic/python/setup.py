from setuptools import setup
from setuptools.dist import Distribution
import os
from pathlib import Path
import shutil

script_dir = Path(__file__).parent

ep_lib_path_str = os.environ.get("BASIC_PLUGIN_EP_LIBRARY_PATH")
assert ep_lib_path_str is not None, "BASIC_PLUGIN_EP_LIBRARY_PATH must be set to the EP library path."

ep_lib_path = Path(ep_lib_path_str)
assert ep_lib_path.is_file(), f"EP library path is not a file: {ep_lib_path}"

dst_libs_dir = script_dir / "onnxruntime_ep_basic"

# copy EP library file to libs dir
dst_libs_dir.mkdir(exist_ok=True)
shutil.copyfile(ep_lib_path, dst_libs_dir / ep_lib_path.name)

class BinaryDistribution(Distribution):
    # This ensures wheel is marked as "non-pure" (has binary files)
    def has_ext_modules(self):
        return True

setup(
    name="onnxruntime-ep-basic",
    version="0.1.0",
    packages=["onnxruntime_ep_basic"],
    include_package_data=True,  # include MANIFEST.in contents
    package_data={
        "onnxruntime_ep_basic": ["*.dll", "*.so"],  # include shared libraries
    },
    distclass=BinaryDistribution,
    description="ONNX Runtime Basic Plugin Execution Provider",
    author="ORT",
    python_requires=">=3.8",
)
