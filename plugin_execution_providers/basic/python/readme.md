# Basic Plugin Execution Provider with Python

## Contents
- `onnxruntime_ep_basic`: Contains files for the basic plugin EP Python package. `__init__.py` provides helper functions to get the EP library path and the EP name.
- `setup.py`: Script to generate the Python package wheel.
- `example_usage`: Contains a script showing example usage of the basic plugin EP Python Package.

## Build Instructions

### Build the native plugin EP library

Follow instructions [here](../readme.md#build-instructions) to build the native library.

### Build the Python package

Set the environment variable `BASIC_PLUGIN_EP_LIBRARY_PATH` to the path to the native plugin EP shared library. E.g., `basic_plugin_ep.dll` on Windows or `libbasic_plugin_ep.so` on Linux.

Run `setup.py` from this directory.

```
python setup.py bdist_wheel
```

The wheel will be generated in the `./dist` directory.

## Run the example usage script

Install the Python package wheel built in the previous step.

```
pip install ./dist/onnxruntime_ep_basic-0.1.0-<version and platform-specific text>.whl
```

Run the Python example usage script.

```
cd example_usage

# install other prerequisites
pip install onnxruntime numpy

python ./example_usage.py
```
