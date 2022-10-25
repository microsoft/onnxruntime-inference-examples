## How to run the application
1.  (Linux) Run ```run_capi_application.sh```, indicate onnxruntime library tarball with -p argument and indicate current onnxruntime-inference-examples/c_cxx/squeezenet directory with -w argument.
```
./run_capi_application.sh -p /home/azureuser/onnxruntime-linux-x64-gpu-1.13.1.tgz -w /home/azureuser/repos/onnxruntime-inference-examples/c_cxx/squeezenet
``` 
2. (Windows) Run ```run_capi_application.bat``` with onnxruntime library zip file and current onnxruntime-inference-examples\c_cxx\squeezenet directory
```
.\run_capi_application.bat D:\onnxruntime-win-x64-gpu-1.13.1.zip D:\repos\onnxruntime-inference-examples\c_cxx\squeezenet
```
3. Or you can manully run cmake and remember to download squeezenet onnx model.
