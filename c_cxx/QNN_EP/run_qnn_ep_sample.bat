set ONNX_MODEL_URL="https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
set ONNX_MODEL="squeezenet.onnx"
SET ORT_ROOT=%1
SET WORKSPACE=%2

set ORT_LIB=%ORT_ROOT%\lib
echo %ORT_LIB%

cd %WORKSPACE%
REM build with Visual Studios 2022 ARM 64-bit Native
cmake.exe -S . -B build\ -G "Visual Studio 17 2022" -DONNXRUNTIME_ROOTDIR=%ORT_ROOT%

REM Copy ORT libraries for linker to build.
cd build
powershell -Command "cp %ORT_LIB%\* ."
MSBuild.exe .\qnn_ep_sample.sln /property:Configuration=Release

REM Copy ORT libraries for executable to run.
cd Release
powershell -Command "cp %ORT_LIB%\* ."
powershell -Command "Invoke-WebRequest %ONNX_MODEL_URL% -Outfile %ONNX_MODEL%"
qnn_ep_sample.exe
