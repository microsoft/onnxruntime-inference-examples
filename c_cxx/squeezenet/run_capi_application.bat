set ONNX_MODEL_URL="https://media.githubusercontent.com/media/onnx/models/main/archive/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
set ONNX_MODEL="squeezenet.onnx"
SET ONNXRUNTIME_ROOTDIR=%1
SET ORT_PACKAGE=%2
SET WORKSPACE=%3

echo The current directory is %CD%

7z.exe x %ORT_PACKAGE% -y
set ORT_LIB=%ORT_PACKAGE:~0,-4%\lib
echo %ORT_LIB%

cd %WORKSPACE%
cmake.exe -S . -B build\ -G "Visual Studio 17 2022" -DONNXRUNTIME_ROOTDIR=%ONNXRUNTIME_ROOTDIR%

REM Copy ORT libraries to same folder for linker to build.
REM For some reasons, setting "LINK" or "LIBPATH" env variables won't help. 
cd build
powershell -Command "cp %ORT_LIB%\* ."
for /f "tokens=*" %%a in ('"C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe" -latest -prerelease -products * -requires Microsoft.Component.MSBuild -find MSBuild\\**\\Bin\\MSBuild.exe') do (
    set MSBUILD_PATH=%%a
)
"%MSBUILD_PATH%" .\capi_test.sln /property:Configuration=Release

REM Copy ORT libraries to same folder for executable to run.
cd Release
powershell -Command "cp %ORT_LIB%\* ."
if exist "C:\local\models\opset8\test_squeezenet\model.onnx" (
    echo "Using local model"
    powershell -Command "cp C:\local\models\opset8\test_squeezenet\model.onnx %ONNX_MODEL%"
) else (
    powershell -Command "Invoke-WebRequest %ONNX_MODEL_URL% -Outfile %ONNX_MODEL%"
)
capi_test.exe
