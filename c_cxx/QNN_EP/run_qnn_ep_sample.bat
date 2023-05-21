ECHO OFF

set ONNX_MODEL_URL="https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
set ONNX_MODEL="squeezenet.onnx"

IF [%1] == [] GOTO Help

SET ORT_ROOT=%1
SET ORT_BIN=%2
SET WORKSPACE=%~dp0

IF NOT EXIST %ORT_ROOT%\include\onnxruntime_cxx_api.h (
    ECHO %ORT_ROOT%\include\onnxruntime_cxx_api.h not found
    GOTO Help
)

IF NOT EXIST %ORT_BIN%\onnxruntime.dll (
    ECHO %ORT_BIN%\onnxruntime.dll not found
    GOTO Help
)

pushd %WORKSPACE%
where /q cmake.exe
IF ERRORLEVEL 1 (
    ECHO Ensure Visual Studio 17 2022 is installed and open a VS Dev Cmd Prompt
    GOTO EXIT
)
REM build with Visual Studios 2022 ARM 64-bit Native
ECHO ON
cmake.exe -S . -B build\ -G "Visual Studio 17 2022" -DONNXRUNTIME_ROOTDIR=%ORT_ROOT%

REM Copy ORT libraries for linker to build.
cd build
copy /Y %ORT_BIN%\onnxruntime.lib .
MSBuild.exe .\qnn_ep_sample.sln /property:Configuration=Release

REM Copy ORT libraries for executable to run.
cd Release
copy /y %ORT_BIN%\onnxruntime.dll .
copy /y %ORT_BIN%\qnncpu.dll .
IF NOT EXIST %ONNX_MODEL% (
    powershell -Command "Invoke-WebRequest %ONNX_MODEL_URL% -Outfile %ONNX_MODEL%" )
qnn_ep_sample.exe

:EXIT
popd
exit /b

:HELP
ECHO HELP:    run_qnn_ep_sample.bat PATH_TO_ORT_ROOT_WITH_INCLUDE_FOLDER PATH_TO_ORT_BINARIES
ECHO Example: run_qnn_ep_sample.bat C:\src\onnxruntime C:\src\onnxruntime\build\Windows\RelWithDebInfo\RelWithDebInfo