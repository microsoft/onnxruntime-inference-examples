@echo off

REM Example: Set the environment variable (can be set outside this script too)
REM set "BASIC_PLUGIN_EP_LIBRARY_PATH=C:\path\to\onnxruntime_ep_basic.dll"

if "%BASIC_PLUGIN_EP_LIBRARY_PATH%"=="" (
    echo ERROR: BASIC_PLUGIN_EP_LIBRARY_PATH environment variable is not set.
    exit /b 1
)

if not exist "%BASIC_PLUGIN_EP_LIBRARY_PATH%" (
    echo ERROR: EP library "%BASIC_PLUGIN_EP_LIBRARY_PATH%" not found.
    exit /b 1
)

set "arch=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "arch=%PROCESSOR_ARCHITEW6432%"

if /i "%arch%"=="AMD64" (
    set "DEST_EP_DLL_FOLDER=.\Contoso.ML.OnnxRuntime.EP.Basic\runtimes\win-x64\native\"
) else if /i "%arch%"=="ARM64" (
    set "DEST_EP_DLL_FOLDER=.\Contoso.ML.OnnxRuntime.EP.Basic\runtimes\win-arm64\native\"
) else (
    echo ERROR: Unknown architecture "%arch%"
    exit /b 1
)


echo Copying EP DLL to "%DEST_EP_DLL_FOLDER%"
copy /Y "%BASIC_PLUGIN_EP_LIBRARY_PATH%" "%DEST_EP_DLL_FOLDER%" >nul

if errorlevel 1 (
    echo ERROR: Failed to EP library to "%DEST_EP_DLL_FOLDER%".
    exit /b 1
)

dotnet build .\Contoso.ML.OnnxRuntime.EP.Basic\Contoso.ML.OnnxRuntime.EP.Basic.csproj -c Debug
dotnet pack .\Contoso.ML.OnnxRuntime.EP.Basic\Contoso.ML.OnnxRuntime.EP.Basic.csproj -c Debug

set "LOCAL_FEED_FOLDER=local_feed"
if not exist "%LOCAL_FEED_FOLDER%" (
    mkdir "%LOCAL_FEED_FOLDER%" || (
        echo ERROR: Failed to create "%LOCAL_FEED_FOLDER%"
    )
)

copy /Y .\Contoso.ML.OnnxRuntime.EP.Basic\bin\Debug\Contoso.ML.OnnxRuntime.EP.Basic.1.0.0.* .\local_feed\
