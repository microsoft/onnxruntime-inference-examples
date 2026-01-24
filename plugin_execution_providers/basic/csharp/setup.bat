@echo off

REM Builds NuGet package that wraps basic plugin EP

IF "%~1"=="" (
    echo ERROR: No build configuration specified.
    echo Usage: .\setup.bat [Debug^|Release]
    exit /b 1
)

SET "BUILD_CONFIG=%~1"

if "%BASIC_PLUGIN_EP_LIBRARY_PATH%"=="" (
    echo ERROR: BASIC_PLUGIN_EP_LIBRARY_PATH environment variable is not set.
    exit /b 1
)

if not exist "%BASIC_PLUGIN_EP_LIBRARY_PATH%" (
    echo ERROR: EP library "%BASIC_PLUGIN_EP_LIBRARY_PATH%" not found.
    exit /b 1
)

set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"

if /i "%ARCH%"=="AMD64" (
    set "DEST_EP_DLL_FOLDER=.\Contoso.ML.OnnxRuntime.EP.Basic\runtimes\win-x64\native\"
) else if /i "%ARCH%"=="ARM64" (
    set "DEST_EP_DLL_FOLDER=.\Contoso.ML.OnnxRuntime.EP.Basic\runtimes\win-arm64\native\"
) else (
    echo ERROR: Unknown architecture "%ARCH%"
    exit /b 1
)

if not exist "%DEST_EP_DLL_FOLDER%" (
    mkdir "%DEST_EP_DLL_FOLDER%" || (
        echo ERROR: Failed to create "%DEST_EP_DLL_FOLDER%".
        exit /b 1
    )
)

echo Copying EP DLL to "%DEST_EP_DLL_FOLDER%"
copy /Y "%BASIC_PLUGIN_EP_LIBRARY_PATH%" "%DEST_EP_DLL_FOLDER%" >nul

if errorlevel 1 (
    echo ERROR: Failed to copy EP library to "%DEST_EP_DLL_FOLDER%".
    exit /b 1
)

echo Building NuGet package ("%BUILD_CONFIG%") ...
dotnet build .\Contoso.ML.OnnxRuntime.EP.Basic\Contoso.ML.OnnxRuntime.EP.Basic.csproj -c "%BUILD_CONFIG%"
dotnet pack .\Contoso.ML.OnnxRuntime.EP.Basic\Contoso.ML.OnnxRuntime.EP.Basic.csproj -c "%BUILD_CONFIG%"

set "LOCAL_FEED_FOLDER=local_feed"
if not exist "%LOCAL_FEED_FOLDER%" (
    mkdir "%LOCAL_FEED_FOLDER%" || (
        echo ERROR: Failed to create "%LOCAL_FEED_FOLDER%"
        exit /b 1
    )
)

copy /Y .\Contoso.ML.OnnxRuntime.EP.Basic\bin\"%BUILD_CONFIG%"\Contoso.ML.OnnxRuntime.EP.Basic.*.nupkg .\local_feed\
copy /Y .\Contoso.ML.OnnxRuntime.EP.Basic\bin\"%BUILD_CONFIG%"\Contoso.ML.OnnxRuntime.EP.Basic.*.snupkg .\local_feed\
