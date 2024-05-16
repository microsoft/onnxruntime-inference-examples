@ECHO OFF

SET WORKSPACE=%~dp0

where /q cmake.exe
IF ERRORLEVEL 1 (
    ECHO Ensure Visual Studio 17 2022 is installed and open a VS Dev Cmd Prompt
    GOTO END
)

IF "%1" == "" (
  ECHO Must provide the path to the ONNX Runtime root directory. The QNN SDK directory is optional.
  GOTO HELP
)

IF NOT EXIST %1 (
  ECHO %1 does not exist
  GOTO HELP
)

SET fullpath_1=%~f1
SET fullpath_2=%~f2

pushd %WORKSPACE%

mkdir build 2>nul

REM If user provided a NuGet package, extract it and create an ONNX Runtime package.
IF "%~x1" == ".nupkg" (
  pushd build
  IF EXIST _ort_extracted_nuget (
    ECHO Removing previously extracted nuget folder
    rmdir _ort_extracted_nuget /S /Q
  )

  IF EXIST _generated_onnxruntime_rootdir (
    ECHO Removing previously generated ONNX Runtime root directory
    rmdir _generated_onnxruntime_rootdir /S /Q
  )

  ECHO Extracting nuget file %fullpath_1%
  copy /y "%fullpath_1%" _nuget_as_zip.zip
  powershell -Command "Expand-Archive -Path _nuget_as_zip.zip -DestinationPath _ort_extracted_nuget"

  mkdir _generated_onnxruntime_rootdir 2>nul
  mkdir _generated_onnxruntime_rootdir\include 2>nul
  mkdir _generated_onnxruntime_rootdir\lib 2>nul
  mkdir _generated_onnxruntime_rootdir\bin 2>nul
  copy /y _ort_extracted_nuget\build\native\include\* _generated_onnxruntime_rootdir\include\
  IF "%PROCESSOR_ARCHITECTURE%" == "ARM64" (
    copy /y _ort_extracted_nuget\runtimes\win-arm64\native\* _generated_onnxruntime_rootdir\lib\
    copy /y _ort_extracted_nuget\runtimes\win-arm64\native\* _generated_onnxruntime_rootdir\bin\
  ) ELSE (
    copy /y _ort_extracted_nuget\runtimes\win-x64\native\* _generated_onnxruntime_rootdir\lib\
    copy /y _ort_extracted_nuget\runtimes\win-x64\native\* _generated_onnxruntime_rootdir\bin\
  )
  copy /y _ort_extracted_nuget\*.txt _generated_onnxruntime_rootdir\
  copy /y _ort_extracted_nuget\*.pdf _generated_onnxruntime_rootdir\
  copy /y _ort_extracted_nuget\LICENSE _generated_onnxruntime_rootdir\
  copy /y _ort_extracted_nuget\README.md _generated_onnxruntime_rootdir\
  copy /y _ort_extracted_nuget\Privacy.md _generated_onnxruntime_rootdir\
  SET ONNXRUNTIME_ROOTDIR=%WORKSPACE%build\_generated_onnxruntime_rootdir
  popd
) ELSE (
  SET ONNXRUNTIME_ROOTDIR=%fullpath_1%
)

SET cmake_args=-DONNXRUNTIME_ROOTDIR=%ONNXRUNTIME_ROOTDIR%

IF NOT "%2" == "" (
  SET cmake_args=%cmake_args% -DQNN_SDK_ROOTDIR=%fullpath_2%
)

IF NOT "%3" == "" (
  SET cmake_args=%cmake_args% -DQNN_HEXAGON_ARCH_VERSION=%3%
)

REM Configure build with CMake
cmake.exe -S . -B build\ -G "Visual Studio 17 2022" %cmake_args% -DCMAKE_BUILD_TYPE=Release

REM Compile build\Release\accuracy_test.exe
cd build
msbuild onnxruntime_accuracy_test.sln /p:Configuration=Release

popd
exit /b

:HELP
ECHO Usage: build.bat ONNXRUNTIME_ROOTDIR [QNN_SDK_ROOTDIR] [QNN_HEXAGON_ARCH_VERSION]
ECHO Example:                  build.bat 'C:\Program Files\onnxruntime'
ECHO Example w/ NuGet package: build.bat '.\microsoft.ml.onnxruntime.qnn.1.16.0.nupkg'
ECHO Example w/ QNN:           build.bat 'C:\Program Files\onnxruntime' 'C:\Qualcomm\AIStack\QNN\2.17.0.231124' 68
:END
