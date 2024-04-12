@ECHO OFF

set ONNX_MODEL_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
set ONNX_MODEL="mobilenetv2-12.onnx"
set KITTEN_IMAGE_URL="https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
set KITTEN_IMAGE="images/kitten.jpg"
set LABEL_FILE_URL="https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
set LABEL_FILE="synset.txt"

IF [%1] == [] GOTO Help

SET ORT_ROOT=%1
SET ORT_BIN=%2
SET WORKSPACE=%~dp0

IF NOT EXIST %ORT_ROOT%\include\onnxruntime\core\session\onnxruntime_cxx_api.h ( REM Source
    IF NOT EXIST %ORT_ROOT%\include\onnxruntime_cxx_api.h (                      REM Drop
        ECHO onnxruntime_cxx_api.h in %ORT_ROOT%\include OR %ORT_ROOT%\include\onnxruntime\core\session NOT FOUND
        GOTO Help
    )
)

IF NOT EXIST %ORT_BIN%\onnxruntime.dll (
    ECHO %ORT_BIN%\onnxruntime.dll not found
    GOTO Help
)

pushd %WORKSPACE%

REM Download mobilenetv2-12 onnx model file
IF NOT EXIST %ONNX_MODEL% (
    powershell -Command "Invoke-WebRequest %ONNX_MODEL_URL% -Outfile %ONNX_MODEL%" )
REM Download kitten.jpg
IF NOT EXIST %KITTEN_IMAGE% (
    mkdir images
    powershell -Command "Invoke-WebRequest %KITTEN_IMAGE_URL% -Outfile %KITTEN_IMAGE%" )
REM Download label file
IF NOT EXIST %LABEL_FILE% (
    powershell -Command "Invoke-WebRequest %LABEL_FILE_URL% -Outfile %LABEL_FILE%" )

REM Generate QDQ model, fixed shape float32 model, fixed shape QDQ model, kitten_input.raw, and kitten_input_nhwc.raw
REM If there are issues installing python pkgs due to long paths see https://github.com/onnx/onnx/issues/5256
IF NOT EXIST mobilenetv2-12_shape.onnx (
    GOTO INSTALL_PYTHON_DEPS_AND_RUN_HELPER
) ELSE IF NOT EXIST kitten_input.raw (
    GOTO INSTALL_PYTHON_DEPS_AND_RUN_HELPER
) ELSE IF NOT EXIST kitten_input_nhwc.raw (
    GOTO INSTALL_PYTHON_DEPS_AND_RUN_HELPER
) ELSE (
    GOTO END_PYTHON
)

:INSTALL_PYTHON_DEPS_AND_RUN_HELPER
@ECHO ON
python -m pip install opencv-python
python -m pip install pillow
python -m pip install onnx
python -m pip install onnxruntime
python mobilenetv2_helper.py
@ECHO OFF
GOTO END_PYTHON

:END_PYTHON

REM Converter float32 model to float16 model
IF NOT EXIST mobilenetv2-12_shape_fp16.onnx (
    GOTO INSTALL_CONVERTER_AND_RUN
) ELSE (
    GOTO END_CONVERTER
)

:INSTALL_CONVERTER_AND_RUN
@ECHO ON
python -m pip install onnxconverter-common
python to_fp16.py
@ECHO OFF
GOTO END_CONVERTER

:END_CONVERTER

REM Download add_trans_cast.py file
set QNN_CTX_ONNX_GEN_SCRIPT_URL="https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/python/tools/qnn/gen_qnn_ctx_onnx_model.py"
set QNN_CTX_ONNX_GEN_SCRIPT="gen_qnn_ctx_onnx_model.py"
IF NOT EXIST %QNN_CTX_ONNX_GEN_SCRIPT% (
    powershell -Command "Invoke-WebRequest %QNN_CTX_ONNX_GEN_SCRIPT_URL% -Outfile %QNN_CTX_ONNX_GEN_SCRIPT%" )

REM based on the input & output information got from QNN converted mobilenetv2-12_net.json file
REM Generate mobilenetv2-12_net_qnn_ctx.onnx with content of libmobilenetv2-12.serialized.bin embedded
python gen_qnn_ctx_onnx_model.py -b libmobilenetv2-12.serialized.bin -q mobilenetv2-12_net.json

where /q cmake.exe
IF ERRORLEVEL 1 (
    ECHO Ensure Visual Studio 17 2022 is installed and open a VS Dev Cmd Prompt
    GOTO EXIT
)

REM build with Visual Studio 2022 
cmake.exe -S . -B build\ -G "Visual Studio 17 2022" -DONNXRUNTIME_BUILDDIR=%ORT_BIN% -DONNXRUNTIME_ROOTDIR=%ORT_ROOT% -DCMAKE_BUILD_TYPE=Release

REM Copy ORT libraries for linker to build - Target is ARM 64-bit Native
cd build
MSBuild.exe .\qnn_ep_sample.sln /property:Configuration=Release /p:Platform="ARM64"

REM Copy ORT libraries for executable to run - These deps required by QC documented in SDK docs/QNN/general/backend.html
cd Release
IF NOT EXIST %ORT_BIN%\QnnHtp.dll (
    ECHO Ensure QNN SDK Binaries are in %ORT_BIN%
    GOTO HELP
)
@ECHO ON
copy /y %ORT_BIN%\onnxruntime.dll .
copy /y %ORT_BIN%\qnncpu.dll .
copy /y %ORT_BIN%\QnnHtp.dll .
copy /y %ORT_BIN%\QnnHtpPrepare.dll .
copy /y %ORT_BIN%\QnnHtpV68Stub.dll .
IF EXIST %ORT_BIN%\QnnHtpV73Stub.dll (
    copy /y %ORT_BIN%\QnnHtpV73Stub.dll .
)
copy /y %ORT_BIN%\QnnSystem.dll .
copy /y %ORT_BIN%\libQnnHtpV68Skel.so .
IF EXIST %ORT_BIN%\libQnnHtpV73Skel.so (
    copy /y %ORT_BIN%\libQnnHtpV73Skel.so
)
IF EXIST %ORT_BIN%\libqnnhtpv73.cat (
    copy /y %ORT_BIN%\libqnnhtpv73.cat
)
copy /y ..\..\mobilenetv2-12_shape.onnx .
copy /y ..\..\mobilenetv2-12_quant_shape.onnx .
copy /y ..\..\mobilenetv2-12_net_qnn_ctx.onnx .
copy /y ..\..\mobilenetv2-12_shape_fp16.onnx .
copy /y ..\..\kitten_input.raw .
copy /y ..\..\kitten_input_nhwc.raw .
copy /y ..\..\synset.txt .

@ECHO ON
REM run mobilenetv2-12_shape.onnx with QNN CPU backend 
qnn_ep_sample.exe --cpu mobilenetv2-12_shape.onnx kitten_input.raw

REM run mobilenetv2-12_quant_shape.onnx with QNN HTP backend
qnn_ep_sample.exe --htp mobilenetv2-12_quant_shape.onnx kitten_input.raw

REM load mobilenetv2-12_quant_shape.onnx with QNN HTP backend, generate mobilenetv2-12_quant_shape.onnx_ctx.onnx which has QNN context binary embedded
REM This does not has to be run on real device with HTP, it can be done on x64 platform also, since it supports offline generation
qnn_ep_sample.exe --htp mobilenetv2-12_quant_shape.onnx kitten_input.raw --gen_ctx

REM TODO Check for mobilenetv2-12_quant_shape.onnx_ctx.onnx

IF EXIST mobilenetv2-12_quant_shape.onnx_ctx.onnx (
    REM run mobilenetv2-12_quant_shape.onnx_ctx.onnx with QNN HTP backend (generted from previous step)
    qnn_ep_sample.exe --htp mobilenetv2-12_quant_shape.onnx_ctx.onnx kitten_input.raw
) ELSE (
    ECHO mobilenetv2-12_quant_shape.onnx_ctx.onnx does not exist. It didn't get generated in previous step. Are you using ONNX 1.17+? or build from latest main branch
)


REM run mobilenetv2-12_net_qnn_ctx.onnx (generated from native QNN) with QNN HTP backend
qnn_ep_sample.exe --qnn mobilenetv2-12_net_qnn_ctx.onnx kitten_input_nhwc.raw

REM only works for v73 and higher
REM run mobilenetv2-12_shape.onnx (float32 model) with QNN HTP backend with FP16 precision
qnn_ep_sample.exe --fp32 mobilenetv2-12_shape.onnx kitten_input.raw

REM only works for v73 and higher
REM run mobilenetv2-12_shape_fp16.onnx (float16 model with float32 IO) with QNN HTP backend 
qnn_ep_sample.exe --fp16 mobilenetv2-12_shape_fp16.onnx kitten_input.raw

:EXIT
popd
exit /b

:HELP
popd
ECHO HELP:    run_qnn_ep_sample.bat PATH_TO_ORT_ROOT_WITH_INCLUDE_FOLDER PATH_TO_ORT_BINARIES_WITH_QNN
ECHO Example (Drop): run_qnn_ep_sample.bat %USERPROFILE%\Downloads\microsoft.ml.onnxruntime.qnn.1.17.0\build\native %USERPROFILE%\Downloads\microsoft.ml.onnxruntime.qnn.1.17.0\runtimes\win-arm64\native
ECHO Example (Src): run_qnn_ep_sample.bat C:\src\onnxruntime C:\src\onnxruntime\build\Windows\RelWithDebInfo\RelWithDebInfo