# build_and_test_trt_ep.ps1

# Change to the directory where this script is located
Set-Location -Path $PSScriptRoot
Write-Host "Current directory set to: $PSScriptRoot"

# Stop on first error
$ErrorActionPreference = "Stop"

# Variables
$SourceDir = "../"
$BuildDir = "./"
$BuildType = "Debug"

# ORT settings
$OrtVersion = "1.23.1"
#$OrtZipUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-gpu-$OrtVersion.zip"
$OrtZipUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-$OrtVersion.zip"
$OrtZipPath = "onnxruntime.zip"
$OrtHome = ".\ort_package"

# Step 1: Download ONNX Runtime package
if (!(Test-Path $OrtHome)) {
    Write-Host "=== Downloading ONNX Runtime $OrtVersion ==="
    Invoke-WebRequest -Uri $OrtZipUrl -OutFile $OrtZipPath

    Write-Host "=== Extracting ONNX Runtime to $OrtHome ==="
    Expand-Archive -Path $OrtZipPath -DestinationPath $OrtHome -Force

    # Clean up zip file
    Remove-Item $OrtZipPath
} else {
    Write-Host "ONNX Runtime directory already exists. Skipping download."
}

# Step 2: Configure CMake
$buildDir = "build"
if (-Not (Test-Path $buildDir)) {
    Write-Host "Creating build directory..."
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}
Set-Location -Path $buildDir

Write-Host "=== Running CMake configure step ==="
$OrtHome = "$PSScriptRoot\ort_package\onnxruntime-win-x64-$OrtVersion"
cmake "-S" $SourceDir `
      "-B" $BuildDir `
      "-DCMAKE_BUILD_TYPE=$BuildType" `
      "-DORT_HOME=$OrtHome"

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed!"
    exit 1
}

# Step 3: Build
Write-Host "=== Building project ==="
cmake --build $BuildDir --config $BuildType --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed!"
    exit 1
}

Write-Host "=== Build completed successfully! ==="
