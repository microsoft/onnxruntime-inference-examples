# PowerShell script to push Phi-3 model files to Android device
# Usage: .\push_model.ps1 -modelDir "path\to\model\directory"

param (
    [Parameter(Mandatory=$true)]
    [string]$modelDir
)

Write-Host "Pushing Phi-3 model files to device..."
Write-Host "Model directory: $modelDir"

# Check if the directory exists
if (-not (Test-Path -Path $modelDir)) {
    Write-Host "ERROR: Model directory does not exist: $modelDir" -ForegroundColor Red
    exit 1
}

# Check if adb is available
try {
    $adbVersion = adb version
    Write-Host "Using ADB: $adbVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: ADB command not found. Make sure Android SDK platform-tools are in your PATH." -ForegroundColor Red
    exit 1
}

# Create directory on device
Write-Host "Creating directory on device..." -ForegroundColor Yellow
adb shell mkdir -p /sdcard/phi-3-model/

# Find ONNX files in the directory
$onnxFiles = Get-ChildItem -Path $modelDir -Filter "*.onnx"
if ($onnxFiles.Count -eq 0) {
    Write-Host "WARNING: No ONNX files found in $modelDir" -ForegroundColor Yellow
}

# Find tokenizer.json
$tokenizerFile = Get-ChildItem -Path $modelDir -Filter "tokenizer.json"
if ($null -eq $tokenizerFile) {
    Write-Host "WARNING: tokenizer.json not found in $modelDir" -ForegroundColor Yellow
}

# Push model files to device
foreach ($file in $onnxFiles) {
    Write-Host "Pushing $($file.Name) to device... (This may take a while)" -ForegroundColor Cyan
    adb push "$($file.FullName)" /sdcard/phi-3-model/
}

if ($null -ne $tokenizerFile) {
    Write-Host "Pushing tokenizer.json to device..." -ForegroundColor Cyan
    adb push "$($tokenizerFile.FullName)" /sdcard/phi-3-model/
}

# Push any other essential files (configs, vocab, etc.)
$otherFiles = Get-ChildItem -Path $modelDir -Exclude "*.onnx","tokenizer.json" | Where-Object { -not $_.PSIsContainer }
foreach ($file in $otherFiles) {
    Write-Host "Pushing $($file.Name) to device..." -ForegroundColor Cyan
    adb push "$($file.FullName)" /sdcard/phi-3-model/
}

# Verify files were transferred correctly
Write-Host "Verifying files on device:" -ForegroundColor Green
adb shell ls -la /sdcard/phi-3-model/

Write-Host "Model files transfer complete!" -ForegroundColor Green
Write-Host "You can now run the AIApp and UIApp to test on-device inference."
