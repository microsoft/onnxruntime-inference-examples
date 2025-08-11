# PowerShell script to test Phi-3 model locally
# Usage: .\test_model_local.ps1 -modelDir "path\to\model\directory"

param (
    [Parameter(Mandatory=$true)]
    [string]$modelDir
)

Write-Host "Testing Phi-3 model locally..." -ForegroundColor Cyan
Write-Host "Model directory: $modelDir" -ForegroundColor Cyan

# Check if model directory exists
if (-not (Test-Path -Path $modelDir)) {
    Write-Host "ERROR: Model directory does not exist: $modelDir" -ForegroundColor Red
    exit 1
}

# Check if model files exist
$onnxFiles = Get-ChildItem -Path $modelDir -Filter "*.onnx"
if ($onnxFiles.Count -eq 0) {
    Write-Host "ERROR: No ONNX files found in directory: $modelDir" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($onnxFiles.Count) ONNX files:" -ForegroundColor Green
foreach ($file in $onnxFiles) {
    Write-Host "  - $($file.Name) ($('{0:N2}' -f ($file.Length / 1MB)) MB)" -ForegroundColor Green
}

# Check for tokenizer.json
$tokenizerFile = Get-ChildItem -Path $modelDir -Filter "tokenizer.json" | Select-Object -First 1
if ($null -eq $tokenizerFile) {
    Write-Host "ERROR: tokenizer.json not found in directory: $modelDir" -ForegroundColor Red
    exit 1
}

Write-Host "Found tokenizer.json" -ForegroundColor Green

# Update test files with the correct model path
Write-Host "Updating test files with model path..." -ForegroundColor Yellow

# Update ModelManagerLocalTest.kt
$modelTestFile = "c:\projects\android_fl\AndroidDualAppProject\AIApp\src\test\java\com\example\aiapp\ModelManagerLocalTest.kt"
if (Test-Path -Path $modelTestFile) {
    (Get-Content -Path $modelTestFile) | 
        ForEach-Object {$_ -replace 'private val LOCAL_MODEL_PATH = ".*"', "private val LOCAL_MODEL_PATH = `"$($modelDir.Replace('\', '\\'))`""} | 
        Set-Content -Path $modelTestFile
    Write-Host "Updated ModelManagerLocalTest.kt with model path" -ForegroundColor Green
}

# Update ModelTest.java
$modelJavaFile = "c:\projects\android_fl\AndroidDualAppProject\ModelTest.java"
if (Test-Path -Path $modelJavaFile) {
    (Get-Content -Path $modelJavaFile) | 
        ForEach-Object {$_ -replace 'private static final String MODEL_PATH = ".*";', "private static final String MODEL_PATH = `"$($modelDir.Replace('\', '\\'))`";"} | 
        Set-Content -Path $modelJavaFile
    Write-Host "Updated ModelTest.java with model path" -ForegroundColor Green
}

Write-Host "`nReady for testing!" -ForegroundColor Cyan
Write-Host "To run the standalone Java test:" -ForegroundColor White
Write-Host "  1. Compile ModelTest.java with the ONNX Runtime GenAI library in classpath" -ForegroundColor White
Write-Host "  2. Run the compiled Java class" -ForegroundColor White
Write-Host "`nTo run the unit tests:" -ForegroundColor White
Write-Host "  1. Open the project in Android Studio" -ForegroundColor White
Write-Host "  2. Right-click on ModelManagerLocalTest.kt and select 'Run ModelManagerLocalTest'" -ForegroundColor White

Write-Host "`nTo test model in Android app:" -ForegroundColor White
Write-Host "  1. Use the push_model.ps1 script to push models to your Android device:" -ForegroundColor White
Write-Host "     .\push_model.ps1 -modelDir `"$modelDir`"" -ForegroundColor White
Write-Host "  2. Install both AIApp and UIApp on your device" -ForegroundColor White
Write-Host "  3. Follow the workflow in the README.md file" -ForegroundColor White
