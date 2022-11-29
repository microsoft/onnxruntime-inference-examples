$ort_version=$args[0]
$url="https://github.com/microsoft/onnxruntime/releases/download/v$ort_version/onnxruntime-win-x64-$ort_version.zip"
Write-Host "Downloading from $url"
Invoke-WebRequest -Uri $url -OutFile onnxruntime.zip
7z x onnxruntime.zip
move onnxruntime-win-x64-$ort_version onnxruntimebin