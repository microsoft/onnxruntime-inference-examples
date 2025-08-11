# PowerShell script to handle stubborn file locks
Write-Host "Attempting to resolve file lock issues..."

$buildPath = "AIApp\build"
$problematicFile = "AIApp\build\intermediates\compile_and_runtime_not_namespaced_r_class_jar\debug\R.jar"

# Stop all Gradle processes
Write-Host "Stopping Gradle daemons..."
& .\gradlew --stop

# Wait for processes to settle
Start-Sleep -Seconds 3

# Try to remove the specific problematic file first
if (Test-Path $problematicFile) {
    Write-Host "Attempting to remove problematic R.jar file..."
    try {
        Remove-Item -Path $problematicFile -Force -ErrorAction Stop
        Write-Host "Successfully removed R.jar"
    } catch {
        Write-Host "R.jar still locked, attempting workaround..."
        
        # Try to move the file to a temp location first
        $tempFile = "$env:TEMP\R_$(Get-Date -Format 'yyyyMMdd_HHmmss').jar"
        try {
            Move-Item -Path $problematicFile -Destination $tempFile -Force
            Write-Host "Moved locked file to temp location: $tempFile"
        } catch {
            Write-Host "Could not move file, it's still locked by a process"
        }
    }
}

# Now try to remove the entire build directory
if (Test-Path $buildPath) {
    Write-Host "Removing build directory..."
    try {
        Remove-Item -Path $buildPath -Recurse -Force -ErrorAction Stop
        Write-Host "Successfully removed build directory"
    } catch {
        Write-Host "Some files in build directory are still locked"
    }
}

# Wait and try Gradle clean
Start-Sleep -Seconds 2
Write-Host "Attempting Gradle clean..."
& .\gradlew clean --no-daemon --no-build-cache

Write-Host "Script completed."
