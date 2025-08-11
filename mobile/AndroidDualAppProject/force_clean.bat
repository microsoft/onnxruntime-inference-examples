@echo off
echo Stopping all Gradle daemons...
gradlew --stop

echo Waiting for processes to release file handles...
timeout /t 5 /nobreak > nul

echo Attempting to delete build directories...
if exist "AIApp\build" (
    echo Deleting AIApp\build...
    rmdir /s /q "AIApp\build" 2>nul
    if exist "AIApp\build" (
        echo Failed to delete AIApp\build with rmdir, trying PowerShell...
        powershell -Command "Get-ChildItem -Path 'AIApp\build' -Recurse | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue"
        powershell -Command "Remove-Item -Path 'AIApp\build' -Force -Recurse -ErrorAction SilentlyContinue"
    )
)

if exist "UIApp\build" (
    echo Deleting UIApp\build...
    rmdir /s /q "UIApp\build" 2>nul
    if exist "UIApp\build" (
        echo Failed to delete UIApp\build with rmdir, trying PowerShell...
        powershell -Command "Get-ChildItem -Path 'UIApp\build' -Recurse | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue"
        powershell -Command "Remove-Item -Path 'UIApp\build' -Force -Recurse -ErrorAction SilentlyContinue"
    )
)

echo Waiting for file system to sync...
timeout /t 3 /nobreak > nul

echo Attempting Gradle clean...
gradlew clean --no-daemon --no-build-cache

echo Done!
