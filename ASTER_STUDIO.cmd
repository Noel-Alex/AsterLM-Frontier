@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0ASTER_STUDIO.ps1" %*
exit /b %ERRORLEVEL%
