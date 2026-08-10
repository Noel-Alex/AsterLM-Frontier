[CmdletBinding()]
param(
    [string]$Distro = $(if ($env:ASTERLM_WSL_DISTRO) { $env:ASTERLM_WSL_DISTRO } else { "Ubuntu" }),
    [string]$HostAddress = $(if ($env:ASTER_STUDIO_HOST) { $env:ASTER_STUDIO_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:ASTER_STUDIO_PORT) { [int]$env:ASTER_STUDIO_PORT } else { 8765 }),
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoWindows = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$repoWsl = (wsl.exe -d $Distro -- wslpath -a $repoWindows).Trim()
if (-not $repoWsl) {
    throw "Could not map the AsterLM checkout into WSL."
}

$python = if ($env:ASTERLM_WSL_PYTHON) {
    $env:ASTERLM_WSL_PYTHON
} else {
    "/root/.venvs/asterlm/bin/python"
}

wsl.exe -d $Distro -- test -x $python
if ($LASTEXITCODE -ne 0) {
    throw "AsterLM's WSL environment is missing at $python. Run the environment setup first."
}

$url = "http://localhost:$Port/"
if (-not $NoOpen) {
    Start-Job -ScriptBlock {
        param($Target)
        Start-Sleep -Seconds 2
        Start-Process $Target
    } -ArgumentList $url | Out-Null
}

Write-Host "AsterLM Studio" -ForegroundColor Cyan
Write-Host "Repository: $repoWindows"
Write-Host "Interface:  $url"
Write-Host "Runtime:    $Distro ($python)"
Write-Host "Press Ctrl+C to stop the control plane. Background research jobs keep their own state."

wsl.exe -d $Distro --cd $repoWsl -- $python studio/server.py --host $HostAddress --port $Port --no-open
exit $LASTEXITCODE
