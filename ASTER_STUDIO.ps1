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
$repoWslInput = $repoWindows.Replace("\", "/")
$repoWsl = (wsl.exe -d $Distro -- wslpath -a -u $repoWslInput).Trim()
if (-not $repoWsl) {
    throw "Could not map the AsterLM checkout into WSL."
}

$python = if ($env:ASTERLM_WSL_PYTHON) {
    $env:ASTERLM_WSL_PYTHON
} else {
    "/root/.venvs/asterlm/bin/python"
}

$modalConfigWindows = if ($env:MODAL_CONFIG_PATH -and (Test-Path -LiteralPath $env:MODAL_CONFIG_PATH)) {
    (Resolve-Path -LiteralPath $env:MODAL_CONFIG_PATH).Path
} else {
    Join-Path $HOME ".modal.toml"
}
$modalConfigWsl = if (Test-Path -LiteralPath $modalConfigWindows) {
    $modalConfigWslInput = $modalConfigWindows.Replace("\", "/")
    (wsl.exe -d $Distro -- wslpath -a -u $modalConfigWslInput).Trim()
} else {
    $null
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
if ($modalConfigWsl) {
    Write-Host "Modal:      provider-native profile store linked"
}
Write-Host "Press Ctrl+C to stop the control plane. Background research jobs keep their own state."

if ($modalConfigWsl) {
    wsl.exe -d $Distro --cd $repoWsl -- env "MODAL_CONFIG_PATH=$modalConfigWsl" $python -m studio.server --host $HostAddress --port $Port --no-open
} else {
    wsl.exe -d $Distro --cd $repoWsl -- $python -m studio.server --host $HostAddress --port $Port --no-open
}
exit $LASTEXITCODE
