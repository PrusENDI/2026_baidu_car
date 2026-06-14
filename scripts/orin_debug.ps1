[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$Delete,
    [string]$Command = "",
    [switch]$SkipFetchLogs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$deployScript = Join-Path $PSScriptRoot "deploy_to_orin.ps1"
$runScript = Join-Path $PSScriptRoot "run_on_orin.ps1"
$fetchScript = Join-Path $PSScriptRoot "fetch_orin_logs.ps1"

& $deployScript -Apply:$Apply -Delete:$Delete
if ($LASTEXITCODE -ne 0) {
    throw "Deploy step failed with exit code $LASTEXITCODE"
}

if (-not $Apply) {
    Write-Host "Dry-run deploy completed. Re-run with -Apply to deploy, run, and fetch logs."
    return
}

try {
    if ([string]::IsNullOrWhiteSpace($Command)) {
        & $runScript
    } else {
        & $runScript -Command $Command
    }
} finally {
    if (-not $SkipFetchLogs) {
        & $fetchScript
    }
}
