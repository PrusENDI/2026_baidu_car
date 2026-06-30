[CmdletBinding()]
param(
    [switch]$PlanOnly,
    [switch]$Verify,
    [string]$LocalRoot,
    [string]$RemoteRoot = "/home/jetson/workspaces/baidu_car_2026_official_run_copy",
    [string]$HostName = "192.168.0.155",
    [string]$User = "jetson",
    [string]$SshKeyPath = "C:\tmp\codex_orin_ed25519"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$SyncScript = Join-Path $ScriptDir "sync_to_orin.ps1"
if ([string]::IsNullOrWhiteSpace($LocalRoot)) {
    $LocalRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
}

$Files = @(
    "smartcar/whalesbot/tools/chassis_pad_control.py",
    "smartcar/whalesbot/tools/servo_calibration_control.py",
    "tests/test_chassis_pad_control.py",
    "tests/test_servo_calibration_control.py"
) -join ","

if ($PlanOnly) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $SyncScript `
        -LocalRoot $LocalRoot `
        -RemoteRoot $RemoteRoot `
        -HostName $HostName `
        -User $User `
        -SshKeyPath $SshKeyPath `
        -Files $Files `
        -PlanOnly
    exit $LASTEXITCODE
}

if ($Verify) {
    Write-Host "Syncing debug tools to Orin and verifying..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $SyncScript `
        -LocalRoot $LocalRoot `
        -RemoteRoot $RemoteRoot `
        -HostName $HostName `
        -User $User `
        -SshKeyPath $SshKeyPath `
        -Files $Files `
        -Verify
    exit $LASTEXITCODE
}

Write-Host "Dry-run debug tools sync. Pass -Verify to perform the real sync."
& powershell -NoProfile -ExecutionPolicy Bypass -File $SyncScript `
    -LocalRoot $LocalRoot `
    -RemoteRoot $RemoteRoot `
    -HostName $HostName `
    -User $User `
    -SshKeyPath $SshKeyPath `
    -Files $Files `
    -DryRun
exit $LASTEXITCODE
