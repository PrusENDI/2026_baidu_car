[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$Delete
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "orin_config.ps1")

Assert-OrinConfig -RequireRsync

$source = ConvertTo-RsyncLocalPath -Path $Script:LocalProjectRoot
$destination = "$($Script:OrinSshTarget):$($Script:RemoteWorkspace.TrimEnd('/'))/"

$rsyncArgs = @(
    "-av",
    "--itemize-changes",
    "--human-readable",
    "-e",
    $Script:RsyncSshCommand
)

if (-not $Apply) {
    $rsyncArgs += "-n"
}

if ($Delete) {
    $rsyncArgs += "--delete"
}

foreach ($exclude in $Script:OrinRsyncExcludes) {
    $rsyncArgs += "--exclude"
    $rsyncArgs += $exclude
}

$rsyncArgs += $source
$rsyncArgs += $destination

Write-Host "Local project root: $($Script:LocalProjectRoot)"
Write-Host "Remote workspace:  $($Script:OrinSshTarget):$($Script:RemoteWorkspace)"

if (-not $Apply) {
    Write-Host "Mode: dry-run. No files will be copied. Re-run with -Apply to deploy."
} else {
    Write-Host "Mode: apply. Files will be copied to the Orin run copy."
    $quotedWorkspace = ConvertTo-RemoteShellSingleQuoted -Value $Script:RemoteWorkspace
    & ssh $Script:OrinSshTarget "mkdir -p -- $quotedWorkspace"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create remote workspace: $($Script:RemoteWorkspace)"
    }
}

if ($Delete) {
    Write-Host "Delete mode: enabled. Remote files absent locally may be deleted by rsync."
} else {
    Write-Host "Delete mode: disabled. Pass -Delete explicitly to include rsync --delete."
}

& rsync @rsyncArgs
if ($LASTEXITCODE -ne 0) {
    throw "rsync failed with exit code $LASTEXITCODE"
}
