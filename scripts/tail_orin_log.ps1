[CmdletBinding()]
param(
    [string]$RemotePath = "logs/latest.log",
    [int]$Lines = 100
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "orin_config.ps1")

Assert-OrinConfig

$remoteFullPath = Join-OrinRemotePath -Base $Script:RemoteWorkspace -Relative $RemotePath
$quotedLogPath = ConvertTo-RemoteShellSingleQuoted -Value $remoteFullPath
$remoteCommand = "tail -n $Lines -F -- $quotedLogPath"

Write-Host "Tailing Orin log: $($Script:OrinSshTarget):$remoteFullPath"
& ssh $Script:OrinSshTarget $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "tail failed with exit code $LASTEXITCODE"
}
