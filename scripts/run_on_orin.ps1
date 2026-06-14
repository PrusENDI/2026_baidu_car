[CmdletBinding()]
param(
    [string]$Command = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "orin_config.ps1")

Assert-OrinConfig

if ([string]::IsNullOrWhiteSpace($Command)) {
    $Command = $Script:RemoteRunCommand
}

Assert-OrinSetting -Name "RemoteRunCommand" -Value $Command

$quotedWorkspace = ConvertTo-RemoteShellSingleQuoted -Value $Script:RemoteWorkspace
$remoteCommand = "cd $quotedWorkspace && $Command"

Write-Host "Running on Orin: $remoteCommand"
& ssh $Script:OrinSshTarget $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "Remote command failed with exit code $LASTEXITCODE"
}
