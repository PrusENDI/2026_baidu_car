[CmdletBinding()]
param(
    [string[]]$RemotePaths = @(),
    [string]$Destination = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "orin_config.ps1")

Assert-OrinConfig -RequireRsync

if ($RemotePaths.Count -eq 0) {
    $RemotePaths = $Script:RemoteLogPaths
}

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = $Script:LocalOrinLogRoot
}

$localDestination = (New-Item -ItemType Directory -Force -Path $Destination).FullName
$localRsyncDestination = ConvertTo-RsyncLocalPath -Path $localDestination

foreach ($remotePath in $RemotePaths) {
    $remoteSourcePath = Join-OrinRemotePath -Base $Script:RemoteWorkspace -Relative $remotePath
    $remoteSource = "$($Script:OrinSshTarget):$remoteSourcePath"

    Write-Host "Fetching Orin logs: $remoteSource -> $localDestination"
    & rsync -av --human-readable -e $Script:RsyncSshCommand $remoteSource $localRsyncDestination
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to fetch remote log path '$remotePath' with exit code $LASTEXITCODE"
    }
}
