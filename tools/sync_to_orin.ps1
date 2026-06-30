[CmdletBinding(DefaultParameterSetName = "Files")]
param(
    [string]$LocalRoot = "",
    # Confirmed on Orin by inspecting /home/jetson/workspaces.
    [string]$RemoteRoot = "/home/jetson/workspaces/baidu_car_2026_official_run_copy",
    [string]$HostName = "192.168.0.155",
    [string]$User = "jetson",
    [Parameter(ParameterSetName = "Files")]
    [string[]]$Files = @(),
    [Parameter(ParameterSetName = "Changed")]
    [switch]$Changed,
    [switch]$DryRun,
    [switch]$PlanOnly,
    [switch]$Verify,
    [switch]$SkipRemoteCheck,
    [string]$RsyncPath = "C:\msys64\usr\bin\rsync.exe",
    [string]$SshPath = "C:\msys64\usr\bin\ssh.exe",
    [string]$SshKeyPath = "C:\tmp\codex_orin_ed25519",
    [string[]]$SshOptions = @("StrictHostKeyChecking=accept-new")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Fail {
    param([string]$Message)
    [Console]::Error.WriteLine($Message)
    exit 1
}

function Convert-ToMsysPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $resolved = [System.IO.Path]::GetFullPath($Path)
    if ($resolved -match "^([A-Za-z]):\\(.*)$") {
        $drive = $matches[1].ToLowerInvariant()
        $rest = $matches[2] -replace "\\", "/"
        return "/$drive/$rest"
    }
    return ($resolved -replace "\\", "/")
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$ChildPath
    )
    $baseFull = [System.IO.Path]::GetFullPath($BasePath).TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
    $childFull = [System.IO.Path]::GetFullPath($ChildPath)
    $baseUri = [System.Uri]::new($baseFull)
    $childUri = [System.Uri]::new($childFull)
    $relative = [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($childUri).ToString())
    return ($relative -replace "\\", "/")
}

function Test-IsUnderRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Candidate
    )
    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate)
    return $candidateFull.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-GitChangedFiles {
    param([Parameter(Mandatory = $true)][string]$Root)
    $gitRoot = & git -C $Root rev-parse --show-toplevel 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitRoot)) {
        Fail "LocalRoot is not inside a git repository: $Root"
    }

    $statusLines = & git -C $Root status --porcelain=v1 -z
    if ($LASTEXITCODE -ne 0) {
        Fail "Unable to read git status under LocalRoot."
    }

    $items = @()
    $parts = $statusLines -split "`0"
    foreach ($part in $parts) {
        if ([string]::IsNullOrWhiteSpace($part)) {
            continue
        }
        if ($part.Length -lt 4) {
            continue
        }
        $pathPart = $part.Substring(3)
        if ($pathPart.Contains(" -> ")) {
            $pathPart = ($pathPart -split " -> ", 2)[1]
        }
        $full = Join-Path $gitRoot $pathPart
        if ((Test-Path -LiteralPath $full -PathType Leaf) -and (Test-IsUnderRoot -Root $Root -Candidate $full)) {
            $items += Get-RelativePath -BasePath $Root -ChildPath $full
        }
    }
    $items | Sort-Object -Unique
}

function Resolve-FileList {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string[]]$ExplicitFiles,
        [bool]$UseChanged
    )
    if ($UseChanged) {
        return @(Get-GitChangedFiles -Root $Root)
    }

    $resolvedFiles = @()
    $expandedFiles = @()
    foreach ($fileArg in $ExplicitFiles) {
        if ([string]::IsNullOrWhiteSpace($fileArg)) {
            continue
        }
        $expandedFiles += ($fileArg -split "," | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }

    foreach ($file in $expandedFiles) {
        if ([string]::IsNullOrWhiteSpace($file)) {
            continue
        }
        $candidate = Join-Path $Root $file
        if (-not (Test-IsUnderRoot -Root $Root -Candidate $candidate)) {
            Fail "Refusing to sync a path outside LocalRoot: $file"
        }
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            Fail "File does not exist under LocalRoot: $file"
        }
        $resolvedFiles += Get-RelativePath -BasePath $Root -ChildPath $candidate
    }
    $resolvedFiles | Sort-Object -Unique
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [string]$FailureMessage = "Command failed."
    )
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        Fail "$FailureMessage Exit code: $LASTEXITCODE"
    }
}

$scriptPath = if (-not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
    $PSCommandPath
} else {
    $MyInvocation.MyCommand.Path
}
$scriptDir = Split-Path -Parent $scriptPath
if ([string]::IsNullOrWhiteSpace($LocalRoot)) {
    $LocalRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
}

$localFull = [System.IO.Path]::GetFullPath($LocalRoot)
if (-not (Test-Path -LiteralPath $localFull -PathType Container)) {
    Fail "LocalRoot does not exist: $localFull"
}
if (-not (Test-Path -LiteralPath $RsyncPath -PathType Leaf)) {
    Fail "rsync.exe not found: $RsyncPath"
}
if (-not (Test-Path -LiteralPath $SshPath -PathType Leaf)) {
    Fail "ssh.exe not found: $SshPath"
}
if (-not (Test-Path -LiteralPath $SshKeyPath -PathType Leaf)) {
    Fail "SSH key not found: $SshKeyPath"
}
if (-not $RemoteRoot.StartsWith("/home/jetson/workspaces/")) {
    Fail "RemoteRoot must stay under /home/jetson/workspaces/: $RemoteRoot"
}

$syncFiles = @(Resolve-FileList -Root $localFull -ExplicitFiles $Files -UseChanged $Changed.IsPresent)
if ($syncFiles.Count -eq 0) {
    Write-Host "No files selected for sync."
    exit 0
}

$mode = if ($PlanOnly) { "PLAN ONLY" } elseif ($DryRun) { "DRY RUN" } else { "SYNC" }
Write-Host "$mode to $User@$HostName`:$RemoteRoot"
foreach ($file in $syncFiles) {
    Write-Host "  $file"
}

if ($PlanOnly) {
    exit 0
}

$sshMsysPath = Convert-ToMsysPath -Path $SshPath
$sshKeyMsysPath = Convert-ToMsysPath -Path $SshKeyPath
$localMsysRoot = Convert-ToMsysPath -Path $localFull
$remoteTarget = "$User@$HostName`:$RemoteRoot/"
$sshBaseArgs = @("-i", $SshKeyPath)
foreach ($option in $SshOptions) {
    if (-not [string]::IsNullOrWhiteSpace($option)) {
        $sshBaseArgs += @("-o", $option)
    }
}

if (-not $SkipRemoteCheck) {
    $remoteCheckArgs = @($sshBaseArgs + @("$User@$HostName", "test -d '$RemoteRoot'"))
    Invoke-Checked -Exe $SshPath -Args $remoteCheckArgs -FailureMessage "RemoteRoot does not exist or is not reachable: $RemoteRoot"
}

$rsyncArgs = @(
    "-az",
    "--relative",
    "--itemize-changes",
    "--prune-empty-dirs",
    "--exclude=.git/",
    "--exclude=__pycache__/",
    "--exclude=*.pyc",
    "--exclude=.pytest_cache/",
    "--exclude=.mypy_cache/",
    "--exclude=dataset/",
    "--exclude=*.log",
    "--exclude=*.pdiparams",
    "--exclude=*.pdmodel",
    "--exclude=*.onnx",
    "--exclude=*.engine",
    "-e",
    "$sshMsysPath -i $sshKeyMsysPath -o StrictHostKeyChecking=accept-new"
)
if ($DryRun) {
    $rsyncArgs += "--dry-run"
}

Push-Location $localFull
try {
    foreach ($file in $syncFiles) {
        $rsyncArgs += "./$file"
    }
    $rsyncArgs += $remoteTarget
    Invoke-Checked -Exe $RsyncPath -Args $rsyncArgs -FailureMessage "rsync failed."
} finally {
    Pop-Location
}

if ($Verify -and -not $DryRun) {
    foreach ($file in $syncFiles) {
        $remotePath = "$RemoteRoot/$file"
        $verifyArgs = @($sshBaseArgs + @("$User@$HostName", "test -f '$remotePath' && stat -c '%n %s %y' '$remotePath'"))
        Invoke-Checked -Exe $SshPath -Args $verifyArgs -FailureMessage "Remote verification failed for $file"
    }
}
