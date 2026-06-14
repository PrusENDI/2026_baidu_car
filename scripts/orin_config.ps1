Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Fill these three values after confirming the Orin target.
$Script:OrinSshTarget = "jetson@192.168.0.155"
$Script:RemoteWorkspace = "/home/jetson/workspaces/baidu_car_2026_official_run_copy/"
$Script:RemoteRunCommand = ""    # Example: python3 car_start_2026.py

$Script:LocalProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Script:LocalOrinLogRoot = Join-Path $Script:LocalProjectRoot "logs\orin"
$Script:RsyncWindowsPathStyle = "msys" # msys: /c/path, cygwin: /cygdrive/c/path
$Script:OrinSshExe = "C:\msys64\usr\bin\ssh.exe"
$Script:OrinSshArgs = @("-i", "/c/tmp/codex_orin_ed25519", "-o", "StrictHostKeyChecking=accept-new")
$Script:RsyncSshCommand = "ssh -i /c/tmp/codex_orin_ed25519 -o StrictHostKeyChecking=accept-new"

# Keep source/model/config files syncable by default. Exclude only local tooling,
# caches, virtual environments, logs, and generated build/runtime output.
$Script:OrinRsyncExcludes = @(
    ".git/",
    ".codex/",
    "WORKSPACE_ID.txt",
    "logs/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".pytype/",
    ".cache/",
    ".tox/",
    ".nox/",
    ".venv/",
    "venv/",
    "env/",
    "ENV/",
    "build/",
    "dist/",
    "temp/",
    "tmp/",
    "*.pyc",
    "*.pyo",
    "*.log",
    "nohup.out"
)

# Relative paths inside RemoteWorkspace that are safe to pull back as logs.
$Script:RemoteLogPaths = @("logs/")

function Assert-OrinSetting {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Name is not configured. Edit scripts\orin_config.ps1 first."
    }
}

function Assert-OrinRemoteWorkspace {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    Assert-OrinSetting -Name "RemoteWorkspace" -Value $Path

    if (-not $Path.StartsWith("/")) {
        throw "RemoteWorkspace must be an absolute Linux path. Current value: $Path"
    }

    $normalizedPath = $Path.TrimEnd("/")
    $dangerousPaths = @("/", "/home", "/tmp", "/var", "/opt", "/usr", "/workspace", "/workspaces")
    if (($dangerousPaths -contains $normalizedPath) -or ($normalizedPath -match "^/home/[^/]+$")) {
        throw "RemoteWorkspace is too broad and unsafe for rsync. Current value: $Path"
    }

    if ($Path -notmatch "/workspaces?/|/workspace/|/baidu|/smartcar|/car") {
        throw "RemoteWorkspace does not look like a dedicated project workspace. Refusing path: $Path"
    }
}

function Assert-OrinConfig {
    param(
        [switch]$RequireRunCommand,
        [switch]$RequireRsync
    )

    Assert-OrinSetting -Name "OrinSshTarget" -Value $Script:OrinSshTarget
    Assert-OrinRemoteWorkspace -Path $Script:RemoteWorkspace

    if ($RequireRunCommand) {
        Assert-OrinSetting -Name "RemoteRunCommand" -Value $Script:RemoteRunCommand
    }

    $sshCommand = Get-Command $Script:OrinSshExe -ErrorAction SilentlyContinue
    if (-not $sshCommand) {
        throw "ssh was not found at $($Script:OrinSshExe). Install MSYS2 OpenSSH or update scripts\orin_config.ps1."
    }

    if ($RequireRsync) {
        $rsyncCommand = Get-Command rsync -ErrorAction SilentlyContinue
        if (-not $rsyncCommand) {
            throw "rsync was not found in PATH. Install Git Bash/MSYS2/cwRsync/WSL rsync and make it available to PowerShell."
        }
    }
}

function Invoke-OrinSsh {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    & $Script:OrinSshExe @Script:OrinSshArgs $Script:OrinSshTarget $Command
}

function ConvertTo-RsyncLocalPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $rsyncPath = $resolved -replace "\\", "/"

    if ($rsyncPath -match "^([A-Za-z]):/(.*)$") {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2]

        switch ($Script:RsyncWindowsPathStyle) {
            "msys" {
                $rsyncPath = "/$drive/$rest"
            }
            "cygwin" {
                $rsyncPath = "/cygdrive/$drive/$rest"
            }
            default {
                throw "Unsupported RsyncWindowsPathStyle '$($Script:RsyncWindowsPathStyle)'. Use 'msys' or 'cygwin'."
            }
        }
    }

    if (-not $rsyncPath.EndsWith("/")) {
        $rsyncPath = "$rsyncPath/"
    }
    return $rsyncPath
}

function Join-OrinRemotePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Base,

        [Parameter(Mandatory = $true)]
        [string]$Relative
    )

    if ($Relative.StartsWith("/")) {
        throw "Remote log/source paths must be relative to RemoteWorkspace. Refusing: $Relative"
    }

    $cleanBase = $Base.TrimEnd("/")
    $cleanRelative = $Relative.TrimStart("./")
    return "$cleanBase/$cleanRelative"
}

function ConvertTo-RemoteShellSingleQuoted {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return "'" + ($Value -replace "'", "'\''") + "'"
}
