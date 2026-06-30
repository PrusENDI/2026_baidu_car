Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$SyncScript = Join-Path $ScriptDir "sync_to_orin.ps1"
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

function New-TestWorkspace {
    $root = Join-Path ([System.IO.Path]::GetTempPath()) ("sync-to-orin-test-" + [System.Guid]::NewGuid().ToString("N"))
    $project = Join-Path $root "project"
    $remote = Join-Path $root "remote"
    $bin = Join-Path $root "bin"
    New-Item -ItemType Directory -Force -Path $project, $remote, $bin | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $project "nested") | Out-Null
    Set-Content -LiteralPath (Join-Path $project "car_wrap_2026.py") -Value "print('car')" -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $project "nested\config.txt") -Value "config" -Encoding UTF8

    $logPath = Join-Path $root "rsync-args.txt"
    $keyPath = Join-Path $root "codex_orin_ed25519"
    Set-Content -LiteralPath $keyPath -Value "fake-key" -Encoding ASCII
    $fakeRsync = Join-Path $bin "rsync.cmd"
    Set-Content -LiteralPath $fakeRsync -Encoding ASCII -Value @"
@echo off
echo %* >> "$logPath"
exit /b 0
"@
    $fakeSsh = Join-Path $bin "ssh.cmd"
    Set-Content -LiteralPath $fakeSsh -Encoding ASCII -Value "@echo off`r`nexit /b 0`r`n"

    [pscustomobject]@{
        Root = $root
        Project = $project
        Remote = $remote
        Bin = $bin
        Log = $logPath
        Key = $keyPath
        RSync = $fakeRsync
        Ssh = $fakeSsh
    }
}

function Invoke-SyncTest {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [int]$ExpectedExitCode = 0
    )

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & powershell -NoProfile -ExecutionPolicy Bypass -File $SyncScript @Arguments 2>&1
    $ErrorActionPreference = $oldErrorActionPreference
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne $ExpectedExitCode) {
        throw "Expected exit code $ExpectedExitCode but got $exitCode. Output:`n$($output -join "`n")"
    }
    $output -join "`n"
}

function Assert-Contains {
    param(
        [string]$Text,
        [string]$Needle,
        [string]$Message
    )
    if (-not $Text.Contains($Needle)) {
        throw "$Message`nMissing: $Needle`nActual:`n$Text"
    }
}

function Assert-NotContains {
    param(
        [string]$Text,
        [string]$Needle,
        [string]$Message
    )
    if ($Text.Contains($Needle)) {
        throw "$Message`nUnexpected: $Needle`nActual:`n$Text"
    }
}

$failures = New-Object System.Collections.Generic.List[string]

try {
    $ws = New-TestWorkspace
    $out = Invoke-SyncTest -Arguments @(
        "-LocalRoot", $ws.Project,
        "-RemoteRoot", "/home/jetson/workspaces/baidu_car_2026_official_run_copy",
        "-HostName", "192.168.0.155",
        "-User", "jetson",
        "-RsyncPath", $ws.RSync,
        "-SshPath", $ws.Ssh,
        "-SshKeyPath", $ws.Key,
        "-Files", "car_wrap_2026.py,nested/config.txt",
        "-DryRun",
        "-SkipRemoteCheck"
    )
    $argsText = Get-Content -LiteralPath $ws.Log -Raw
    Assert-Contains $argsText "--dry-run" "Dry-run mode should be passed to rsync."
    Assert-Contains $argsText "--relative" "Relative mode should preserve project paths."
    Assert-Contains $argsText "--exclude=.git/" "Git metadata should be excluded."
    Assert-Contains $argsText "-i" "Rsync SSH command should include an identity file option."
    Assert-Contains $argsText "codex_orin_ed25519" "Rsync SSH command should use the configured Orin identity file."
    Assert-Contains $argsText "StrictHostKeyChecking=accept-new" "Rsync SSH command should accept the Orin host key on first use."
    Assert-Contains $argsText "jetson@192.168.0.155:/home/jetson/workspaces/baidu_car_2026_official_run_copy/" "Remote target should include user, host, and root."
    Assert-Contains $argsText "./car_wrap_2026.py" "Explicit root file should be synced relative to project root."
    Assert-Contains $argsText "./nested/config.txt" "Explicit nested file should be synced relative to project root."
    Assert-Contains $out "DRY RUN" "User output should make dry-run mode obvious."
} catch {
    $failures.Add("explicit-files: $($_.Exception.Message)")
} finally {
    if ($ws -and (Test-Path -LiteralPath $ws.Root)) {
        Remove-Item -LiteralPath $ws.Root -Recurse -Force
    }
}

try {
    $ws = New-TestWorkspace
    $out = Invoke-SyncTest -Arguments @(
        "-RsyncPath", $ws.RSync,
        "-SshPath", $ws.Ssh,
        "-SshKeyPath", $ws.Key,
        "-Files", "car_wrap_2026.py",
        "-PlanOnly"
    )
    Assert-Contains $out "PLAN ONLY" "Running from the project root without -LocalRoot should still compute the project root."
    Assert-Contains $out "car_wrap_2026.py" "Default LocalRoot should allow selecting project files."
} catch {
    $failures.Add("default-local-root: $($_.Exception.Message)")
} finally {
    if ($ws -and (Test-Path -LiteralPath $ws.Root)) {
        Remove-Item -LiteralPath $ws.Root -Recurse -Force
    }
}

try {
    $ws = New-TestWorkspace
    $out = Invoke-SyncTest -Arguments @(
        "-LocalRoot", $ws.Project,
        "-RsyncPath", $ws.RSync,
        "-SshPath", $ws.Ssh,
        "-SshKeyPath", $ws.Key,
        "-Files", "car_wrap_2026.py",
        "-PlanOnly"
    )
    Assert-Contains $out "jetson@192.168.0.155:/home/jetson/workspaces/baidu_car_2026_official_run_copy" "Default remote root should target the confirmed Orin run copy."
} catch {
    $failures.Add("default-remote-root: $($_.Exception.Message)")
} finally {
    if ($ws -and (Test-Path -LiteralPath $ws.Root)) {
        Remove-Item -LiteralPath $ws.Root -Recurse -Force
    }
}

try {
    $ws = New-TestWorkspace
    $out = Invoke-SyncTest -Arguments @(
        "-LocalRoot", $ws.Project,
        "-RemoteRoot", "/home/jetson/workspaces/baidu_car_2026_official_run_copy",
        "-HostName", "192.168.0.155",
        "-User", "jetson",
        "-RsyncPath", $ws.RSync,
        "-SshPath", $ws.Ssh,
        "-SshKeyPath", $ws.Key,
        "-Files", "car_wrap_2026.py",
        "-PlanOnly"
    )
    Assert-Contains $out "PLAN ONLY" "Plan-only mode should be visible in output."
    if (Test-Path -LiteralPath $ws.Log) {
        $argsText = Get-Content -LiteralPath $ws.Log -Raw
        if (-not [string]::IsNullOrWhiteSpace($argsText)) {
            throw "Plan-only mode should not invoke rsync. Actual rsync args:`n$argsText"
        }
    }
} catch {
    $failures.Add("plan-only: $($_.Exception.Message)")
} finally {
    if ($ws -and (Test-Path -LiteralPath $ws.Root)) {
        Remove-Item -LiteralPath $ws.Root -Recurse -Force
    }
}

try {
    $ws = New-TestWorkspace
    $out = Invoke-SyncTest -Arguments @(
        "-LocalRoot", $ws.Project,
        "-RemoteRoot", "/home/jetson/workspaces/baidu_car_2026_official_run_copy",
        "-HostName", "192.168.0.155",
        "-User", "jetson",
        "-RsyncPath", $ws.RSync,
        "-SshPath", $ws.Ssh,
        "-SshKeyPath", $ws.Key,
        "-Files", "missing.py",
        "-DryRun",
        "-SkipRemoteCheck"
    ) -ExpectedExitCode 1
    Assert-Contains $out "File does not exist under LocalRoot" "Missing files should stop before rsync."
} catch {
    $failures.Add("missing-file: $($_.Exception.Message)")
} finally {
    if ($ws -and (Test-Path -LiteralPath $ws.Root)) {
        Remove-Item -LiteralPath $ws.Root -Recurse -Force
    }
}

try {
    $ws = New-TestWorkspace
    Push-Location $ws.Project
    try {
        git init | Out-Null
        git config user.email "test@example.invalid"
        git config user.name "Test User"
        git add car_wrap_2026.py nested/config.txt
        git commit -m "initial" | Out-Null
        Set-Content -LiteralPath (Join-Path $ws.Project "car_wrap_2026.py") -Value "print('changed')" -Encoding UTF8
    } finally {
        Pop-Location
    }

    Invoke-SyncTest -Arguments @(
        "-LocalRoot", $ws.Project,
        "-RemoteRoot", "/home/jetson/workspaces/baidu_car_2026_official_run_copy",
        "-HostName", "192.168.0.155",
        "-User", "jetson",
        "-RsyncPath", $ws.RSync,
        "-SshPath", $ws.Ssh,
        "-SshKeyPath", $ws.Key,
        "-Changed",
        "-DryRun",
        "-SkipRemoteCheck"
    ) | Out-Null
    $argsText = Get-Content -LiteralPath $ws.Log -Raw
    Assert-Contains $argsText "./car_wrap_2026.py" "Changed mode should include modified files."
    Assert-NotContains $argsText "./nested/config.txt" "Changed mode should not include unchanged files."
} catch {
    $failures.Add("changed-files: $($_.Exception.Message)")
} finally {
    if ($ws -and (Test-Path -LiteralPath $ws.Root)) {
        Remove-Item -LiteralPath $ws.Root -Recurse -Force
    }
}

if ($failures.Count -gt 0) {
    Write-Error ("Sync script tests failed:`n" + ($failures -join "`n"))
    exit 1
}

Write-Host "All sync script tests passed."
