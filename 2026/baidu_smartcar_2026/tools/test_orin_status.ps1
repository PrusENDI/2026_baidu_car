Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$StatusScript = Join-Path $ScriptDir "orin_status.ps1"

function New-TestWorkspace {
    $root = Join-Path ([System.IO.Path]::GetTempPath()) ("orin-status-test-" + [System.Guid]::NewGuid().ToString("N"))
    $bin = Join-Path $root "bin"
    $snapshots = Join-Path $root "snapshots"
    New-Item -ItemType Directory -Force -Path $bin, $snapshots | Out-Null
    $keyPath = Join-Path $root "codex_orin_ed25519"
    Set-Content -LiteralPath $keyPath -Value "fake-key" -Encoding ASCII

    $argsLog = Join-Path $root "ssh-args.txt"
    $fakeSsh = Join-Path $bin "ssh.ps1"
    Set-Content -LiteralPath $fakeSsh -Encoding UTF8 -Value @"
`$args -join ' ' | Set-Content -LiteralPath '$argsLog' -Encoding UTF8
Set-Content -LiteralPath '$root\ssh-invoked.txt' -Value 'invoked' -Encoding UTF8
Write-Output '=== ORIN STATUS SNAPSHOT ==='
Write-Output 'host=ubuntu'
Write-Output 'user=jetson'
Write-Output 'remote_root=/home/jetson/workspaces/baidu_car_2026_official_run_copy'
Write-Output '=== RECENT ERRORS ==='
Write-Output 'Traceback simulated error line'
exit 0
"@

    [pscustomobject]@{
        Root = $root
        Bin = $bin
        Snapshots = $snapshots
        Ssh = $fakeSsh
        Key = $keyPath
        ArgsLog = $argsLog
    }
}

function Invoke-StatusTest {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [int]$ExpectedExitCode = 0
    )

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & powershell -NoProfile -ExecutionPolicy Bypass -File $StatusScript @Arguments 2>&1
    $ErrorActionPreference = $oldErrorActionPreference
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne $ExpectedExitCode) {
        throw "Expected exit code $ExpectedExitCode but got $exitCode. Output:`n$($output -join "`n")"
    }
    $output -join "`n"
}

function Assert-Contains {
    param([string]$Text, [string]$Needle, [string]$Message)
    if (-not $Text.Contains($Needle)) {
        throw "$Message`nMissing: $Needle`nActual:`n$Text"
    }
}

function Assert-NotContains {
    param([string]$Text, [string]$Needle, [string]$Message)
    if ($Text.Contains($Needle)) {
        throw "$Message`nUnexpected: $Needle`nActual:`n$Text"
    }
}

$failures = New-Object System.Collections.Generic.List[string]

try {
    $ws = New-TestWorkspace
    $out = Invoke-StatusTest -Arguments @(
        "-SshPath", $ws.Ssh,
        "-SshKeyPath", $ws.Key,
        "-SnapshotDir", $ws.Snapshots
    )
    Assert-Contains $out "=== ORIN STATUS SNAPSHOT ===" "Status output should be printed to console."
    Assert-Contains $out "Traceback simulated error line" "Recent errors should be included in console output."

    if (-not (Test-Path -LiteralPath (Join-Path $ws.Root "ssh-invoked.txt"))) {
        throw "Fake SSH was not invoked."
    }
    $sshArgs = Get-Content -LiteralPath $ws.ArgsLog -Raw
    Assert-Contains $sshArgs "-i" "Status SSH command should include an identity file option."
    Assert-Contains $sshArgs "codex_orin_ed25519" "Status SSH command should use the configured Orin identity file."
    Assert-Contains $sshArgs "StrictHostKeyChecking=accept-new" "Status SSH command should accept the Orin host key on first use."

    $scriptText = Get-Content -LiteralPath $StatusScript -Raw
    Assert-Contains $scriptText '$HostName = "192.168.0.155"' "Default SSH host should be Orin."
    Assert-Contains $scriptText '$User = "jetson"' "Default SSH user should be jetson."
    Assert-Contains $scriptText "/home/jetson/workspaces/baidu_car_2026_official_run_copy" "Default remote root should be the confirmed run copy."
    Assert-Contains $scriptText "journalctl" "Status command should include system journal inspection."
    Assert-Contains $scriptText "tail" "Status command should include project log tailing."
    Assert-NotContains $scriptText " rm " "Status command should not remove remote files."
    Assert-NotContains $scriptText " > " "Status command should not redirect output on Orin."

    $snapshots = @(Get-ChildItem -LiteralPath $ws.Snapshots -Filter "orin-status-*.log")
    if ($snapshots.Count -ne 1) {
        throw "Expected one snapshot log, got $($snapshots.Count)."
    }
    $snapshotText = Get-Content -LiteralPath $snapshots[0].FullName -Raw
    Assert-Contains $snapshotText "Traceback simulated error line" "Snapshot should contain the SSH output."
} catch {
    $failures.Add("status-snapshot: $($_.Exception.Message)")
} finally {
    if ($ws -and (Test-Path -LiteralPath $ws.Root)) {
        Remove-Item -LiteralPath $ws.Root -Recurse -Force
    }
}

try {
    $ws = New-TestWorkspace
    Set-Content -LiteralPath $ws.Ssh -Encoding UTF8 -Value @"
Write-Error 'stderr warning from ssh'
Write-Output '=== ORIN STATUS SNAPSHOT ==='
Write-Output 'host=ubuntu'
Write-Output '=== RECENT ERRORS ==='
Write-Output 'no errors'
exit 0
"@
    $out = Invoke-StatusTest -Arguments @(
        "-SshPath", $ws.Ssh,
        "-SshKeyPath", $ws.Key,
        "-SnapshotDir", $ws.Snapshots
    )
    Assert-Contains $out "stderr warning from ssh" "SSH stderr should be captured without failing when exit code is zero."
    Assert-Contains $out "=== ORIN STATUS SNAPSHOT ===" "SSH stdout should still be captured."
} catch {
    $failures.Add("ssh-stderr-warning: $($_.Exception.Message)")
} finally {
    if ($ws -and (Test-Path -LiteralPath $ws.Root)) {
        Remove-Item -LiteralPath $ws.Root -Recurse -Force
    }
}

try {
    $ws = New-TestWorkspace
    $projectSnapshotDir = Resolve-Path (Join-Path $ScriptDir "..\logs\orin-status") -ErrorAction SilentlyContinue
    $beforeSnapshots = @()
    if ($projectSnapshotDir) {
        $beforeSnapshots = @(Get-ChildItem -LiteralPath $projectSnapshotDir.Path -Filter "orin-status-*.log" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    }
    $oldLocation = Get-Location
    Set-Location $ws.Root
    try {
        $out = Invoke-StatusTest -Arguments @(
            "-SshPath", $ws.Ssh,
            "-SshKeyPath", $ws.Key,
            "-SnapshotDir", $ws.Snapshots
        )
    } finally {
        Set-Location $oldLocation
    }
    Assert-Contains $out "Snapshot saved:" "Snapshot path should be reported."
    $projectSnapshotDir = Resolve-Path (Join-Path $ScriptDir "..\logs\orin-status") -ErrorAction SilentlyContinue
    if ($projectSnapshotDir) {
        $afterSnapshots = @(Get-ChildItem -LiteralPath $projectSnapshotDir.Path -Filter "orin-status-*.log" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
        foreach ($snapshot in $afterSnapshots) {
            if ($beforeSnapshots -notcontains $snapshot) {
                Remove-Item -LiteralPath $snapshot -Force
            }
        }
    }
} catch {
    $failures.Add("default-snapshot-dir: $($_.Exception.Message)")
} finally {
    if ($ws -and (Test-Path -LiteralPath $ws.Root)) {
        Remove-Item -LiteralPath $ws.Root -Recurse -Force
    }
}

if ($failures.Count -gt 0) {
    Write-Error ("Orin status tests failed:`n" + ($failures -join "`n"))
    exit 1
}

Write-Host "All Orin status tests passed."
