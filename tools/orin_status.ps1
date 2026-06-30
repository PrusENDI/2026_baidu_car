[CmdletBinding()]
param(
    [string]$HostName = "192.168.0.155",
    [string]$User = "jetson",
    [string]$RemoteRoot = "/home/jetson/workspaces/baidu_car_2026_official_run_copy",
    [string]$SshPath = "C:\msys64\usr\bin\ssh.exe",
    [string]$SshKeyPath = "C:\tmp\codex_orin_ed25519",
    [string[]]$SshOptions = @("StrictHostKeyChecking=accept-new"),
    [string]$SnapshotDir = "",
    [int]$LogLines = 120,
    [switch]$NoSnapshot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Fail {
    param([string]$Message)
    [Console]::Error.WriteLine($Message)
    exit 1
}

if (-not (Test-Path -LiteralPath $SshPath -PathType Leaf)) {
    Fail "ssh.exe not found: $SshPath"
}
if (-not (Test-Path -LiteralPath $SshKeyPath -PathType Leaf)) {
    Fail "SSH key not found: $SshKeyPath"
}
if ([string]::IsNullOrWhiteSpace($SnapshotDir)) {
    $scriptPath = if (-not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
        $PSCommandPath
    } else {
        $MyInvocation.MyCommand.Path
    }
    $scriptDir = Split-Path -Parent $scriptPath
    $projectRoot = Resolve-Path (Join-Path $scriptDir "..")
    $SnapshotDir = Join-Path $projectRoot.Path "logs\orin-status"
}
if (-not $RemoteRoot.StartsWith("/home/jetson/workspaces/")) {
    Fail "RemoteRoot must stay under /home/jetson/workspaces/: $RemoteRoot"
}
if ($LogLines -lt 1 -or $LogLines -gt 2000) {
    Fail "LogLines must be between 1 and 2000."
}

$remoteScript = @"
set +e
REMOTE_ROOT='$RemoteRoot'
LOG_LINES=$LogLines
echo '=== ORIN STATUS SNAPSHOT ==='
echo "time=`$(date -Is)"
echo "host=`$(hostname)"
echo "user=`$(whoami)"
echo "remote_root=`$REMOTE_ROOT"
echo
echo '=== SYSTEM ==='
uname -a 2>/dev/null
uptime 2>/dev/null
free -h 2>/dev/null
df -h / /home /home/jetson/workspaces 2>/dev/null
echo
echo '=== JETSON ==='
if command -v tegrastats >/dev/null 2>&1; then
  timeout 3s tegrastats --interval 1000 2>/dev/null | head -n 3
else
  echo 'tegrastats not found'
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi 2>/dev/null
fi
echo
echo '=== PROJECT ROOT ==='
if [ -d "`$REMOTE_ROOT" ]; then
  pwd
  find "`$REMOTE_ROOT" -maxdepth 1 -mindepth 1 -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort
else
  echo "missing remote root: `$REMOTE_ROOT"
fi
echo
echo '=== PYTHON AND INFERENCE PROCESSES ==='
ps -eo pid,ppid,stat,etime,pcpu,pmem,args 2>/dev/null | grep -E 'python|car_start|infer_back_end|infer_front|collect_control|zmq|paddle' | grep -v grep
echo
echo '=== LISTENING PORTS ==='
if command -v ss >/dev/null 2>&1; then
  ss -ltnp 2>/dev/null | grep -E ':(5000|5001|5002|5003|5004|5005|5006)\b' || true
elif command -v netstat >/dev/null 2>&1; then
  netstat -ltnp 2>/dev/null | grep -E ':(5000|5001|5002|5003|5004|5005|5006)\b' || true
else
  echo 'ss/netstat not found'
fi
echo
echo '=== RECENT PROJECT LOGS ==='
for dir in "`$REMOTE_ROOT/logs" "`$REMOTE_ROOT/log" "`$REMOTE_ROOT/debug" "`$REMOTE_ROOT"; do
  if [ -d "`$dir" ]; then
    echo "--- dir: `$dir ---"
    find "`$dir" -maxdepth 2 -type f \( -name '*.log' -o -name '*.txt' -o -name '*.out' -o -name '*.err' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 8 | cut -d' ' -f2- | while IFS= read -r f; do
      echo "--- tail: `$f ---"
      tail -n "`$LOG_LINES" "`$f" 2>/dev/null
    done
  fi
done
echo
echo '=== RECENT ERRORS ==='
for dir in "`$REMOTE_ROOT/logs" "`$REMOTE_ROOT/log" "`$REMOTE_ROOT/debug" "`$REMOTE_ROOT"; do
  if [ -d "`$dir" ]; then
    find "`$dir" -maxdepth 2 -type f \( -name '*.log' -o -name '*.txt' -o -name '*.out' -o -name '*.err' \) -print0 2>/dev/null | xargs -0 grep -nEi 'traceback|exception|error|failed|fatal|permission denied|no such file|camera|serial|zmq|paddle' 2>/dev/null | tail -n "`$LOG_LINES"
  fi
done
echo
echo '=== JOURNAL RECENT ERRORS ==='
journalctl -p warning..alert --since '30 minutes ago' --no-pager -n "`$LOG_LINES" 2>/dev/null || true
"@

$sshTarget = "$User@$HostName"
$sshArgs = @("-i", $SshKeyPath)
foreach ($option in $SshOptions) {
    if (-not [string]::IsNullOrWhiteSpace($option)) {
        $sshArgs += @("-o", $option)
    }
}
$sshArgs += @($sshTarget, $remoteScript)
$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$outputLines = & $SshPath @sshArgs 2>&1
$ErrorActionPreference = $oldErrorActionPreference
$exitCode = $LASTEXITCODE
$outputText = $outputLines -join [Environment]::NewLine
Write-Output $outputText

if (-not $NoSnapshot) {
    New-Item -ItemType Directory -Force -Path $SnapshotDir | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $safeHost = $HostName -replace "[^A-Za-z0-9_.-]", "_"
    $snapshotPath = Join-Path $SnapshotDir "orin-status-$safeHost-$stamp.log"
    Set-Content -LiteralPath $snapshotPath -Value $outputText -Encoding UTF8
    Write-Host "Snapshot saved: $snapshotPath"
}

if ($exitCode -ne 0) {
    Fail "SSH status command failed. Exit code: $exitCode"
}
