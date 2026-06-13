# ============================================================================
# Hermes Agent Offline Setup (Windows)
# ============================================================================
# Self-contained offline installer. Zero network access required.
# Shipped via electron-builder extraResources.
#
# Target: Windows Server 2016+, PowerShell 5.1+
# Install root: C:\hermes (fixed, does not follow %LOCALAPPDATA%)
#
# Layout after install:
#   C:\hermes\hermes-agent\             - Agent source + venv
#     venv\Scripts\                     - Python 3.11 + all deps
#       python.exe
#       python311._pth                  - includes source dir
#       Lib\site-packages\
#   C:\hermes\git\                      - PortableGit (cmd\git.exe)
#   C:\hermes\node\                     - Node.js
#   C:\hermes\maven\                    - Apache Maven
#
# Pre-uninstalls existing Git/Node/Python/Maven to ensure our versions win.
# ALL COMMENTS IN ASCII - PowerShell 5.1 without BOM mangles non-ASCII.
# ============================================================================

param(
    [string]$HermesHome = "C:\hermes",
    [string]$InstallDir = "C:\hermes\hermes-agent",
    [string]$PayloadDir = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ============================================================================
# CRITICAL FIX: Write-Host throws HostException in no-console environments
# (Electron windowsHide subprocess). Override to write to [Console]::Out
# (stdout pipe) so bootstrap-runner can capture all output.
# ============================================================================
function script:Write-Host {
    param(
        [Parameter(Position=0)][object]$Object,
        [object]$ForegroundColor,
        [object]$BackgroundColor,
        [switch]$NoNewline,
        [object]$Separator
    )
    $text = if ($null -ne $Object) { $Object.ToString() } else { "" }
    if ($NoNewline) {
        [Console]::Out.Write($text)
    } else {
        [Console]::Out.WriteLine($text)
    }
}

# Force C:\hermes regardless of what was passed in
if ($HermesHome -ne "C:\hermes") {
    [Console]::Out.WriteLine("[INFO] Forcing HermesHome to C:\hermes (offline bundle standard)")
    $HermesHome = "C:\hermes"
    $InstallDir = "C:\hermes\hermes-agent"
}

$VenvDir        = Join-Path $InstallDir "venv"
$ScriptsDir     = Join-Path $VenvDir "Scripts"
# Bootstrap marker is written by JS bootstrap-runner.cjs (correct schema).
$DevToolsDir    = Join-Path $PayloadDir "devtools"
$GitDir         = Join-Path $HermesHome "git"
$NodeDir        = Join-Path $HermesHome "node"
$MavenDir       = Join-Path $HermesHome "maven"

# ============================================================================
# Invoke-NativeCmd: Run a native executable and capture output WITHOUT 2>&1
#
# PowerShell 5.1 + $ErrorActionPreference="Stop" + native stderr via 2>&1 =
# terminating error. Even relaxing EAP to Continue is unreliable. This helper
# uses Start-Process with file redirection, sidestepping the error stream
# entirely. Returns @{ ExitCode=N; Output="..." }.
# ============================================================================
function Invoke-NativeCmd {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string]$ArgumentList,
        [int]$TimeoutSec = 60
    )
    $_outFile = Join-Path $env:TEMP "hermes_native_out_$([System.Guid]::NewGuid().ToString('N').Substring(0,8)).txt"
    $_errFile = Join-Path $env:TEMP "hermes_native_err_$([System.Guid]::NewGuid().ToString('N').Substring(0,8)).txt"
    try {
        $_p = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $_outFile -RedirectStandardError $_errFile
        $_output = ""
        if (Test-Path $_outFile) { $_output += (Get-Content $_outFile -Raw) }
        if (Test-Path $_errFile) { $_output += (Get-Content $_errFile -Raw) }
        return @{ ExitCode = $_p.ExitCode; Output = $_output }
    } finally {
        Remove-Item $_outFile, $_errFile -Force -ErrorAction SilentlyContinue
    }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem

# --- Helper functions ---

function Write-Info  { param([string]$Message) [Console]::Out.WriteLine("-> $Message") }
function Write-OK    { param([string]$Message) [Console]::Out.WriteLine("[OK] $Message") }
function Write-Warn2 { param([string]$Message) [Console]::Out.WriteLine("[!]  $Message") }
function Write-Err2  { param([string]$Message) [Console]::Out.WriteLine("[X]  $Message") }

# Safe ZIP extraction - handles existing destination dirs.
# .NET zip extraction throws IOException if dest exists, so we extract
# file-by-file with overwrite.
function Safe-ExtractZip {
    param([string]$ZipPath, [string]$DestDir)
    if (-not (Test-Path $DestDir)) {
        New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    }

    # Method 1 (preferred): Windows built-in tar (Server 2019+/Win10 1803+)
    # tar natively supports zip format, avoids .NET ExtractToFile file-lock/path issues
    $tarExe = Get-Command tar -ErrorAction SilentlyContinue
    if ($tarExe) {
        # bsdtar (Windows) does NOT support --force-local (GNU tar only).
        # Use Start-Process to avoid 2>&1 + EAP issues entirely.
        $_tarOut = Join-Path $env:TEMP "hermes_tar_out.txt"
        $_tarErr = Join-Path $env:TEMP "hermes_tar_err.txt"
        try {
            $_tp = Start-Process -FilePath "tar" -ArgumentList "-xf `"$ZipPath`" -C `"$DestDir`"" `
                -NoNewWindow -Wait -PassThru `
                -RedirectStandardOutput $_tarOut -RedirectStandardError $_tarErr
            if ($_tp.ExitCode -eq 0) { return }
            Write-Warning "[hermes] tar extraction failed (exit $($_tp.ExitCode)), falling back to .NET method"
        } finally {
            Remove-Item $_tarOut, $_tarErr -Force -ErrorAction SilentlyContinue
        }
    }

    # Method 2 (fallback): .NET per-file extraction with retry
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipArchive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($entry in $zipArchive.Entries) {
            $entryName = $entry.FullName -replace '/', '\\'
            if ($entryName.EndsWith('\\') -or $entryName.EndsWith('/')) { continue }

            $destPath = Join-Path $DestDir $entryName
            $destParent = Split-Path $destPath -Parent
            if (-not (Test-Path $destParent)) {
                New-Item -ItemType Directory -Path $destParent -Force | Out-Null
            }

            # Antivirus may lock .pyd/.dll files; 5 retries with increasing delay
            $extracted = $false
            for ($attempt = 1; $attempt -le 5; $attempt++) {
                try {
                    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
                    $extracted = $true
                    break
                } catch {
                    if ($attempt -lt 5) {
                        Start-Sleep -Milliseconds (500 * $attempt)
                    } else {
                        throw "Failed to extract entry '$entryName' from '$ZipPath': $($_.Exception.Message)"
                    }
                }
            }
        }
    } finally {
        $zipArchive.Dispose()
    }
}

function Add-ToUserPath {
    param([string]$PathToAdd)
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -and $currentPath.Split(';') -contains $PathToAdd) { return }
    $newPath = if ($currentPath) { "$PathToAdd;$currentPath" } else { $PathToAdd }
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    $env:PATH = "$PathToAdd;$env:PATH"
    [Console]::Out.WriteLine("[OK] PATH updated: $PathToAdd")
}

function Remove-FromPath {
    param([string]$PathToRemove)
    foreach ($scope in @("User", "Machine")) {
        try {
            $currentPath = [Environment]::GetEnvironmentVariable("PATH", $scope)
            if (-not $currentPath) { continue }
            $entries = $currentPath.Split(';') | Where-Object {
                $_ -and ($_.TrimEnd('\').ToLower() -ne $PathToRemove.TrimEnd('\').ToLower())
            }
            $newPath = $entries -join ';'
            if ($newPath -ne $currentPath) {
                [Environment]::SetEnvironmentVariable("PATH", $newPath, $scope)
                [Console]::Out.WriteLine("[!] Removed from $scope PATH: $PathToRemove")
            }
        } catch {
            # Machine scope may need admin; ignore
        }
    }
}

# ============================================================================
# Top-level try/catch - any uncaught error outputs details and exits non-zero
# ============================================================================
try {

# ============================================================================
# Stage 0: Uninstall existing Git / Node / Python / Maven
# ============================================================================
[Console]::Out.WriteLine("-> Stage 0/10: Cleaning existing dev tools...")

$uninstallKeys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
)

$toolsFound = @()
foreach ($key in $uninstallKeys) {
    try {
        $items = Get-ItemProperty $key -ErrorAction SilentlyContinue
        foreach ($item in $items) {
            $name = $item.DisplayName
            if (-not $name) { continue }
            if ($name -match "^(Git\s|Git for Windows|Node\.js|Python\s+\d)" -and $item.UninstallString) {
                $toolsFound += [PSCustomObject]@{
                    Name                  = $name
                    UninstallString       = $item.UninstallString
                    QuietUninstallString  = $item.QuietUninstallString
                    InstallLocation       = $item.InstallLocation
                }
            }
        }
    } catch {}
}

foreach ($tool in $toolsFound) {
    [Console]::Out.WriteLine("[!] Uninstalling: $($tool.Name)")
    try {
        $cmd = $tool.QuietUninstallString
        if (-not $cmd) { $cmd = $tool.UninstallString }
        if ($cmd) {
            if ($cmd -match "^msiexec") {
                Start-Process "msiexec.exe" -ArgumentList ($cmd -replace "^msiexec\s+", ""), "/qn", "/norestart" -Wait -WindowStyle Hidden -ErrorAction SilentlyContinue
            } else {
                $exePath = ($cmd -split '"')[1]
                if (-not $exePath) {
                    $parts = $cmd -split ' ', 2
                    $exePath = $parts[0]
                }
                if (Test-Path $exePath) {
                    Start-Process $exePath -ArgumentList "/S" -Wait -WindowStyle Hidden -ErrorAction SilentlyContinue
                }
            }
        }
    } catch {}
    if ($tool.InstallLocation -and (Test-Path $tool.InstallLocation)) {
        try { Remove-Item -Recurse -Force $tool.InstallLocation -ErrorAction SilentlyContinue } catch {}
    }
}

# Clean PATH entries that contain old tools
$pathEntries = $env:PATH -split ';'
foreach ($entry in $pathEntries) {
    if (-not $entry) { continue }
    $trimmed = $entry.TrimEnd('\')
    if ((Test-Path (Join-Path $trimmed "git.exe") -ErrorAction SilentlyContinue) -or
        (Test-Path (Join-Path $trimmed "node.exe") -ErrorAction SilentlyContinue) -or
        (Test-Path (Join-Path $trimmed "npm.cmd") -ErrorAction SilentlyContinue) -or
        (Test-Path (Join-Path $trimmed "mvn.cmd") -ErrorAction SilentlyContinue))
    {
        Remove-FromPath $entry
    }
}

# Clean known old install directories
$oldDirs = @(
    "C:\hermes-offline",
    "C:\Git",
    "C:\Node",
    "C:\nodejs",
    "$env:LOCALAPPDATA\Programs\Git",
    "$env:LOCALAPPDATA\Programs\Python"
)
foreach ($old in $oldDirs) {
    if (Test-Path $old) {
        [Console]::Out.WriteLine("[!] Removing old directory: $old")
        try { Remove-Item -Recurse -Force $old -ErrorAction SilentlyContinue } catch {}
    }
}
# Wildcard: C:\Python*
foreach ($pyDir in (Get-Item "C:\Python*" -ErrorAction SilentlyContinue)) {
    [Console]::Out.WriteLine("[!] Removing old directory: $($pyDir.FullName)")
    try { Remove-Item -Recurse -Force $pyDir.FullName -ErrorAction SilentlyContinue } catch {}
}

[Console]::Out.WriteLine("[OK] Stage 0 complete")

# ============================================================================
# Stage 1: Create directories + extract source code
# ============================================================================
[Console]::Out.WriteLine("-> Stage 1/10: Creating directories and extracting source...")

New-Item -ItemType Directory -Path $HermesHome -Force | Out-Null

# Source must be extracted FIRST (before venv) to avoid directory conflicts
if (Test-Path $InstallDir) {
    # Preserve existing venv if re-running bootstrap; delete everything else
    Get-ChildItem $InstallDir -Exclude "venv" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
} else {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$sourceZip = Join-Path $PayloadDir "hermes-agent-source.zip"
if (-not (Test-Path $sourceZip)) {
    throw "Missing source zip: $sourceZip"
}
Safe-ExtractZip $sourceZip $InstallDir

if (-not (Test-Path (Join-Path $InstallDir "hermes_cli"))) {
    throw "Source extraction failed - hermes_cli not found in $InstallDir"
}
[Console]::Out.WriteLine("[OK] Source code extracted")

# ============================================================================
# Stage 2: Extract Python embeddable into venv\Scripts\
# ============================================================================
[Console]::Out.WriteLine("-> Stage 2/10: Extracting Python 3.11...")

$pythonZip = Join-Path $PayloadDir "python-3.11.9-embed-amd64.zip"
if (-not (Test-Path $pythonZip)) {
    throw "Missing Python zip: $pythonZip"
}

# Fresh venv - delete old if exists
if (Test-Path $VenvDir) {
    Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $ScriptsDir -Force | Out-Null

Safe-ExtractZip $pythonZip $ScriptsDir

$pythonExe = Join-Path $ScriptsDir "python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python extraction failed - python.exe not found at $pythonExe"
}
[Console]::Out.WriteLine("[OK] Python 3.11 extracted")

# ============================================================================
# Stage 3: Configure embeddable Python (_pth file)
# ============================================================================
[Console]::Out.WriteLine("-> Stage 3/10: Configuring Python...")

# python311._pth makes PYTHONPATH ignored; we add source dir inside it
$pthFile = Join-Path $ScriptsDir "python311._pth"
$pthLines = @(
    "python311.zip",
    ".",
    "Lib\site-packages",
    "import site",
    "# Source dir - replaces PYTHONPATH (ignored in embeddable mode)",
    $InstallDir
)
$pthLines -join "`r`n" | Set-Content $pthFile -Encoding ASCII

$sitePackages = Join-Path $ScriptsDir "Lib\site-packages"
New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null
[Console]::Out.WriteLine("[OK] Python configured (_pth includes: $InstallDir)")

# ============================================================================
# Stage 4: Extract ALL wheels to site-packages (no pip needed)
# ============================================================================
[Console]::Out.WriteLine("-> Stage 4/10: Extracting all packages to site-packages...")

$wheelsDir = Join-Path $PayloadDir "wheels"
if (-not (Test-Path $wheelsDir)) {
    throw "Missing wheels directory: $wheelsDir"
}

# A .whl file IS a zip archive. Extracting it into site-packages gives a
# fully functional installation: package code + .dist-info metadata.
# This avoids running pip entirely, which is unreliable on embeddable Python
# (missing site.py initialization, missing distutils, etc.).
$allWheels = Get-ChildItem $wheelsDir -Filter "*.whl"
$failCount = 0
$okCount = 0
$totalWheels = $allWheels.Count
$idx = 0
foreach ($whl in $allWheels) {
    $idx++
    [Console]::Out.WriteLine("  [$idx/$totalWheels] $($whl.Name)")
    try {
        Safe-ExtractZip $whl.FullName $sitePackages
        $okCount++
    } catch {
        [Console]::Out.WriteLine("  [!] FAIL: $($whl.Name) - $_")
        $failCount++
    }
}
[Console]::Out.WriteLine("[OK] $okCount/$totalWheels extracted, $failCount failed")
if ($failCount -gt 0) {
    [Console]::Out.WriteLine("[!] Warning: $failCount wheels failed to extract")
}
$pkgCount = (Get-ChildItem $sitePackages -Directory).Count
[Console]::Out.WriteLine("[OK] $okCount wheels extracted ($pkgCount dirs in site-packages, $failCount failed)")

# ============================================================================
# Stage 5: Verify hermes_cli import
# ============================================================================
[Console]::Out.WriteLine("-> Stage 5/10: Verifying Python environment...")

# hermes_cli source is already in $InstallDir (extracted in Stage 1).
# The _pth file includes $InstallDir, so imports work without pip install -e.
$_quickCheckFile = Join-Path $env:TEMP "hermes_quick_check.py"
"import hermes_cli;print('OK')" | Set-Content $_quickCheckFile -Encoding ASCII
$_result = Invoke-NativeCmd -FilePath $pythonExe -ArgumentList $_quickCheckFile
Remove-Item $_quickCheckFile -Force -ErrorAction SilentlyContinue
if ($_result.ExitCode -ne 0 -or $_result.Output.Trim() -ne "OK") {
    [Console]::Out.WriteLine("[X] Import test failed: $($_result.Output)")
    throw "hermes_cli import test failed"
}
[Console]::Out.WriteLine("[OK] hermes_cli import verified")

# ============================================================================
# Stage 8: Install PortableGit
# ============================================================================
[Console]::Out.WriteLine("-> Stage 6/10: Installing Git...")

$gitArchive = Join-Path $DevToolsDir "PortableGit-2.54.0-64-bit.7z.exe"
if (Test-Path $gitArchive) {
    if (Test-Path $GitDir) {
        Remove-Item -Recurse -Force $GitDir -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Path $GitDir -Force | Out-Null

    # PortableGit self-extracting 7z SFX: -o"dir" -y
    # MUST pass as single argument string, NOT array. Start-Process -ArgumentList
    # array mode double-quotes embedded quotes (-o\"C:\...\"), SFX can't parse it,
    # falls back to GUI dialog waiting for a click. With -WindowStyle Hidden the
    # dialog is invisible -> infinite hang (the bug that caused "stuck at Git").
    $_gitArg = "-o`"$GitDir`" -y"
    $_r = Invoke-NativeCmd -FilePath $gitArchive -ArgumentList $_gitArg -TimeoutSec 120
    if ($_r.ExitCode -ne 0) {
        throw "PortableGit extraction failed (exit code $($_r.ExitCode))`n$($_r.Output)"
    }

    $gitExe = Join-Path $GitDir "cmd\git.exe"
    if (-not (Test-Path $gitExe)) {
        throw "git.exe not found at $gitExe after extraction"
    }

    Add-ToUserPath (Join-Path $GitDir "cmd")
    Add-ToUserPath (Join-Path $GitDir "bin")
    $_r = Invoke-NativeCmd -FilePath $gitExe -ArgumentList '--version'
    [Console]::Out.WriteLine("[OK] Git: $($_r.Output.Trim())")
} else {
    [Console]::Out.WriteLine("[!] Git archive not found at $gitArchive, skipping")
}

# ============================================================================
# Stage 9: Install Node.js
# ============================================================================
[Console]::Out.WriteLine("-> Stage 7/10: Installing Node.js...")

$nodeZip = Join-Path $DevToolsDir "node-win-x64.zip"
if (Test-Path $nodeZip) {
    if (Test-Path $NodeDir) {
        Remove-Item -Recurse -Force $NodeDir -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Path $NodeDir -Force | Out-Null

    # Node zip has an inner node-vXX.X.X-win-x64 directory
    $nodeTemp = Join-Path $env:TEMP "hermes-node-extract"
    if (Test-Path $nodeTemp) {
        Remove-Item -Recurse -Force $nodeTemp -ErrorAction SilentlyContinue
    }
    Safe-ExtractZip $nodeZip $nodeTemp

    $innerNodeDir = Get-ChildItem $nodeTemp -Directory | Select-Object -First 1
    if ($innerNodeDir) {
        robocopy $innerNodeDir.FullName $NodeDir /E /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Null
    }
    Remove-Item -Recurse -Force $nodeTemp -ErrorAction SilentlyContinue

    if (-not (Test-Path (Join-Path $NodeDir "node.exe"))) {
        throw "node.exe not found after extraction"
    }

    Add-ToUserPath $NodeDir

    $npmPrefix = Join-Path $HermesHome "npm-global"
    New-Item -ItemType Directory -Path $npmPrefix -Force | Out-Null
    [Environment]::SetEnvironmentVariable("NPM_CONFIG_PREFIX", $npmPrefix, "User")
    $env:NPM_CONFIG_PREFIX = $npmPrefix
    Add-ToUserPath $npmPrefix

    $_r = Invoke-NativeCmd -FilePath (Join-Path $NodeDir "node.exe") -ArgumentList '--version'
    [Console]::Out.WriteLine("[OK] Node: $($_r.Output.Trim())")
} else {
    [Console]::Out.WriteLine("[!] Node archive not found at $nodeZip, skipping")
}

# ============================================================================
# Stage 10: Install Maven
# ============================================================================
[Console]::Out.WriteLine("-> Stage 8/10: Installing Maven...")

$mavenZip = Join-Path $DevToolsDir "apache-maven-3.9.9-bin.zip"
if (Test-Path $mavenZip) {
    if (Test-Path $MavenDir) {
        Remove-Item -Recurse -Force $MavenDir -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Path $MavenDir -Force | Out-Null

    $mavenTemp = Join-Path $env:TEMP "hermes-maven-extract"
    if (Test-Path $mavenTemp) {
        Remove-Item -Recurse -Force $mavenTemp -ErrorAction SilentlyContinue
    }
    Safe-ExtractZip $mavenZip $mavenTemp

    $innerMavenDir = Get-ChildItem $mavenTemp -Directory | Select-Object -First 1
    if ($innerMavenDir) {
        robocopy $innerMavenDir.FullName $MavenDir /E /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Null
    }
    Remove-Item -Recurse -Force $mavenTemp -ErrorAction SilentlyContinue

    if (-not (Test-Path (Join-Path $MavenDir "bin\mvn.cmd"))) {
        throw "mvn.cmd not found after extraction"
    }

    Add-ToUserPath (Join-Path $MavenDir "bin")
    [Environment]::SetEnvironmentVariable("MAVEN_HOME", $MavenDir, "User")
    $env:MAVEN_HOME = $MavenDir

    $_r = Invoke-NativeCmd -FilePath (Join-Path $MavenDir "bin\mvn.cmd") -ArgumentList '--version'
    [Console]::Out.WriteLine("[OK] Maven: $($_r.Output.Split("`n")[0].Trim())")
} else {
    [Console]::Out.WriteLine("[!] Maven archive not found at $mavenZip, skipping")
}

# ============================================================================
# Stage 11: Set environment variables
# ============================================================================
[Console]::Out.WriteLine("-> Stage 9/10: Setting environment variables...")

[Environment]::SetEnvironmentVariable("HERMES_HOME", $HermesHome, "User")
$env:HERMES_HOME = $HermesHome
[Console]::Out.WriteLine("[OK] HERMES_HOME=$HermesHome")

# ============================================================================
# Stage 11: Post-install verification
# ============================================================================
# After installing all files, smoke-test the Python environment by:
#   1. Importing every module the dashboard needs at runtime
#   2. Verifying HERMES_HOME is writable
# Dashboard boot test removed: embeddable Python cold start + plugin discovery
# makes it unreliable within any fixed timeout on offline Windows. Electron's
# bootstrap-runner has its own readiness probe for the real startup.
# ============================================================================
[Console]::Out.WriteLine("")
[Console]::Out.WriteLine("==========================================")
[Console]::Out.WriteLine("  Post-Install Verification")
[Console]::Out.WriteLine("==========================================")

# --- 11a: Import check ---
[Console]::Out.WriteLine("[1/2] Python module imports...")

$_checkScript = @"
import sys
failures = []
mods = [
    ('uvicorn', 'uvicorn'),
    ('websockets', 'websockets'),
    ('fastapi', 'fastapi'),
    ('httptools', 'httptools'),
    ('pydantic', 'pydantic'),
    ('starlette', 'starlette'),
    ('multipart', 'python-multipart'),
    ('yaml', 'pyyaml'),
    ('dotenv', 'python-dotenv'),
    ('jwt', 'pyjwt'),
    ('cryptography', 'cryptography'),
    ('openai', 'openai'),
    ('httpx', 'httpx'),
    ('tui_gateway.ws', 'tui_gateway.ws'),
    ('tui_gateway.server', 'tui_gateway.server'),
    ('hermes_cli.web_server', 'hermes_cli.web_server (dashboard)'),
    ('hermes_cli.main', 'hermes_cli.main (CLI entry)'),
]
for mod, label in mods:
    try:
        __import__(mod)
    except Exception as e:
        failures.append(f'{label}: {e}')
if failures:
    print('FAIL')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
else:
    print('OK')
    sys.exit(0)
"@

# PowerShell 5.1 mangles multi-line strings passed via -c to native commands.
# Write to a temp .py file and execute that instead.
$_checkScriptFile = Join-Path $env:TEMP "hermes_import_check.py"
$_checkScript | Set-Content $_checkScriptFile -Encoding ASCII
$_result = Invoke-NativeCmd -FilePath $pythonExe -ArgumentList $_checkScriptFile
Remove-Item $_checkScriptFile -Force -ErrorAction SilentlyContinue
[Console]::Out.WriteLine($_result.Output)
if ($_result.ExitCode -ne 0) {
    throw "Post-install verification FAILED: critical Python imports failed. See output above."
}
Write-OK "All imports passed"

# --- 11b: Verify file write capability ---
# Dashboard/WS smoke test removed: embeddable Python cold start makes the
# dashboard too slow to boot within a fixed timeout during install. The
# Electron app starts its own dashboard at runtime; import check (11a) +
# file write test here are sufficient to verify the install.
[Console]::Out.WriteLine("[2/2] File write capability test...")

$_writeTestPath = Join-Path $HermesHome ".write-test"
try {
    "write-test-ok" | Set-Content $_writeTestPath -Encoding UTF8
    $_readBack = Get-Content $_writeTestPath -Raw
    if ($_readBack.Trim() -ne "write-test-ok") {
        throw "Content mismatch: expected 'write-test-ok', got '$_readBack'"
    }
    Remove-Item $_writeTestPath -Force
    Write-OK "File write/read verified at $HermesHome"
} catch {
    throw "Cannot write to $HermesHome : $_"
}

[Console]::Out.WriteLine("")
[Console]::Out.WriteLine("[OK] Post-install verification PASSED (imports + file I/O)")
[Console]::Out.WriteLine("")

# ============================================================================
# Stage 12: Print install summary (marker is written by JS bootstrap-runner)
# ============================================================================
# The .hermes-bootstrap-complete marker must follow the JS-side schema
# (schemaVersion + pinnedCommit) that isBootstrapComplete() checks.
# JS bootstrap-runner.cjs writes it after we exit 0, so we must NOT write
# it here -- a schema-mismatched marker would make isBootstrapComplete()
# return false forever, causing an infinite bootstrap loop.
[Console]::Out.WriteLine("-> Final: Install summary...")

$toolSummary = @()
if (Test-Path (Join-Path $GitDir "cmd\git.exe")) {
    $_r = Invoke-NativeCmd -FilePath (Join-Path $GitDir 'cmd\git.exe') -ArgumentList '--version'
    $toolSummary += "git=$($_r.Output.Trim())"
}
if (Test-Path (Join-Path $NodeDir "node.exe")) {
    $_r = Invoke-NativeCmd -FilePath (Join-Path $NodeDir 'node.exe') -ArgumentList '--version'
    $toolSummary += "node=$($_r.Output.Trim())"
}
if (Test-Path (Join-Path $MavenDir "bin\mvn.cmd")) {
    $_r = Invoke-NativeCmd -FilePath (Join-Path $MavenDir 'bin\mvn.cmd') -ArgumentList '--version'
    $toolSummary += "maven=$($_r.Output.Split("`n")[0].Trim())"
}
$_r = Invoke-NativeCmd -FilePath $pythonExe -ArgumentList '--version'
$toolSummary += "python=$($_r.Output.Trim())"

[Console]::Out.WriteLine("[OK] Tools: $($toolSummary -join '; ')")
[Console]::Out.WriteLine("[OK] Marker will be written by bootstrap-runner (JS-side)")
[Console]::Out.WriteLine("")
[Console]::Out.WriteLine("==========================================")
[Console]::Out.WriteLine("  Hermes Agent installed successfully!    ")
[Console]::Out.WriteLine("==========================================")
[Console]::Out.WriteLine("  Install root:  $HermesHome")
[Console]::Out.WriteLine("  Python:        $pythonExe")
[Console]::Out.WriteLine("  Verified:      imports, file I/O")
[Console]::Out.WriteLine("  Tools:         $($toolSummary -join ', ')")
[Console]::Out.WriteLine("")

} catch {
    [Console]::Out.WriteLine("")
    [Console]::Out.WriteLine("[FATAL] setup-offline.ps1 failed:")
    [Console]::Out.WriteLine("  $_")
    [Console]::Out.WriteLine("")
    [Console]::Out.WriteLine("Stack Trace:")
    [Console]::Out.WriteLine("  $($_.ScriptStackTrace)")
    [Console]::Out.WriteLine("")
    [Console]::Out.WriteLine("Error Record:")
    [Console]::Out.WriteLine("  $($_.Exception.Message)")
    [Console]::Out.WriteLine("")
    exit 1
}
