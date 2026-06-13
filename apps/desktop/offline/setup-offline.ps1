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

# Force C:\hermes regardless of what was passed in
if ($HermesHome -ne "C:\hermes") {
    Write-Host "[INFO] Forcing HermesHome to C:\hermes (offline bundle standard)"
    $HermesHome = "C:\hermes"
    $InstallDir = "C:\hermes\hermes-agent"
}

$VenvDir        = Join-Path $InstallDir "venv"
$ScriptsDir     = Join-Path $VenvDir "Scripts"
$BootstrapMarker = Join-Path $InstallDir ".hermes-bootstrap-complete"
$DevToolsDir    = Join-Path $PayloadDir "devtools"
$GitDir         = Join-Path $HermesHome "git"
$NodeDir        = Join-Path $HermesHome "node"
$MavenDir       = Join-Path $HermesHome "maven"

Add-Type -AssemblyName System.IO.Compression.FileSystem

# --- Helper functions ---

function Write-Info  { param([string]$Message) Write-Host "-> $Message" }
function Write-OK    { param([string]$Message) Write-Host "[OK] $Message" }
function Write-Warn2 { param([string]$Message) Write-Host "[!]  $Message" }
function Write-Err2  { param([string]$Message) Write-Host "[X]  $Message" }

# Safe ZIP extraction - handles existing destination dirs.
# .NET ExtractToDirectory throws IOException if dest exists, so we extract
# file-by-file with overwrite.
function Safe-ExtractZip {
    param([string]$ZipPath, [string]$DestDir)
    if (-not (Test-Path $DestDir)) {
        New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    }
    $zipArchive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($entry in $zipArchive.Entries) {
            $destPath = Join-Path $DestDir $entry.FullName
            $destParent = Split-Path $destPath -Parent
            if (-not (Test-Path $destParent)) {
                New-Item -ItemType Directory -Path $destParent -Force | Out-Null
            }
            # Skip directory entries (end with / or \)
            if (-not $entry.FullName.EndsWith('/') -and -not $entry.FullName.EndsWith('\')) {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
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
    Write-Host "[OK] PATH updated: $PathToAdd"
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
                Write-Host "[!] Removed from $scope PATH: $PathToRemove"
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
Write-Host "-> Stage 0/12: Cleaning existing dev tools..."

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
    Write-Host "[!] Uninstalling: $($tool.Name)"
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
        Write-Host "[!] Removing old directory: $old"
        try { Remove-Item -Recurse -Force $old -ErrorAction SilentlyContinue } catch {}
    }
}
# Wildcard: C:\Python*
foreach ($pyDir in (Get-Item "C:\Python*" -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Removing old directory: $($pyDir.FullName)"
    try { Remove-Item -Recurse -Force $pyDir.FullName -ErrorAction SilentlyContinue } catch {}
}

Write-Host "[OK] Stage 0 complete"

# ============================================================================
# Stage 1: Create directories + extract source code
# ============================================================================
Write-Host "-> Stage 1/12: Creating directories and extracting source..."

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
Write-Host "[OK] Source code extracted"

# ============================================================================
# Stage 2: Extract Python embeddable into venv\Scripts\
# ============================================================================
Write-Host "-> Stage 2/12: Extracting Python 3.11..."

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
Write-Host "[OK] Python 3.11 extracted"

# ============================================================================
# Stage 3: Configure embeddable Python (_pth file)
# ============================================================================
Write-Host "-> Stage 3/12: Configuring Python..."

# python311._pth makes PYTHONPATH ignored; we add source dir inside it
$pthFile = Join-Path $ScriptsDir "python311._pth"
$pthLines = @(
    ".",
    "Lib",
    "Lib\site-packages",
    "import site",
    "# Source dir - replaces PYTHONPATH (ignored in embeddable mode)",
    $InstallDir
)
$pthLines -join "`r`n" | Set-Content $pthFile -Encoding ASCII

$sitePackages = Join-Path $ScriptsDir "Lib\site-packages"
New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null
Write-Host "[OK] Python configured (_pth includes: $InstallDir)"

# ============================================================================
# Stage 4: Bootstrap pip from wheel
# ============================================================================
Write-Host "-> Stage 4/12: Installing pip..."

$wheelsDir = Join-Path $PayloadDir "wheels"
if (-not (Test-Path $wheelsDir)) {
    throw "Missing wheels directory: $wheelsDir"
}

$pipWheel = Get-ChildItem $wheelsDir -Filter "pip-*-py3-none-any.whl" | Select-Object -First 1
if (-not $pipWheel) {
    throw "pip wheel not found in $wheelsDir"
}

$tempDir = Join-Path $env:TEMP "hermes-pip-bootstrap"
if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    Safe-ExtractZip $pipWheel.FullName $tempDir
    $bootstrapCode = @"
import sys, runpy
sys.path.insert(0, r'$tempDir')
runpy.run_module('pip', run_name='__main__')
"@
    $bootstrapFile = Join-Path $tempDir "_bootstrap.py"
    Set-Content -Path $bootstrapFile -Value $bootstrapCode -Encoding UTF8
    & $pythonExe -S $bootstrapFile 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "pip bootstrap failed (exit code $LASTEXITCODE)"
    }
} finally {
    Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
}
Write-Host "[OK] pip installed"

# ============================================================================
# Stage 5: Install setuptools, wheel, packaging
# ============================================================================
Write-Host "-> Stage 5/12: Installing setuptools/wheel/packaging..."

foreach ($pkg in @("setuptools", "wheel", "packaging")) {
    $whl = Get-ChildItem $wheelsDir -Filter "$pkg-*-py3-none-any.whl" | Select-Object -First 1
    if ($whl) {
        & $pythonExe -m pip install --no-index --find-links $wheelsDir $pkg 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[!] Warning: failed to install $pkg, continuing..."
        }
    }
}
Write-Host "[OK] Build tools installed"

# ============================================================================
# Stage 6: Install all agent dependencies from wheels
# ============================================================================
Write-Host "-> Stage 6/12: Installing agent dependencies..."

$allWheels = Get-ChildItem $wheelsDir -Filter "*.whl" | Where-Object {
    $_.BaseName -notmatch '^(pip|setuptools|wheel|packaging)-'
}
$failCount = 0
foreach ($whl in $allWheels) {
    & $pythonExe -m pip install --no-index --find-links $wheelsDir $whl.FullName 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Warning: failed to install $($whl.Name)"
        $failCount++
    }
}
if ($failCount -gt 5) {
    throw "Too many dependency failures ($failCount wheels failed)"
}
$pkgCount = (Get-ChildItem $sitePackages -Directory).Count
Write-Host "[OK] Dependencies installed ($pkgCount packages, $failCount skipped)"

# ============================================================================
# Stage 7: Install hermes-agent package + verify import
# ============================================================================
Write-Host "-> Stage 7/12: Installing hermes-agent package..."

& $pythonExe -m pip install --no-index --find-links $wheelsDir --no-deps -e $InstallDir 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Warning: pip install -e failed; _pth will handle imports"
}

$importResult = & $pythonExe -c "import hermes_cli;print('OK')" 2>&1
if ($importResult -ne "OK") {
    Write-Host "[X] Import test failed: $importResult"
    throw "hermes_cli import test failed"
}
Write-Host "[OK] hermes_cli import verified"

# ============================================================================
# Stage 8: Install PortableGit
# ============================================================================
Write-Host "-> Stage 8/12: Installing Git..."

$gitArchive = Join-Path $DevToolsDir "PortableGit-2.54.0-64-bit.7z.exe"
if (Test-Path $gitArchive) {
    if (Test-Path $GitDir) {
        Remove-Item -Recurse -Force $GitDir -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Path $GitDir -Force | Out-Null

    # PortableGit self-extracting: -o"dir" -y
    $proc = Start-Process -FilePath $gitArchive `
        -ArgumentList "-o`"$GitDir`"", "-y" `
        -WindowStyle Hidden -Wait -PassThru

    if ($proc.ExitCode -ne 0) {
        throw "PortableGit extraction failed (exit code $($proc.ExitCode))"
    }

    $gitExe = Join-Path $GitDir "cmd\git.exe"
    if (-not (Test-Path $gitExe)) {
        throw "git.exe not found at $gitExe after extraction"
    }

    Add-ToUserPath (Join-Path $GitDir "cmd")
    Add-ToUserPath (Join-Path $GitDir "bin")
    $gitVersion = & $gitExe --version 2>&1
    Write-Host "[OK] Git: $gitVersion"
} else {
    Write-Host "[!] Git archive not found at $gitArchive, skipping"
}

# ============================================================================
# Stage 9: Install Node.js
# ============================================================================
Write-Host "-> Stage 9/12: Installing Node.js..."

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

    $nodeVersion = & (Join-Path $NodeDir "node.exe") --version 2>&1
    Write-Host "[OK] Node: $nodeVersion"
} else {
    Write-Host "[!] Node archive not found at $nodeZip, skipping"
}

# ============================================================================
# Stage 10: Install Maven
# ============================================================================
Write-Host "-> Stage 10/12: Installing Maven..."

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

    $mvnVersion = & (Join-Path $MavenDir "bin\mvn.cmd") --version 2>&1 | Select-Object -First 1
    Write-Host "[OK] Maven: $mvnVersion"
} else {
    Write-Host "[!] Maven archive not found at $mavenZip, skipping"
}

# ============================================================================
# Stage 11: Set environment variables
# ============================================================================
Write-Host "-> Stage 11/12: Setting environment variables..."

[Environment]::SetEnvironmentVariable("HERMES_HOME", $HermesHome, "User")
$env:HERMES_HOME = $HermesHome
Write-Host "[OK] HERMES_HOME=$HermesHome"

# ============================================================================
# Stage 12: Write bootstrap-complete marker
# ============================================================================
Write-Host "-> Stage 12/12: Writing completion marker..."

$toolSummary = @()
if (Test-Path (Join-Path $GitDir "cmd\git.exe")) {
    $toolSummary += "git=$(& (Join-Path $GitDir 'cmd\git.exe') --version 2>&1)"
}
if (Test-Path (Join-Path $NodeDir "node.exe")) {
    $toolSummary += "node=$(& (Join-Path $NodeDir 'node.exe') --version 2>&1)"
}
if (Test-Path (Join-Path $MavenDir "bin\mvn.cmd")) {
    $toolSummary += "maven=$(& (Join-Path $MavenDir 'bin\mvn.cmd') --version 2>&1 | Select-Object -First 1)"
}
$toolSummary += "python=$(& $pythonExe --version 2>&1)"

$markerData = @{
    status         = "complete"
    installed_at   = (Get-Date -Format "o")
    version        = "offline-bundle"
    install_method = "offline"
    hermes_home    = $HermesHome
    tools          = ($toolSummary -join "; ")
} | ConvertTo-Json
$markerData | Set-Content $BootstrapMarker -Encoding UTF8

Write-Host "[OK] Bootstrap marker written"
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Hermes Agent installed successfully!    " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Install root:  $HermesHome"
Write-Host "  Python:        $pythonExe"
Write-Host "  Tools:         $($toolSummary -join ', ')"
Write-Host ""

} catch {
    Write-Host ""
    Write-Host "[FATAL] setup-offline.ps1 failed:" -ForegroundColor Red
    Write-Host "  $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Stack Trace:" -ForegroundColor Yellow
    Write-Host "  $($_.ScriptStackTrace)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Error Record:" -ForegroundColor Yellow
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
