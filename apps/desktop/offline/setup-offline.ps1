# ============================================================================
# Hermes Agent Offline Setup (Windows)
# ============================================================================
# 自包含离线安装，全程零网络请求。
# 随桌面安装器通过 extraResources 分发。
#
# 目标环境：Windows Server 2016+，PowerShell 5.1+
# 安装位置：C:\hermes\（固定，不跟随 %LOCALAPPDATA%）
#
# 安装后目录布局：
#   C:\hermes\hermes-agent\             - Agent 源码 + venv
#     └── venv\Scripts\                 - Python 3.11 + 所有依赖
#         ├── python.exe
#         ├── python311.dll
#         ├── python311._pth            - 含源码目录，不依赖 PYTHONPATH
#         └── Lib\site-packages\
#   C:\hermes\git\                      - PortableGit (cmd\git.exe)
#   C:\hermes\node\                     - Node.js v22
#   C:\hermes\maven\                    - Apache Maven 3.9.9
#
# 安装前会先卸载系统已有的 Git/Node/Python/Maven，确保完全使用我们的版本。
# ============================================================================

param(
    [string]$HermesHome = "C:\hermes",
    [string]$InstallDir = "C:\hermes\hermes-agent",
    [string]$PayloadDir = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# 强制使用 C:\hermes，忽略传入的其他路径
if ($HermesHome -ne "C:\hermes") {
    Write-Host "[INFO] Forcing HermesHome to C:\hermes (offline bundle standard)"
    $HermesHome = "C:\hermes"
    $InstallDir = "C:\hermes\hermes-agent"
}

$VenvDir     = Join-Path $InstallDir "venv"
$ScriptsDir  = Join-Path $VenvDir "Scripts"
$BootstrapMarker = Join-Path $InstallDir ".hermes-bootstrap-complete"
$DevToolsDir = Join-Path $PayloadDir "devtools"

$GitDir   = Join-Path $HermesHome "git"
$NodeDir  = Join-Path $HermesHome "node"
$MavenDir = Join-Path $HermesHome "maven"

Add-Type -AssemblyName System.IO.Compression.FileSystem

# ── 辅助函数 ──

function Write-Info  { param([string]$Message) Write-Host "-> $Message" }
function Write-OK    { param([string]$Message) Write-Host "[OK] $Message" }
function Write-Warn  { param([string]$Message) Write-Host "[!]  $Message" }
function Write-Err   { param([string]$Message) Write-Host "[X]  $Message" }

# 安全解压 ZIP——自动处理目标目录已存在的情况
# ExtractToDirectory 会在目标已存在时抛 IOException，这里逐文件解压
function Safe-ExtractZip {
    param([string]$ZipPath, [string]$DestDir)
    if (-not (Test-Path $DestDir)) {
        New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    }
    $zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($entry in $zip.Entries) {
            $destPath = Join-Path $DestDir $entry.FullName
            $destDir2 = Split-Path $destPath -Parent
            if (-not (Test-Path $destDir2)) {
                New-Item -ItemType Directory -Path $destDir2 -Force | Out-Null
            }
            if (-not $entry.FullName.EndsWith('/') -and -not $entry.FullName.EndsWith('\')) {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
            }
        }
    } finally {
        $zip.Dispose()
    }
}

# 将路径持久化写入用户级 PATH
function Add-ToUserPath {
    param([string]$PathToAdd)
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -and $currentPath.Split(';') -contains $PathToAdd) { return }
    $newPath = if ($currentPath) { "$PathToAdd;$currentPath" } else { $PathToAdd }
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    $env:PATH = "$PathToAdd;$env:PATH"
    Write-OK "PATH updated: $PathToAdd"
}

# 从用户级和系统级 PATH 中移除指定路径
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
                Write-Warn "Removed from $scope PATH: $PathToRemove"
            }
        } catch {}
    }
}

# ============================================================================
# 顶层 try/catch —— 任何未捕获的错误都不会静默退出
# ============================================================================
try {

# ============================================================================
# Stage 0: 卸载系统已有的 Git / Node / Python / Maven
# ============================================================================
Write-Info "Stage 0/12: Cleaning existing dev tools..."

$uninstallKeys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
)

$toolsToUninstall = @()
foreach ($key in $uninstallKeys) {
    try {
        $items = Get-ItemProperty $key -ErrorAction SilentlyContinue
        foreach ($item in $items) {
            $name = $item.DisplayName
            if (-not $name) { continue }
            if ($name -match "^(Git\s|Git for Windows|Node\.js|Python\s+\d)" -and $item.UninstallString) {
                $toolsToUninstall += [PSCustomObject]@{
                    Name = $name
                    UninstallString = $item.UninstallString
                    QuietUninstallString = $item.QuietUninstallString
                    InstallLocation = $item.InstallLocation
                }
            }
        }
    } catch {}
}

foreach ($tool in $toolsToUninstall) {
    Write-Warn "Uninstalling: $($tool.Name)"
    try {
        $cmd = $tool.QuietUninstallString
        if (-not $cmd) { $cmd = $tool.UninstallString }
        if ($cmd) {
            if ($cmd -match "^msiexec") {
                Start-Process "msiexec.exe" -ArgumentList ($cmd -replace "^msiexec\s+", ""), "/qn", "/norestart" -Wait -WindowStyle Hidden -ErrorAction SilentlyContinue
            } else {
                $exePath = ($cmd -split '"')[1]
                if (-not $exePath) { $parts = $cmd -split ' ', 2; $exePath = $parts[0] }
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

# 清理 PATH 中包含旧工具的条目
$pathEntries = $env:PATH -split ';'
foreach ($entry in $pathEntries) {
    if (-not $entry) { continue }
    $t = $entry.TrimEnd('\')
    if ((Test-Path (Join-Path $t "git.exe") -EA SilentlyContinue) -or
        (Test-Path (Join-Path $t "node.exe") -EA SilentlyContinue) -or
        (Test-Path (Join-Path $t "npm.cmd") -EA SilentlyContinue) -or
        (Test-Path (Join-Path $t "mvn.cmd") -EA SilentlyContinue))
    {
        Remove-FromPath $entry
    }
}

# 清理旧的手动安装目录
foreach ($old in @("C:\hermes-offline", "C:\Git", "C:\Node", "C:\nodejs", "$env:LOCALAPPDATA\Programs\Git", "$env:LOCALAPPDATA\Programs\Python")) {
    if (Test-Path $old) {
        Write-Warn "Removing old directory: $old"
        try { Remove-Item -Recurse -Force $old -ErrorAction SilentlyContinue } catch {}
    }
}
# 通配符匹配 C:\Python*
foreach ($oldDir in (Get-Item "C:\Python*" -EA SilentlyContinue)) {
    Write-Warn "Removing old directory: $($oldDir.FullName)"
    try { Remove-Item -Recurse -Force $oldDir.FullName -EA SilentlyContinue } catch {}
}

Write-OK "Stage 0 complete"

# ============================================================================
# Stage 1: 创建目录结构 + 提取源代码
# ============================================================================
Write-Info "Stage 1/12: Creating directories and extracting source..."

New-Item -ItemType Directory -Path $HermesHome -Force | Out-Null

# 源代码必须先提取——否则 venv 目录创建后 ExtractToDirectory 会冲突
if (Test-Path $InstallDir) {
    # 保留已有的 venv（如果 bootstrap 重跑），删除其他内容
    Get-ChildItem $InstallDir -Exclude "venv" | Remove-Item -Recurse -Force -EA SilentlyContinue
} else {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$sourceZip = Join-Path $PayloadDir "hermes-agent-source.zip"
if (-not (Test-Path $sourceZip)) { throw "Missing source zip: $sourceZip" }
Safe-ExtractZip $sourceZip $InstallDir

if (-not (Test-Path (Join-Path $InstallDir "hermes_cli"))) {
    throw "Source extraction failed - hermes_cli not found"
}
Write-OK "Source code extracted"

# ============================================================================
# Stage 2: 提取 Python embeddable
# ============================================================================
Write-Info "Stage 2/12: Extracting Python 3.11..."

$pythonZip = Join-Path $PayloadDir "python-3.11.9-embed-amd64.zip"
if (-not (Test-Path $pythonZip)) { throw "Missing Python zip: $pythonZip" }

# 清理旧 venv，全新创建
if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir -EA SilentlyContinue }
New-Item -ItemType Directory -Path $ScriptsDir -Force | Out-Null

Safe-ExtractZip $pythonZip $ScriptsDir

$pythonExe = Join-Path $ScriptsDir "python.exe"
if (-not (Test-Path $pythonExe)) { throw "Python extraction failed - python.exe not found" }
Write-OK "Python 3.11 extracted"

# ============================================================================
# Stage 3: 配置 embeddable Python（_pth）
# ============================================================================
Write-Info "Stage 3/12: Configuring Python..."

$pthFile = Join-Path $ScriptsDir "python311._pth"
@(
    ".",
    "Lib",
    "Lib\site-packages",
    "import site",
    "# 源码目录——替代 PYTHONPATH（embeddable 模式下 PYTHONPATH 被忽略）",
    $InstallDir
) -join "`r`n" | Set-Content $pthFile -Encoding ASCII

$sitePackages = Join-Path $ScriptsDir "Lib\site-packages"
New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null
Write-OK "Python configured"

# ============================================================================
# Stage 4: 从 wheel 引导 pip
# ============================================================================
Write-Info "Stage 4/12: Installing pip..."

$wheelsDir = Join-Path $PayloadDir "wheels"
if (-not (Test-Path $wheelsDir)) { throw "Missing wheels directory: $wheelsDir" }

$pipWheel = Get-ChildItem $wheelsDir -Filter "pip-*-py3-none-any.whl" | Select-Object -First 1
if (-not $pipWheel) { throw "pip wheel not found in $wheelsDir" }

$tempDir = Join-Path $env:TEMP "hermes-pip-bootstrap"
if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir -EA SilentlyContinue }
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    Safe-ExtractZip $pipWheel.FullName $tempDir
    @"
import sys, runpy
sys.path.insert(0, r'$tempDir')
runpy.run_module('pip', run_name='__main__')
"@ | Set-Content (Join-Path $tempDir "_bootstrap.py") -Encoding UTF8
    & $pythonExe -S (Join-Path $tempDir "_bootstrap.py") 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed (exit $LASTEXITCODE)" }
} finally {
    Remove-Item -Recurse -Force $tempDir -EA SilentlyContinue
}
Write-OK "pip installed"

# ============================================================================
# Stage 5: 安装 setuptools + wheel + packaging
# ============================================================================
Write-Info "Stage 5/12: Installing setuptools/wheel/packaging..."

foreach ($pkg in @("setuptools", "wheel", "packaging")) {
    $whl = Get-ChildItem $wheelsDir -Filter "$pkg-*-py3-none-any.whl" | Select-Object -First 1
    if ($whl) {
        & $pythonExe -m pip install --no-index --find-links $wheelsDir $pkg 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Warn "Failed to install $pkg, continuing..." }
    }
}
Write-OK "Build tools installed"

# ============================================================================
# Stage 6: 安装所有 agent 依赖
# ============================================================================
Write-Info "Stage 6/12: Installing agent dependencies..."

$wheelFiles = Get-ChildItem $wheelsDir -Filter "*.whl" | Where-Object {
    $_.BaseName -notmatch '^(pip|setuptools|wheel|packaging)-'
}
$failCount = 0
foreach ($whl in $wheelFiles) {
    & $pythonExe -m pip install --no-index --find-links $wheelsDir $whl.FullName 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Failed: $($whl.Name)"
        $failCount++
    }
}
if ($failCount -gt 5) {
    throw "Too many dependency failures ($failCount)"
}
$pkgCount = (Get-ChildItem $sitePackages -Directory).Count
Write-OK "Dependencies installed ($pkgCount packages, $failCount skipped)"

# ============================================================================
# Stage 7: 安装 hermes-agent 包 + 验证导入
# ============================================================================
Write-Info "Stage 7/12: Installing hermes-agent package..."

& $pythonExe -m pip install --no-index --find-links $wheelsDir --no-deps -e $InstallDir 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pip install -e failed; _pth will handle imports"
}

$imp = & $pythonExe -c "import hermes_cli;print('OK')" 2>&1
if ($imp -ne "OK") {
    Write-Err "Import test failed: $imp"
    throw "hermes_cli import test failed"
}
Write-OK "hermes_cli import verified"

# ============================================================================
# Stage 8: 安装 PortableGit
# ============================================================================
Write-Info "Stage 8/12: Installing Git..."

$gitArchive = Join-Path $DevToolsDir "PortableGit-2.54.0-64-bit.7z.exe"
if (Test-Path $gitArchive) {
    if (Test-Path $GitDir) { Remove-Item -Recurse -Force $GitDir -EA SilentlyContinue }
    New-Item -ItemType Directory -Path $GitDir -Force | Out-Null

    # PortableGit 自解压：-o"目录" -y
    # 用 -WindowStyle Hidden 替代 -NoNewWindow（无 console 环境下更可靠）
    $proc = Start-Process -FilePath $gitArchive `
        -ArgumentList "-o`"$GitDir`"", "-y" `
        -WindowStyle Hidden -Wait -PassThru

    if ($proc.ExitCode -ne 0) {
        throw "PortableGit extraction failed (exit code $($proc.ExitCode))"
    }

    $gitExe = Join-Path $GitDir "cmd\git.exe"
    if (-not (Test-Path $gitExe)) { throw "git.exe not found at $gitExe after extraction" }

    Add-ToUserPath (Join-Path $GitDir "cmd")
    Add-ToUserPath (Join-Path $GitDir "bin")
    Write-OK "Git: $(& $gitExe --version 2>&1)"
} else {
    Write-Warn "Git archive not found at $gitArchive, skipping"
}

# ============================================================================
# Stage 9: 安装 Node.js
# ============================================================================
Write-Info "Stage 9/12: Installing Node.js..."

$nodeZip = Join-Path $DevToolsDir "node-win-x64.zip"
if (Test-Path $nodeZip) {
    if (Test-Path $NodeDir) { Remove-Item -Recurse -Force $NodeDir -EA SilentlyContinue }
    New-Item -ItemType Directory -Path $NodeDir -Force | Out-Null

    # Node zip 内有一层 node-vXX.X.X-win-x64 目录
    $nodeTemp = Join-Path $env:TEMP "hermes-node-extract"
    if (Test-Path $nodeTemp) { Remove-Item -Recurse -Force $nodeTemp -EA SilentlyContinue }
    Safe-ExtractZip $nodeZip $nodeTemp

    $innerDir = Get-ChildItem $nodeTemp -Directory | Select-Object -First 1
    if ($innerDir) {
        # robocopy 比 Copy-Item 快得多且不报错
        robocopy $innerDir.FullName $NodeDir /E /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Null
    }
    Remove-Item -Recurse -Force $nodeTemp -EA SilentlyContinue

    if (-not (Test-Path (Join-Path $NodeDir "node.exe"))) {
        throw "node.exe not found after extraction"
    }

    Add-ToUserPath $NodeDir

    $npmPrefix = Join-Path $HermesHome "npm-global"
    New-Item -ItemType Directory -Path $npmPrefix -Force | Out-Null
    [Environment]::SetEnvironmentVariable("NPM_CONFIG_PREFIX", $npmPrefix, "User")
    $env:NPM_CONFIG_PREFIX = $npmPrefix
    Add-ToUserPath $npmPrefix

    Write-OK "Node: $(& (Join-Path $NodeDir "node.exe") --version 2>&1)"
} else {
    Write-Warn "Node archive not found at $nodeZip, skipping"
}

# ============================================================================
# Stage 10: 安装 Maven
# ============================================================================
Write-Info "Stage 10/12: Installing Maven..."

$mavenZip = Join-Path $DevToolsDir "apache-maven-3.9.9-bin.zip"
if (Test-Path $mavenZip) {
    if (Test-Path $MavenDir) { Remove-Item -Recurse -Force $MavenDir -EA SilentlyContinue }
    New-Item -ItemType Directory -Path $MavenDir -Force | Out-Null

    $mavenTemp = Join-Path $env:TEMP "hermes-maven-extract"
    if (Test-Path $mavenTemp) { Remove-Item -Recurse -Force $mavenTemp -EA SilentlyContinue }
    Safe-ExtractZip $mavenZip $mavenTemp

    $innerDir = Get-ChildItem $mavenTemp -Directory | Select-Object -First 1
    if ($innerDir) {
        robocopy $innerDir.FullName $MavenDir /E /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Null
    }
    Remove-Item -Recurse -Force $mavenTemp -EA SilentlyContinue

    if (-not (Test-Path (Join-Path $MavenDir "bin\mvn.cmd"))) {
        throw "mvn.cmd not found after extraction"
    }

    Add-ToUserPath (Join-Path $MavenDir "bin")
    [Environment]::SetEnvironmentVariable("MAVEN_HOME", $MavenDir, "User")
    $env:MAVEN_HOME = $MavenDir

    $mvnVer = & (Join-Path $MavenDir "bin\mvn.cmd") --version 2>&1 | Select-Object -First 1
    Write-OK "Maven: $mvnVer"
} else {
    Write-Warn "Maven archive not found at $mavenZip, skipping"
}

# ============================================================================
# Stage 11: 设置环境变量 + 写入 marker
# ============================================================================
Write-Info "Stage 11/12: Setting environment variables..."

[Environment]::SetEnvironmentVariable("HERMES_HOME", $HermesHome, "User")
$env:HERMES_HOME = $HermesHome
Write-OK "HERMES_HOME=$HermesHome"

# ============================================================================
# Stage 12: 写入 bootstrap marker
# ============================================================================
Write-Info "Stage 12/12: Writing completion marker..."

$toolSummary = @()
if (Test-Path (Join-Path $GitDir "cmd\git.exe")) {
    $toolSummary += "git=$(& (Join-Path $GitDir "cmd\git.exe") --version 2>&1)"
}
if (Test-Path (Join-Path $NodeDir "node.exe")) {
    $toolSummary += "node=$(& (Join-Path $NodeDir "node.exe") --version 2>&1)"
}
if (Test-Path (Join-Path $MavenDir "bin\mvn.cmd")) {
    $toolSummary += "maven=$(& (Join-Path $MavenDir "bin\mvn.cmd") --version 2>&1 | Select-Object -First 1)"
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

Write-OK "Bootstrap marker written"
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Hermes Agent installed successfully!    " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Install root:  $HermesHome" -ForegroundColor White
Write-Host "  Python:        $pythonExe" -ForegroundColor White
Write-Host "  Tools:         $($toolSummary -join ', ')" -ForegroundColor White
Write-Host ""

} catch {
    # ── 顶层错误捕获：输出详细错误信息 ──
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

    # 确保 exit code 非 0，让 bootstrap-runner 能捕获
    exit 1
}
