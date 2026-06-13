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
#   C:\hermes\hermes-agent\            - Agent 源码 + venv
#     └── venv\Scripts\                - Python 3.11 + 所有依赖
#         ├── python.exe               - getVenvPython() 期望此路径
#         ├── python311.dll
#         ├── python311._pth           - 含源码目录，不依赖 PYTHONPATH
#         └── Lib\site-packages\
#   C:\hermes\git\                     - PortableGit (cmd\git.exe + bin\bash.exe)
#   C:\hermes\node\                    - Node.js v22
#   C:\hermes\maven\                   - Apache Maven 3.9.9
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
    Write-Host "[INFO] Forcing HermesHome to C:\hermes (offline bundle standard)" -ForegroundColor Yellow
    $HermesHome = "C:\hermes"
    $InstallDir = "C:\hermes\hermes-agent"
}

$VenvDir = Join-Path $InstallDir "venv"
$ScriptsDir = Join-Path $VenvDir "Scripts"
$BootstrapMarker = Join-Path $InstallDir ".hermes-bootstrap-complete"
$DevToolsDir = Join-Path $PayloadDir "devtools"

$GitDir = Join-Path $HermesHome "git"
$NodeDir = Join-Path $HermesHome "node"
$MavenDir = Join-Path $HermesHome "maven"

Add-Type -AssemblyName System.IO.Compression.FileSystem

# ============================================================================
# 辅助函数
# ============================================================================

function Write-Info  { param([string]$Message) Write-Host "-> $Message" -ForegroundColor Cyan }
function Write-OK    { param([string]$Message) Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Warn  { param([string]$Message) Write-Host "[!]  $Message" -ForegroundColor Yellow }
function Write-Err   { param([string]$Message) Write-Host "[X]  $Message" -ForegroundColor Red }

# 将路径持久化写入用户级 PATH（不覆盖已有值）
function Add-ToUserPath {
    param([string]$PathToAdd)
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -and $currentPath.Split(';') -contains $PathToAdd) {
        return
    }
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
            $entries = $currentPath.Split(';') | Where-Object { $_ -and ($_.TrimEnd('\').ToLower() -ne $PathToRemove.TrimEnd('\').ToLower()) }
            $newPath = $entries -join ';'
            if ($newPath -ne $currentPath) {
                [Environment]::SetEnvironmentVariable("PATH", $newPath, $scope)
                Write-Warn "Removed from $scope PATH: $PathToRemove"
            }
        } catch {
            # Machine scope 可能需要管理员权限，忽略错误
        }
    }
}

# ============================================================================
# Stage 0: 卸载系统已有的 Git / Node / Python / Maven
# ============================================================================
# 用户要求：已安装的先卸载，完全使用我们的版本。
# 策略：
# 1. 扫描注册表卸载条目，找到 Git/Node/Python 的卸载程序并静默执行
# 2. 扫描 PATH，移除包含 git.exe / node.exe / python.exe / mvn.cmd 的目录
# 3. 清理已知的旧安装目录
# ============================================================================

Write-Info "Stage 0/12: Uninstalling existing development tools..."

$uninstallKeys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
)

# 扫描注册表，找匹配工具的卸载条目
$toolsToUninstall = @()
foreach ($key in $uninstallKeys) {
    try {
        $items = Get-ItemProperty $key -ErrorAction SilentlyContinue
        foreach ($item in $items) {
            $name = $item.DisplayName
            if (-not $name) { continue }
            # 匹配 Git for Windows / Node.js / Python
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

# 执行卸载
foreach ($tool in $toolsToUninstall) {
    Write-Warn "Uninstalling: $($tool.Name)"
    $uninstallCmd = $tool.QuietUninstallString
    if (-not $uninstallCmd) {
        $uninstallCmd = $item.UninstallString
    }
    if ($uninstallCmd) {
        try {
            # 解析卸载命令——可能是 "C:\path\uninstall.exe" /S 或 MSI
            if ($uninstallCmd -match "^msiexec") {
                $proc = Start-Process "msiexec.exe" -ArgumentList ($uninstallCmd -replace "^msiexec\s+", ""), "/qn", "/norestart" -Wait -NoNewWindow -PassThru -ErrorAction SilentlyContinue
            } else {
                # 尝试静默卸载：大部分安装器支持 /S 或 /silent
                $exePath = ($uninstallCmd -split '"')[1]
                if (-not $exePath) {
                    $parts = $uninstallCmd -split ' ', 2
                    $exePath = $parts[0]
                }
                if (Test-Path $exePath) {
                    $proc = Start-Process $exePath -ArgumentList "/S", "/silent" -Wait -NoNewWindow -PassThru -ErrorAction SilentlyContinue
                }
            }
            Write-OK "Uninstalled: $($tool.Name)"
        } catch {
            Write-Warn "Could not auto-uninstall $($tool.Name) (will shadow via PATH)"
        }
    }
    # 清理安装目录
    if ($tool.InstallLocation -and (Test-Path $tool.InstallLocation)) {
        try {
            Remove-Item -Recurse -Force $tool.InstallLocation -ErrorAction SilentlyContinue
        } catch {}
    }
}

# 清理 PATH 中旧的工具路径
Write-Info "  Cleaning PATH of old tool entries..."
$pathEntries = $env:PATH -split ';'
$dirtyPaths = @()
foreach ($entry in $pathEntries) {
    if (-not $entry) { continue }
    $entryTrimmed = $entry.TrimEnd('\')
    # 检查路径下是否有我们要替换的工具
    $hasGit = Test-Path (Join-Path $entryTrimmed "git.exe") -ErrorAction SilentlyContinue
    $hasNode = Test-Path (Join-Path $entryTrimmed "node.exe") -ErrorAction SilentlyContinue
    $hasNpm = Test-Path (Join-Path $entryTrimmed "npm.cmd") -ErrorAction SilentlyContinue
    $hasPython = (Test-Path (Join-Path $entryTrimmed "python.exe") -ErrorAction SilentlyContinue) -and -not (Test-Path (Join-Path $entryTrimmed "python311._pth"))
    $hasMvn = Test-Path (Join-Path $entryTrimmed "mvn.cmd") -ErrorAction SilentlyContinue
    if ($hasGit -or $hasNode -or $hasNpm -or $hasPython -or $hasMvn) {
        $dirtyPaths += $entry
        Remove-FromPath $entry
    }
}
if ($dirtyPaths.Count -gt 0) {
    Write-Warn "Removed $($dirtyPaths.Count) old tool path(s) from PATH"
} else {
    Write-OK "No old tool paths found in PATH"
}

# 清理已知的旧手动安装目录
$oldDirs = @(
    "C:\hermes-offline",
    "C:\Git",
    "C:\Node",
    "C:\nodejs",
    "C:\Python*",
    "$env:LOCALAPPDATA\Programs\Git",
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:LOCALAPPDATA\hermes"
)
foreach ($old in $oldDirs) {
    $resolved = Get-Item $old -ErrorAction SilentlyContinue
    foreach ($dir in $resolved) {
        # 不要删除 C:\hermes（我们自己要用的）
        if ($dir.FullName -and $dir.FullName.ToLower() -ne "c:\hermes") {
            Write-Warn "Removing old directory: $($dir.FullName)"
            try { Remove-Item -Recurse -Force $dir.FullName -ErrorAction SilentlyContinue } catch {}
        }
    }
}

Write-OK "Stage 0 complete: old tools cleaned"

# ============================================================================
# Stage 1: 创建目录结构
# ============================================================================
Write-Info "Stage 1/12: Creating directory structure..."

New-Item -ItemType Directory -Path $HermesHome -Force | Out-Null
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

# 如果 venv 已存在，删除重建（确保干净）
if (Test-Path $VenvDir) {
    Remove-Item -Recurse -Force $VenvDir
}
New-Item -ItemType Directory -Path $ScriptsDir -Force | Out-Null

Write-OK "Directories created: $HermesHome"

# ============================================================================
# Stage 2: 提取 Python embeddable 到 venv\Scripts\
# ============================================================================
Write-Info "Stage 2/12: Extracting Python 3.11..."

$pythonZip = Join-Path $PayloadDir "python-3.11.9-embed-amd64.zip"
if (-not (Test-Path $pythonZip)) {
    throw "Python zip not found at $pythonZip"
}

[System.IO.Compression.ZipFile]::ExtractToDirectory($pythonZip, $ScriptsDir)

$pythonExe = Join-Path $ScriptsDir "python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python extraction failed - python.exe not found at $pythonExe"
}
Write-OK "Python extracted to $ScriptsDir"

# ============================================================================
# Stage 3: 配置 embeddable Python
# ============================================================================
# python311._pth 存在时 PYTHONPATH 被完全忽略。
# 我们把源码目录写进 ._pth，让 hermes_cli 可直接导入。
# ============================================================================
Write-Info "Stage 3/12: Configuring Python..."

$pthFile = Join-Path $ScriptsDir "python311._pth"
$pthContent = @"
.
Lib
Lib\site-packages
import site
# 源码目录——替代 PYTHONPATH（embeddable 模式下 PYTHONPATH 被忽略）
$InstallDir
"@
$pthContent | Set-Content $pthFile -Encoding ASCII

$sitePackages = Join-Path $ScriptsDir "Lib\site-packages"
New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null
Write-OK "Python configured (_pth includes source dir: $InstallDir)"

# ============================================================================
# Stage 4: 从 wheel 引导安装 pip
# ============================================================================
Write-Info "Stage 4/12: Installing pip..."

$wheelsDir = Join-Path $PayloadDir "wheels"
if (-not (Test-Path $wheelsDir)) {
    throw "Missing wheels directory: $wheelsDir"
}

$pipWheel = Get-ChildItem $wheelsDir -Filter "pip-*-py3-none-any.whl" | Select-Object -First 1
if (-not $pipWheel) {
    throw "pip wheel not found in $wheelsDir"
}

$tempDir = Join-Path $env:TEMP "hermes-pip-bootstrap"
if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    [System.IO.Compression.ZipFile]::ExtractToDirectory($pipWheel.FullName, $tempDir)
    $bootstrapScript = @"
import sys, runpy
sys.path.insert(0, r'$tempDir')
runpy.run_module('pip', run_name='__main__')
"@
    $scriptFile = Join-Path $tempDir "_bootstrap.py"
    $bootstrapScript | Set-Content $scriptFile -Encoding UTF8
    & $pythonExe -S $scriptFile 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "pip bootstrap failed (exit code $LASTEXITCODE)"
    }
} finally {
    Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
}
Write-OK "pip installed"

# ============================================================================
# Stage 5: 安装 setuptools 和 wheel
# ============================================================================
Write-Info "Stage 5/12: Installing setuptools and wheel..."

foreach ($pkg in @("setuptools", "wheel", "packaging")) {
    $whl = Get-ChildItem $wheelsDir -Filter "$pkg-*-py3-none-any.whl" | Select-Object -First 1
    if ($whl) {
        & $pythonExe -m pip install --no-index --find-links $wheelsDir $pkg 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install $pkg (exit code $LASTEXITCODE)"
        }
    }
}
Write-OK "setuptools and wheel installed"

# ============================================================================
# Stage 6: 安装所有 agent 依赖
# ============================================================================
Write-Info "Stage 6/12: Installing agent dependencies..."

$wheelFiles = Get-ChildItem $wheelsDir -Filter "*.whl"
foreach ($whl in $wheelFiles) {
    $name = $whl.BaseName -replace '-cp.*$|-(py3|py2).*$', ''
    if ($name -match '^(pip|setuptools|wheel|packaging)-') {
        continue
    }
    & $pythonExe -m pip install --no-index --find-links $wheelsDir $whl.FullName 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to install $($whl.Name)"
        throw "Dependency installation failed"
    }
}

$pkgCount = (Get-ChildItem $sitePackages -Directory).Count
Write-OK "Dependencies installed ($pkgCount packages)"

# ============================================================================
# Stage 7: 提取源代码
# ============================================================================
Write-Info "Stage 7/12: Extracting source code..."

$sourceZip = Join-Path $PayloadDir "hermes-agent-source.zip"
if (-not (Test-Path $sourceZip)) {
    throw "Missing source code archive: $sourceZip"
}

# 清理安装目录但保留 venv
Get-ChildItem $InstallDir -Exclude "venv" | Remove-Item -Recurse -Force

[System.IO.Compression.ZipFile]::ExtractToDirectory($sourceZip, $InstallDir)

$hermesCliDir = Join-Path $InstallDir "hermes_cli"
if (-not (Test-Path $hermesCliDir)) {
    throw "Source extraction failed - hermes_cli directory not found"
}
Write-OK "Source code extracted"

# ============================================================================
# Stage 8: 安装 hermes-agent 包（editable）
# ============================================================================
Write-Info "Stage 8/12: Installing hermes-agent package..."

& $pythonExe -m pip install --no-index --find-links $wheelsDir --no-deps -e $InstallDir 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pip install -e failed, relying on _pth for import path"
} else {
    Write-OK "hermes-agent package installed (editable)"
}

# 关键验证：hermes_cli 能否导入
$testResult = & $pythonExe -c "import hermes_cli; print('OK')" 2>&1
if ($testResult -ne "OK") {
    Write-Err "hermes_cli import failed: $testResult"
    throw "hermes_cli import test failed"
}
Write-OK "hermes_cli import verified"

# ============================================================================
# Stage 9: 安装 PortableGit
# ============================================================================
Write-Info "Stage 9/12: Installing Git..."

$gitArchive = Join-Path $DevToolsDir "PortableGit-2.54.0-64-bit.7z.exe"
if (Test-Path $gitArchive) {
    if (Test-Path $GitDir) {
        Remove-Item -Recurse -Force $GitDir
    }
    New-Item -ItemType Directory -Path $GitDir -Force | Out-Null

    # PortableGit 自解压：-o"目标目录" -y
    $extractProc = Start-Process -FilePath $gitArchive `
        -ArgumentList "-o`"$GitDir`"", "-y" `
        -NoNewWindow -Wait -PassThru

    if ($extractProc.ExitCode -ne 0) {
        throw "PortableGit extraction failed (exit code $($extractProc.ExitCode))"
    }

    $gitExe = Join-Path $GitDir "cmd\git.exe"
    if (-not (Test-Path $gitExe)) {
        throw "PortableGit extraction did not produce cmd\git.exe at $gitExe"
    }

    Add-ToUserPath (Join-Path $GitDir "cmd")
    Add-ToUserPath (Join-Path $GitDir "bin")

    $gitVer = & $gitExe --version 2>&1
    Write-OK "Git installed: $gitVer"
} else {
    Write-Warn "Git archive not found, skipping"
}

# ============================================================================
# Stage 10: 安装 Node.js
# ============================================================================
Write-Info "Stage 10/12: Installing Node.js..."

$nodeZip = Join-Path $DevToolsDir "node-win-x64.zip"
if (Test-Path $nodeZip) {
    if (Test-Path $NodeDir) {
        Remove-Item -Recurse -Force $NodeDir
    }
    New-Item -ItemType Directory -Path $NodeDir -Force | Out-Null

    # Node zip 内有一层 node-vXX.X.X-win-x64 目录
    $nodeTemp = Join-Path $env:TEMP "hermes-node-extract"
    if (Test-Path $nodeTemp) { Remove-Item -Recurse -Force $nodeTemp }
    [System.IO.Compression.ZipFile]::ExtractToDirectory($nodeZip, $nodeTemp)

    $innerDir = Get-ChildItem $nodeTemp -Directory | Select-Object -First 1
    if ($innerDir) {
        Copy-Item -Path (Join-Path $innerDir.FullName "*") -Destination $NodeDir -Recurse -Force
    } else {
        Copy-Item -Path (Join-Path $nodeTemp "*") -Destination $NodeDir -Recurse -Force
    }
    Remove-Item -Recurse -Force $nodeTemp -ErrorAction SilentlyContinue

    $nodeExe = Join-Path $NodeDir "node.exe"
    if (-not (Test-Path $nodeExe)) {
        throw "Node extraction succeeded but node.exe not found at $nodeExe"
    }

    Add-ToUserPath $NodeDir

    # npm 全局安装目录设到 hermes 下
    $npmPrefix = Join-Path $HermesHome "npm-global"
    New-Item -ItemType Directory -Path $npmPrefix -Force | Out-Null
    [Environment]::SetEnvironmentVariable("NPM_CONFIG_PREFIX", $npmPrefix, "User")
    $env:NPM_CONFIG_PREFIX = $npmPrefix
    Add-ToUserPath $npmPrefix

    $nodeVer = & $nodeExe --version 2>&1
    Write-OK "Node.js installed: $nodeVer"
} else {
    Write-Warn "Node archive not found, skipping"
}

# ============================================================================
# Stage 11: 安装 Maven
# ============================================================================
Write-Info "Stage 11/12: Installing Maven..."

$mavenZip = Join-Path $DevToolsDir "apache-maven-3.9.9-bin.zip"
if (Test-Path $mavenZip) {
    if (Test-Path $MavenDir) {
        Remove-Item -Recurse -Force $MavenDir
    }
    New-Item -ItemType Directory -Path $MavenDir -Force | Out-Null

    $mavenTemp = Join-Path $env:TEMP "hermes-maven-extract"
    if (Test-Path $mavenTemp) { Remove-Item -Recurse -Force $mavenTemp }
    [System.IO.Compression.ZipFile]::ExtractToDirectory($mavenZip, $mavenTemp)

    $innerDir = Get-ChildItem $mavenTemp -Directory | Select-Object -First 1
    if ($innerDir) {
        Copy-Item -Path (Join-Path $innerDir.FullName "*") -Destination $MavenDir -Recurse -Force
    } else {
        Copy-Item -Path (Join-Path $mavenTemp "*") -Destination $MavenDir -Recurse -Force
    }
    Remove-Item -Recurse -Force $mavenTemp -ErrorAction SilentlyContinue

    $mvnCmd = Join-Path $MavenDir "bin\mvn.cmd"
    if (-not (Test-Path $mvnCmd)) {
        throw "Maven extraction succeeded but mvn.cmd not found at $mvnCmd"
    }

    Add-ToUserPath (Join-Path $MavenDir "bin")

    [Environment]::SetEnvironmentVariable("MAVEN_HOME", $MavenDir, "User")
    $env:MAVEN_HOME = $MavenDir

    $mvnVer = & $mvnCmd --version 2>&1 | Select-Object -First 1
    Write-OK "Maven installed: $mvnVer"
} else {
    Write-Warn "Maven archive not found, skipping"
}

# ============================================================================
# Stage 12: 设置环境变量 + 写入 bootstrap marker
# ============================================================================
Write-Info "Stage 12/12: Setting environment variables and writing marker..."

# 持久化 HERMES_HOME
[Environment]::SetEnvironmentVariable("HERMES_HOME", $HermesHome, "User")
$env:HERMES_HOME = $HermesHome
Write-OK "HERMES_HOME set to: $HermesHome"

# 生成工具摘要
$toolSummary = @()
if (Test-Path (Join-Path $GitDir "cmd\git.exe")) {
    $toolSummary += "git: $(& (Join-Path $GitDir "cmd\git.exe") --version 2>&1)"
}
if (Test-Path (Join-Path $NodeDir "node.exe")) {
    $toolSummary += "node: $(& (Join-Path $NodeDir "node.exe") --version 2>&1)"
}
if (Test-Path (Join-Path $MavenDir "bin\mvn.cmd")) {
    $toolSummary += "maven: $(& (Join-Path $MavenDir "bin\mvn.cmd") --version 2>&1 | Select-Object -First 1)"
}
$toolSummary += "python: $(& $pythonExe --version 2>&1)"

# 写入 marker
$markerData = @{
    status = "complete"
    installed_at = (Get-Date -Format "o")
    version = "offline-bundle"
    install_method = "offline"
    hermes_home = $HermesHome
    tools = ($toolSummary -join "; ")
} | ConvertTo-Json
$markerData | Set-Content $BootstrapMarker -Encoding UTF8
Write-OK "Bootstrap marker written"

# ============================================================================
# 完成
# ============================================================================
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Hermes Agent installed successfully!    " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Install root:  C:\hermes" -ForegroundColor White
Write-Host "  Agent source:  $InstallDir" -ForegroundColor White
Write-Host "  Python:        $pythonExe" -ForegroundColor White
if (Test-Path (Join-Path $GitDir "cmd\git.exe"))   { Write-Host "  Git:           $GitDir\cmd\git.exe" -ForegroundColor White }
if (Test-Path (Join-Path $NodeDir "node.exe"))     { Write-Host "  Node:          $NodeDir\node.exe" -ForegroundColor White }
if (Test-Path (Join-Path $MavenDir "bin\mvn.cmd")) { Write-Host "  Maven:         $MavenDir\bin\mvn.cmd" -ForegroundColor White }
Write-Host ""
Write-Host "  HERMES_HOME=C:\hermes (persisted)" -ForegroundColor Yellow
Write-Host "  All tools added to User PATH." -ForegroundColor Yellow
Write-Host ""
