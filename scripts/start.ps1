#Requires -Version 5.1
# =============================================================
# Local Model Optimizer - Windows Start Script
# =============================================================
# Usage: .\scripts\start.ps1 [-Port PORT] [-Host HOST] [-Workers N]
# =============================================================

param(
    [int]$Port = 0,
    [string]$Host = "",
    [int]$Workers = 0,
    [string]$LogLevel = "",
    [string]$LogFile = "",
    [switch]$Dev,
    [switch]$NoBrowser,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# --- Paths ---
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VenvDir = Join-Path $ProjectRoot ".venv"
$ConfigFile = Join-Path $ProjectRoot "config\default.yaml"
$EnvFile = Join-Path $ProjectRoot ".env"
$LogDir = Join-Path $ProjectRoot "logs"

# --- Color helpers ---
function Write-Info {
    Write-Host "[INFO] " -ForegroundColor Blue -NoNewline
    Write-Host $args
}

function Write-Ok {
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $args
}

function Write-Warn {
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $args
}

function Write-Err {
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $args
    exit 1
}

# =============================================================
# Help
# =============================================================
if ($Help) {
    @"
Usage: .\scripts\start.ps1 [OPTIONS]

Options:
  -Port PORT       Server port (default: 8000)
  -Host HOST       Server host (default: 127.0.0.1)
  -Workers N       Number of workers (default: 1)
  -LogLevel LVL    Log level: debug|info|warning|error (default: info)
  -LogFile FILE    Log to file instead of stdout
  -Dev             Development mode with auto-reload
  -NoBrowser       Don't open browser on start
  -Help            Show this help

Environment Variables:
  LMO_HOST         Server host (overridden by -Host)
  LMO_PORT         Server port (overridden by -Port)
  LMO_WORKERS      Worker count (overridden by -Workers)
  LMO_LOG_LEVEL    Log level (overridden by -LogLevel)

Examples:
  .\scripts\start.ps1                        # Start with defaults
  .\scripts\start.ps1 -Port 9000             # Custom port
  .\scripts\start.ps1 -Dev                   # Development mode
  .\scripts\start.ps1 -Workers 4 -LogLevel debug
"@
    exit 0
}

# =============================================================
# Banner
# =============================================================
Write-Host @"

  _                    _    ___ _   ___
 | |   ___  __ _ __ _ / _  / _ \ | | __|
 | |__/ _ \/ _` / _` | (_) | (_) | | _|
 |____\___/\__, |\__,_|\___/ \___/  |_|
           |___/ Optimizer Server

"@ -ForegroundColor Cyan

# =============================================================
# Pre-flight Checks
# =============================================================

# Check virtual environment
if (-not (Test-Path $VenvDir)) {
    Write-Err @"
Virtual environment not found at $VenvDir
Run the installer first: .\scripts\install.ps1
"@
}

# Activate virtual environment
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    . $ActivateScript
    Write-Ok "Activated virtual environment"
} else {
    Write-Err "Activation script not found at $ActivateScript"
}

# Load .env file if exists
if (Test-Path $EnvFile) {
    Write-Info "Loading environment from $EnvFile"
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and !$line.StartsWith("#") -and $line -match "^([^=]+)=(.*)$") {
            $key = $Matches[1].Trim()
            $value = $Matches[2].Trim()
            # Remove surrounding quotes if present
            if ($value -match "^['""](.*)['""]$") {
                $value = $Matches[1]
            }
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Load config file info
if (Test-Path $ConfigFile) {
    Write-Info "Using config: $ConfigFile"
}

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# =============================================================
# Resolve Configuration (env vars -> params -> defaults)
# =============================================================
if ($Host -eq "") {
    $Host = if ($env:LMO_HOST) { $env:LMO_HOST } else { "127.0.0.1" }
}
if ($Port -eq 0) {
    $Port = if ($env:LMO_PORT) { [int]$env:LMO_PORT } else { 8000 }
}
if ($Workers -eq 0) {
    $Workers = if ($env:LMO_WORKERS) { [int]$env:LMO_WORKERS } else { 1 }
}
if ($LogLevel -eq "") {
    $LogLevel = if ($env:LMO_LOG_LEVEL) { $env:LMO_LOG_LEVEL.ToLower() } else { "info" }
}

# =============================================================
# Check for Models
# =============================================================
function Test-Models {
    $modelDir = Join-Path $ProjectRoot "models"

    if (-not (Test-Path $modelDir)) {
        Write-Warn "Models directory not found at $modelDir"
        return
    }

    $modelFiles = Get-ChildItem -Path $modelDir -Recurse -Include "*.gguf", "*.bin" -ErrorAction SilentlyContinue

    if ($modelFiles.Count -eq 0) {
        Write-Warn "No model files found in $modelDir"
        Write-Warn "Download a model before making inference requests:"
        Write-Warn "  .\scripts\download_model.ps1"
        Write-Host ""
    } else {
        Write-Ok "Found $($modelFiles.Count) model file(s)"
    }
}

Test-Models

# =============================================================
# Display Server Information
# =============================================================
Write-Host ""
Write-Info "Starting Local Model Optimizer..."
Write-Host "  Host:      $Host"
Write-Host "  Port:      $Port"
Write-Host "  Workers:   $Workers"
Write-Host "  Log Level: $LogLevel"
if ($LogFile -ne "") {
    Write-Host "  Log File:  $LogFile"
}
if ($Dev) {
    Write-Host "  Mode:      Development (auto-reload)"
}
Write-Host ""

# Security warning for binding to all interfaces
if ($Host -eq "0.0.0.0") {
    Write-Warn "Server is binding to ALL network interfaces (0.0.0.0)"
    Write-Warn "This exposes the API to your entire network."
    Write-Warn "For local use only, set LMO_HOST=127.0.0.1 or use -Host 127.0.0.1"
    Write-Host ""
}

# =============================================================
# Check Port Availability
# =============================================================
function Test-Port {
    param([int]$Port)

    try {
        $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
        $portInUse = $listeners | Where-Object { $_.Port -eq $Port }

        if ($portInUse) {
            Write-Warn "Port $Port is already in use"
            $response = Read-Host "Continue anyway? [y/N]"
            if ($response -notmatch "^[Yy]$") {
                exit 0
            }
        }
    } catch {
        # Ignore errors in port check
    }
}

Test-Port -Port $Port

# =============================================================
# Start Server
# =============================================================
Set-Location $ProjectRoot

# Build uvicorn arguments
$uvicornArgs = @(
    "src.api.server:app",
    "--host", $Host,
    "--port", $Port,
    "--log-level", $LogLevel
)

if ($Dev) {
    $uvicornArgs += "--reload"
    $uvicornArgs += "--reload-dir"
    $uvicornArgs += "src"
} else {
    if ($Workers -gt 1) {
        $uvicornArgs += "--workers"
        $uvicornArgs += $Workers
    }
}

# Trap for graceful shutdown
$serverProcess = $null

function Stop-Server {
    Write-Host ""
    Write-Info "Shutting down server..."
    if ($serverProcess -and -not $serverProcess.HasExited) {
        $serverProcess.Kill()
        $serverProcess.WaitForExit(5000)
    }
    Write-Ok "Server stopped"
}

# Register cleanup on Ctrl+C
[Console]::TreatControlCAsInput = $false

try {
    Write-Ok "Starting uvicorn server..."
    Write-Host "API Documentation: http://${Host}:${Port}/docs" -ForegroundColor Cyan
    Write-Host ""

    # 使用 venv 中的 Python 可执行文件
    $PythonExe = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path $PythonExe)) {
        $PythonExe = "python"  # 回退到系统 Python
    }

    if ($LogFile -ne "") {
        # Start with log file
        $logPath = Join-Path $ProjectRoot $LogFile
        $allArgs = @("-m", "uvicorn") + $uvicornArgs
        $serverProcess = Start-Process -FilePath $PythonExe `
            -ArgumentList $allArgs `
            -NoNewWindow `
            -PassThru `
            -RedirectStandardOutput $logPath `
            -RedirectStandardError (Join-Path $LogDir "stderr.log")
    } else {
        # Start in current window
        $allArgs = @("-m", "uvicorn") + $uvicornArgs
        $serverProcess = Start-Process -FilePath $PythonExe `
            -ArgumentList $allArgs `
            -NoNewWindow `
            -PassThru
    }

    # Wait for server to start
    Start-Sleep -Seconds 2

    if ($serverProcess -and -not $serverProcess.HasExited) {
        Write-Ok "Server started (PID: $($serverProcess.Id))"
        Write-Host ""
        Write-Host "============================================" -ForegroundColor Green
        Write-Host "  Server is running!" -ForegroundColor Green
        Write-Host "============================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "  API Docs:  http://${Host}:${Port}/docs"
        Write-Host "  Health:    http://${Host}:${Port}/health"
        Write-Host ""
        Write-Host "  Press Ctrl+C to stop the server"
        Write-Host ""

        # Open browser if not disabled
        if (-not $NoBrowser -and -not $Dev) {
            try {
                Start-Process "http://localhost:${Port}/docs"
            } catch {
                # Ignore browser open errors
            }
        }

        # Wait for the server process to exit
        $serverProcess.WaitForExit()
    } else {
        Write-Err "Server failed to start. Check the logs for details."
    }
} catch {
    Write-Err "Error starting server: $_"
} finally {
    Stop-Server
}
