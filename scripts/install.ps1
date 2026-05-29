#Requires -Version 5.1
# =============================================================
# Local Model Optimizer - Windows Installation Script
# =============================================================
# Usage: .\scripts\install.ps1 [-SkipVenv] [-Force] [-NoGpu]
# =============================================================

param(
    [switch]$SkipVenv,
    [switch]$Force,
    [switch]$NoGpu,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# --- Paths ---
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VenvDir = Join-Path $ProjectRoot ".venv"
$ConfigExample = Join-Path $ProjectRoot "config\default.example.yaml"
$ConfigFile = Join-Path $ProjectRoot "config\default.yaml"
$EnvExample = Join-Path $ProjectRoot ".env.example"
$EnvFile = Join-Path $ProjectRoot ".env"

# --- Python requirements ---
$PythonMinVersion = [version]"3.10.0"

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

function Write-Step {
    Write-Host ""
    Write-Host ">>> $args" -ForegroundColor Magenta
}

# =============================================================
# Help
# =============================================================
if ($Help) {
    @"
Usage: .\scripts\install.ps1 [OPTIONS]

Options:
  -SkipVenv    Skip virtual environment creation
  -Force       Force recreate virtual environment
  -NoGpu       Skip GPU detection, use CPU-only build
  -Help        Show this help

Examples:
  .\scripts\install.ps1                    # Full installation
  .\scripts\install.ps1 -Force             # Recreate venv from scratch
  .\scripts\install.ps1 -NoGpu             # CPU-only installation
  .\scripts\install.ps1 -SkipVenv          # Install deps without venv
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
           |___/ Optimizer Installer

"@ -ForegroundColor Cyan

# =============================================================
# Step 1: Detect and Validate Python
# =============================================================
Write-Step "Detecting Python..."

$PythonExe = $null
$PythonVersion = $null

# Try common Python commands
$pythonCommands = @("python3.12", "python3.11", "python3.10", "python3", "python")

foreach ($cmd in $pythonCommands) {
    try {
        $output = & $cmd --version 2>&1
        if ($output -match "Python (\d+\.\d+\.\d+)") {
            $ver = [version]$Matches[1]
            if ($ver -ge $PythonMinVersion) {
                $PythonExe = (Get-Command $cmd -ErrorAction Stop).Source
                $PythonVersion = $ver
                break
            }
        }
    } catch {
        continue
    }
}

if (-not $PythonExe) {
    Write-Err @"
Python >= 3.10 not found.
Please install Python from https://www.python.org/downloads/
Make sure to check 'Add Python to PATH' during installation.
"@
}

Write-Ok "Found: Python $PythonVersion at $PythonExe"

# Check for pip
try {
    & $PythonExe -m pip --version | Out-Null
} catch {
    Write-Warn "pip not found. Installing pip..."
    & $PythonExe -m ensurepip --upgrade
}

# =============================================================
# Step 2: Create Virtual Environment
# =============================================================
if (-not $SkipVenv) {
    Write-Step "Creating virtual environment..."

    if (Test-Path $VenvDir) {
        if ($Force) {
            Write-Warn "Removing existing virtual environment (-Force)"
            Remove-Item -Recurse -Force $VenvDir
        } else {
            Write-Warn "Virtual environment already exists at $VenvDir"
            $response = Read-Host "Recreate? [y/N]"
            if ($response -notmatch "^[Yy]$") {
                Write-Info "Using existing virtual environment"
            } else {
                Remove-Item -Recurse -Force $VenvDir
            }
        }
    }

    if (-not (Test-Path $VenvDir)) {
        & $PythonExe -m venv $VenvDir
        Write-Ok "Virtual environment created at $VenvDir"
    }

    # Activate virtual environment
    $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $ActivateScript) {
        . $ActivateScript
    } else {
        Write-Err "Activation script not found at $ActivateScript"
    }

    # Upgrade pip
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel 2>&1 | Select-Object -Last 1
} else {
    Write-Warn "Skipping virtual environment creation (-SkipVenv)"
    if (Test-Path $VenvDir) {
        $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
        if (Test-Path $ActivateScript) {
            . $ActivateScript
        }
    }
}

# =============================================================
# Step 3: Detect GPU Hardware
# =============================================================
Write-Step "Detecting GPU hardware..."

$GpuType = "none"
$GpuName = ""
$GpuMemory = 0

if (-not $NoGpu) {
    # Try NVIDIA first
    try {
        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
        $gpuInfo = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>&1 | Select-Object -First 1

        if ($gpuInfo -match "(.+?),\s*(\d+),\s*(.+)") {
            $GpuType = "nvidia"
            $GpuName = $Matches[1].Trim()
            $GpuMemory = [int]$Matches[2]
            $driverVersion = $Matches[3].Trim()

            # Get CUDA version
            $cudaVersion = "unknown"
            try {
                $nvidiaSmiOutput = & nvidia-smi 2>&1
                if ($nvidiaSmiOutput -match "CUDA Version:\s+([\d.]+)") {
                    $cudaVersion = $Matches[1]
                }
            } catch {}

            Write-Ok "NVIDIA GPU: $GpuName (${GpuMemory}MB, Driver: $driverVersion, CUDA: $cudaVersion)"
        }
    } catch {
        # Try AMD
        try {
            $amdGpu = Get-WmiObject Win32_VideoController -ErrorAction Stop |
                Where-Object { $_.Name -match "AMD|Radeon" } |
                Select-Object -First 1

            if ($amdGpu) {
                $GpuType = "amd"
                $GpuName = $amdGpu.Name
                Write-Ok "AMD GPU: $GpuName"
            }
        } catch {}

        # Try Intel Arc
        if ($GpuType -eq "none") {
            try {
                $intelGpu = Get-WmiObject Win32_VideoController -ErrorAction Stop |
                    Where-Object { $_.Name -match "Intel.*Arc|Intel.*Graphics" } |
                    Select-Object -First 1

                if ($intelGpu) {
                    $GpuType = "intel"
                    $GpuName = $intelGpu.Name
                    Write-Ok "Intel GPU: $GpuName"
                }
            } catch {}
        }
    }
} else {
    Write-Warn "GPU detection skipped (-NoGpu), using CPU-only mode"
}

if ($GpuType -eq "none") {
    Write-Warn "No GPU detected, will build llama-cpp-python for CPU-only mode"
}

# =============================================================
# Step 4: Install Dependencies
# =============================================================
Write-Step "Installing Python dependencies..."

Set-Location $ProjectRoot

# Install core dependencies (excluding llama-cpp-python)
Write-Info "Installing core dependencies..."
$requirements = Get-Content "requirements.txt" |
    Where-Object { $_ -notmatch "^\s*#" } |
    Where-Object { $_ -notmatch "^\s*$" } |
    Where-Object { $_ -notmatch "llama-cpp-python" }

foreach ($dep in $requirements) {
    pip install $dep 2>&1 | Select-Object -Last 1
}

Write-Ok "Core dependencies installed"

# Install llama-cpp-python with GPU support
Write-Step "Installing llama-cpp-python..."

switch ($GpuType) {
    "nvidia" {
        Write-Info "Building with CUDA support..."
        $env:CMAKE_ARGS = "-DGGML_CUDA=on"
        $env:FORCE_CMAKE = "1"
        pip install llama-cpp-python --force-rebuild --no-cache-dir 2>&1 | Select-Object -Last 5
    }
    "amd" {
        Write-Info "Building with ROCm support..."
        $env:CMAKE_ARGS = "-DGGML_HIP=on"
        $env:FORCE_CMAKE = "1"
        pip install llama-cpp-python --force-rebuild --no-cache-dir 2>&1 | Select-Object -Last 5
    }
    "intel" {
        Write-Info "Building with SYCL support..."
        $env:CMAKE_ARGS = "-DGGML_SYCL=on"
        $env:FORCE_CMAKE = "1"
        pip install llama-cpp-python --force-rebuild --no-cache-dir 2>&1 | Select-Object -Last 5
    }
    default {
        Write-Info "Building CPU-only version..."
        pip install llama-cpp-python 2>&1 | Select-Object -Last 5
    }
}

# Clean up environment variables
Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue
Remove-Item Env:FORCE_CMAKE -ErrorAction SilentlyContinue

Write-Ok "llama-cpp-python installed"

# =============================================================
# Step 5: Create Project Directories
# =============================================================
Write-Step "Creating project directories..."

$dirs = @(
    # Model directories
    "models",
    "models\llm",
    "models\vision",
    "models\audio",

    # Data directories
    "data",
    "data\layer_cache",

    # Log directory
    "logs",

    # Config directory
    "config",

    # Cache directories
    ".cache",
    ".cache\huggingface",

    # Benchmark results
    "benchmark_results"
)

foreach ($dir in $dirs) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}

# Create .gitkeep files for empty directories
$gitkeepDirs = @("models\llm", "models\vision", "models\audio", "data", "data\layer_cache", "logs")
foreach ($dir in $gitkeepDirs) {
    $gitkeepPath = Join-Path $ProjectRoot "$dir\.gitkeep"
    if (-not (Test-Path $gitkeepPath)) {
        New-Item -ItemType File -Path $gitkeepPath -Force | Out-Null
    }
}

Write-Ok "Project directories created"

# =============================================================
# Step 6: Create Default Configuration Files
# =============================================================
Write-Step "Setting up configuration files..."

# Create default.yaml from example
if (-not (Test-Path $ConfigFile)) {
    if (Test-Path $ConfigExample) {
        Copy-Item $ConfigExample $ConfigFile
        Write-Ok "Created config\default.yaml from example"
    } else {
        Write-Warn "Config example not found at $ConfigExample"
    }
} else {
    Write-Info "Config file already exists at $ConfigFile"
}

# Create .env from example
if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Write-Ok "Created .env from example"
    } else {
        Write-Warn ".env example not found at $EnvExample"
    }
} else {
    Write-Info ".env file already exists at $EnvFile"
}

# =============================================================
# Step 7: Verify Installation
# =============================================================
Write-Step "Verifying installation..."

$errors = 0

# Check critical imports
$criticalModules = @("fastapi", "uvicorn", "pydantic", "psutil", "yaml")

foreach ($module in $criticalModules) {
    try {
        python -c "import $module" 2>&1 | Out-Null
        Write-Ok "Import: $module"
    } catch {
        Write-Warn "Failed to import: $module"
        $errors++
    }
}

# Check llama-cpp-python
try {
    python -c "import llama_cpp" 2>&1 | Out-Null
    Write-Ok "Import: llama_cpp"

    try {
        $llamaVer = python -c "import llama_cpp; print(llama_cpp.__version__)" 2>&1
        Write-Info "llama-cpp-python version: $llamaVer"
    } catch {}
} catch {
    Write-Warn "Failed to import: llama_cpp"
    $errors++
}

if ($errors -gt 0) {
    Write-Warn "$errors module(s) failed to import. Installation may be incomplete."
} else {
    Write-Ok "All critical modules verified"
}

# =============================================================
# Print Summary
# =============================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Python:        $PythonVersion"
Write-Host "  Virtual Env:   $VenvDir"
Write-Host "  GPU:           $GpuType $GpuName"
Write-Host "  Project Root:  $ProjectRoot"
Write-Host ""
Write-Host "  Next Steps:" -ForegroundColor Cyan
Write-Host "    1. Activate virtual environment:"
Write-Host "       .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "    2. Download a model:"
Write-Host "       .\scripts\download_model.ps1"
Write-Host ""
Write-Host "    3. Start the service:"
Write-Host "       .\scripts\start.ps1"
Write-Host ""
Write-Host "  Quick Start:" -ForegroundColor Cyan
Write-Host "       .\scripts\download_model.ps1 -Repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF -Quant Q4_K_M"
Write-Host "       .\scripts\start.ps1"
Write-Host ""
