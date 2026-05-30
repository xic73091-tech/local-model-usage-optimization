#!/usr/bin/env bash
# =============================================================
# Local Model Optimizer - Linux/macOS Installation Script
# =============================================================
# Usage: bash scripts/install.sh [--skip-venv] [--force] [--no-gpu]
# =============================================================

set -e

# --- Color definitions ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# --- Paths ---
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
CONFIG_EXAMPLE="$PROJECT_ROOT/config/default.example.yaml"
CONFIG_FILE="$PROJECT_ROOT/config/default.yaml"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"
ENV_FILE="$PROJECT_ROOT/.env"

# --- Python requirements ---
PYTHON_MIN_VERSION="3.10"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# --- Flags ---
SKIP_VENV=false
FORCE_RECREATE=false
NO_GPU=false

# --- Helpers ---
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step()    { echo -e "\n${MAGENTA}>>> $1${NC}"; }

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-venv)  SKIP_VENV=true; shift ;;
        --force)      FORCE_RECREATE=true; shift ;;
        --no-gpu)     NO_GPU=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-venv    Skip virtual environment creation"
            echo "  --force        Force recreate virtual environment"
            echo "  --no-gpu       Skip GPU detection, use CPU-only build"
            echo "  -h, --help     Show this help"
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# =============================================================
# Banner
# =============================================================
print_banner() {
    echo -e "${CYAN}"
    cat << 'BANNER'
  _                    _    ___ _   ___
 | |   ___  __ _ __ _ / _  / _ \ | | __|
 | |__/ _ \/ _` / _` | (_) | (_) | | _|
 |____\___/\__, |\__,_|\___/ \___/  |_|
           |___/ Optimizer Installer
BANNER
    echo -e "${NC}"
}

# =============================================================
# Step 1: Detect Operating System
# =============================================================
detect_os() {
    step "Detecting operating system..."

    OS="unknown"
    OS_VERSION=""

    case "$(uname -s)" in
        Linux*)
            OS="Linux"
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                OS_VERSION="$NAME $VERSION_ID"
            fi
            ;;
        Darwin*)
            OS="macOS"
            OS_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
            ;;
        CYGWIN*|MINGW*|MSYS*)
            error "This script is for Linux/macOS. Use scripts/install.ps1 for Windows."
            ;;
        *)
            error "Unsupported OS: $(uname -s)"
            ;;
    esac

    ARCH=$(uname -m)
    success "Detected: $OS $OS_VERSION ($ARCH)"
}

# =============================================================
# Step 2: Detect and Validate Python
# =============================================================
detect_python() {
    step "Detecting Python..."

    PYTHON=""
    PYTHON_VERSION=""

    # Try common Python commands in order of preference
    for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            version_output=$("$cmd" --version 2>&1)
            if [[ "$version_output" =~ Python[[:space:]]+([0-9]+\.[0-9]+(\.[0-9]+)?) ]]; then
                version="${BASH_REMATCH[1]}"
                major=$(echo "$version" | cut -d. -f1)
                minor=$(echo "$version" | cut -d. -f2)

                if [ "$major" -ge "$PYTHON_MIN_MAJOR" ] && [ "$minor" -ge "$PYTHON_MIN_MINOR" ]; then
                    PYTHON="$cmd"
                    PYTHON_VERSION="$version"
                    break
                fi
            fi
        fi
    done

    if [ -z "$PYTHON" ]; then
        error "Python >= $PYTHON_MIN_VERSION not found.
  Please install Python $PYTHON_MIN_VERSION+ from https://www.python.org/downloads/
  Or use your system package manager:
    Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip
    macOS: brew install python@3.12"
    fi

    PYTHON_PATH=$(command -v "$PYTHON")
    success "Found: Python $PYTHON_VERSION at $PYTHON_PATH"

    # Check for pip
    if ! "$PYTHON" -m pip --version &>/dev/null; then
        warn "pip not found. Installing pip..."
        "$PYTHON" -m ensurepip --upgrade 2>/dev/null || {
            error "Failed to install pip. Please install pip manually."
        }
    fi
}

# =============================================================
# Step 3: Create Virtual Environment
# =============================================================
create_venv() {
    if [ "$SKIP_VENV" = true ]; then
        warn "Skipping virtual environment creation (--skip-venv)"
        if [ -d "$VENV_DIR" ]; then
            source "$VENV_DIR/bin/activate"
        fi
        return
    fi

    step "Creating virtual environment..."

    if [ -d "$VENV_DIR" ]; then
        if [ "$FORCE_RECREATE" = true ]; then
            warn "Removing existing virtual environment (--force)"
            rm -rf "$VENV_DIR"
        else
            warn "Virtual environment already exists at $VENV_DIR"
            read -p "Recreate? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                info "Using existing virtual environment"
                source "$VENV_DIR/bin/activate"
                return
            fi
            rm -rf "$VENV_DIR"
        fi
    fi

    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"

    # Activate
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel 2>&1 | tail -1
}

# =============================================================
# Step 4: Detect GPU Hardware
# =============================================================
detect_gpu() {
    if [ "$NO_GPU" = true ]; then
        GPU_TYPE="none"
        GPU_NAME=""
        GPU_MEMORY=0
        warn "GPU detection skipped (--no-gpu), using CPU-only mode"
        return
    fi

    step "Detecting GPU hardware..."

    GPU_TYPE="none"
    GPU_NAME=""
    GPU_MEMORY=0
    CUDA_VERSION=""

    # --- NVIDIA GPU ---
    if command -v nvidia-smi &>/dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_info" ]; then
            GPU_TYPE="nvidia"
            GPU_NAME=$(echo "$gpu_info" | cut -d',' -f1 | xargs)
            GPU_MEMORY=$(echo "$gpu_info" | cut -d',' -f2 | xargs)
            driver_version=$(echo "$gpu_info" | cut -d',' -f3 | xargs)

            # Get CUDA version if available
            CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s+\K[0-9.]+' || echo "unknown")

            success "NVIDIA GPU: $GPU_NAME (${GPU_MEMORY}MB, Driver: $driver_version, CUDA: $CUDA_VERSION)"
        fi

    # --- AMD GPU (Linux) ---
    elif [ "$OS" = "Linux" ]; then
        if command -v rocminfo &>/dev/null; then
            amd_gpu=$(rocminfo 2>/dev/null | grep -A5 "Marketing" | grep "Marketing Name" | head -1 | cut -d: -f2 | xargs)
            if [ -n "$amd_gpu" ]; then
                GPU_TYPE="amd"
                GPU_NAME="$amd_gpu"
                success "AMD GPU: $GPU_NAME (ROCm)"
            fi
        elif lspci 2>/dev/null | grep -qi "amd\|radeon"; then
            GPU_TYPE="amd"
            GPU_NAME=$(lspci 2>/dev/null | grep -i "vga" | grep -i "amd\|radeon" | head -1 | sed 's/.*: //' | xargs)
            warn "AMD GPU detected: $GPU_NAME (ROCm may not be installed)"
        fi

    # --- Apple Silicon (macOS) ---
    elif [ "$OS" = "macOS" ]; then
        chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
        if echo "$chip" | grep -qi "apple"; then
            GPU_TYPE="apple"
            GPU_NAME="$chip"
            # Unified memory - use total system memory as reference
            total_mem=$(sysctl -n hw.memsize 2>/dev/null)
            if [ -n "$total_mem" ]; then
                GPU_MEMORY=$((total_mem / 1024 / 1024))
            fi
            success "Apple Silicon: $GPU_NAME (Unified Memory)"
        fi
    fi

    if [ "$GPU_TYPE" = "none" ]; then
        warn "No GPU detected, will build llama-cpp-python for CPU-only mode"
    fi
}

# =============================================================
# Step 5: Install Dependencies
# =============================================================
install_dependencies() {
    step "Installing Python dependencies..."

    cd "$PROJECT_ROOT"

    # Install core dependencies (excluding llama-cpp-python which needs special handling)
    info "Installing core dependencies..."
    local install_errors=0
    while read -r dep; do
        [ -z "$dep" ] && continue
        if ! pip install "$dep" 2>&1 | tail -1; then
            warn "Failed to install: $dep"
            ((install_errors++))
        fi
    done < <(grep -v "^#" requirements.txt | grep -v "llama-cpp-python" | grep -v "^$")

    if [ $install_errors -gt 0 ]; then
        warn "$install_errors dependency(ies) failed to install"
    fi
    success "Core dependencies installed"

    # Install llama-cpp-python with appropriate GPU support
    install_llama_cpp
}

install_llama_cpp() {
    step "Installing llama-cpp-python..."

    case "$GPU_TYPE" in
        nvidia)
            info "Building with CUDA support..."
            # Detect CUDA version for proper build
            if command -v nvcc &>/dev/null; then
                cuda_ver=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "")
                info "NVCC CUDA version: ${cuda_ver:-unknown}"
            fi
            CMAKE_ARGS="-DGGML_CUDA=on" \
            FORCE_CMAKE=1 \
            pip install llama-cpp-python --force-rebuild --no-cache-dir 2>&1 | tail -5
            ;;
        amd)
            info "Building with ROCm support..."
            CMAKE_ARGS="-DGGML_HIP=on" \
            FORCE_CMAKE=1 \
            pip install llama-cpp-python --force-rebuild --no-cache-dir 2>&1 | tail -5
            ;;
        apple)
            info "Building with Metal support..."
            CMAKE_ARGS="-DGGML_METAL=on" \
            FORCE_CMAKE=1 \
            pip install llama-cpp-python --force-rebuild --no-cache-dir 2>&1 | tail -5
            ;;
        *)
            info "Building CPU-only version..."
            pip install llama-cpp-python 2>&1 | tail -5
            ;;
    esac

    success "llama-cpp-python installed"
}

# =============================================================
# Step 6: Create Project Directories
# =============================================================
create_directories() {
    step "Creating project directories..."

    local dirs=(
        # Model directories
        "$PROJECT_ROOT/models"
        "$PROJECT_ROOT/models/llm"
        "$PROJECT_ROOT/models/vision"
        "$PROJECT_ROOT/models/audio"

        # Data directories
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/data/layer_cache"

        # Log directory
        "$PROJECT_ROOT/logs"

        # Config directory
        "$PROJECT_ROOT/config"

        # Cache directories
        "$PROJECT_ROOT/.cache"
        "$PROJECT_ROOT/.cache/huggingface"

        # Benchmark results
        "$PROJECT_ROOT/benchmark_results"
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done

    # Create .gitkeep files for empty directories
    for dir in "models/llm" "models/vision" "models/audio" "data" "data/layer_cache" "logs"; do
        gitkeep="$PROJECT_ROOT/$dir/.gitkeep"
        if [ ! -f "$gitkeep" ]; then
            touch "$gitkeep" 2>/dev/null || true
        fi
    done

    success "Project directories created"
}

# =============================================================
# Step 7: Create Default Configuration Files
# =============================================================
create_config_files() {
    step "Setting up configuration files..."

    # Create default.yaml from example
    if [ ! -f "$CONFIG_FILE" ]; then
        if [ -f "$CONFIG_EXAMPLE" ]; then
            cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
            success "Created config/default.yaml from example"
        else
            warn "Config example not found at $CONFIG_EXAMPLE"
        fi
    else
        info "Config file already exists at $CONFIG_FILE"
    fi

    # Create .env from example
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "$ENV_EXAMPLE" ]; then
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            success "Created .env from example"
        else
            warn ".env example not found at $ENV_EXAMPLE"
        fi
    else
        info ".env file already exists at $ENV_FILE"
    fi
}

# =============================================================
# Step 8: Verify Installation
# =============================================================
verify_installation() {
    step "Verifying installation..."

    local errors=0

    # Check critical imports
    for module in fastapi uvicorn pydantic psutil yaml; do
        if python -c "import $module" 2>/dev/null; then
            success "Import: $module"
        else
            warn "Failed to import: $module"
            ((errors++))
        fi
    done

    # Check llama-cpp-python
    if python -c "import llama_cpp" 2>/dev/null; then
        success "Import: llama_cpp"
        # Get version
        llama_ver=$(python -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null || echo "unknown")
        info "llama-cpp-python version: $llama_ver"
    else
        warn "Failed to import: llama_cpp"
        ((errors++))
    fi

    if [ $errors -gt 0 ]; then
        warn "$errors module(s) failed to import. Installation may be incomplete."
    else
        success "All critical modules verified"
    fi
}

# =============================================================
# Print Summary
# =============================================================
print_summary() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Installation Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "  OS:            $OS $OS_VERSION ($ARCH)"
    echo "  Python:        $PYTHON_VERSION"
    echo "  Virtual Env:   $VENV_DIR"
    echo "  GPU:           ${GPU_TYPE:-none} ${GPU_NAME}"
    echo "  Project Root:  $PROJECT_ROOT"
    echo ""
    echo -e "${CYAN}  Next Steps:${NC}"
    echo "    1. Activate virtual environment:"
    echo "       source .venv/bin/activate"
    echo ""
    echo "    2. Download a model:"
    echo "       bash scripts/download_model.sh"
    echo ""
    echo "    3. Start the service:"
    echo "       bash scripts/start.sh"
    echo ""
    echo -e "${CYAN}  Quick Start:${NC}"
    echo "       bash scripts/download_model.sh --repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF --quant Q4_K_M"
    echo "       bash scripts/start.sh"
    echo ""
}

# =============================================================
# Main Installation Flow
# =============================================================
main() {
    print_banner

    detect_os
    detect_python
    create_venv
    detect_gpu
    install_dependencies
    create_directories
    create_config_files
    verify_installation
    print_summary
}

main "$@"
