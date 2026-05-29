#!/usr/bin/env bash
# =============================================================
# Local Model Optimizer - Linux/macOS Start Script
# =============================================================
# Usage: bash scripts/start.sh [OPTIONS]
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
CONFIG_FILE="$PROJECT_ROOT/config/default.yaml"
ENV_FILE="$PROJECT_ROOT/.env"
LOG_DIR="$PROJECT_ROOT/logs"

# --- Helpers ---
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- Default configuration ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"
LOG_FILE="${LOG_FILE:-}"
DEV_MODE=false
NO_BROWSER=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)       PORT="$2"; shift 2 ;;
        --host)       HOST="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --log-level)  LOG_LEVEL="$2"; shift 2 ;;
        --log-file)   LOG_FILE="$2"; shift 2 ;;
        --dev)        DEV_MODE=true; shift ;;
        --no-browser) NO_BROWSER=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT       Server port (default: 8000)"
            echo "  --host HOST       Server host (default: 0.0.0.0)"
            echo "  --workers N       Number of workers (default: 1)"
            echo "  --log-level LVL   Log level: debug|info|warning|error (default: info)"
            echo "  --log-file FILE   Log to file instead of stdout"
            echo "  --dev             Development mode with auto-reload"
            echo "  --no-browser      Don't open browser on start"
            echo "  -h, --help        Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  HOST              Server host (overridden by --host)"
            echo "  PORT              Server port (overridden by --port)"
            echo "  WORKERS           Worker count (overridden by --workers)"
            echo "  LOG_LEVEL         Log level (overridden by --log-level)"
            exit 0
            ;;
        *) error "Unknown option: $1. Use --help for usage." ;;
    esac
done

# =============================================================
# Banner
# =============================================================
echo -e "${CYAN}"
cat << 'BANNER'
  _                    _    ___ _   ___
 | |   ___  __ _ __ _ / _  / _ \ | | __|
 | |__/ _ \/ _` / _` | (_) | (_) | | _|
 |____\___/\__, |\__,_|\___/ \___/  |_|
           |___/ Optimizer Server
BANNER
echo -e "${NC}"

# =============================================================
# Pre-flight Checks
# =============================================================

# Check virtual environment
if [ ! -d "$VENV_DIR" ]; then
    error "Virtual environment not found at $VENV_DIR
  Run the installer first: bash scripts/install.sh"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
success "Activated virtual environment"

# Load .env file if exists
if [ -f "$ENV_FILE" ]; then
    info "Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
fi

# Load config file info
if [ -f "$CONFIG_FILE" ]; then
    info "Using config: $CONFIG_FILE"
fi

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# =============================================================
# Check for Models
# =============================================================
check_models() {
    local model_dir="$PROJECT_ROOT/models"
    local has_models=false

    # Check for any model files
    if find "$model_dir" -name "*.gguf" -o -name "*.bin" 2>/dev/null | head -1 | grep -q .; then
        has_models=true
    fi

    if [ "$has_models" = false ]; then
        warn "No model files found in $model_dir"
        warn "Download a model before making inference requests:"
        warn "  bash scripts/download_model.sh"
        echo ""
    else
        local model_count=$(find "$model_dir" -name "*.gguf" -o -name "*.bin" 2>/dev/null | wc -l)
        success "Found $model_count model file(s)"
    fi
}

check_models

# =============================================================
# Display Server Information
# =============================================================
echo ""
info "Starting Local Model Optimizer..."
echo "  Host:      $HOST"
echo "  Port:      $PORT"
echo "  Workers:   $WORKERS"
echo "  Log Level: $LOG_LEVEL"
if [ -n "$LOG_FILE" ]; then
    echo "  Log File:  $LOG_FILE"
fi
if [ "$DEV_MODE" = true ]; then
    echo "  Mode:      Development (auto-reload)"
fi
echo ""

# =============================================================
# Check Port Availability
# =============================================================
check_port() {
    local port=$1
    if command -v lsof &>/dev/null; then
        if lsof -i :$port -sTCP:LISTEN &>/dev/null; then
            warn "Port $port is already in use"
            local pid=$(lsof -ti :$port -sTCP:LISTEN 2>/dev/null | head -1)
            if [ -n "$pid" ]; then
                local proc_name=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
                warn "Process using port: $proc_name (PID: $pid)"
            fi
            read -p "Continue anyway? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 0
            fi
        fi
    elif command -v ss &>/dev/null; then
        if ss -tlnp 2>/dev/null | grep -q ":$port "; then
            warn "Port $port may already be in use"
        fi
    fi
}

check_port "$PORT"

# =============================================================
# Start Server
# =============================================================
cd "$PROJECT_ROOT"

# Build uvicorn command
UVICORN_ARGS=(
    "src.api.server:app"
    "--host" "$HOST"
    "--port" "$PORT"
    "--log-level" "$LOG_LEVEL"
)

if [ "$DEV_MODE" = true ]; then
    UVICORN_ARGS+=("--reload" "--reload-dir" "src")
else
    if [ "$WORKERS" -gt 1 ]; then
        UVICORN_ARGS+=("--workers" "$WORKERS")
    fi
fi

# Add log file if specified
if [ -n "$LOG_FILE" ]; then
    UVICORN_ARGS+=("--log-config" "/dev/null")
fi

# Trap for graceful shutdown
cleanup() {
    echo ""
    info "Shutting down server..."
    # Send SIGTERM to the server process
    if [ -n "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    success "Server stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start the server
success "Starting uvicorn server..."
echo -e "${CYAN}API Documentation: http://$HOST:$PORT/docs${NC}"
echo ""

if [ -n "$LOG_FILE" ]; then
    # Log to file and stdout
    uvicorn "${UVICORN_ARGS[@]}" 2>&1 | tee "$LOG_FILE" &
    SERVER_PID=$!
else
    uvicorn "${UVICORN_ARGS[@]}" &
    SERVER_PID=$!
fi

# Wait for server to start
sleep 2

if kill -0 "$SERVER_PID" 2>/dev/null; then
    success "Server started (PID: $SERVER_PID)"
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Server is running!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "  API Docs:  http://$HOST:$PORT/docs"
    echo "  Health:    http://$HOST:$PORT/health"
    echo ""
    echo "  Press Ctrl+C to stop the server"
    echo ""

    # Open browser if not disabled and not in dev mode
    if [ "$NO_BROWSER" = false ] && [ "$DEV_MODE" = false ]; then
        if command -v xdg-open &>/dev/null; then
            xdg-open "http://localhost:$PORT/docs" 2>/dev/null || true
        elif command -v open &>/dev/null; then
            open "http://localhost:$PORT/docs" 2>/dev/null || true
        fi
    fi

    # Wait for the server process
    wait "$SERVER_PID"
else
    error "Server failed to start. Check the logs for details."
fi
