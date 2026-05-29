# Local Model Optimizer - Scripts

This directory contains installation, startup, and model management scripts for the Local Model Optimizer project.

## Quick Start

### Linux / macOS

```bash
# 1. Install dependencies
bash scripts/install.sh

# 2. Download a model (interactive)
bash scripts/download_model.sh

# 3. Start the service
bash scripts/start.sh
```

### Windows (PowerShell)

```powershell
# 1. Install dependencies
.\scripts\install.ps1

# 2. Download a model (interactive)
.\scripts\download_model.ps1

# 3. Start the service
.\scripts\start.ps1
```

---

## Scripts Reference

### install.sh / install.ps1

Installs the project dependencies and sets up the environment.

**Features:**
- Detects operating system (Linux/macOS/Windows)
- Validates Python version (requires 3.10+)
- Creates isolated virtual environment
- Detects GPU hardware (NVIDIA/AMD/Apple Silicon/Intel)
- Builds llama-cpp-python with appropriate GPU acceleration:
  - CUDA for NVIDIA GPUs
  - ROCm for AMD GPUs
  - Metal for Apple Silicon
  - SYCL for Intel GPUs
  - CPU-only fallback
- Creates all required project directories (including `data/layer_cache`)
- Generates default configuration files (`config/default.yaml`, `.env`)
- Verifies installation by testing critical imports

**Options:**

| Option | Description |
|--------|-------------|
| `--skip-venv` | Skip virtual environment creation |
| `--force` | Force recreate virtual environment |
| `--no-gpu` | Skip GPU detection, use CPU-only build |
| `-h, --help` | Show help |

**Examples:**

```bash
# Full installation
bash scripts/install.sh

# Recreate venv from scratch
bash scripts/install.sh --force

# CPU-only installation
bash scripts/install.sh --no-gpu
```

---

### start.sh / start.ps1

Starts the Local Model Optimizer API server.

**Features:**
- Loads environment from `.env` file
- Checks for available models
- Verifies port availability
- Supports development mode with auto-reload
- Graceful shutdown handling
- Optional browser auto-open

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--port PORT` | Server port | 8000 |
| `--host HOST` | Server host | 0.0.0.0 |
| `--workers N` | Number of workers | 1 |
| `--log-level LVL` | Log level (debug/info/warning/error) | info |
| `--log-file FILE` | Log to file | stdout |
| `--dev` | Development mode with auto-reload | false |
| `--no-browser` | Don't open browser | false |
| `-h, --help` | Show help | |

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `LMO_HOST` | Server host | 0.0.0.0 |
| `LMO_PORT` | Server port | 8000 |
| `LMO_WORKERS` | Worker count | 1 |
| `LMO_LOG_LEVEL` | Log level | info |

**Examples:**

```bash
# Start with defaults
bash scripts/start.sh

# Custom port
bash scripts/start.sh --port 9000

# Development mode
bash scripts/start.sh --dev

# Production with multiple workers
bash scripts/start.sh --workers 4 --log-level warning
```

**API Endpoints:**

Once started, the following endpoints are available:
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

---

### download_model.sh / download_model.ps1

Downloads GGUF models from HuggingFace Hub.

**Features:**
- Interactive model selection from popular models
- Quantization type selection with descriptions
- VRAM requirements estimation per quantization level
- Comparison table showing all quantization options
- Supports custom HuggingFace repo IDs
- HuggingFace model search
- Can set downloaded model as default
- Shows file sizes after download

**Options:**

| Option | Description |
|--------|-------------|
| `--repo ID` | HuggingFace repo ID |
| `--quant TYPE` | Quantization type (e.g., Q4_K_M) |
| `--filename NAME` | Specific GGUF filename to download |
| `--list` | List available models |
| `--search QUERY` | Search HuggingFace for models |
| `--activate` | Set as default model after download |
| `-h, --help` | Show help |

**Available Models:**

| # | Model | Parameters |
|---|-------|------------|
| 1 | Llama 2 7B Chat | 7B |
| 2 | Llama 2 13B Chat | 13B |
| 3 | Llama 3 8B Instruct | 8B |
| 4 | Mistral 7B Instruct v0.2 | 7B |
| 5 | Mistral 7B Instruct v0.3 | 7B |
| 6 | Mixtral 8x7B Instruct | 47B |
| 7 | Qwen2 7B Instruct | 7B |
| 8 | Qwen2 14B Instruct | 14B |
| 9 | CodeLlama 7B Instruct | 7B |
| 10 | DeepSeek Coder 6.7B | 6.7B |
| 11 | Phi-3 Mini 4K | 3.8B |
| 12 | Gemma 2 2B | 2B |
| 13 | TinyLlama 1.1B | 1.1B |
| 14 | Custom | - |

**Quantization Options:**

| # | Type | Description | Size vs Q4_K_M |
|---|------|-------------|----------------|
| 1 | Q2_K | Smallest (2-bit) | ~0.6x |
| 2 | Q3_K_S | Small (3-bit) | ~0.75x |
| 3 | Q3_K_M | Medium-small (3-bit) | ~0.85x |
| 4 | Q4_0 | Legacy 4-bit | ~0.9x |
| 5 | Q4_K_S | Small (4-bit) | ~0.95x |
| 6 | Q4_K_M | Medium (4-bit) RECOMMENDED | ~1.0x |
| 7 | Q5_K_S | Large (5-bit) | ~1.15x |
| 8 | Q5_K_M | Larger (5-bit) | ~1.2x |
| 9 | Q6_K | Very large (6-bit) | ~1.35x |
| 10 | Q8_0 | Largest (8-bit) | ~1.7x |
| 11 | F16 | Half precision | ~2.0x |

**VRAM Estimation:**

The script provides VRAM estimates based on model size and quantization:

| Model Size | Q4_K_M (Recommended) | Q2_K (Smallest) | Q8_0 (Best) |
|------------|---------------------|------------------|--------------|
| 2B | ~1.5 GB | ~0.9 GB | ~2.6 GB |
| 7B | ~4.8 GB | ~3.5 GB | ~8.0 GB |
| 13B | ~9.0 GB | ~6.5 GB | ~15.0 GB |
| 70B | ~44.0 GB | ~32.0 GB | ~72.0 GB |

> Note: Add ~1-2 GB overhead for inference buffers. Longer contexts require more VRAM.

**Examples:**

```bash
# Interactive mode
bash scripts/download_model.sh

# Direct download with options
bash scripts/download_model.sh --repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF --quant Q4_K_M

# Search for models
bash scripts/download_model.sh --search "llama 7b gguf"

# List available models
bash scripts/download_model.sh --list

# Download and set as default
bash scripts/download_model.sh --repo Qwen/Qwen2-7B-Instruct --quant Q4_K_M --activate
```

---

## Directory Structure

After installation, the project structure is:

```
.
├── .venv/                    # Virtual environment (created by install)
├── .env                      # Environment variables (created by install)
├── config/
│   ├── default.yaml          # Main configuration file
│   └── default.example.yaml  # Configuration template
├── models/
│   ├── llm/                  # Language models (GGUF files)
│   ├── vision/               # Vision models
│   └── audio/                # Audio models
├── data/
│   └── layer_cache/          # Model layer cache
├── logs/                     # Application logs
├── .cache/
│   └── huggingface/          # HuggingFace download cache
└── benchmark_results/        # Benchmark output
```

---

## Configuration

### Environment Variables (.env)

Key environment variables (see `.env.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `LMO_HOST` | Server host | 0.0.0.0 |
| `LMO_PORT` | Server port | 8000 |
| `LMO_WORKERS` | Worker count | 1 |
| `LMO_DEFAULT_MODEL` | Default model path | - |
| `LMO_MODEL_DIR` | Model directory | models |
| `LMO_N_CTX` | Context window size | 4096 |
| `LMO_N_GPU_LAYERS` | GPU offload layers (-1=all) | -1 |
| `LMO_LOG_LEVEL` | Log level | INFO |
| `HF_TOKEN` | HuggingFace API token | - |

### Configuration File (config/default.yaml)

The main configuration file supports all settings from `.env` plus additional options for:
- Model-specific overrides
- Inference parameters (temperature, top_p, etc.)
- Scheduler settings
- Hardware detection
- Monitoring and thresholds
- Logging configuration
- API authentication

See `config/default.example.yaml` for all available options.

---

## Troubleshooting

### Python not found

Install Python 3.10+ from https://www.python.org/downloads/

Or use your system package manager:
```bash
# Ubuntu/Debian
sudo apt install python3 python3-venv python3-pip

# macOS
brew install python@3.12

# Windows
winget install Python.Python.3.12
```

### CUDA build fails

Ensure CUDA toolkit is installed:
```bash
# Ubuntu
sudo apt install nvidia-cuda-toolkit

# Or install from NVIDIA
# https://developer.nvidia.com/cuda-downloads
```

### ROCm build fails (AMD)

Install ROCm:
```bash
# Ubuntu
sudo apt install rocm-dev

# Or follow: https://rocm.docs.amd.com/
```

### Permission denied

```bash
chmod +x scripts/*.sh
```

### Models directory empty

Download a model first:
```bash
bash scripts/download_model.sh
```

### Port already in use

```bash
# Check what's using the port
lsof -i :8000

# Or use a different port
bash scripts/start.sh --port 9000
```

### Virtual environment issues

```bash
# Recreate venv
bash scripts/install.sh --force

# Or manually
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Advanced Usage

### Custom Model Download

```bash
# Download specific file from repo
bash scripts/download_model.sh --repo TheBloke/Llama-2-7B-GGUF --filename llama-2-7b.Q4_K_M.gguf

# Download from private repo (requires HF_TOKEN)
export HF_TOKEN="your_token_here"
bash scripts/download_model.sh --repo meta-llama/Llama-2-7b-chat-hf --quant Q4_K_M
```

### Production Deployment

```bash
# Start with multiple workers
bash scripts/start.sh --workers 4 --log-level warning --log-file logs/production.log

# Or use systemd/PM2 for process management
```

### Development Workflow

```bash
# Install in development mode
bash scripts/install.sh

# Start with auto-reload
bash scripts/start.sh --dev

# The server will restart automatically on code changes
```
