#!/usr/bin/env bash
# =============================================================
# Local Model Optimizer - Model Download Script
# =============================================================
# Usage: bash scripts/download_model.sh [OPTIONS]
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
MODEL_DIR="$PROJECT_ROOT/models/llm"
HF_CACHE="$PROJECT_ROOT/.cache/huggingface"

# --- Helpers ---
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step()    { echo -e "\n${MAGENTA}>>> $1${NC}"; }

# =============================================================
# Popular GGUF Models Database
# =============================================================
declare -A MODELS=(
    # Llama family
    ["1"]="meta-llama/Llama-2-7b-chat-hf|Llama 2 7B Chat|7B"
    ["2"]="meta-llama/Llama-2-13b-chat-hf|Llama 2 13B Chat|13B"
    ["3"]="meta-llama/Meta-Llama-3-8B-Instruct|Llama 3 8B Instruct|8B"

    # Mistral family
    ["4"]="mistralai/Mistral-7B-Instruct-v0.2|Mistral 7B Instruct v0.2|7B"
    ["5"]="mistralai/Mistral-7B-Instruct-v0.3|Mistral 7B Instruct v0.3|7B"
    ["6"]="mistralai/Mixtral-8x7B-Instruct-v0.1|Mixtral 8x7B Instruct|47B"

    # Qwen family
    ["7"]="Qwen/Qwen2-7B-Instruct|Qwen2 7B Instruct|7B"
    ["8"]="Qwen/Qwen2-14B-Instruct|Qwen2 14B Instruct|14B"

    # Code models
    ["9"]="codellama/CodeLlama-7b-Instruct-hf|CodeLlama 7B Instruct|7B"
    ["10"]="deepseek-ai/deepseek-coder-6.7b-instruct|DeepSeek Coder 6.7B|6.7B"

    # Small/Fast models
    ["11"]="microsoft/Phi-3-mini-4k-instruct|Phi-3 Mini 4K|3.8B"
    ["12"]="google/gemma-2-2b-it|Gemma 2 2B|2B"
    ["13"]="TinyLlama/TinyLlama-1.1B-Chat-v1.0|TinyLlama 1.1B|1.1B"

    # Custom
    ["14"]="Custom (enter HuggingFace repo ID)|Custom|?"
)

# GGUF Quantization types with metadata
declare -A QUANTS=(
    ["1"]="Q2_K|Q2_K - Smallest (2-bit)|~0.6"
    ["2"]="Q3_K_S|Q3_K_S - Small (3-bit)|~0.75"
    ["3"]="Q3_K_M|Q3_K_M - Medium-small (3-bit)|~0.85"
    ["4"]="Q4_0|Q4_0 - Legacy 4-bit|~0.9"
    ["5"]="Q4_K_S|Q4_K_S - Small (4-bit)|~0.95"
    ["6"]="Q4_K_M|Q4_K_M - Medium (4-bit) RECOMMENDED|~1.0"
    ["7"]="Q5_K_S|Q5_K_S - Large (5-bit)|~1.15"
    ["8"]="Q5_K_M|Q5_K_M - Larger (5-bit)|~1.2"
    ["9"]="Q6_K|Q6_K - Very large (6-bit)|~1.35"
    ["10"]="Q8_0|Q8_0 - Largest (8-bit)|~1.7"
    ["11"]="F16|F16 - Half precision|~2.0"
)

# =============================================================
# VRAM Estimation Database (approximate GB per model size)
# =============================================================
# Format: base_params -> [Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16]
declare -A VRAM_1B=(   ["Q2_K"]="0.8" ["Q3_K_M"]="0.9" ["Q4_K_M"]="1.0" ["Q5_K_M"]="1.1" ["Q6_K"]="1.2" ["Q8_0"]="1.5" ["F16"]="2.5")
declare -A VRAM_3B=(   ["Q2_K"]="1.8" ["Q3_K_M"]="2.2" ["Q4_K_M"]="2.5" ["Q5_K_M"]="2.8" ["Q6_K"]="3.2" ["Q8_0"]="4.0" ["F16"]="7.0")
declare -A VRAM_7B=(   ["Q2_K"]="3.5" ["Q3_K_M"]="4.2" ["Q4_K_M"]="4.8" ["Q5_K_M"]="5.5" ["Q6_K"]="6.2" ["Q8_0"]="8.0" ["F16"]="14.0")
declare -A VRAM_13B=(  ["Q2_K"]="6.5" ["Q3_K_M"]="7.8" ["Q4_K_M"]="9.0" ["Q5_K_M"]="10.2" ["Q6_K"]="11.5" ["Q8_0"]="15.0" ["F16"]="26.0")
declare -A VRAM_14B=(  ["Q2_K"]="7.0" ["Q3_K_M"]="8.5" ["Q4_K_M"]="9.8" ["Q5_K_M"]="11.0" ["Q6_K"]="12.5" ["Q8_0"]="16.0" ["F16"]="28.0")
declare -A VRAM_34B=(  ["Q2_K"]="16.0" ["Q3_K_M"]="19.0" ["Q4_K_M"]="22.0" ["Q5_K_M"]="25.0" ["Q6_K"]="28.0" ["Q8_0"]="36.0" ["F16"]="65.0")
declare -A VRAM_70B=(  ["Q2_K"]="32.0" ["Q3_K_M"]="38.0" ["Q4_K_M"]="44.0" ["Q5_K_M"]="50.0" ["Q6_K"]="56.0" ["Q8_0"]="72.0" ["F16"]="130.0")
declare -A VRAM_47B=(  ["Q2_K"]="22.0" ["Q3_K_M"]="26.0" ["Q4_K_M"]="30.0" ["Q5_K_M"]="34.0" ["Q6_K"]="38.0" ["Q8_0"]="48.0" ["F16"]="90.0")

# =============================================================
# Flags and defaults
# =============================================================
REPO_ID=""
QUANT_TYPE=""
FILENAME=""
LIST_ONLY=false
ACTIVATE=false
SEARCH_QUERY=""

# =============================================================
# Parse arguments
# =============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)       REPO_ID="$2"; shift 2 ;;
        --quant)      QUANT_TYPE="$2"; shift 2 ;;
        --filename)   FILENAME="$2"; shift 2 ;;
        --list)       LIST_ONLY=true; shift ;;
        --activate)   ACTIVATE=true; shift ;;
        --search)     SEARCH_QUERY="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --repo ID        HuggingFace repo ID (e.g., TheBloke/Mistral-7B-Instruct-v0.2-GGUF)"
            echo "  --quant TYPE     Quantization type (e.g., Q4_K_M, Q5_K_M)"
            echo "  --filename NAME  Specific GGUF filename to download"
            echo "  --list           List available models and exit"
            echo "  --search QUERY   Search HuggingFace for models"
            echo "  --activate       Set downloaded model as default after download"
            echo "  -h, --help       Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --list"
            echo "  $0 --repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF --quant Q4_K_M"
            echo "  $0 --search 'llama 7b gguf'"
            echo "  $0  # Interactive mode"
            exit 0
            ;;
        *) error "Unknown option: $1. Use --help for usage." ;;
    esac
done

# =============================================================
# Check virtual environment
# =============================================================
if [ ! -d "$VENV_DIR" ]; then
    error "Virtual environment not found. Run: bash scripts/install.sh"
fi

source "$VENV_DIR/bin/activate"

# =============================================================
# Display functions
# =============================================================
list_models() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}  Available Models${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    for key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r repo desc params <<< "${MODELS[$key]}"
        printf "  ${GREEN}%2s${NC}. %-35s ${YELLOW}(%s)${NC}\n" "$key" "$desc" "$params"
        echo "      Repo: $repo"
    done
    echo ""
}

list_quants() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}  Quantization Options${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    for key in $(echo "${!QUANTS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r quant desc ratio <<< "${QUANTS[$key]}"
        printf "  ${GREEN}%2s${NC}. %-40s ${YELLOW}(~%sx of Q4_K_M)${NC}\n" "$key" "$desc" "$ratio"
    done
    echo ""
}

# =============================================================
# VRAM Estimation
# =============================================================
estimate_vram() {
    local params="$1"
    local quant="$2"

    # Determine model size category
    local vram_ref=""
    local size_label=""

    # Parse parameter count
    local num=$(echo "$params" | grep -oP '[\d.]+' | head -1)
    local unit=$(echo "$params" | grep -oP '[BM]' | head -1)

    # Convert to numeric for comparison
    local size_gb=0
    if [ "$unit" = "B" ]; then
        size_gb=$num
    elif [ "$unit" = "M" ]; then
        size_gb=$(echo "$num / 1000" | bc -l 2>/dev/null | head -c 4)
    fi

    # Select appropriate VRAM reference
    # Use awk for float comparison
    local category=$(awk "BEGIN { n=$num; u=\"$unit\"; if(u==\"M\") n=n/1000; if(n<=1.5) print \"1B\"; else if(n<=4) print \"3B\"; else if(n<=8) print \"7B\"; else if(n<=14) print \"14B\"; else if(n<=20) print \"14B\"; else if(n<=36) print \"34B\"; else if(n<=50) print \"47B\"; else print \"70B\" }")

    # Get VRAM estimate
    local vram=""
    case $category in
        1B)  vram="${VRAM_1B[$quant]}" ;;
        3B)  vram="${VRAM_3B[$quant]}" ;;
        7B)  vram="${VRAM_7B[$quant]}" ;;
        13B) vram="${VRAM_13B[$quant]}" ;;
        14B) vram="${VRAM_14B[$quant]}" ;;
        34B) vram="${VRAM_34B[$quant]}" ;;
        47B) vram="${VRAM_47B[$quant]}" ;;
        70B) vram="${VRAM_70B[$quant]}" ;;
        *)   vram="unknown" ;;
    esac

    echo "$vram"
}

get_quant_label() {
    local quant="$1"
    for key in $(echo "${!QUANTS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r q desc ratio <<< "${QUANTS[$key]}"
        if [ "$q" = "$quant" ]; then
            echo "$desc"
            return
        fi
    done
    echo "$quant"
}

# =============================================================
# Interactive selection
# =============================================================
select_model() {
    list_models
    read -p "Select model [1-14]: " choice

    if [ -z "${MODELS[$choice]}" ]; then
        error "Invalid selection"
    fi

    if [ "$choice" = "14" ]; then
        read -p "Enter HuggingFace repo ID: " REPO_ID
        MODEL_PARAMS=""
        MODEL_DESC="Custom Model"
    else
        IFS='|' read -r REPO_ID MODEL_DESC MODEL_PARAMS <<< "${MODELS[$choice]}"
    fi
}

select_quant() {
    list_quants
    read -p "Select quantization [1-11] (default: 6 for Q4_K_M): " choice
    choice=${choice:-6}

    if [ -z "${QUANTS[$choice]}" ]; then
        error "Invalid selection"
    fi

    IFS='|' read -r QUANT_TYPE _ _ <<< "${QUANTS[$choice]}"
}

# =============================================================
# Search HuggingFace
# =============================================================
search_models() {
    local query="$1"
    step "Searching HuggingFace for: $query"

    python << PYEOF
from huggingface_hub import HfApi
import sys

api = HfApi()
results = api.list_models(search="$query", limit=10, sort="downloads", direction=-1)

print("\n  Search Results:")
print("  " + "=" * 60)

for i, model in enumerate(results, 1):
    tags = [t for t in (model.tags or []) if "gguf" in t.lower()]
    gguf_marker = " [GGUF]" if tags else ""
    downloads = model.downloads or 0
    print(f"  {i:2d}. {model.id}{gguf_marker}")
    print(f"      Downloads: {downloads:,} | Likes: {model.likes or 0}")
    if model.pipeline_tag:
        print(f"      Task: {model.pipeline_tag}")
    print()

PYEOF
}

# =============================================================
# Download model
# =============================================================
download_model() {
    step "Downloading model..."

    info "  Repo:      $REPO_ID"
    info "  Quant:     ${QUANT_TYPE:-auto}"
    info "  Target:    $MODEL_DIR"
    echo ""

    mkdir -p "$MODEL_DIR"
    mkdir -p "$HF_CACHE"

    # Build Python download command
    python << PYEOF
import sys
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("[ERROR] huggingface-hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)

repo_id = "$REPO_ID"
quant_type = "${QUANT_TYPE:-}"
filename = "${FILENAME:-}"
model_dir = "$MODEL_DIR"
hf_cache = "$HF_CACHE"

# Set HF cache directory
os.environ["HF_HOME"] = hf_cache

print(f"[INFO] Downloading from: {repo_id}")

try:
    if filename:
        # Download specific file
        print(f"[INFO] Downloading file: {filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        print(f"[OK] Downloaded to: {path}")
    elif quant_type:
        # Download files matching quantization pattern
        print(f"[INFO] Filtering for: *{quant_type}*")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[f"*{quant_type}*", "*.json", "*.txt", "README*"],
        )
        print(f"[OK] Downloaded to: {path}")
    else:
        # Download all files
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Downloaded to: {path}")

except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    sys.exit(1)
PYEOF

    success "Download complete!"
}

# =============================================================
# Post-download: Find and report model info
# =============================================================
post_download_info() {
    local model_name=$(basename "$REPO_ID")
    local model_path="$MODEL_DIR/$model_name"

    # Find GGUF files
    local gguf_files=$(find "$model_path" -name "*.gguf" -type f 2>/dev/null)

    if [ -z "$gguf_files" ]; then
        warn "No GGUF files found in $model_path"
        return
    fi

    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Downloaded Model Files${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    while IFS= read -r gguf_file; do
        local filename=$(basename "$gguf_file")
        local filesize=$(du -h "$gguf_file" 2>/dev/null | cut -f1)

        echo -e "  ${GREEN}File:${NC} $filename"
        echo -e "  ${GREEN}Size:${NC} $filesize"
        echo -e "  ${GREEN}Path:${NC} $gguf_file"
        echo ""
    done <<< "$gguf_files"

    # VRAM estimation
    if [ -n "$MODEL_PARAMS" ] && [ -n "$QUANT_TYPE" ]; then
        local vram=$(estimate_vram "$MODEL_PARAMS" "$QUANT_TYPE")
        local quant_label=$(get_quant_label "$QUANT_TYPE")

        if [ "$vram" != "unknown" ] && [ -n "$vram" ]; then
            echo -e "${CYAN}========================================${NC}"
            echo -e "${CYAN}  VRAM Requirements Estimate${NC}"
            echo -e "${CYAN}========================================${NC}"
            echo ""
            echo -e "  Model:         $MODEL_DESC ($MODEL_PARAMS)"
            echo -e "  Quantization:  $quant_label"
            echo -e "  VRAM Needed:   ${YELLOW}~${vram} GB${NC}"
            echo ""
            echo -e "  ${BLUE}Notes:${NC}"
            echo "  - This is an estimate for model loading only"
            echo "  - Actual usage depends on context length and batch size"
            echo "  - Add ~1-2 GB overhead for inference buffers"
            echo "  - Longer contexts require more VRAM"
            echo ""
        fi
    fi

    # Show VRAM table for this model
    if [ -n "$MODEL_PARAMS" ]; then
        show_vram_table "$MODEL_PARAMS"
    fi
}

# =============================================================
# Show VRAM comparison table
# =============================================================
show_vram_table() {
    local params="$1"

    local category=$(awk "BEGIN { n=$params; gsub(/[^0-9.]/,\"\",n); if(n+0<=1.5) print \"1B\"; else if(n+0<=4) print \"3B\"; else if(n+0<=8) print \"7B\"; else if(n+0<=14) print \"14B\"; else if(n+0<=20) print \"14B\"; else if(n+0<=36) print \"34B\"; else if(n+0<=50) print \"47B\"; else print \"70B\" }")

    local -n vram_ref="VRAM_${category}"

    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  VRAM Estimates by Quantization${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    printf "  %-12s %-35s %s\n" "Quant" "Description" "VRAM (GB)"
    echo "  --------   ---------------------------   ---------"

    for key in $(echo "${!QUANTS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r quant desc ratio <<< "${QUANTS[$key]}"
        local vram="${vram_ref[$quant]:-N/A}"
        local marker=""
        if [ "$quant" = "$QUANT_TYPE" ]; then
            marker=" ${GREEN}<-- selected${NC}"
        fi
        printf "  %-12s %-35s ~%s GB%b\n" "$quant" "$(echo "$desc" | cut -d'|' -f1)" "$vram" "$marker"
    done
    echo ""
}

# =============================================================
# Set as default model
# =============================================================
set_default() {
    local model_name=$(basename "$REPO_ID")
    local model_path="$MODEL_DIR/$model_name"

    # Find GGUF file
    local gguf_file=$(find "$model_path" -name "*.gguf" -type f | head -1)

    if [ -z "$gguf_file" ]; then
        warn "No GGUF file found in $model_path"
        return
    fi

    # Make path relative to project root
    local rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$gguf_file")

    # Update config using Python
    python << PYEOF
import yaml
from pathlib import Path

config_file = Path("$CONFIG_FILE")
if config_file.exists():
    with open(config_file) as f:
        config = yaml.safe_load(f) or {}

    if "model" not in config:
        config["model"] = {}

    config["model"]["default_model"] = "$rel_path"

    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("[OK] Set as default model: $rel_path")
else:
    print("[WARN] Config file not found at $config_file")
PYEOF
}

# =============================================================
# Main
# =============================================================
main() {
    echo -e "${CYAN}"
    cat << 'BANNER'
  _                    _    ___ _   ___
 | |   ___  __ _ __ _ / _  / _ \ | | __|
 | |__/ _ \/ _` / _` | (_) | (_) | | _|
 |____\___/\__, |\__,_|\___/ \___/  |_|
           |___/ Model Downloader
BANNER
    echo -e "${NC}"

    # Search mode
    if [ -n "$SEARCH_QUERY" ]; then
        search_models "$SEARCH_QUERY"
        exit 0
    fi

    # List mode
    if [ "$LIST_ONLY" = true ]; then
        list_models
        exit 0
    fi

    # Interactive mode if no args
    if [ -z "$REPO_ID" ]; then
        select_model
    fi

    if [ -z "$QUANT_TYPE" ] && [ -z "$FILENAME" ]; then
        select_quant
    fi

    # Download
    download_model

    # Post-download info
    MODEL_DESC="${MODEL_DESC:-Custom Model}"
    MODEL_PARAMS="${MODEL_PARAMS:-?}"
    post_download_info

    # Set as default if requested
    if [ "$ACTIVATE" = true ]; then
        set_default
    fi

    # Summary
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Model Download Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "  Model:  $(basename "$REPO_ID")"
    echo "  Path:   $MODEL_DIR/$(basename "$REPO_ID")"
    echo ""
    echo "  Next steps:"
    echo "    Start the service:"
    echo "      bash scripts/start.sh"
    echo ""
}

main "$@"
