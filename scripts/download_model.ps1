#Requires -Version 5.1
# =============================================================
# Local Model Optimizer - Windows Model Download Script
# =============================================================
# Usage: .\scripts\download_model.ps1 [-Repo REPO_ID] [-Quant QUANT_TYPE]
# =============================================================

param(
    [string]$Repo = "",
    [string]$Quant = "",
    [string]$Filename = "",
    [string]$Search = "",
    [switch]$List,
    [switch]$Activate,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# --- Paths ---
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VenvDir = Join-Path $ProjectRoot ".venv"
$ModelDir = Join-Path $ProjectRoot "models\llm"
$HfCache = Join-Path $ProjectRoot ".cache\huggingface"
$ConfigFile = Join-Path $ProjectRoot "config\default.yaml"

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
# Model Catalog
# =============================================================
$Models = @{
    "1"  = @{ Repo = "meta-llama/Llama-2-7b-chat-hf"; Name = "Llama 2 7B Chat"; Params = "7B" }
    "2"  = @{ Repo = "meta-llama/Llama-2-13b-chat-hf"; Name = "Llama 2 13B Chat"; Params = "13B" }
    "3"  = @{ Repo = "meta-llama/Meta-Llama-3-8B-Instruct"; Name = "Llama 3 8B Instruct"; Params = "8B" }
    "4"  = @{ Repo = "mistralai/Mistral-7B-Instruct-v0.2"; Name = "Mistral 7B Instruct v0.2"; Params = "7B" }
    "5"  = @{ Repo = "mistralai/Mistral-7B-Instruct-v0.3"; Name = "Mistral 7B Instruct v0.3"; Params = "7B" }
    "6"  = @{ Repo = "mistralai/Mixtral-8x7B-Instruct-v0.1"; Name = "Mixtral 8x7B Instruct"; Params = "47B" }
    "7"  = @{ Repo = "Qwen/Qwen2-7B-Instruct"; Name = "Qwen2 7B Instruct"; Params = "7B" }
    "8"  = @{ Repo = "Qwen/Qwen2-14B-Instruct"; Name = "Qwen2 14B Instruct"; Params = "14B" }
    "9"  = @{ Repo = "codellama/CodeLlama-7b-Instruct-hf"; Name = "CodeLlama 7B Instruct"; Params = "7B" }
    "10" = @{ Repo = "deepseek-ai/deepseek-coder-6.7b-instruct"; Name = "DeepSeek Coder 6.7B"; Params = "6.7B" }
    "11" = @{ Repo = "microsoft/Phi-3-mini-4k-instruct"; Name = "Phi-3 Mini 4K"; Params = "3.8B" }
    "12" = @{ Repo = "google/gemma-2-2b-it"; Name = "Gemma 2 2B"; Params = "2B" }
    "13" = @{ Repo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"; Name = "TinyLlama 1.1B"; Params = "1.1B" }
    "14" = @{ Repo = "Custom"; Name = "Custom (enter HuggingFace repo ID)"; Params = "?" }
}

# =============================================================
# Quantization Options with VRAM Multipliers
# =============================================================
$Quants = @{
    "1"  = @{ Type = "Q2_K";  Desc = "Q2_K - Smallest (2-bit)";              Ratio = 0.6 }
    "2"  = @{ Type = "Q3_K_S"; Desc = "Q3_K_S - Small (3-bit)";              Ratio = 0.75 }
    "3"  = @{ Type = "Q3_K_M"; Desc = "Q3_K_M - Medium-small (3-bit)";       Ratio = 0.85 }
    "4"  = @{ Type = "Q4_0";  Desc = "Q4_0 - Legacy 4-bit";                  Ratio = 0.9 }
    "5"  = @{ Type = "Q4_K_S"; Desc = "Q4_K_S - Small (4-bit)";              Ratio = 0.95 }
    "6"  = @{ Type = "Q4_K_M"; Desc = "Q4_K_M - Medium (4-bit) RECOMMENDED"; Ratio = 1.0 }
    "7"  = @{ Type = "Q5_K_S"; Desc = "Q5_K_S - Large (5-bit)";              Ratio = 1.15 }
    "8"  = @{ Type = "Q5_K_M"; Desc = "Q5_K_M - Larger (5-bit)";             Ratio = 1.2 }
    "9"  = @{ Type = "Q6_K";  Desc = "Q6_K - Very large (6-bit)";            Ratio = 1.35 }
    "10" = @{ Type = "Q8_0";  Desc = "Q8_0 - Largest (8-bit)";               Ratio = 1.7 }
    "11" = @{ Type = "F16";   Desc = "F16 - Half precision";                  Ratio = 2.0 }
}

# =============================================================
# VRAM Estimation Table (base VRAM for Q4_K_M)
# =============================================================
$VramBase = @{
    "1B"   = 1.0
    "1.1B" = 1.0
    "2B"   = 1.5
    "3B"   = 2.5
    "3.8B" = 3.0
    "6.7B" = 4.5
    "7B"   = 4.8
    "8B"   = 5.2
    "13B"  = 9.0
    "14B"  = 9.8
    "34B"  = 22.0
    "47B"  = 30.0
    "70B"  = 44.0
}

# =============================================================
# Help
# =============================================================
if ($Help) {
    @"
Usage: .\scripts\download_model.ps1 [OPTIONS]

Options:
  -Repo ID        HuggingFace repo ID (e.g., TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
  -Quant TYPE     Quantization type (e.g., Q4_K_M, Q5_K_M)
  -Filename NAME  Specific GGUF filename to download
  -List           List available models and exit
  -Search QUERY   Search HuggingFace for models
  -Activate       Set downloaded model as default after download
  -Help           Show this help

Examples:
  .\scripts\download_model.ps1 -List
  .\scripts\download_model.ps1 -Repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF -Quant Q4_K_M
  .\scripts\download_model.ps1 -Search "llama 7b gguf"
  .\scripts\download_model.ps1   # Interactive mode
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
           |___/ Model Downloader

"@ -ForegroundColor Cyan

# =============================================================
# Display Functions
# =============================================================
function Show-Models {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Available Models" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    foreach ($key in ($Models.Keys | Sort-Object { [int]$_ })) {
        $m = $Models[$key]
        $params = if ($m.Params -ne "?") { "($($m.Params))" } else { "" }
        Write-Host "  $($key.PadLeft(2)). $($m.Name) $params" -ForegroundColor Green
        Write-Host "      Repo: $($m.Repo)"
    }
    Write-Host ""
}

function Show-Quants {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Quantization Options" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    foreach ($key in ($Quants.Keys | Sort-Object { [int]$_ })) {
        $q = $Quants[$key]
        $ratioStr = "~$($q.Ratio)x of Q4_K_M"
        Write-Host "  $($key.PadLeft(2)). $($q.Desc.PadRight(42)) ($ratioStr)" -ForegroundColor Green
    }
    Write-Host ""
}

# =============================================================
# VRAM Estimation
# =============================================================
function Get-VramEstimate {
    param(
        [string]$Params,
        [string]$QuantType
    )

    # Find base VRAM for this parameter size
    $baseVram = $null

    if ($VramBase.ContainsKey($Params)) {
        $baseVram = $VramBase[$Params]
    } else {
        # Try to find closest match
        $num = [double]($Params -replace '[^0-9.]', '')
        $unit = ($Params -replace '[0-9.]', '').ToUpper()

        if ($unit -eq "M") {
            $num = $num / 1000
        }

        $closest = $null
        $minDiff = [double]::MaxValue

        foreach ($key in $VramBase.Keys) {
            $keyNum = [double]($key -replace '[^0-9.]', '')
            $diff = [Math]::Abs($keyNum - $num)
            if ($diff -lt $minDiff) {
                $minDiff = $diff
                $closest = $key
            }
        }

        if ($closest -and $minDiff -lt 5) {
            $baseVram = $VramBase[$closest]
        }
    }

    if (-not $baseVram) {
        return $null
    }

    # Find quant ratio
    $ratio = 1.0
    foreach ($q in $Quants.Values) {
        if ($q.Type -eq $QuantType) {
            $ratio = $q.Ratio
            break
        }
    }

    return [Math]::Round($baseVram * $ratio, 1)
}

function Get-QuantLabel {
    param([string]$QuantType)

    foreach ($q in $Quants.Values) {
        if ($q.Type -eq $QuantType) {
            return $q.Desc
        }
    }
    return $QuantType
}

function Show-VramTable {
    param(
        [string]$Params,
        [string]$SelectedQuant = ""
    )

    # Find base VRAM
    $baseVram = $null
    if ($VramBase.ContainsKey($Params)) {
        $baseVram = $VramBase[$Params]
    }

    if (-not $baseVram) {
        return
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  VRAM Estimates by Quantization" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("  {0,-12} {1,-38} {2}" -f "Quant", "Description", "VRAM (GB)")
    Write-Host "  --------   ---------------------------   ---------"

    foreach ($key in ($Quants.Keys | Sort-Object { [int]$_ })) {
        $q = $Quants[$key]
        $vram = [Math]::Round($baseVram * $q.Ratio, 1)
        $marker = ""
        if ($q.Type -eq $SelectedQuant) {
            $marker = " <-- selected"
            Write-Host ("  {0,-12} {1,-38} ~{2} GB{3}" -f $q.Type, $q.Desc, $vram, $marker) -ForegroundColor Green
        } else {
            Write-Host ("  {0,-12} {1,-38} ~{2} GB{3}" -f $q.Type, $q.Desc, $vram, $marker)
        }
    }
    Write-Host ""
}

# =============================================================
# Interactive Selection
# =============================================================
function Select-Model {
    Show-Models
    $choice = Read-Host "Select model [1-14]"

    if (-not $Models.ContainsKey($choice)) {
        Write-Err "Invalid selection"
    }

    if ($choice -eq "14") {
        $repo = Read-Host "Enter HuggingFace repo ID"
        return @{
            Repo = $repo
            Name = "Custom Model"
            Params = "?"
        }
    }

    return $Models[$choice]
}

function Select-Quant {
    Show-Quants
    $choice = Read-Host "Select quantization [1-11] (default: 6 for Q4_K_M)"
    if ([string]::IsNullOrEmpty($choice)) { $choice = "6" }

    if (-not $Quants.ContainsKey($choice)) {
        Write-Err "Invalid selection"
    }

    return $Quants[$choice]
}

# =============================================================
# Search HuggingFace
# =============================================================
function Search-Models {
    param([string]$Query)

    Write-Step "Searching HuggingFace for: $Query"

    python -c @"
from huggingface_hub import HfApi

api = HfApi()
results = api.list_models(search='$Query', limit=10, sort='downloads', direction=-1)

print()
print('  Search Results:')
print('  ' + '=' * 60)

for i, model in enumerate(results, 1):
    tags = [t for t in (model.tags or []) if 'gguf' in t.lower()]
    gguf_marker = ' [GGUF]' if tags else ''
    downloads = model.downloads or 0
    print(f'  {i:2d}. {model.id}{gguf_marker}')
    print(f'      Downloads: {downloads:,} | Likes: {model.likes or 0}')
    if model.pipeline_tag:
        print(f'      Task: {model.pipeline_tag}')
    print()
"@
}

# =============================================================
# Download Model
# =============================================================
function Download-Model {
    param(
        [string]$RepoId,
        [string]$QuantType,
        [string]$FilePattern
    )

    Write-Step "Downloading model..."

    Write-Info "  Repo:      $RepoId"
    Write-Info "  Quant:     $(if ($QuantType) { $QuantType } else { 'auto' })"
    Write-Info "  Target:    $ModelDir"
    Write-Host ""

    # Ensure directories exist
    if (-not (Test-Path $ModelDir)) {
        New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
    }
    if (-not (Test-Path $HfCache)) {
        New-Item -ItemType Directory -Path $HfCache -Force | Out-Null
    }

    # Escape paths for Python
    $escapedModelDir = $ModelDir.Replace('\', '\\')
    $escapedHfCache = $HfCache.Replace('\', '\\')

    # Build Python download script
    $pyScript = @"
import sys
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("[ERROR] huggingface-hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)

repo_id = "$RepoId"
quant_type = "$QuantType"
filename = "$FilePattern"
model_dir = r"$escapedModelDir"
hf_cache = r"$escapedHfCache"

os.environ["HF_HOME"] = hf_cache

print(f"[INFO] Downloading from: {repo_id}")

try:
    if filename:
        print(f"[INFO] Downloading file: {filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        print(f"[OK] Downloaded to: {path}")
    elif quant_type:
        print(f"[INFO] Filtering for: *{quant_type}*")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[f"*{quant_type}*", "*.json", "*.txt", "README*"],
        )
        print(f"[OK] Downloaded to: {path}")
    else:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Downloaded to: {path}")
except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    sys.exit(1)
"@

    python -c $pyScript

    Write-Ok "Download complete!"

    return Join-Path $ModelDir ($RepoId.Split("/")[-1])
}

# =============================================================
# Post-Download Info
# =============================================================
function Show-PostDownloadInfo {
    param(
        [string]$ModelPath,
        [string]$ModelName,
        [string]$ModelParams,
        [string]$QuantType
    )

    # Find GGUF files
    $ggufFiles = Get-ChildItem -Path $ModelPath -Filter "*.gguf" -Recurse -ErrorAction SilentlyContinue

    if ($ggufFiles.Count -eq 0) {
        Write-Warn "No GGUF files found in $ModelPath"
        return
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Downloaded Model Files" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    foreach ($file in $ggufFiles) {
        $sizeMB = [Math]::Round($file.Length / 1MB, 1)
        $sizeStr = if ($sizeMB -gt 1024) { "$([Math]::Round($sizeMB / 1024, 1)) GB" } else { "$sizeMB MB" }

        Write-Host "  File: $($file.Name)" -ForegroundColor Green
        Write-Host "  Size: $sizeStr"
        Write-Host "  Path: $($file.FullName)"
        Write-Host ""
    }

    # VRAM estimation
    if ($ModelParams -ne "?" -and $QuantType) {
        $vram = Get-VramEstimate -Params $ModelParams -QuantType $QuantType
        $quantLabel = Get-QuantLabel -QuantType $QuantType

        if ($vram) {
            Write-Host "========================================" -ForegroundColor Cyan
            Write-Host "  VRAM Requirements Estimate" -ForegroundColor Cyan
            Write-Host "========================================" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "  Model:         $ModelName ($ModelParams)"
            Write-Host "  Quantization:  $quantLabel"
            Write-Host "  VRAM Needed:   ~${vram} GB" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  Notes:" -ForegroundColor Blue
            Write-Host "  - This is an estimate for model loading only"
            Write-Host "  - Actual usage depends on context length and batch size"
            Write-Host "  - Add ~1-2 GB overhead for inference buffers"
            Write-Host "  - Longer contexts require more VRAM"
            Write-Host ""
        }
    }

    # Show VRAM comparison table
    if ($ModelParams -ne "?") {
        Show-VramTable -Params $ModelParams -SelectedQuant $QuantType
    }
}

# =============================================================
# Set as Default Model
# =============================================================
function Set-DefaultModel {
    param([string]$ModelPath)

    $ggufFile = Get-ChildItem -Path $ModelPath -Filter "*.gguf" -Recurse | Select-Object -First 1

    if (-not $ggufFile) {
        Write-Warn "No GGUF file found in $ModelPath"
        return
    }

    # Make path relative to project root
    $relPath = $ggufFile.FullName.Replace($ProjectRoot, "").TrimStart('\')

    if (Test-Path $ConfigFile) {
        try {
            python -c @"
import yaml
from pathlib import Path

config_file = Path(r'$($ConfigFile.Replace('\', '\\'))')
if config_file.exists():
    with open(config_file) as f:
        config = yaml.safe_load(f) or {}

    if 'model' not in config:
        config['model'] = {}

    config['model']['default_model'] = '$relPath'

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print('[OK] Set as default model: $relPath')
else:
    print('[WARN] Config file not found')
"@
        } catch {
            Write-Warn "Failed to update config: $_"
        }
    }
}

# =============================================================
# Main
# =============================================================

# Search mode
if ($Search -ne "") {
    Search-Models -Query $Search
    exit 0
}

# List mode
if ($List) {
    Show-Models
    exit 0
}

# Check virtual environment
if (-not (Test-Path $VenvDir)) {
    Write-Err "Virtual environment not found. Run: .\scripts\install.ps1"
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    . $ActivateScript
}

# Store model info for later
$ModelInfo = $null
$ModelParams = "?"

# Interactive mode if no args
if ($Repo -eq "") {
    $ModelInfo = Select-Model
    $Repo = $ModelInfo.Repo
    $ModelParams = $ModelInfo.Params
} else {
    # Find params from catalog
    foreach ($m in $Models.Values) {
        if ($m.Repo -eq $Repo) {
            $ModelParams = $m.Params
            break
        }
    }
}

$QuantType = ""
if ($Quant -eq "" -and $Filename -eq "") {
    $quantInfo = Select-Quant
    $QuantType = $quantInfo.Type
} else {
    $QuantType = $Quant
}

$modelPath = Download-Model -RepoId $Repo -QuantType $QuantType -FilePattern $Filename

# Post-download info
$modelName = if ($ModelInfo) { $ModelInfo.Name } else { $Repo.Split("/")[-1] }
Show-PostDownloadInfo -ModelPath $modelPath -ModelName $modelName -ModelParams $ModelParams -QuantType $QuantType

if ($Activate) {
    Set-DefaultModel -ModelPath $modelPath
}

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Model Download Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Model:  $($Repo.Split("/")[-1])"
Write-Host "  Path:   $modelPath"
Write-Host ""
Write-Host "  Next steps:"
Write-Host "    Start the service:"
Write-Host "      .\scripts\start.ps1"
Write-Host ""
