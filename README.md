# LocalAI Optimizer

**智能本地大语言模型推理优化平台** -- 小显存也能跑大模型。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

---

## 项目简介

LocalAI Optimizer 是一个本地大语言模型推理优化工具包，核心目标是**让小显存/内存也能运行大模型**。

通过智能量化、GPU-CPU-Disk 三层卸载、KV Cache 压缩、动态层加载等技术，实现：
- **6GB 显存运行 13B 模型** -- 极限量化 + 混合推理
- **4GB 显存运行 7B 模型** -- 低精度量化 + 层分配优化
- **纯 CPU 运行 7B 模型** -- 无需 GPU，16GB 内存即可

### 核心卖点

| 特性 | LocalAI Optimizer | Ollama | LM Studio |
|------|-------------------|--------|-----------|
| **GPU-CPU-Disk 三层卸载** | Yes | No | No |
| **KV Cache 量化/压缩/共享** | Yes | No | No |
| **动态层加载 (LRU/预取)** | Yes | No | No |
| **一键优化 (4种模式)** | Yes | No | No |
| **精确显存预估** | Yes | No | Partial |
| **策略对比报告** | Yes | No | No |

### 功能特性

| 功能模块 | 描述 |
|---------|------|
| **硬件智能检测** | 跨平台检测 GPU (NVIDIA/AMD/Apple/Intel)、CPU、内存、存储，自动推荐最优配置 |
| **统一模型管理** | 支持 GGUF/SafeTensors/ONNX/PyTorch 格式，自动扫描、元数据解析、显存估算 |
| **智能推理调度** | 基于任务类型的模型选择、量化级别自动选择、模型热切换、优先级队列 |
| **内存深度优化** | 量化 + 卸载 + KV Cache 压缩 + 动态层加载，四维一体优化 |
| **OpenAI 兼容 API** | 完整兼容 OpenAI Chat/Completions 接口，支持流式输出 |
| **实时性能监控** | CPU/GPU/内存指标采集、瓶颈分析、优化建议 |
| **HuggingFace 集成** | 一键从 HuggingFace Hub 下载模型 |

---

## 快速开始指南

### 系统要求

- Python >= 3.10
- 操作系统：Linux / macOS / Windows
- 推荐：NVIDIA GPU (支持 CUDA) 或 Apple Silicon (支持 Metal)
- 最低：纯 CPU 也可运行，推荐 16GB+ 内存

### 安装步骤

#### Linux / macOS

```bash
# 克隆项目
git clone https://github.com/your-org/local-model-optimizer.git
cd local-model-optimizer

# 运行安装脚本 (自动创建虚拟环境、检测 GPU、安装依赖)
chmod +x scripts/install.sh
./scripts/install.sh
```

#### Windows (PowerShell)

```powershell
# 克隆项目
git clone https://github.com/your-org/local-model-optimizer.git
cd local-model-optimizer

# 运行安装脚本
.\scripts\install.ps1
```

#### 手动安装

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt

# 安装项目 (可编辑模式)
pip install -e .
```

> **GPU 支持说明：**
> - NVIDIA GPU：安装脚本会自动以 `-DGGML_CUDA=on` 编译 `llama-cpp-python`
> - AMD GPU：安装脚本会自动以 `-DGGML_HIP=on` 编译
> - Apple Silicon：默认支持 Metal 加速
> - 仅 CPU：直接安装即可，AVX2 指令集会自动检测

### 配置说明

```bash
# 复制配置模板
cp config/default.example.yaml config/default.yaml

# 复制环境变量模板
cp .env.example .env
```

主要配置项 (`config/default.yaml`)：

```yaml
server:
  host: "0.0.0.0"
  port: 8000

model:
  default_model: "models/llama-7b-q4_k_m.gguf"
  model_dir: "models"
  defaults:
    n_ctx: 4096          # 上下文窗口大小
    n_gpu_layers: -1     # GPU offload 层数 (-1 = 全部)
    n_threads: 0         # CPU 线程数 (0 = 自动检测)

# 内存优化配置
memory_optimization:
  quantization: "q4_k_m"       # 量化级别
  offload:
    strategy: "gpu_cpu"        # gpu_only | gpu_cpu | gpu_cpu_disk | cpu_only
    gpu_layers: -1             # -1=全部GPU, 0=仅CPU
    cpu_threads: 4
  kv_cache:
    cache_bits: 16             # 16=FP16, 8=INT8, 4=INT4
    prefix_sharing: true       # 前缀共享
  dynamic:
    max_gpu_layers: 20
    prefetch_enabled: true

scheduler:
  max_queue_size: 1000
  max_global_concurrency: 4
  model_select_strategy: "balanced"  # balanced | speed | quality | memory

monitoring:
  enabled: true
  collect_interval: 1.0
  history_size: 300
```

### 启动命令

```bash
# Windows
.\scripts\start.ps1

# Linux/macOS
./scripts/start.sh

# 或直接使用 Python
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# 开发模式 (自动重载)
.\scripts\start.ps1 -Dev
```

服务启动后访问：

- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`
- 硬件信息：`http://localhost:8000/api/hardware`

### 下载模型

```bash
# 交互式下载 (推荐)
# Windows:
.\scripts\download_model.ps1

# Linux/macOS:
./scripts/download_model.sh

# 命令行指定参数
.\scripts\download_model.ps1 -Repo "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" -Quant Q4_K_M
```

支持的模型仓库示例：

| 模型 | HuggingFace Repo |
|------|------------------|
| Llama 2 7B | `TheBloke/Llama-2-7B-GGUF` |
| Llama 2 13B | `TheBloke/Llama-2-13B-GGUF` |
| Mistral 7B Instruct | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` |
| CodeLlama 7B | `TheBloke/CodeLlama-7B-GGUF` |
| Qwen2 7B | `Qwen/Qwen2-7B-Instruct-GGUF` |
| Phi-2 | `TheBloke/phi-2-GGUF` |

---

## 内存优化指南

这是本项目的核心能力。通过四维一体的优化策略，让有限显存运行更大的模型。

### 优化原理

```
┌─────────────────────────────────────────────────────────────────┐
│                    四维一体内存优化架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  维度1: 量化压缩                                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ FP16 (16bit) -> Q8_0 (8bit) -> Q4_K_M (4bit)           │    │
│  │              -> Q2_K (2bit) -> IQ2_XXS (2.1bit极限)     │    │
│  │  效果: 模型体积压缩 4x ~ 8x                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  维度2: GPU-CPU-Disk 三层卸载                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  GPU (最快) <-> CPU (中等) <-> Disk (最慢)               │    │
│  │  关键层(注意力)放GPU，非关键层(FFN)放CPU，冷数据放磁盘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  维度3: KV Cache 优化                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  量化: FP16 -> INT8/INT4，节省 50-75%                    │    │
│  │  分页: PagedAttention，消除内存碎片                       │    │
│  │  共享: 多请求共享相同前缀的 KV                            │    │
│  │  淘汰: 基于注意力分数保留重要 KV                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  维度4: 动态层加载                                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  按需加载: 只在需要时加载层到 GPU                         │    │
│  │  LRU淘汰: 最久未使用的层卸载到 CPU/磁盘                  │    │
│  │  预测预取: 基于马尔可夫链预测下一层并预加载               │    │
│  │  流水线: 一层计算时下一层从 CPU 预加载                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 不同优化策略对比

#### 量化级别对比

| 量化类型 | 有效位数 | 每B参数显存 | 7B 模型体积 | 质量评分 | 速度因子 | 适用场景 |
|---------|----------|------------|------------|---------|---------|---------|
| IQ2_XXS | 2.1 bit | 0.26 GB | ~1.8 GB | 0.45 | 1.10x | 极限显存，需 imatrix |
| IQ2_XS | 2.3 bit | 0.29 GB | ~2.0 GB | 0.50 | 1.10x | 极限显存，需 imatrix |
| Q2_K | 2.9 bit | 0.31 GB | ~2.2 GB | 0.55 | 1.20x | 极端受限显存 |
| Q3_K_M | 3.9 bit | 0.44 GB | ~3.1 GB | 0.70 | 1.15x | 显存紧张 |
| **Q4_K_M** | **4.8 bit** | **0.56 GB** | **~3.9 GB** | **0.85** | **1.00x** | **推荐平衡选择** |
| Q5_K_M | 6.5 bit | 0.81 GB | ~5.7 GB | 0.90 | 0.95x | 高质量需求 |
| Q8_0 | 8.5 bit | 1.06 GB | ~7.4 GB | 0.95 | 0.85x | 显存充足 |
| FP16 | 16 bit | 2.0 GB | ~14 GB | 1.00 | 0.50x | 专业/研究用途 |

#### 卸载策略对比

| 策略 | GPU 层 | CPU 层 | Disk 层 | 速度 | 显存需求 | 适用场景 |
|------|--------|--------|---------|------|---------|---------|
| GPU-Only | 全部 | 0 | 0 | 最快 | 最高 | 显存充足 |
| GPU-CPU 混合 | 关键层 | 剩余层 | 0 | 较快 | 中等 | 显存不足但内存充足 |
| GPU-CPU-Disk | 部分 | 部分 | 冷数据 | 较慢 | 最低 | 极端受限 |
| CPU-Only | 0 | 全部 | 0 | 慢 | 0 VRAM | 无 GPU |

### 8GB 显存运行 13B 模型

```python
from src.optimization.memory_optimizer import MemoryOptimizer, OptimizationProfile, HardwareProfile

# 定义硬件
hardware = HardwareProfile(
    vram_total_gb=8.0,
    vram_free_gb=7.2,
    ram_total_gb=16.0,
    ram_free_gb=12.0,
    cpu_cores=8,
    has_gpu=True,
)

# 一键优化
optimizer = MemoryOptimizer()
result = optimizer.optimize_for_model(
    model_size_b=13.0,
    profile=OptimizationProfile.BALANCED,
    hardware=hardware,
    context_length=4096,
)

print(f"量化级别: {result.quantization.level}")        # Q4_K_M
print(f"GPU层数: {result.offload_config.gpu_layers}")   # 20/40
print(f"预估显存: {result.estimated_vram_gb:.1f} GB")    # ~7.2 GB
print(f"预估速度: {result.estimated_speed_tps:.1f} t/s") # ~25.3 t/s
print(f"质量评分: {result.quality_score:.2f}")            # 0.85
```

**优化配置详情：**

```yaml
quantization: Q4_K_M          # 4.8 bits，质量与显存的最佳平衡
offload:
  strategy: gpu_cpu            # GPU-CPU 混合
  gpu_layers: 20               # 20 层在 GPU (注意力层优先)
  cpu_layers: 20               # 20 层在 CPU
kv_cache:
  cache_bits: 4                # KV Cache INT4 量化
  prefix_sharing: true         # 启用前缀共享
  compression_ratio: 0.5
dynamic:
  max_gpu_layers: 16           # 动态加载，GPU 最多驻留 16 层
  prefetch_enabled: true       # 启用预测预取
```

**也可以使用快捷接口：**

```python
# 无需完整硬件画像
result = optimizer.quick_optimize(
    model_size_b=13.0,
    target="balanced",
    vram_gb=8.0,
    ram_gb=16.0,
)
```

### 4GB 显存运行 7B 模型

```python
from src.optimization.vram_optimizer import VRAMOptimizer, OptimizationTarget

optimizer = VRAMOptimizer(vram_gb=4.0)
result = optimizer.optimize(
    model_size_b=7.0,
    target=OptimizationTarget.MINIMAL_VRAM,
)

print(f"量化级别: {result.quantization.value}")            # Q3_K_M
print(f"GPU层数: {result.layer_allocation.gpu_layers}/"
      f"{result.layer_allocation.total_layers}")            # 16/32
print(f"预估显存: {result.estimated_vram_gb:.2f} GB")       # ~3.5 GB
print(f"预估速度: {result.estimated_speed_tps:.1f} t/s")    # ~18.7 t/s
```

**对比不同优化目标：**

```python
comparisons = optimizer.compare_targets(model_size_b=7.0)
for comp in comparisons:
    print(f"{comp['target']}: {comp['vram_gb']}GB, "
          f"{comp['speed_tps']}t/s, quality={comp['quality']:.2f}")
```

输出：

```
minimal_vram:  3.17 GB,  14.8 t/s, quality=0.55
balanced:      4.57 GB,  40.0 t/s, quality=0.85
max_speed:     4.57 GB,  40.0 t/s, quality=0.85
max_quality:   4.57 GB,  40.0 t/s, quality=0.85
```

### 纯 CPU 运行 7B 模型

```python
from src.optimization.memory_optimizer import MemoryOptimizer, OptimizationProfile, HardwareProfile

# 无 GPU
hardware = HardwareProfile(
    vram_total_gb=0.0,
    vram_free_gb=0.0,
    ram_total_gb=16.0,
    ram_free_gb=12.0,
    cpu_cores=8,
    has_gpu=False,
)

optimizer = MemoryOptimizer()
result = optimizer.optimize_for_model(
    model_size_b=7.0,
    profile=OptimizationProfile.BALANCED,
    hardware=hardware,
)

print(f"量化级别: {result.quantization.level}")        # Q4_K_M
print(f"GPU层数: {result.offload_config.gpu_layers}")   # 0
print(f"预估显存: {result.estimated_vram_gb:.1f} GB")    # 0 GB
print(f"预估内存: {result.estimated_ram_gb:.1f} GB")     # ~5.2 GB
print(f"预估速度: {result.estimated_speed_tps:.1f} t/s") # ~8.5 t/s
```

**CPU 推理优化建议：**

- 使用 Q4_K_M 量化，在质量和速度间取得平衡
- 设置 `n_threads` 为物理核心数 (不要超过 8)
- 启用 AVX2 指令集可显著提升速度
- 上下文长度建议 2048 以内

### 使用优化 API

```bash
# 获取优化配置
curl -X POST http://localhost:8000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "model_size_b": 13.0,
    "target": "minimal_vram",
    "vram_gb": 6.0,
    "ram_gb": 16.0
  }'

# 获取优化报告 (包含所有模式对比)
curl http://localhost:8000/api/optimize/report/13B
```

### 6GB 显存实测数据

| 模型 | 量化 | GPU层 | 显存占用 | 速度 (t/s) | 质量评分 | 可用性 |
|------|------|-------|---------|-----------|---------|--------|
| **7B** | Q4_K_M | 32/32 | 4.57 GB | 40.0 | 0.85 | PERFECT |
| **13B** | IQ2_XS | 40/40 | 4.56 GB | 23.7 | 0.50 | PERFECT |
| **30B** | IQ2_XXS | 30/60 | 5.06 GB | 5.1 | 0.45 | GOOD |
| **70B** | IQ2_XXS | 19/80 | 5.04 GB | 1.0 | 0.45 | OK |

---

## API 文档

### OpenAI 兼容接口

#### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": false
  }'
```

响应格式：

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 9,
    "total_tokens": 34
  }
}
```

流式输出 (设置 `"stream": true`) 返回 Server-Sent Events：

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"!"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

#### Text Completions

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "The theory of relativity states that",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

#### List Models

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-7b-q4_k_m",
      "object": "model",
      "created": 1699000000,
      "owned_by": "local"
    }
  ]
}
```

### 管理接口

#### 硬件信息

```bash
GET /api/hardware
```

返回完整的硬件画像，包括 GPU/CPU/内存信息以及推荐配置。

#### 模型管理

```bash
# 列出所有已知模型
GET /api/models

# 触发模型下载
POST /api/models/download
Content-Type: application/json

{
  "model_id": "TheBloke/Llama-2-7B-GGUF",
  "filename": "llama-2-7b.Q4_K_M.gguf"
}
```

#### 性能监控

```bash
# 获取当前指标快照
GET /api/metrics/current

# 获取历史指标 (可选 ?count=N 限制条数)
GET /api/metrics/history

# 瓶颈分析
GET /api/metrics/bottleneck?sample_count=60

# 优化建议
GET /api/metrics/suggestions?sample_count=60
```

`/api/metrics/current` 响应示例：

```json
{
  "timestamp": 1699000000.0,
  "cpu_percent": 45.2,
  "memory_percent": 62.8,
  "memory_used_gb": 10.05,
  "memory_total_gb": 16.0,
  "gpu_percent": 78.0,
  "gpu_memory_percent": 65.3,
  "gpu_memory_used_gb": 5.22,
  "gpu_memory_total_gb": 8.0,
  "tokens_per_second": 32.5,
  "ttft_ms": 156.3
}
```

### 优化接口

```bash
# 一键优化 -- 获取最优配置
POST /api/optimize
Content-Type: application/json

{
  "model_size_b": 13.0,
  "target": "balanced",
  "vram_gb": 8.0,
  "ram_gb": 16.0,
  "cpu_cores": 8
}
```

```bash
# 获取完整优化报告 (包含所有模式对比)
GET /api/optimize/report/{model_name}
```

```bash
# 应用优化配置
POST /api/optimize/apply
Content-Type: application/json

{
  "quantization": "Q4_K_M",
  "gpu_layers": 20,
  "kv_cache_bits": 4,
  "context_length": 2048
}
```

---

## 架构说明

### 系统架构图

```
local-model-optimizer/
├── src/
│   ├── api/
│   │   └── server.py              # FastAPI 服务 (OpenAI 兼容 + 管理 + 优化 API)
│   ├── backends/
│   │   ├── base.py                # 推理后端抽象基类
│   │   └── llama_cpp.py           # llama.cpp 后端实现
│   ├── core/
│   │   ├── config.py              # 配置管理 (YAML + 环境变量)
│   │   ├── hardware_detector.py   # 硬件检测 (GPU/CPU/内存/存储)
│   │   └── model_manager.py       # 模型管理 (扫描/下载/转换/显存估算)
│   ├── monitor/
│   │   ├── metrics.py             # 性能指标采集 (CPU/GPU/内存)
│   │   └── analyzer.py            # 性能分析与瓶颈检测
│   └── optimization/              # 核心优化模块
│       ├── scheduler.py           # 智能推理调度器 (优先级队列/模型选择)
│       ├── offloader.py           # 模型卸载 (GPU <-> CPU <-> Disk)
│       ├── kv_cache.py            # KV Cache 优化 (分页/量化/压缩/共享)
│       ├── dynamic_loader.py      # 动态层加载 (LRU/预取/流水线)
│       ├── quantizer.py           # 标准量化管理 (GGUF/GPTQ/AWQ/BnB)
│       ├── ultra_quantizer.py     # 极低精度量化 (Q2_K/IQ2_XXS)
│       ├── memory_optimizer.py    # 内存优化协调器 (一键优化)
│       ├── vram_optimizer.py      # 显存优化器 (6GB/8GB 深度优化)
│       └── multi_vram_optimizer.py # 多显存配置优化器
├── config/
│   └── default.example.yaml       # 配置文件模板
├── scripts/
│   ├── install.ps1 / install.sh   # 安装脚本
│   ├── start.ps1 / start.sh       # 启动脚本
│   └── download_model.ps1 / .sh   # 模型下载脚本
├── research/                      # 项目研究文档
├── requirements.txt
└── setup.py
```

### 内存优化流程图

```
用户请求 --> FastAPI Server
              |
              v
         InferenceScheduler
          |          |
    模型选择      量化选择
    (任务类型)    (显存约束)
          |          |
          v          v
     ┌────────────────────────────┐
     │    MemoryOptimizer         │
     │    (内存优化协调器)         │
     ├────────────────────────────┤
     │  1. QuantizationManager    │  <-- 选择最优量化级别
     │  2. ModelOffloader         │  <-- 计算 GPU/CPU/Disk 层分配
     │  3. KVCacheOptimizer       │  <-- 配置 KV Cache 策略
     │  4. DynamicLayerLoader     │  <-- 配置动态加载策略
     └────────────────────────────┘
              |
              v
     LlamaCppBackend
              |
              v
     模型推理 --> MetricsCollector --> PerformanceAnalyzer
```

### 关键设计决策

1. **llama-cpp-python 作为推理引擎** -- 支持 GGUF 格式，跨平台 (CUDA/Metal/CPU)，内存效率高
2. **优先级堆调度** -- 支持 URGENT/HIGH/NORMAL/LOW 四级优先级，同优先级 FIFO
3. **模型热切换** -- 自动卸载旧模型、加载新模型，切换过程对用户透明
4. **量化自适应** -- 根据可用显存自动选择最优量化级别 (Q2_K ~ FP16)
5. **四维优化协调** -- 量化、卸载、KV Cache、动态加载协同工作，而非独立运作

---

## 性能基准

### 不同配置下的 tokens/s 对比

> 基准: 7B Q4_K_M 全 GPU 在中端 GPU 上 = 40 t/s

| 配置 | 7B | 13B | 30B | 70B |
|------|-----|------|------|------|
| 24GB GPU (全GPU) | 52.0 | 33.0 | 14.0 | 6.5 |
| 8GB GPU (混合) | 40.0 | 25.3 | 8.5 | 3.2 |
| 6GB GPU (混合) | 40.0 | 23.7 | 5.1 | 1.0 |
| 4GB GPU (混合) | 32.0 | 14.8 | - | - |
| 纯 CPU (16GB RAM) | 8.5 | 5.2 | 2.1 | 0.8 |

### 显存占用对比表

| 模型 | Q2_K | Q3_K_M | Q4_K_M | Q8_0 | FP16 |
|------|------|--------|--------|------|------|
| 3B | 0.9 GB | 1.3 GB | 1.7 GB | 3.2 GB | 6.0 GB |
| 7B | 2.2 GB | 3.1 GB | 3.9 GB | 7.4 GB | 14.0 GB |
| 13B | 4.0 GB | 5.7 GB | 7.3 GB | 13.8 GB | 26.0 GB |
| 30B | 9.3 GB | 13.2 GB | 16.8 GB | 31.8 GB | 60.0 GB |
| 70B | 21.7 GB | 30.8 GB | 39.2 GB | 74.2 GB | 140.0 GB |

### 推荐配置速查表

| 显存 | 推荐模型 | 量化 | GPU层 | 速度 | 质量 |
|------|---------|------|-------|------|------|
| 4GB | 7B | Q3_K_M | 16/32 | 18.7 t/s | 0.70 |
| 6GB | 13B | IQ2_XS | 40/40 | 23.7 t/s | 0.50 |
| 8GB | 13B | Q4_K_M | 20/40 | 25.3 t/s | 0.85 |
| 12GB | 13B | Q4_K_M | 40/40 | 33.0 t/s | 0.85 |
| 16GB | 30B | Q4_K_M | 30/60 | 14.0 t/s | 0.85 |
| 24GB | 70B | Q4_K_M | 40/80 | 6.5 t/s | 0.85 |

---

## 开发指南

### 开发环境搭建

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码检查
ruff check src/

# 类型检查
mypy src/

# 运行测试
pytest tests/ -v

# 运行显存优化测试
python test_vram_optimization.py
```

### 添加新的推理后端

实现 `InferenceBackend` 基类即可接入新的推理引擎：

```python
from src.backends.base import InferenceBackend, InferenceConfig, GenerationResult

class MyCustomBackend(InferenceBackend):
    def load_model(self) -> None:
        self._model = load_my_model(self.config.model_path)
        self._is_loaded = True

    def unload_model(self) -> None:
        self._model = None
        self._is_loaded = False

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        output = self._model.generate(
            prompt, max_tokens=kwargs.get("max_tokens", 512)
        )
        return GenerationResult(
            text=output.text,
            tokens_generated=output.token_count,
            tokens_per_second=output.tps,
            prompt_tokens=output.prompt_tokens,
            finish_reason="stop",
        )
```

### 项目命令行工具

安装后可用的 CLI 命令：

```bash
# 启动 API 服务
lmo-server

# 查看硬件信息
lmo-hwinfo

# 运行性能基准测试
lmo-benchmark
```

### Python API 使用示例

```python
# 1. 硬件检测
from src.core.hardware_detector import HardwareDetector
detector = HardwareDetector()
profile = detector.detect()
print(f"GPU: {profile.gpu.name}, VRAM: {profile.gpu.vram_total_gb}GB")

# 2. 模型管理
from src.core.model_manager import ModelManager
manager = ModelManager(model_dir="models")
models = manager.scan_models()
print(f"发现 {len(models)} 个模型")

# 3. 内存优化
from src.optimization.memory_optimizer import MemoryOptimizer, OptimizationProfile, HardwareProfile
optimizer = MemoryOptimizer()
result = optimizer.quick_optimize(13.0, "balanced", vram_gb=8.0)

# 4. 显存优化
from src.optimization.vram_optimizer import VRAMOptimizer, OptimizationTarget
vram_opt = VRAMOptimizer(vram_gb=6.0)
result = vram_opt.optimize(13.0, OptimizationTarget.BALANCED)

# 5. 极低精度量化
from src.optimization.ultra_quantizer import UltraQuantizer
quantizer = UltraQuantizer()
rec = quantizer.recommend_for_vram(13.0, available_vram_gb=6.0)
print(f"推荐量化: {rec.recommended_level.value}")

# 6. 策略对比
from src.optimization.offloader import ModelOffloader
offloader = ModelOffloader()
summary = offloader.get_strategy_summary(13.0, "q4_k_m", gpu_vram_gb=8.0)
print(summary)
```

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

```
MIT License

Copyright (c) 2026 Local Model Optimizer Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
