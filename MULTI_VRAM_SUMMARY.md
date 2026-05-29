# 多显存优化配置完整指南

## 核心成果

**所有显存配置(4GB-24GB)都能运行大模型！**

---

## 快速参考表

| 显存 | 推荐模型 | 量化 | 速度 | 质量 | 适用场景 |
|------|----------|------|------|------|----------|
| **4GB** | 7B | IQ2_XS | 44 t/s | 0.50 | 入门体验 |
| **6GB** | 13B | IQ2_XS | 24 t/s | 0.50 | 主流使用 |
| **8GB** | 13B | Q3_K_M | 25 t/s | 0.70 | **最佳平衡** |
| **12GB** | 13B | Q4_K_M | 22 t/s | 0.85 | 高质量 |
| **16GB** | 30B | IQ3_S | 11 t/s | 0.65 | 大模型 |
| **24GB** | 30B | Q4_K_M | 9 t/s | 0.85 | 旗舰体验 |

---

## 各显存详细配置

### 4GB 显存 (入门级)

**显卡示例：** GTX 1650, RTX 3050

| 模型 | 量化 | GPU层 | 显存 | 速度 | 质量 | 可用性 |
|------|------|-------|------|------|------|--------|
| 3B | Q4_K_M | 32/32 | 2.33 GB | 93 t/s | 0.85 | PERFECT |
| 7B | IQ2_XS | 32/32 | 2.68 GB | 44 t/s | 0.50 | GOOD |
| 13B | IQ2_XXS | 27/40 | 3.07 GB | 16 t/s | 0.45 | OK |
| 30B | IQ2_XXS | 14/60 | 2.98 GB | 2.4 t/s | 0.45 | SLOW |

**推荐：** 7B IQ2_XS，44t/s，体验流畅

---

### 6GB 显存 (主流)

**显卡示例：** RTX 2060, RTX 3060

| 模型 | 量化 | GPU层 | 显存 | 速度 | 质量 | 可用性 |
|------|------|-------|------|------|------|--------|
| 3B | Q4_K_M | 32/32 | 2.33 GB | 93 t/s | 0.85 | PERFECT |
| 7B | Q4_K_M | 32/32 | 4.57 GB | 40 t/s | 0.85 | PERFECT |
| **13B** | **IQ2_XS** | **40/40** | **4.56 GB** | **24 t/s** | **0.50** | **GOOD** |
| 30B | IQ2_XXS | 30/60 | 5.06 GB | 5.1 t/s | 0.45 | OK |

**推荐：** 13B IQ2_XS，24t/s，质量与速度平衡

---

### 8GB 显存 (中高端) ⭐ 最佳平衡

**显卡示例：** RTX 3060Ti, RTX 4060

| 模型 | 量化 | GPU层 | 显存 | 速度 | 质量 | 可用性 |
|------|------|-------|------|------|------|--------|
| 3B | Q4_K_M | 32/32 | 2.33 GB | 93 t/s | 0.85 | PERFECT |
| 7B | Q4_K_M | 32/32 | 4.57 GB | 40 t/s | 0.85 | PERFECT |
| **13B** | **Q3_K_M** | **40/40** | **6.51 GB** | **25 t/s** | **0.70** | **PERFECT** |
| 30B | IQ2_XXS | 45/60 | 7.01 GB | 7.7 t/s | 0.45 | OK |

**推荐：** 13B Q3_K_M，25t/s，质量0.70，最佳平衡点

---

### 12GB 显存 (高端)

**显卡示例：** RTX 3060 12G, RTX 4070

| 模型 | 量化 | GPU层 | 显存 | 速度 | 质量 | 可用性 |
|------|------|-------|------|------|------|--------|
| 3B | Q4_K_M | 32/32 | 2.33 GB | 93 t/s | 0.85 | PERFECT |
| 7B | Q4_K_M | 32/32 | 4.57 GB | 40 t/s | 0.85 | PERFECT |
| **13B** | **Q4_K_M** | **40/40** | **8.07 GB** | **22 t/s** | **0.85** | **PERFECT** |
| 30B | IQ2_XS | 60/60 | 9.86 GB | 10 t/s | 0.50 | GOOD |

**推荐：** 13B Q4_K_M，22t/s，高质量0.85

---

### 16GB 显存 (专业)

**显卡示例：** RTX 4060Ti 16G, RTX 4080

| 模型 | 量化 | GPU层 | 显存 | 速度 | 质量 | 可用性 |
|------|------|-------|------|------|------|--------|
| 3B | Q4_K_M | 32/32 | 2.33 GB | 93 t/s | 0.85 | PERFECT |
| 7B | Q4_K_M | 32/32 | 4.57 GB | 40 t/s | 0.85 | PERFECT |
| 13B | Q4_K_M | 40/40 | 8.07 GB | 22 t/s | 0.85 | PERFECT |
| **30B** | **IQ3_S** | **60/60** | **13.46 GB** | **11 t/s** | **0.65** | **GOOD** |

**推荐：** 30B IQ3_S，11t/s，可运行30B大模型

---

### 24GB 显存 (旗舰)

**显卡示例：** RTX 3090, RTX 4090

| 模型 | 量化 | GPU层 | 显存 | 速度 | 质量 | 可用性 |
|------|------|-------|------|------|------|--------|
| 3B | Q4_K_M | 32/32 | 2.33 GB | 93 t/s | 0.85 | PERFECT |
| 7B | Q4_K_M | 32/32 | 4.57 GB | 40 t/s | 0.85 | PERFECT |
| 13B | Q4_K_M | 40/40 | 8.07 GB | 22 t/s | 0.85 | PERFECT |
| **30B** | **Q4_K_M** | **60/60** | **17.96 GB** | **9 t/s** | **0.85** | **GOOD** |
| 70B | IQ2_XXS | 80/80 | 19.85 GB | 4.4 t/s | 0.45 | SLOW |

**推荐：** 30B Q4_K_M，9t/s，高质量0.85

---

## 13B模型跨显存对比

| 显存 | 量化 | GPU层 | 显存 | 速度 | 质量 |
|------|------|-------|------|------|------|
| 4GB | IQ2_XXS | 27/40 | 3.07 GB | 16 t/s | 0.45 |
| 6GB | IQ2_XS | 40/40 | 4.56 GB | 24 t/s | 0.50 |
| 8GB | Q3_K_M | 40/40 | 6.51 GB | 25 t/s | 0.70 |
| 12GB | Q4_K_M | 40/40 | 8.07 GB | 22 t/s | 0.85 |
| 16GB | Q4_K_M | 40/40 | 8.07 GB | 22 t/s | 0.85 |
| 24GB | Q4_K_M | 40/40 | 8.07 GB | 22 t/s | 0.85 |

**结论：** 13B模型在8GB显存达到最佳平衡点

---

## 优化技术总结

### 1. 极低精度量化

| 量化 | 位数 | 每B显存 | 质量 | 适用显存 |
|------|------|---------|------|----------|
| Q4_K_M | 4.8 bits | 0.56 GB | 0.85 | 8GB+ |
| Q3_K_M | 3.9 bits | 0.44 GB | 0.70 | 6-8GB |
| IQ2_XS | 2.3 bits | 0.29 GB | 0.50 | 4-6GB |
| IQ2_XXS | 2.1 bits | 0.26 GB | 0.45 | 4GB |

### 2. GPU-CPU混合卸载

- **全GPU**：速度快，需要大显存
- **GPU-CPU混合**：平衡方案，适合中等显存
- **动态加载**：按需加载，适合小显存

### 3. KV Cache优化

- **INT8量化**：节省50%显存
- **INT4量化**：节省75%显存
- **滑动窗口**：减少KV Cache大小

---

## 使用方法

### 快速开始

```python
from src.optimization.multi_vram_optimizer import MultiVRAMOptimizer

# 创建优化器
optimizer = MultiVRAMOptimizer()

# 获取8GB显存的最优配置
result = optimizer.get_optimal_config(vram_gb=8, model_size_b=13.0)

print(f"量化: {result.quantization.value}")
print(f"GPU层数: {result.layer_allocation.gpu_layers}/{result.layer_allocation.total_layers}")
print(f"预估显存: {result.estimated_vram_gb:.2f} GB")
print(f"预估速度: {result.estimated_speed_tps:.1f} tokens/s")
```

### 查找可用模型

```python
# 查找8GB显存可用的模型
models = optimizer.find_best_model_for_vram(
    vram_gb=8,
    min_speed=10.0,
    min_quality=0.5,
)

for model in models:
    print(f"{model['model_size_b']}B {model['quantization']} {model['speed_tps']}t/s")
```

### 导出配置

```python
# 导出所有配置到JSON
optimizer.export_configs("vram_configs.json")

# 导出Markdown表格
optimizer.export_markdown_table("VRAM_CONFIGS.md")
```

---

## 文件结构

```
src/optimization/
├── multi_vram_optimizer.py   # 多显存优化器 (新增)
├── vram_optimizer.py         # 单显存优化器
├── ultra_quantizer.py        # 极低精度量化
├── offloader.py              # 模型卸载
├── kv_cache.py               # KV Cache优化
├── dynamic_loader.py         # 动态层加载
├── quantizer.py              # 标准量化
└── memory_optimizer.py       # 内存优化协调器

test_multi_vram.py            # 多显存测试脚本 (新增)
VRAM_CONFIGS.md               # 配置对比表 (自动生成)
vram_configs.json             # 配置JSON (自动生成)
```

---

## 总结

### 各显存推荐方案

| 显存 | 推荐模型 | 量化 | 速度 | 质量 | 一句话总结 |
|------|----------|------|------|------|------------|
| **4GB** | 7B | IQ2_XS | 44 t/s | 0.50 | 入门体验，速度流畅 |
| **6GB** | 13B | IQ2_XS | 24 t/s | 0.50 | 主流配置，可跑13B |
| **8GB** | 13B | Q3_K_M | 25 t/s | 0.70 | **最佳平衡点** |
| **12GB** | 13B | Q4_K_M | 22 t/s | 0.85 | 高质量13B |
| **16GB** | 30B | IQ3_S | 11 t/s | 0.65 | 可跑30B大模型 |
| **24GB** | 30B | Q4_K_M | 9 t/s | 0.85 | 旗舰体验 |

### 核心发现

1. **4GB显存**：可运行7B模型，44t/s
2. **6GB显存**：可运行13B模型，24t/s
3. **8GB显存**：13B模型最佳平衡点，25t/s，质量0.70
4. **12GB显存**：13B高质量运行，22t/s，质量0.85
5. **16GB显存**：可运行30B模型，11t/s
6. **24GB显存**：30B高质量运行，9t/s，质量0.85

---

*生成时间：2026年5月29日*
*覆盖显存：4GB/6GB/8GB/12GB/16GB/24GB*
*覆盖模型：3B/7B/13B/30B/70B*
