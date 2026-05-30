"""
模型卸载 (Model Offloading) 模块

项目核心模块：让小显存也能运行大模型。

支持四种卸载策略:
  - GPU-Only:        全部层在GPU，需要大显存
  - GPU-CPU 混合:    关键层(注意力)在GPU，非关键层(FFN)在CPU
  - GPU-CPU-Disk:    三层卸载，冷数据交换到磁盘
  - CPU-Only:        纯CPU推理

核心能力:
  - 根据模型大小和可用显存智能计算最优GPU层数分配
  - 预估各策略下的显存/内存/磁盘占用
  - 预估各策略下的推理吞吐 (tokens/s)
  - 自动推荐最优卸载配置
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 枚举与常量
# ============================================================

class OffloadStrategy(Enum):
    """卸载策略枚举"""
    GPU_ONLY = "gpu_only"
    GPU_CPU = "gpu_cpu"
    GPU_CPU_DISK = "gpu_cpu_disk"
    CPU_ONLY = "cpu_only"


# 量化级别 -> 每参数字节数 (近似)
_QUANT_BYTES_PER_PARAM: Dict[str, float] = {
    "q2_k":   0.5625,   # ~4.5 bits/param
    "q3_k_s": 0.6875,   # ~5.5 bits/param
    "q3_k_m": 0.75,     # ~6 bits/param
    "q4_0":   0.5625,   # 4 bits + scale, 约 4.5
    "q4_k_s": 0.625,    # 4 bits + improved, ~5
    "q4_k_m": 0.6875,   # ~5.5 bits/param
    "q5_k_s": 0.75,     # ~6 bits/param
    "q5_k_m": 0.8125,   # ~6.5 bits/param
    "q6_k":   0.875,    # ~7 bits/param
    "q8_0":   1.0625,   # 8 bits + scale, ~8.5
    "fp16":   2.0,      # 16 bits
    "f32":    4.0,      # 32 bits
}

# 量化级别 -> 相对推理速度因子 (基于位宽精度的计算量比)
_QUANT_SPEED_FACTOR: Dict[str, float] = {
    "q2_k":   1.30,
    "q3_k_s": 1.25,
    "q3_k_m": 1.20,
    "q4_0":   1.20,
    "q4_k_s": 1.15,
    "q4_k_m": 1.10,
    "q5_k_s": 1.05,
    "q5_k_m": 1.00,
    "q6_k":   0.95,
    "q8_0":   0.85,
    "fp16":   0.50,
    "f32":    0.25,
}

# 典型Transformer层中注意力层占比 (注意力层是关键路径)
_ATTENTION_LAYER_RATIO: float = 0.40   # 注意力占每层 ~40%
_FFN_LAYER_RATIO: float = 0.60         # FFN 占每层 ~60%

# 硬件基准推理速率 (tokens/s) - 用于估算
_BASELINE_GPU_TPS_7B: float = 40.0     # 7B Q4_K_M 在中端NVIDIA GPU上的基线
_BASELINE_CPU_TPS_7B: float = 8.0      # 7B Q4_K_M 在中端CPU上的基线
_DISK_PENALTY_FACTOR: float = 0.15     # 磁盘卸载的速度惩罚系数

# GPU厂商性能因子 (相对于NVIDIA CUDA)
_GPU_VENDOR_PERF_FACTOR: Dict[str, float] = {
    "nvidia": 1.00,          # NVIDIA CUDA: 基线
    "amd_linux": 0.75,       # AMD ROCm Linux: ~75% of CUDA
    "amd_windows": 0.50,     # AMD ROCm Windows: ~50% (ROCm支持有限)
    "apple": 0.85,           # Apple Metal: ~85% (高效但生态不同)
    "intel": 0.40,           # Intel SYCL: ~40% (早期支持)
    "cpu": 1.00,             # CPU-only: 不受GPU厂商因子影响
}

# KV Cache 估算: 每层每1K上下文约占用 (MB)
_KV_CACHE_PER_LAYER_PER_1K_CTX: float = 0.5


# ============================================================
# 数据类
# ============================================================

@dataclass
class OffloadConfig:
    """卸载配置

    Attributes:
        gpu_layers:       卸载到GPU的层数。-1表示全部在GPU，0表示全部在CPU。
        cpu_threads:      CPU推理线程数。
        disk_offload:     是否启用磁盘卸载(三层策略)。
        swap_space_gb:    磁盘交换空间大小(GB)。
        prefetch_layers:  从磁盘预取的层数(预加载到CPU内存)。
        strategy:         卸载策略(由系统自动设置或手动指定)。
        context_length:   上下文长度，影响KV Cache显存占用。
        batch_size:       批处理大小。
    """
    gpu_layers: int = -1            # -1: 全部GPU, 0: 全部CPU
    cpu_threads: int = 4
    disk_offload: bool = False
    swap_space_gb: float = 0.0
    prefetch_layers: int = 2
    strategy: OffloadStrategy = OffloadStrategy.GPU_ONLY
    context_length: int = 4096
    batch_size: int = 512

    @property
    def is_gpu_only(self) -> bool:
        return self.gpu_layers == -1 and not self.disk_offload

    @property
    def is_cpu_only(self) -> bool:
        return self.gpu_layers == 0 and not self.disk_offload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_layers": self.gpu_layers,
            "cpu_threads": self.cpu_threads,
            "disk_offload": self.disk_offload,
            "swap_space_gb": self.swap_space_gb,
            "prefetch_layers": self.prefetch_layers,
            "strategy": self.strategy.value,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
        }


@dataclass
class MemoryEstimate:
    """内存占用预估结果

    Attributes:
        gpu_vram_mb:   GPU显存需求 (MB)
        cpu_ram_mb:    CPU内存需求 (MB)
        disk_mb:       磁盘空间需求 (MB)
        kv_cache_mb:   KV Cache占用 (MB)
        total_mb:      总内存需求 (MB)
        model_size_mb: 模型文件大小 (MB)
    """
    gpu_vram_mb: float = 0.0
    cpu_ram_mb: float = 0.0
    disk_mb: float = 0.0
    kv_cache_mb: float = 0.0
    total_mb: float = 0.0
    model_size_mb: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "gpu_vram_mb": round(self.gpu_vram_mb, 1),
            "cpu_ram_mb": round(self.cpu_ram_mb, 1),
            "disk_mb": round(self.disk_mb, 1),
            "kv_cache_mb": round(self.kv_cache_mb, 1),
            "total_mb": round(self.total_mb, 1),
            "model_size_mb": round(self.model_size_mb, 1),
        }


@dataclass
class PerformanceEstimate:
    """性能预估结果

    Attributes:
        estimated_tps:       预估推理速率 (tokens/s)
        latency_factor:      相对于GPU-Only的延迟倍数
        bottleneck:          性能瓶颈描述
        gpu_utilization:     GPU利用率预估 (0.0 - 1.0)
        suitable_for_realtime: 是否适合实时对话 (>= 5 tokens/s)
    """
    estimated_tps: float = 0.0
    latency_factor: float = 1.0
    bottleneck: str = ""
    gpu_utilization: float = 0.0
    suitable_for_realtime: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_tps": round(self.estimated_tps, 1),
            "latency_factor": round(self.latency_factor, 2),
            "bottleneck": self.bottleneck,
            "gpu_utilization": round(self.gpu_utilization, 2),
            "suitable_for_realtime": self.suitable_for_realtime,
        }


@dataclass
class OffloadReport:
    """完整卸载报告：包含配置、内存预估、性能预估和对比

    Attributes:
        config:              卸载配置
        memory:              内存占用预估
        performance:         性能预估
        strategy_name:       策略名称
        total_layers:        模型总层数
        gpu_layers_actual:   实际GPU层数
        cpu_layers_actual:   实际CPU层数
        disk_layers_actual:  实际磁盘层数
    """
    config: OffloadConfig = field(default_factory=OffloadConfig)
    memory: MemoryEstimate = field(default_factory=MemoryEstimate)
    performance: PerformanceEstimate = field(default_factory=PerformanceEstimate)
    strategy_name: str = ""
    total_layers: int = 0
    gpu_layers_actual: int = 0
    cpu_layers_actual: int = 0
    disk_layers_actual: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_layers": self.total_layers,
            "gpu_layers": self.gpu_layers_actual,
            "cpu_layers": self.cpu_layers_actual,
            "disk_layers": self.disk_layers_actual,
            "config": self.config.to_dict(),
            "memory": self.memory.to_dict(),
            "performance": self.performance.to_dict(),
        }


# ============================================================
# 工具函数
# ============================================================

def _estimate_model_layers(model_size_b: float) -> int:
    """根据模型参数量估算Transformer层数

    基于主流开源模型的架构规律:
      - 1B-3B:   ~24-28 层
      - 7B:      ~32 层
      - 13B:     ~40 层
      - 30B-34B: ~60 层
      - 65B-70B: ~80 层

    Args:
        model_size_b: 模型参数量 (B = 十亿)

    Returns:
        估算的层数
    """
    if model_size_b <= 0:
        return 0
    # 经验公式: layers ≈ 6.7 * sqrt(params_B) + 16, 取整到最近的4的倍数
    raw = 6.7 * math.sqrt(model_size_b) + 16
    layers = max(1, int(round(raw / 4) * 4))
    return layers


def _get_bytes_per_param(quantization: str) -> float:
    """获取量化级别的每参数字节数

    Args:
        quantization: 量化级别字符串 (如 'q4_k_m', 'fp16')

    Returns:
        每参数字节数
    """
    key = quantization.lower().replace("-", "_").replace(" ", "_")
    return _QUANT_BYTES_PER_PARAM.get(key, 0.6875)  # 默认 Q4_K_M


def _get_speed_factor(quantization: str) -> float:
    """获取量化级别的速度因子

    Args:
        quantization: 量化级别字符串

    Returns:
        速度因子 (>1 更快, <1 更慢)
    """
    key = quantization.lower().replace("-", "_").replace(" ", "_")
    return _QUANT_SPEED_FACTOR.get(key, 1.0)


def _estimate_kv_cache_mb(
    n_layers: int, context_length: int, model_size_b: float
) -> float:
    """估算KV Cache显存占用

    KV Cache大小与层数、上下文长度成正比，
    同时与隐藏维度有关(大模型隐藏维度更大)。

    Args:
        n_layers: 模型层数
        context_length: 上下文长度
        model_size_b: 模型参数量 (B)

    Returns:
        KV Cache占用 (MB)
    """
    # 隐藏维度随模型大小增长: dim ≈ 64 * sqrt(params_B) * 4
    # 但KV Cache主要取决于层数和上下文，简化为经验公式
    hidden_scale = math.sqrt(model_size_b / 7.0)  # 以7B为基准
    ctx_factor = context_length / 1024.0
    kv_mb = n_layers * _KV_CACHE_PER_LAYER_PER_1K_CTX * ctx_factor * hidden_scale
    return max(kv_mb, 0.0)


# ============================================================
# 核心类
# ============================================================

class ModelOffloader:
    """模型卸载管理器

    核心卖点：让小显存也能运行大模型。

    提供三大核心能力:
      1. 内存预估: 给定模型大小和卸载策略，计算GPU/CPU/磁盘各需多少空间
      2. 策略推荐: 给定硬件资源，自动推荐最优卸载配置
      3. 性能预估: 对比不同策略下的推理速率

    典型用法::

        offloader = ModelOffloader()

        # 推荐配置
        config = offloader.recommend_offload_strategy(
            model_size_b=13.0,
            gpu_vram_gb=6.0,
            cpu_ram_gb=16.0,
        )

        # 预估内存占用
        mem = offloader.estimate_memory_usage(
            model_size_b=13.0,
            quantization="q4_k_m",
            config=config,
        )

        # 对比各策略性能
        comparison = offloader.compare_strategies(
            model_size_b=13.0,
            quantization="q4_k_m",
            gpu_vram_gb=6.0,
        )
    """

    # ----------------------------------------------------------
    # 内存预估
    # ----------------------------------------------------------

    def estimate_memory_usage(
        self,
        model_size_b: float,
        quantization: str,
        config: OffloadConfig,
    ) -> MemoryEstimate:
        """预估指定配置下的内存占用

        Args:
            model_size_b:  模型参数量 (B = 十亿), 如 7.0 表示 7B
            quantization:  量化级别 (如 'q4_k_m', 'fp16')
            config:        卸载配置

        Returns:
            MemoryEstimate: GPU显存、CPU内存、磁盘空间需求
        """
        bytes_per_param = _get_bytes_per_param(quantization)
        model_size_mb = model_size_b * 1e9 * bytes_per_param / (1024 * 1024)
        total_layers = _estimate_model_layers(model_size_b)

        # 确定实际GPU/CPU/Disk层数分配
        gpu_l, cpu_l, disk_l = self._resolve_layer_allocation(
            total_layers, config
        )

        # 每层大小
        per_layer_mb = model_size_mb / max(total_layers, 1)

        # GPU 显存: 模型层 + KV Cache + 开销
        gpu_model_mb = per_layer_mb * gpu_l
        kv_cache_mb = _estimate_kv_cache_mb(
            total_layers, config.context_length, model_size_b
        )
        # GPU上的KV Cache按GPU层数比例分配
        kv_on_gpu = kv_cache_mb * (gpu_l / max(total_layers, 1))
        # 显存额外开销 (CUDA context, buffers等)
        gpu_overhead = 300.0 if gpu_l > 0 else 0.0
        gpu_vram_mb = gpu_model_mb + kv_on_gpu + gpu_overhead

        # CPU 内存: 模型层 + KV Cache剩余 + 工作内存
        cpu_model_mb = per_layer_mb * cpu_l
        kv_on_cpu = kv_cache_mb - kv_on_gpu
        # CPU推理额外工作内存
        cpu_work_mb = config.batch_size * 0.01 if cpu_l > 0 else 0.0
        cpu_ram_mb = cpu_model_mb + max(kv_on_cpu, 0.0) + cpu_work_mb

        # 磁盘: 仅在三层策略时使用
        disk_mb = per_layer_mb * disk_l

        # 总需求
        total_mb = gpu_vram_mb + cpu_ram_mb

        estimate = MemoryEstimate(
            gpu_vram_mb=gpu_vram_mb,
            cpu_ram_mb=cpu_ram_mb,
            disk_mb=disk_mb,
            kv_cache_mb=kv_cache_mb,
            total_mb=total_mb,
            model_size_mb=model_size_mb,
        )

        logger.debug(
            "内存预估: model=%sB quant=%s gpu_layers=%d/%d  "
            "GPU=%.0fMB CPU=%.0fMB Disk=%.0fMB KV=%.0fMB",
            model_size_b, quantization, gpu_l, total_layers,
            gpu_vram_mb, cpu_ram_mb, disk_mb, kv_cache_mb,
        )

        return estimate

    # ----------------------------------------------------------
    # 最优GPU层数计算
    # ----------------------------------------------------------

    def calculate_optimal_gpu_layers(
        self,
        model_size_b: float,
        quantization: str,
        available_vram_gb: float,
    ) -> int:
        """计算在给定显存约束下可卸载到GPU的最优层数

        算法:
          1. 计算模型总大小和每层大小
          2. 预留显存给KV Cache和系统开销
          3. 用剩余显存除以每层大小得到GPU层数
          4. 限制在 [0, total_layers] 范围内

        Args:
            model_size_b:      模型参数量 (B)
            quantization:      量化级别
            available_vram_gb: 可用GPU显存 (GB)

        Returns:
            最优GPU层数 (0 表示全部在CPU)
        """
        if available_vram_gb <= 0:
            return 0

        bytes_per_param = _get_bytes_per_param(quantization)
        model_size_mb = model_size_b * 1e9 * bytes_per_param / (1024 * 1024)
        total_layers = _estimate_model_layers(model_size_b)

        if total_layers <= 0 or model_size_mb <= 0:
            return 0

        per_layer_mb = model_size_mb / total_layers
        available_vram_mb = available_vram_gb * 1024

        # 预留: KV Cache (估算1/3放GPU) + CUDA开销
        kv_reserve = _estimate_kv_cache_mb(
            total_layers, 4096, model_size_b
        ) * 0.4
        overhead_reserve = 300.0  # CUDA context等
        usable_vram = available_vram_mb - kv_reserve - overhead_reserve

        if usable_vram <= 0:
            return 0

        gpu_layers = int(usable_vram / per_layer_mb)
        gpu_layers = max(0, min(gpu_layers, total_layers))

        logger.debug(
            "最优GPU层数: model=%sB quant=%s vram=%.1fGB -> %d/%d layers",
            model_size_b, quantization, available_vram_gb,
            gpu_layers, total_layers,
        )

        return gpu_layers

    # ----------------------------------------------------------
    # 策略推荐
    # ----------------------------------------------------------

    def recommend_offload_strategy(
        self,
        model_size_b: float,
        gpu_vram_gb: float,
        cpu_ram_gb: float,
        disk_available_gb: float = 100.0,
        target_tps: float = 10.0,
        quantization: str = "q4_k_m",
    ) -> OffloadConfig:
        """根据硬件资源推荐最优卸载配置

        决策逻辑:
          1. 如果GPU显存足够装下整个模型 -> GPU-Only
          2. 如果GPU显存不够但CPU内存够 -> GPU-CPU混合，尽量多放GPU层
          3. 如果GPU+CPU都不够 -> GPU-CPU-Disk三层卸载
          4. 如果没有GPU -> CPU-Only

        Args:
            model_size_b:        模型参数量 (B)
            gpu_vram_gb:         可用GPU显存 (GB)
            cpu_ram_gb:          可用CPU内存 (GB)
            disk_available_gb:   可用磁盘空间 (GB)
            target_tps:          目标推理速率 (tokens/s)
            quantization:        量化级别

        Returns:
            推荐的OffloadConfig
        """
        bytes_per_param = _get_bytes_per_param(quantization)
        model_size_mb = model_size_b * 1e9 * bytes_per_param / (1024 * 1024)
        model_size_gb = model_size_mb / 1024
        total_layers = _estimate_model_layers(model_size_b)

        # 情况1: 无GPU -> CPU-Only
        if gpu_vram_gb <= 0:
            config = OffloadConfig(
                gpu_layers=0,
                cpu_threads=self._recommend_cpu_threads(),
                disk_offload=False,
                swap_space_gb=0.0,
                strategy=OffloadStrategy.CPU_ONLY,
            )
            logger.info(
                "推荐策略: CPU-Only  model=%sB (无GPU可用)", model_size_b
            )
            return config

        # 计算最优GPU层数
        optimal_gpu = self.calculate_optimal_gpu_layers(
            model_size_b, quantization, gpu_vram_gb
        )

        # 情况2: GPU能装下全部模型 -> GPU-Only
        if optimal_gpu >= total_layers:
            config = OffloadConfig(
                gpu_layers=-1,
                cpu_threads=self._recommend_cpu_threads(),
                disk_offload=False,
                swap_space_gb=0.0,
                strategy=OffloadStrategy.GPU_ONLY,
            )
            logger.info(
                "推荐策略: GPU-Only  model=%sB  vram=%.1fGB  layers=%d",
                model_size_b, gpu_vram_gb, total_layers,
            )
            return config

        # 计算剩余CPU需要装多少层
        cpu_layers_needed = total_layers - optimal_gpu
        per_layer_mb = model_size_mb / max(total_layers, 1)
        cpu_needed_mb = per_layer_mb * cpu_layers_needed
        # CPU还需要容纳KV Cache和工作内存
        kv_cache_mb = _estimate_kv_cache_mb(total_layers, 4096, model_size_b)
        cpu_total_needed = cpu_needed_mb + kv_cache_mb + 500  # 500MB工作内存

        # 情况3: CPU内存够装剩余层 -> GPU-CPU混合
        if cpu_total_needed <= cpu_ram_gb * 1024 * 0.8:
            config = OffloadConfig(
                gpu_layers=optimal_gpu,
                cpu_threads=self._recommend_cpu_threads(),
                disk_offload=False,
                swap_space_gb=0.0,
                strategy=OffloadStrategy.GPU_CPU,
            )
            logger.info(
                "推荐策略: GPU-CPU混合  model=%sB  gpu=%d/%d layers  "
                "vram=%.1fGB  ram=%.1fGB",
                model_size_b, optimal_gpu, total_layers,
                gpu_vram_gb, cpu_ram_gb,
            )
            return config

        # 情况4: CPU内存也不够 -> GPU-CPU-Disk 三层卸载
        # CPU能装多少层就装多少，剩余放磁盘
        cpu_budget_mb = cpu_ram_gb * 1024 * 0.7 - kv_cache_mb - 500
        cpu_affordable = max(0, int(cpu_budget_mb / per_layer_mb))
        disk_layers = total_layers - optimal_gpu - cpu_affordable

        swap_gb = max(disk_layers * per_layer_mb / 1024, 0) * 1.2  # 预留20%
        swap_gb = min(swap_gb, disk_available_gb * 0.5)  # 不超过磁盘50%

        config = OffloadConfig(
            gpu_layers=optimal_gpu,
            cpu_threads=self._recommend_cpu_threads(),
            disk_offload=True,
            swap_space_gb=round(swap_gb, 1),
            prefetch_layers=min(2, cpu_affordable),
            strategy=OffloadStrategy.GPU_CPU_DISK,
        )
        logger.info(
            "推荐策略: GPU-CPU-Disk  model=%sB  gpu=%d  cpu=%d  disk=%d  "
            "swap=%.1fGB",
            model_size_b, optimal_gpu, cpu_affordable, disk_layers, swap_gb,
        )
        return config

    # ----------------------------------------------------------
    # 性能预估
    # ----------------------------------------------------------

    def estimate_performance(
        self,
        model_size_b: float,
        quantization: str,
        config: OffloadConfig,
        gpu_vram_gb: float = 0.0,
        cpu_ram_gb: float = 16.0,
        gpu_vendor: str = "nvidia",
    ) -> PerformanceEstimate:
        """预估指定配置下的推理性能

        基于经验公式，考虑:
          - 模型大小与量化级别的计算量
          - GPU/CPU层数比例对速度的影响
          - 磁盘I/O带来的额外延迟
          - 硬件规模的扩展性
          - GPU厂商性能差异 (NVIDIA/AMD/Apple/Intel)

        Args:
            model_size_b:  模型参数量 (B)
            quantization:  量化级别
            config:        卸载配置
            gpu_vram_gb:   GPU显存 (GB)，用于估算GPU算力
            cpu_ram_gb:    CPU内存 (GB)
            gpu_vendor:    GPU厂商 ('nvidia', 'amd', 'apple', 'intel', 'cpu')

        Returns:
            PerformanceEstimate: 预估tokens/s、延迟倍数等
        """
        total_layers = _estimate_model_layers(model_size_b)
        gpu_l, cpu_l, disk_l = self._resolve_layer_allocation(
            total_layers, config
        )

        quant_factor = _get_speed_factor(quantization)

        # 基准速率 (以7B Q4_K_M中端NVIDIA GPU为基准)
        # 参数量越大速度越慢: tps ∝ 1/sqrt(params)
        size_penalty = math.sqrt(7.0 / max(model_size_b, 0.5))

        # GPU厂商性能因子
        import sys
        if gpu_vendor == "amd":
            vendor_key = "amd_windows" if sys.platform == "win32" else "amd_linux"
        else:
            vendor_key = gpu_vendor
        vendor_factor = _GPU_VENDOR_PERF_FACTOR.get(vendor_key, 1.0)

        # GPU速率估算
        gpu_tps = 0.0
        if gpu_l > 0 and gpu_vram_gb > 0:
            # GPU算力随显存大致线性增长(简化)
            vram_factor = min(gpu_vram_gb / 8.0, 3.0)
            gpu_ratio = gpu_l / max(total_layers, 1)
            gpu_tps = _BASELINE_GPU_TPS_7B * size_penalty * quant_factor * vram_factor * gpu_ratio * vendor_factor

        # CPU速率估算
        cpu_tps = 0.0
        if cpu_l > 0:
            thread_factor = min(config.cpu_threads / 4.0, 2.0)
            cpu_ratio = cpu_l / max(total_layers, 1)
            cpu_tps = _BASELINE_CPU_TPS_7B * size_penalty * quant_factor * thread_factor * cpu_ratio

        # 磁盘惩罚: 磁盘层严重拖慢整体 (I/O瓶颈)
        disk_penalty = 0.0
        if disk_l > 0:
            disk_ratio = disk_l / max(total_layers, 1)
            disk_penalty = disk_ratio * _DISK_PENALTY_FACTOR

        # 总体速率: 取GPU和CPU的较小值(瓶颈层)，再减去磁盘惩罚
        # 混合推理时，实际速率取决于最慢的组件
        if gpu_l > 0 and cpu_l > 0:
            # GPU-CPU混合: 受PCIe带宽和同步开销影响
            estimated_tps = min(gpu_tps, cpu_tps * 1.5) * 0.85  # 15% PCIe开销
        elif gpu_l > 0:
            estimated_tps = gpu_tps
        else:
            estimated_tps = cpu_tps

        # 磁盘惩罚
        if disk_l > 0:
            estimated_tps *= (1.0 - disk_penalty)

        estimated_tps = max(estimated_tps, 0.1)

        # 相对延迟因子 (相对于GPU-Only)
        gpu_only_tps = _BASELINE_GPU_TPS_7B * size_penalty * quant_factor * vendor_factor
        latency_factor = gpu_only_tps / max(estimated_tps, 0.1)

        # 瓶颈判定
        bottleneck = self._identify_bottleneck(
            gpu_l, cpu_l, disk_l, gpu_tps, cpu_tps, gpu_vram_gb
        )

        # GPU利用率
        gpu_util = gpu_l / max(total_layers, 1) if total_layers > 0 else 0.0

        return PerformanceEstimate(
            estimated_tps=round(estimated_tps, 1),
            latency_factor=round(latency_factor, 2),
            bottleneck=bottleneck,
            gpu_utilization=round(gpu_util, 2),
            suitable_for_realtime=estimated_tps >= 5.0,
        )

    # ----------------------------------------------------------
    # 策略对比
    # ----------------------------------------------------------

    def compare_strategies(
        self,
        model_size_b: float,
        quantization: str,
        gpu_vram_gb: float = 8.0,
        cpu_ram_gb: float = 16.0,
        disk_available_gb: float = 100.0,
        gpu_vendor: str = "nvidia",
    ) -> List[OffloadReport]:
        """对比所有可行的卸载策略

        返回每种策略的完整报告（内存占用、性能预估），
        便于用户选择最适合的方案。

        Args:
            model_size_b:        模型参数量 (B)
            quantization:        量化级别
            gpu_vram_gb:         GPU显存 (GB)
            cpu_ram_gb:          CPU内存 (GB)
            disk_available_gb:   可用磁盘 (GB)

        Returns:
            OffloadReport列表，按推荐优先级排序
        """
        reports: List[OffloadReport] = []
        total_layers = _estimate_model_layers(model_size_b)

        # 策略1: GPU-Only
        if gpu_vram_gb > 0:
            config = OffloadConfig(
                gpu_layers=-1,
                cpu_threads=self._recommend_cpu_threads(),
                strategy=OffloadStrategy.GPU_ONLY,
            )
            mem = self.estimate_memory_usage(model_size_b, quantization, config)
            # 检查是否真的能放下
            if mem.gpu_vram_mb <= gpu_vram_gb * 1024:
                perf = self.estimate_performance(
                    model_size_b, quantization, config, gpu_vram_gb, cpu_ram_gb, gpu_vendor
                )
                reports.append(OffloadReport(
                    config=config, memory=mem, performance=perf,
                    strategy_name="GPU-Only (全GPU)",
                    total_layers=total_layers,
                    gpu_layers_actual=total_layers,
                    cpu_layers_actual=0,
                    disk_layers_actual=0,
                ))

        # 策略2: GPU-CPU 混合
        if gpu_vram_gb > 0:
            optimal_gpu = self.calculate_optimal_gpu_layers(
                model_size_b, quantization, gpu_vram_gb
            )
            if 0 < optimal_gpu < total_layers:
                config = OffloadConfig(
                    gpu_layers=optimal_gpu,
                    cpu_threads=self._recommend_cpu_threads(),
                    strategy=OffloadStrategy.GPU_CPU,
                )
                mem = self.estimate_memory_usage(model_size_b, quantization, config)
                perf = self.estimate_performance(
                    model_size_b, quantization, config, gpu_vram_gb, cpu_ram_gb, gpu_vendor
                )
                cpu_actual = total_layers - optimal_gpu
                reports.append(OffloadReport(
                    config=config, memory=mem, performance=perf,
                    strategy_name="GPU-CPU混合",
                    total_layers=total_layers,
                    gpu_layers_actual=optimal_gpu,
                    cpu_layers_actual=cpu_actual,
                    disk_layers_actual=0,
                ))

        # 策略3: GPU-CPU-Disk 三层卸载
        if gpu_vram_gb > 0:
            optimal_gpu = self.calculate_optimal_gpu_layers(
                model_size_b, quantization, gpu_vram_gb
            )
            bytes_per_param = _get_bytes_per_param(quantization)
            model_size_mb = model_size_b * 1e9 * bytes_per_param / (1024 * 1024)
            per_layer_mb = model_size_mb / max(total_layers, 1)
            kv_mb = _estimate_kv_cache_mb(total_layers, 4096, model_size_b)
            cpu_budget_mb = cpu_ram_gb * 1024 * 0.7 - kv_mb - 500
            cpu_affordable = max(0, int(cpu_budget_mb / per_layer_mb))
            disk_l = total_layers - optimal_gpu - cpu_affordable

            if disk_l > 0:
                swap_gb = disk_l * per_layer_mb / 1024 * 1.2
                swap_gb = min(swap_gb, disk_available_gb * 0.5)
                config = OffloadConfig(
                    gpu_layers=optimal_gpu,
                    cpu_threads=self._recommend_cpu_threads(),
                    disk_offload=True,
                    swap_space_gb=round(swap_gb, 1),
                    prefetch_layers=min(2, cpu_affordable),
                    strategy=OffloadStrategy.GPU_CPU_DISK,
                )
                mem = self.estimate_memory_usage(model_size_b, quantization, config)
                perf = self.estimate_performance(
                    model_size_b, quantization, config, gpu_vram_gb, cpu_ram_gb, gpu_vendor
                )
                reports.append(OffloadReport(
                    config=config, memory=mem, performance=perf,
                    strategy_name="GPU-CPU-Disk (三层卸载)",
                    total_layers=total_layers,
                    gpu_layers_actual=optimal_gpu,
                    cpu_layers_actual=cpu_affordable,
                    disk_layers_actual=disk_l,
                ))

        # 策略4: CPU-Only
        config = OffloadConfig(
            gpu_layers=0,
            cpu_threads=self._recommend_cpu_threads(),
            strategy=OffloadStrategy.CPU_ONLY,
        )
        mem = self.estimate_memory_usage(model_size_b, quantization, config)
        # CPU-Only需要检查内存是否够
        if mem.cpu_ram_mb <= cpu_ram_gb * 1024 * 0.8:
            perf = self.estimate_performance(
                model_size_b, quantization, config, 0, cpu_ram_gb, gpu_vendor
            )
            reports.append(OffloadReport(
                config=config, memory=mem, performance=perf,
                strategy_name="CPU-Only (纯CPU)",
                total_layers=total_layers,
                gpu_layers_actual=0,
                cpu_layers_actual=total_layers,
                disk_layers_actual=0,
            ))

        # 按性能降序排列
        reports.sort(key=lambda r: r.performance.estimated_tps, reverse=True)

        return reports

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _resolve_layer_allocation(
        self, total_layers: int, config: OffloadConfig
    ) -> Tuple[int, int, int]:
        """解析配置，返回 (gpu_layers, cpu_layers, disk_layers)

        处理 gpu_layers 的特殊值:
          - -1: 全部在GPU
          -  0: 全部在CPU (或根据disk_offload决定)
          - >0: 指定层数

        Args:
            total_layers: 模型总层数
            config:       卸载配置

        Returns:
            (gpu层数, cpu层数, disk层数) 三者之和 = total_layers
        """
        if total_layers <= 0:
            return 0, 0, 0

        # 确定GPU层数
        if config.gpu_layers == -1:
            gpu_l = total_layers
        elif config.gpu_layers == 0:
            gpu_l = 0
        else:
            gpu_l = min(config.gpu_layers, total_layers)

        remaining = total_layers - gpu_l

        if config.disk_offload and remaining > 0:
            # 三层分配: prefetch_layers在CPU(热数据), 其余在磁盘(冷数据)
            cpu_l = min(config.prefetch_layers, remaining)
            disk_l = remaining - cpu_l
        else:
            cpu_l = remaining
            disk_l = 0

        return gpu_l, cpu_l, disk_l

    @staticmethod
    def _recommend_cpu_threads() -> int:
        """推荐CPU线程数

        基于系统CPU核心数，取物理核心数(不超过8)。
        """
        import os
        logical = os.cpu_count() or 4
        physical = max(1, logical // 2)
        return min(physical, 8)

    @staticmethod
    def _identify_bottleneck(
        gpu_l: int, cpu_l: int, disk_l: int,
        gpu_tps: float, cpu_tps: float,
        gpu_vram_gb: float,
    ) -> str:
        """识别性能瓶颈

        Returns:
            瓶颈描述字符串
        """
        if disk_l > 0 and gpu_l > 0 and cpu_l > 0:
            return "磁盘I/O (三层卸载，冷层从磁盘加载延迟高)"
        if disk_l > 0:
            return "磁盘I/O (大量层在磁盘，严重影响推理速度)"

        if gpu_l > 0 and cpu_l > 0:
            if gpu_tps > cpu_tps:
                return "CPU计算 (GPU等待CPU层完成)"
            else:
                return "PCIe带宽 (GPU-CPU数据传输)"

        if gpu_l > 0:
            if gpu_vram_gb < 4:
                return "GPU显存 (显存紧张，KV Cache受限)"
            return "GPU计算 (正常瓶颈)"

        return "CPU计算 (纯CPU推理)"

    # ----------------------------------------------------------
    # 便捷方法
    # ----------------------------------------------------------

    def get_strategy_summary(
        self,
        model_size_b: float,
        quantization: str = "q4_k_m",
        gpu_vram_gb: float = 8.0,
        cpu_ram_gb: float = 16.0,
        gpu_vendor: str = "nvidia",
    ) -> str:
        """获取策略对比的文本摘要 (适合打印/日志)

        Args:
            model_size_b:  模型参数量 (B)
            quantization:  量化级别
            gpu_vram_gb:   GPU显存 (GB)
            cpu_ram_gb:    CPU内存 (GB)

        Returns:
            格式化的策略对比文本
        """
        reports = self.compare_strategies(
            model_size_b, quantization, gpu_vram_gb, cpu_ram_gb, gpu_vendor=gpu_vendor
        )

        if not reports:
            return "无可行策略: 硬件资源不足以运行该模型"

        lines = [
            f"{'='*60}",
            f"模型卸载策略对比: {model_size_b}B ({quantization})",
            f"硬件: GPU {gpu_vram_gb:.0f}GB | CPU {cpu_ram_gb:.0f}GB",
            f"{'='*60}",
            "",
        ]

        for i, r in enumerate(reports, 1):
            rec_mark = " [推荐]" if i == 1 else ""
            lines.append(f"策略{i}: {r.strategy_name}{rec_mark}")
            lines.append(f"  层分配: GPU={r.gpu_layers_actual}  CPU={r.cpu_layers_actual}  Disk={r.disk_layers_actual}")
            lines.append(f"  显存需求: {r.memory.gpu_vram_mb:.0f} MB")
            lines.append(f"  内存需求: {r.memory.cpu_ram_mb:.0f} MB")
            if r.memory.disk_mb > 0:
                lines.append(f"  磁盘需求: {r.memory.disk_mb:.0f} MB")
            lines.append(f"  KV Cache: {r.memory.kv_cache_mb:.0f} MB")
            lines.append(f"  预估速度: {r.performance.estimated_tps:.1f} tokens/s")
            lines.append(f"  速度倍率: {r.performance.latency_factor:.1f}x (相对GPU-Only)")
            lines.append(f"  实时对话: {'适合' if r.performance.suitable_for_realtime else '不适合'}")
            lines.append(f"  瓶颈: {r.performance.bottleneck}")
            lines.append("")

        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行演示: 展示卸载策略对比"""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    offloader = ModelOffloader()

    # 演示不同模型大小在不同显存下的策略对比
    scenarios = [
        (7.0, 4.0, 16.0),    # 7B 模型 + 4GB 显存
        (7.0, 8.0, 16.0),    # 7B 模型 + 8GB 显存
        (13.0, 6.0, 16.0),   # 13B 模型 + 6GB 显存
        (13.0, 24.0, 32.0),  # 13B 模型 + 24GB 显存
        (70.0, 8.0, 32.0),   # 70B 模型 + 8GB 显存 (极端场景)
    ]

    for model_b, vram, ram in scenarios:
        print(offloader.get_strategy_summary(model_b, "q4_k_m", vram, ram))
        print()


if __name__ == "__main__":
    main()
