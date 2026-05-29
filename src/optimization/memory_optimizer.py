"""
内存优化协调器

核心协调模块，整合所有优化策略:
- ModelOffloader (offloader.py): 模型卸载 (GPU <-> CPU <-> Disk)
- KVCacheOptimizer (kv_cache.py): KV Cache 优化 (分页/量化/压缩/前缀共享)
- DynamicLayerLoader (dynamic_loader.py): 动态层加载 (LRU淘汰/预测预取)
- QuantizationManager (quantizer.py): 量化管理 (GGUF/GPTQ/AWQ/BnB)

提供一键优化配置功能:
    输入: 模型参数量、目标优化模式、硬件画像
    输出: 完整优化配置 (量化 + 卸载 + Cache策略 + 动态加载)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .offloader import (
    ModelOffloader,
    OffloadConfig,
    OffloadStrategy,
    OffloadReport,
    MemoryEstimate,
    PerformanceEstimate,
    _estimate_model_layers,
    _get_bytes_per_param,
    _get_speed_factor,
)
from .kv_cache import (
    KVCacheOptimizer,
    KVCacheConfig,
)
from .quantizer import (
    QuantizationManager as _QuantManager,
    ALL_PROFILES,
    QuantizationProfile,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 枚举定义
# ===========================================================================


class OptimizationProfile(Enum):
    """优化目标模式"""
    MINIMAL_VRAM = "minimal_vram"   # 最小显存占用 (可以慢)
    BALANCED = "balanced"           # 平衡模式
    MAX_SPEED = "max_speed"         # 最快速度 (需要更多显存)
    QUALITY = "quality"             # 最高质量


# ===========================================================================
# 配置数据类
# ===========================================================================


@dataclass
class DynamicConfig:
    """动态层加载配置 (用于 memory_optimizer 的简化视图)

    对应 dynamic_loader.py 中的 DynamicConfig。
    """
    max_gpu_layers: int = 0              # GPU 最大驻留层数 (0 = 不使用动态加载)
    max_cpu_layers: int = 0              # CPU 最大驻留层数
    prefetch_enabled: bool = False       # 是否启用预测预取
    prefetch_count: int = 0              # 预取层数
    swap_enabled: bool = False           # 是否启用层交换 (磁盘)
    eviction_strategy: str = "lru"       # 淘汰策略: lru / lfu / adaptive

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_gpu_layers": self.max_gpu_layers,
            "max_cpu_layers": self.max_cpu_layers,
            "prefetch_enabled": self.prefetch_enabled,
            "prefetch_count": self.prefetch_count,
            "swap_enabled": self.swap_enabled,
            "eviction_strategy": self.eviction_strategy,
        }


@dataclass
class QuantizationConfig:
    """量化配置 (用于 memory_optimizer 的输出)"""
    level: str = "q4_k_m"               # 量化级别
    vram_per_b: float = 0.0              # 每B参数显存 (GB)
    quality_score: float = 0.0           # 质量评分 (0-1)
    speed_score: float = 0.0             # 速度评分 (0-1)
    bits: int = 4                        # 量化位数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "vram_per_b": round(self.vram_per_b, 3),
            "quality_score": round(self.quality_score, 3),
            "speed_score": round(self.speed_score, 3),
            "bits": self.bits,
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    quantization: QuantizationConfig
    offload_config: OffloadConfig
    kv_cache_config: KVCacheConfig
    dynamic_config: DynamicConfig
    profile: OptimizationProfile
    # 预估指标
    estimated_vram_gb: float = 0.0       # 预估显存使用 (GB)
    estimated_ram_gb: float = 0.0        # 预估系统内存使用 (GB)
    estimated_speed_tps: float = 0.0     # 预估生成速度 (tokens/s)
    quality_score: float = 0.0           # 质量评分 (0-1)
    # 详情
    total_model_size_gb: float = 0.0     # 模型权重大小 (GB)
    kv_cache_size_gb: float = 0.0        # KV Cache 大小 (GB)
    overhead_gb: float = 0.0             # 运行时开销 (GB)
    gpu_utilization: float = 0.0         # GPU 利用率 (0-1)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.value,
            "quantization": self.quantization.to_dict(),
            "offload_config": self.offload_config.to_dict(),
            "kv_cache_config": {
                "cache_bits": self.kv_cache_config.cache_bits,
                "max_cache_size_gb": self.kv_cache_config.max_cache_size_gb,
                "eviction_policy": self.kv_cache_config.eviction_policy,
                "prefix_sharing": self.kv_cache_config.prefix_sharing,
                "page_size": self.kv_cache_config.page_size,
            },
            "dynamic_config": self.dynamic_config.to_dict(),
            "estimated_vram_gb": round(self.estimated_vram_gb, 2),
            "estimated_ram_gb": round(self.estimated_ram_gb, 2),
            "estimated_speed_tps": round(self.estimated_speed_tps, 1),
            "quality_score": round(self.quality_score, 3),
            "total_model_size_gb": round(self.total_model_size_gb, 2),
            "kv_cache_size_gb": round(self.kv_cache_size_gb, 3),
            "overhead_gb": round(self.overhead_gb, 2),
            "gpu_utilization": round(self.gpu_utilization, 3),
            "notes": self.notes,
        }


# ===========================================================================
# 硬件画像
# ===========================================================================


@dataclass
class HardwareProfile:
    """硬件画像 (简化版, 用于内存优化器)

    可以从 core.hardware_detector.HardwareProfile 转换而来。
    """
    vram_total_gb: float = 0.0           # GPU 显存总量 (GB)
    vram_free_gb: float = 0.0            # GPU 可用显存 (GB)
    ram_total_gb: float = 0.0            # 系统内存总量 (GB)
    ram_free_gb: float = 0.0             # 系统可用内存 (GB)
    gpu_name: str = ""                   # GPU 名称
    cpu_cores: int = 4                   # CPU 物理核心数
    has_gpu: bool = False                # 是否有独立 GPU
    is_unified_memory: bool = False      # 是否统一内存 (Apple Silicon)
    disk_speed_mbps: float = 0.0         # 磁盘顺序读速度 (MB/s)
    gpu_vendor: str = ""                 # GPU 厂商: nvidia/amd/apple/intel

    @property
    def model_memory_budget_gb(self) -> float:
        """可用于模型的内存预算 (GB)"""
        if self.has_gpu and not self.is_unified_memory:
            return self.vram_free_gb * 0.85
        elif self.is_unified_memory:
            return (self.vram_free_gb + self.ram_free_gb) * 0.5
        else:
            return self.ram_free_gb * 0.6

    @classmethod
    def from_detector(cls, detector_profile: Any) -> "HardwareProfile":
        """从 core.hardware_detector.HardwareProfile 转换"""
        return cls(
            vram_total_gb=detector_profile.gpu.vram_total_gb,
            vram_free_gb=detector_profile.gpu.vram_free_mb / 1024,
            ram_total_gb=detector_profile.memory.total_gb,
            ram_free_gb=detector_profile.memory.available_gb,
            gpu_name=detector_profile.gpu.name,
            cpu_cores=detector_profile.cpu.physical_cores,
            has_gpu=detector_profile.gpu.has_gpu,
            is_unified_memory=detector_profile.gpu.unified_memory,
            disk_speed_mbps=detector_profile.memory.disk_speed.sequential_read_mbps,
            gpu_vendor=detector_profile.gpu.vendor.value if detector_profile.gpu.vendor else "",
        )


# ===========================================================================
# 速度估算
# ===========================================================================

_BASELINE_GPU_TPS_7B: float = 40.0     # 7B Q4_K_M 中端 GPU 基线
_BASELINE_CPU_TPS_7B: float = 8.0      # 7B Q4_K_M 中端 CPU 基线


def _estimate_base_speed_tps(
    model_size_b: float,
    hardware: HardwareProfile,
    quant_speed_factor: float,
    gpu_layers_ratio: float = 1.0,
) -> float:
    """估算基础生成速度 (tokens/s)

    基于 offloader.py 的速度估算逻辑, 结合硬件信息。
    """
    import math

    size_penalty = math.sqrt(7.0 / max(model_size_b, 0.5))

    if hardware.has_gpu:
        vram_factor = min(hardware.vram_free_gb / 8.0, 3.0)
        gpu_tps = _BASELINE_GPU_TPS_7B * size_penalty * quant_speed_factor * vram_factor * gpu_layers_ratio

        # GPU 厂商修正
        name_lower = hardware.gpu_name.lower()
        if any(k in name_lower for k in ("rtx 4090", "rtx 4080")):
            gpu_tps *= 1.5
        elif any(k in name_lower for k in ("rtx 3090", "rtx 3080")):
            gpu_tps *= 1.2
        elif any(k in name_lower for k in ("rtx 4060", "rtx 3060")):
            gpu_tps *= 0.85

        # 混合推理惩罚
        if gpu_layers_ratio < 1.0:
            cpu_ratio = 1.0 - gpu_layers_ratio
            thread_factor = min(hardware.cpu_cores / 4.0, 2.0)
            cpu_tps = _BASELINE_CPU_TPS_7B * size_penalty * quant_speed_factor * thread_factor * cpu_ratio
            # PCIe 开销
            return min(gpu_tps, cpu_tps * 1.5) * 0.85

        return gpu_tps
    else:
        thread_factor = min(hardware.cpu_cores / 4.0, 2.0)
        return _BASELINE_CPU_TPS_7B * size_penalty * quant_speed_factor * thread_factor


# ===========================================================================
# 内存优化协调器 (核心)
# ===========================================================================


class MemoryOptimizer:
    """内存优化协调器

    整合所有优化模块, 提供一键优化配置:
    - ModelOffloader: 模型卸载 (GPU <-> CPU <-> Disk)
    - KVCacheOptimizer: KV Cache 优化 (分页/量化/压缩/前缀共享)
    - DynamicLayerLoader: 动态层加载 (LRU淘汰/预测预取)
    - QuantizationManager: 量化管理 (GGUF/GPTQ/AWQ/BnB)

    用法::

        optimizer = MemoryOptimizer()

        # 一键优化
        result = optimizer.optimize_for_model(
            model_size_b=7.0,
            profile=OptimizationProfile.BALANCED,
            hardware=hardware_profile,
        )
        print(result.to_dict())

        # 详细报告
        report = optimizer.get_optimization_report(7.0, hardware)

        # 快捷优化 (无需完整硬件画像)
        result = optimizer.quick_optimize(7.0, "balanced", vram_gb=8)
    """

    def __init__(self):
        self.offloader = ModelOffloader()
        self.kv_optimizer = KVCacheOptimizer()
        self.quant_manager = _QuantManager()
        logger.info("MemoryOptimizer 已初始化 (整合 offloader/kv_cache/quantizer/dynamic_loader)")

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def optimize_for_model(
        self,
        model_size_b: float,
        profile: OptimizationProfile,
        hardware: HardwareProfile,
        context_length: int = 4096,
        batch_size: int = 1,
    ) -> OptimizationResult:
        """为指定模型生成优化配置

        整合 quantizer / offloader / kv_cache / dynamic_loader 的决策。

        Args:
            model_size_b: 模型参数量 (十亿, e.g. 7.0 = 7B)
            profile: 优化目标模式
            hardware: 硬件画像
            context_length: 目标上下文长度
            batch_size: 批处理大小

        Returns:
            OptimizationResult 完整优化配置
        """
        logger.info(
            "开始优化: model=%sB, profile=%s, vram=%.1fGB, ram=%.1fGB",
            model_size_b, profile.value, hardware.vram_free_gb, hardware.ram_free_gb,
        )

        notes: List[str] = []
        num_layers = _estimate_model_layers(model_size_b)
        notes.append(f"模型参数量: {model_size_b}B, 估算层数: {num_layers}")

        # ------------------------------------------------------------------
        # 1. 选择量化级别 (使用 quantizer.py)
        # ------------------------------------------------------------------
        quant_level = self._select_quantization(model_size_b, hardware, profile)
        quant_profile = ALL_PROFILES.get(quant_level)
        if quant_profile is None:
            # 降级到 Q4_K_M
            quant_profile = ALL_PROFILES["Q4_K_M"]
            quant_level = "Q4_K_M"

        quant_config = QuantizationConfig(
            level=quant_level,
            vram_per_b=quant_profile.vram_per_billion_params,
            quality_score=quant_profile.quality_score,
            speed_score=quant_profile.speed_score,
            bits=quant_profile.bits,
        )
        quantized_size_gb = model_size_b * quant_profile.vram_per_billion_params
        notes.append(
            f"量化级别: {quant_level} ({quant_profile.bits}bit), "
            f"量化后权重: {quantized_size_gb:.2f}GB"
        )

        # ------------------------------------------------------------------
        # 2. 配置卸载策略 (使用 offloader.py)
        # ------------------------------------------------------------------
        offload_config = self.offloader.recommend_offload_strategy(
            model_size_b=model_size_b,
            gpu_vram_gb=hardware.vram_free_gb,
            cpu_ram_gb=hardware.ram_free_gb,
            disk_available_gb=100.0,
            quantization=quant_level,
        )
        # 覆盖 context_length 和 batch_size
        offload_config.context_length = context_length
        offload_config.batch_size = batch_size

        # 获取内存预估
        mem_estimate = self.offloader.estimate_memory_usage(
            model_size_b, quant_level, offload_config,
        )
        # 获取性能预估
        perf_estimate = self.offloader.estimate_performance(
            model_size_b, quant_level, offload_config,
            hardware.vram_free_gb, hardware.ram_free_gb,
        )
        notes.append(
            f"卸载策略: {offload_config.strategy.value}, "
            f"GPU层: {offload_config.gpu_layers if offload_config.gpu_layers >= 0 else '全部'}, "
            f"CPU线程: {offload_config.cpu_threads}"
        )

        # ------------------------------------------------------------------
        # 3. 配置 KV Cache (使用 kv_cache.py)
        # ------------------------------------------------------------------
        kv_config = self.kv_optimizer.recommend_config(
            available_memory_gb=hardware.vram_free_gb + hardware.ram_free_gb * 0.3,
            typical_seq_length=context_length,
            num_layers=num_layers,
            batch_size=batch_size,
        )
        # 根据 profile 调整 KV Cache 配置
        kv_config = self._adjust_kv_config(kv_config, profile, hardware)

        kv_size_gb = self.kv_optimizer.estimate_cache_size(
            seq_length=context_length,
            num_layers=num_layers,
            num_heads=32,   # 典型值
            head_dim=128,   # 典型值
            batch_size=batch_size,
        )
        notes.append(
            f"KV Cache: {kv_config.cache_bits}bit, "
            f"预估大小: {kv_size_gb:.3f}GB, "
            f"前缀共享: {'启用' if kv_config.prefix_sharing else '禁用'}"
        )

        # ------------------------------------------------------------------
        # 4. 配置动态层加载 (基于 dynamic_loader.py 的概念)
        # ------------------------------------------------------------------
        dynamic_config = self._configure_dynamic_loading(
            model_size_b, num_layers, hardware, profile, offload_config,
        )
        if dynamic_config.max_gpu_layers > 0:
            notes.append(
                f"动态层加载: 启用, GPU层上限={dynamic_config.max_gpu_layers}, "
                f"CPU层上限={dynamic_config.max_cpu_layers}, "
                f"预取={'启用' if dynamic_config.prefetch_enabled else '禁用'}"
            )

        # ------------------------------------------------------------------
        # 5. 计算最终预估指标
        # ------------------------------------------------------------------
        # 动态加载的速度惩罚
        dyn_speed_factor = 1.0
        if dynamic_config.max_gpu_layers > 0 and dynamic_config.max_gpu_layers < num_layers:
            # 动态加载有额外开销, 但可以减少显存占用
            dyn_speed_factor = 0.85 if dynamic_config.prefetch_enabled else 0.70

        # GPU 层比例
        if offload_config.gpu_layers == -1:
            gpu_layers_ratio = 1.0
        elif offload_config.gpu_layers == 0:
            gpu_layers_ratio = 0.0
        else:
            gpu_layers_ratio = offload_config.gpu_layers / max(num_layers, 1)

        # 速度估算
        speed_factor = _get_speed_factor(quant_level)
        final_speed = _estimate_base_speed_tps(
            model_size_b, hardware, speed_factor, gpu_layers_ratio,
        ) * dyn_speed_factor

        # 显存/内存
        estimated_vram = mem_estimate.gpu_vram_mb / 1024 + kv_size_gb * gpu_layers_ratio
        estimated_ram = mem_estimate.cpu_ram_mb / 1024 + kv_size_gb * (1 - gpu_layers_ratio)
        overhead = 0.3 + 0.5  # GPU CUDA context + CPU 工作内存

        # GPU 利用率
        if hardware.vram_free_gb > 0:
            gpu_util = min(estimated_vram / hardware.vram_free_gb, 1.0)
        else:
            gpu_util = 0.0

        notes.append(
            f"预估速度: {final_speed:.1f} tokens/s, "
            f"质量评分: {quant_profile.quality_score:.2f}"
        )

        if estimated_vram > hardware.vram_free_gb:
            notes.append(
                f"警告: 预估显存 ({estimated_vram:.1f}GB) 超过可用显存 ({hardware.vram_free_gb:.1f}GB)"
            )

        result = OptimizationResult(
            quantization=quant_config,
            offload_config=offload_config,
            kv_cache_config=kv_config,
            dynamic_config=dynamic_config,
            profile=profile,
            estimated_vram_gb=round(estimated_vram, 2),
            estimated_ram_gb=round(estimated_ram, 2),
            estimated_speed_tps=round(final_speed, 1),
            quality_score=quant_profile.quality_score,
            total_model_size_gb=round(quantized_size_gb, 2),
            kv_cache_size_gb=round(kv_size_gb, 3),
            overhead_gb=round(overhead, 2),
            gpu_utilization=round(gpu_util, 3),
            notes=notes,
        )

        logger.info(
            "优化完成: quant=%s, vram=%.1fGB, ram=%.1fGB, speed=%.1f t/s, quality=%.2f",
            quant_level, estimated_vram, estimated_ram, final_speed, quant_profile.quality_score,
        )

        return result

    def get_optimization_report(
        self,
        model_size_b: float,
        hardware: HardwareProfile,
        context_length: int = 4096,
    ) -> Dict[str, Any]:
        """生成详细优化报告

        对所有优化模式生成配置, 并比较效果。
        同时包含 offloader.py 的策略对比和 quantizer.py 的量化对比。

        Args:
            model_size_b: 模型参数量 (十亿)
            hardware: 硬件画像
            context_length: 目标上下文长度

        Returns:
            包含所有模式优化结果的报告字典
        """
        report: Dict[str, Any] = {
            "model_size_b": model_size_b,
            "hardware": {
                "gpu": hardware.gpu_name or ("无GPU" if not hardware.has_gpu else "未知"),
                "vram_total_gb": round(hardware.vram_total_gb, 1),
                "vram_free_gb": round(hardware.vram_free_gb, 1),
                "ram_total_gb": round(hardware.ram_total_gb, 1),
                "ram_free_gb": round(hardware.ram_free_gb, 1),
                "cpu_cores": hardware.cpu_cores,
            },
            "model_layers": _estimate_model_layers(model_size_b),
            "profiles": {},
            "comparison": {},
            "offload_strategies": {},
            "quantization_comparison": {},
            "recommendation": "",
        }

        # 对每种优化模式生成配置
        results: Dict[str, OptimizationResult] = {}
        for profile in OptimizationProfile:
            result = self.optimize_for_model(
                model_size_b=model_size_b,
                profile=profile,
                hardware=hardware,
                context_length=context_length,
            )
            results[profile.value] = result
            report["profiles"][profile.value] = result.to_dict()

        # 生成对比表
        comparison = {
            "vram_gb": {},
            "ram_gb": {},
            "speed_tps": {},
            "quality_score": {},
            "quantization": {},
        }
        for name, r in results.items():
            comparison["vram_gb"][name] = r.estimated_vram_gb
            comparison["ram_gb"][name] = r.estimated_ram_gb
            comparison["speed_tps"][name] = r.estimated_speed_tps
            comparison["quality_score"][name] = r.quality_score
            comparison["quantization"][name] = r.quantization.level
        report["comparison"] = comparison

        # offloader.py 的策略对比
        offload_reports = self.offloader.compare_strategies(
            model_size_b=model_size_b,
            quantization="q4_k_m",
            gpu_vram_gb=hardware.vram_free_gb,
            cpu_ram_gb=hardware.ram_free_gb,
        )
        report["offload_strategies"] = [
            {
                "name": r.strategy_name,
                "gpu_layers": r.gpu_layers_actual,
                "cpu_layers": r.cpu_layers_actual,
                "disk_layers": r.disk_layers_actual,
                "vram_mb": round(r.memory.gpu_vram_mb, 0),
                "ram_mb": round(r.memory.cpu_ram_mb, 0),
                "speed_tps": r.performance.estimated_tps,
                "bottleneck": r.performance.bottleneck,
            }
            for r in offload_reports
        ]

        # quantizer.py 的量化对比
        quant_table = self.quant_manager.get_comparison_table()
        report["quantization_comparison"] = [
            {
                "name": q["name"],
                "bits": q["bits"],
                "quality": q["quality_score"],
                "speed": q["speed_score"],
                "vram_7b": q["vram_7b"],
                "vram_13b": q["vram_13b"],
            }
            for q in quant_table[:12]  # 只取前 12 个
        ]

        # 推荐建议
        vram_budget = hardware.vram_free_gb
        quantized_7b = model_size_b * 0.56  # Q4_K_M 大约 0.56 GB/B
        if quantized_7b <= vram_budget * 0.5:
            report["recommendation"] = (
                f"模型较小, 显存充足。建议使用 MAX_SPEED 或 QUALITY 模式以获得最佳体验。"
            )
        elif quantized_7b <= vram_budget:
            report["recommendation"] = (
                f"模型中等大小, 显存基本满足。建议使用 BALANCED 模式, "
                f"或 MAX_SPEED 模式配合量化。"
            )
        elif quantized_7b <= vram_budget * 2:
            report["recommendation"] = (
                f"模型较大, 显存不足。建议使用 BALANCED 模式 + 量化 + 部分层 CPU 卸载。"
            )
        else:
            report["recommendation"] = (
                f"模型很大, 显存严重不足。建议使用 MINIMAL_VRAM 模式 + 低量化 + GPU-CPU 混合推理, "
                f"或考虑使用更小的模型。"
            )

        return report

    def estimate_performance(self, config: OptimizationResult) -> Dict[str, float]:
        """预估给定配置的性能指标

        Args:
            config: 优化结果配置

        Returns:
            性能指标字典
        """
        return {
            "estimated_tokens_per_second": config.estimated_speed_tps,
            "estimated_vram_gb": config.estimated_vram_gb,
            "estimated_ram_gb": config.estimated_ram_gb,
            "quality_score": config.quality_score,
            "gpu_utilization": config.gpu_utilization,
        }

    def estimate_memory(self, config: OptimizationResult) -> Dict[str, float]:
        """预估给定配置的内存使用

        Args:
            config: 优化结果配置

        Returns:
            内存使用字典
        """
        return {
            "model_weights_gb": config.total_model_size_gb,
            "kv_cache_gb": config.kv_cache_size_gb,
            "overhead_gb": config.overhead_gb,
            "total_vram_gb": config.estimated_vram_gb,
            "total_ram_gb": config.estimated_ram_gb,
        }

    # ------------------------------------------------------------------
    # 组合优化分析
    # ------------------------------------------------------------------

    def analyze_combinations(
        self,
        model_size_b: float,
        hardware: HardwareProfile,
        context_length: int = 4096,
    ) -> List[Dict[str, Any]]:
        """分析不同优化组合的效果

        计算以下组合:
        1. 仅量化 (基线)
        2. 量化 + GPU-CPU 卸载
        3. 量化 + 动态层加载
        4. 量化 + KV Cache 压缩
        5. 全部组合

        Returns:
            组合效果列表, 按综合评分排序
        """
        num_layers = _estimate_model_layers(model_size_b)
        combinations: List[Dict[str, Any]] = []

        # 测试多种量化级别 (大小写不敏感查找)
        test_quants_raw = ["Q4_K_M", "Q2_K", "Q8_0"]
        # 构建大小写不敏感索引
        _profile_index = {k.upper(): k for k in ALL_PROFILES.keys()}
        test_quants = []
        for tq in test_quants_raw:
            actual_key = _profile_index.get(tq.upper())
            if actual_key:
                test_quants.append(actual_key)

        for quant_level in test_quants:
            profile = ALL_PROFILES.get(quant_level)
            if profile is None:
                continue

            quant_size_gb = model_size_b * profile.vram_per_billion_params
            speed_factor = _get_speed_factor(quant_level)

            # 基础速度 (全 GPU)
            base_speed = _estimate_base_speed_tps(model_size_b, hardware, speed_factor, 1.0)

            # 组合 1: 仅量化
            combinations.append({
                "name": f"仅量化 ({quant_level})",
                "quantization": quant_level,
                "offload": "gpu_only",
                "kv_strategy": "standard",
                "dynamic": False,
                "vram_gb": round(quant_size_gb + 0.3, 2),
                "ram_gb": round(0.5, 2),
                "speed_tps": round(base_speed, 1),
                "quality": profile.quality_score,
            })

            # 组合 2: 量化 + GPU-CPU 卸载
            offload_cfg = self.offloader.recommend_offload_strategy(
                model_size_b, hardware.vram_free_gb, hardware.ram_free_gb,
                quantization=quant_level,
            )
            mem_est = self.offloader.estimate_memory_usage(
                model_size_b, quant_level, offload_cfg,
            )
            perf_est = self.offloader.estimate_performance(
                model_size_b, quant_level, offload_cfg,
                hardware.vram_free_gb, hardware.ram_free_gb,
            )
            combinations.append({
                "name": f"量化 ({quant_level}) + GPU-CPU 卸载",
                "quantization": quant_level,
                "offload": offload_cfg.strategy.value,
                "kv_strategy": "standard",
                "dynamic": False,
                "vram_gb": round(mem_est.gpu_vram_mb / 1024, 2),
                "ram_gb": round(mem_est.cpu_ram_mb / 1024, 2),
                "speed_tps": round(perf_est.estimated_tps, 1),
                "quality": profile.quality_score,
            })

            # 组合 3: 量化 + 动态层加载 (假设 60% 层在 GPU)
            gpu_ratio = 0.6
            dyn_speed = _estimate_base_speed_tps(
                model_size_b, hardware, speed_factor, gpu_ratio,
            ) * 0.85  # 动态加载惩罚
            dyn_vram = quant_size_gb * gpu_ratio + 0.3
            dyn_ram = quant_size_gb * (1 - gpu_ratio) + 0.5
            combinations.append({
                "name": f"量化 ({quant_level}) + 动态层加载",
                "quantization": quant_level,
                "offload": "dynamic",
                "kv_strategy": "standard",
                "dynamic": True,
                "vram_gb": round(dyn_vram, 2),
                "ram_gb": round(dyn_ram, 2),
                "speed_tps": round(dyn_speed, 1),
                "quality": profile.quality_score,
            })

            # 组合 4: 量化 + KV Cache 压缩 (INT4 KV)
            kv_size_int4 = self.kv_optimizer.estimate_cache_size(
                seq_length=context_length,
                num_layers=num_layers,
                num_heads=32,
                head_dim=128,
                batch_size=1,
            )
            # INT4 KV 是 FP16 的 1/4
            kv_size_int4 *= 0.25
            kv_vram = quant_size_gb + kv_size_int4 + 0.3
            kv_speed = base_speed * 0.95
            combinations.append({
                "name": f"量化 ({quant_level}) + KV Cache INT4",
                "quantization": quant_level,
                "offload": "gpu_only",
                "kv_strategy": "quantized_int4",
                "dynamic": False,
                "vram_gb": round(kv_vram, 2),
                "ram_gb": round(0.5, 2),
                "speed_tps": round(kv_speed, 1),
                "quality": profile.quality_score * 0.98,
            })

            # 组合 5: 全部组合
            all_speed = dyn_speed * 0.95
            all_vram = dyn_vram * 0.7 + kv_size_int4 + 0.3
            all_ram = dyn_ram + quant_size_gb * 0.3 + 0.5
            combinations.append({
                "name": f"全部组合 ({quant_level})",
                "quantization": quant_level,
                "offload": "gpu_cpu_dynamic",
                "kv_strategy": "quantized_int4",
                "dynamic": True,
                "vram_gb": round(all_vram, 2),
                "ram_gb": round(all_ram, 2),
                "speed_tps": round(all_speed, 1),
                "quality": profile.quality_score * 0.98,
            })

        # 计算综合评分并排序
        for combo in combinations:
            speed_norm = min(combo["speed_tps"] / 100.0, 1.0)
            vram_efficiency = max(0, 1.0 - combo["vram_gb"] / max(hardware.vram_free_gb, 1.0))
            combo["composite_score"] = round(
                combo["quality"] * 0.3 + speed_norm * 0.3 + vram_efficiency * 0.4, 3
            )

        combinations.sort(key=lambda c: c["composite_score"], reverse=True)
        return combinations

    # ------------------------------------------------------------------
    # 快捷方法
    # ------------------------------------------------------------------

    def quick_optimize(
        self,
        model_size_b: float,
        target: str = "balanced",
        vram_gb: float = 8.0,
        ram_gb: float = 16.0,
        cpu_cores: int = 4,
    ) -> OptimizationResult:
        """快捷优化 (无需完整硬件画像)

        Args:
            model_size_b: 模型参数量 (十亿)
            target: 目标模式 ("minimal_vram", "balanced", "max_speed", "quality")
            vram_gb: 可用显存 (GB)
            ram_gb: 可用内存 (GB)
            cpu_cores: CPU 核心数

        Returns:
            OptimizationResult
        """
        profile = OptimizationProfile(target)
        hardware = HardwareProfile(
            vram_total_gb=vram_gb,
            vram_free_gb=vram_gb * 0.9,
            ram_total_gb=ram_gb,
            ram_free_gb=ram_gb * 0.7,
            cpu_cores=cpu_cores,
            has_gpu=vram_gb > 0,
        )
        return self.optimize_for_model(model_size_b, profile, hardware)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _select_quantization(
        self,
        model_size_b: float,
        hardware: HardwareProfile,
        profile: OptimizationProfile,
    ) -> str:
        """选择量化级别

        使用 quantizer.py 的推荐逻辑, 根据 profile 调整策略。
        MINIMAL_VRAM 直接选择最小可用量化, 其他模式使用智能推荐。
        """
        if profile == OptimizationProfile.MINIMAL_VRAM:
            # 最小显存模式: 找能放进显存的最小量化
            min_quant = self.quant_manager.get_min_vram_quant(
                model_size_b, hardware.vram_free_gb,
            )
            if min_quant:
                return min_quant
            # 连最小的都放不下, 返回最小的 GGUF 量化
            return "Q2_K"

        task_map = {
            OptimizationProfile.QUALITY: "code",        # 高质量要求
            OptimizationProfile.MAX_SPEED: "chat",      # 速度优先, 质量要求低
            OptimizationProfile.BALANCED: "general",    # 平衡
        }
        task_type = task_map.get(profile, "general")

        try:
            # 对 MAX_SPEED 和 QUALITY 做自定义选择, 而非完全依赖 quantizer
            if profile == OptimizationProfile.MAX_SPEED:
                return self._select_fastest_quant(model_size_b, hardware)
            elif profile == OptimizationProfile.QUALITY:
                return self._select_highest_quality_quant(model_size_b, hardware)

            return self.quant_manager.recommend_quantization(
                model_size_b=model_size_b,
                available_vram_gb=hardware.vram_free_gb,
                task_type=task_type,
            )
        except ValueError:
            # 显存不足, 降级到最小量化
            logger.warning("显存不足以运行任何量化级别, 降级到 Q2_K")
            return "Q2_K"

    def _select_fastest_quant(
        self, model_size_b: float, hardware: HardwareProfile,
    ) -> str:
        """选择速度最快的量化级别 (在显存约束内)"""
        best = None
        best_speed = 0.0
        effective_vram = hardware.vram_free_gb * 0.90

        for name, profile in ALL_PROFILES.items():
            vram = self.quant_manager.estimate_vram(model_size_b, name)
            if vram > effective_vram:
                continue
            if profile.speed_score > best_speed:
                best_speed = profile.speed_score
                best = name

        return best or "Q2_K"

    def _select_highest_quality_quant(
        self, model_size_b: float, hardware: HardwareProfile,
    ) -> str:
        """选择质量最高的量化级别 (在显存约束内)"""
        best = None
        best_quality = 0.0
        effective_vram = hardware.vram_free_gb * 0.90

        for name, profile in ALL_PROFILES.items():
            vram = self.quant_manager.estimate_vram(model_size_b, name)
            if vram > effective_vram:
                continue
            if profile.quality_score > best_quality:
                best_quality = profile.quality_score
                best = name

        return best or "Q8_0"

    def _adjust_kv_config(
        self,
        kv_config: KVCacheConfig,
        profile: OptimizationProfile,
        hardware: HardwareProfile,
    ) -> KVCacheConfig:
        """根据优化目标调整 KV Cache 配置"""
        if profile == OptimizationProfile.QUALITY:
            kv_config.cache_bits = 16
            kv_config.compression_ratio = 0.8
        elif profile == OptimizationProfile.MAX_SPEED:
            # 速度模式: 使用较高精度但启用前缀共享
            kv_config.cache_bits = min(kv_config.cache_bits, 16)
            kv_config.prefix_sharing = True
        elif profile == OptimizationProfile.MINIMAL_VRAM:
            # 显存模式: 使用最低精度
            kv_config.cache_bits = 4
            kv_config.compression_ratio = 0.3
            kv_config.prefix_sharing = True
        # BALANCED 保持 recommend_config 的默认值

        return kv_config

    def _configure_dynamic_loading(
        self,
        model_size_b: float,
        num_layers: int,
        hardware: HardwareProfile,
        profile: OptimizationProfile,
        offload_config: OffloadConfig,
    ) -> DynamicConfig:
        """配置动态层加载

        当 GPU 显存不足以装下全部模型层时启用。
        """
        # 只在需要时启用动态加载
        if offload_config.strategy == OffloadStrategy.GPU_ONLY:
            return DynamicConfig()

        if profile == OptimizationProfile.MAX_SPEED:
            # 速度模式不使用动态加载
            return DynamicConfig()

        # 计算 GPU 可驻留的层数
        bytes_per_param = _get_bytes_per_param(
            offload_config.strategy.value  # 这里不准确, 用默认
        )
        # 简化: 用 offloader 计算的 GPU 层数
        gpu_layers = offload_config.gpu_layers
        if gpu_layers == -1:
            gpu_layers = num_layers
        elif gpu_layers == 0:
            return DynamicConfig()

        # 动态加载: GPU 层数比实际少一些, 留给动态调度
        max_gpu = max(4, int(gpu_layers * 0.8))
        max_cpu = num_layers - max_gpu

        return DynamicConfig(
            max_gpu_layers=max_gpu,
            max_cpu_layers=max_cpu,
            prefetch_enabled=profile != OptimizationProfile.MINIMAL_VRAM,
            prefetch_count=2 if profile != OptimizationProfile.MINIMAL_VRAM else 0,
            swap_enabled=offload_config.disk_offload or profile == OptimizationProfile.MINIMAL_VRAM,
            eviction_strategy="lru",
        )
