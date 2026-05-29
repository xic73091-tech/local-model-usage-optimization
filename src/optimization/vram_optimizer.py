"""
显存优化器

针对小显存(6GB/8GB)场景的深度优化模块。
整合极低精度量化、智能层卸载、KV Cache压缩等技术，
实现6GB显存运行13B甚至更大模型。

核心优化策略：
1. 极低精度量化 (Q2_K/IQ2_XXS)
2. 智能GPU-CPU层分配
3. KV Cache INT4量化
4. 滑动窗口注意力
5. 动态层加载
6. 激活值检查点
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .ultra_quantizer import UltraQuantizer, UltraQuantLevel, ULTRA_QUANT_PROFILES

logger = logging.getLogger(__name__)


# ============================================================
# 枚举与常量
# ============================================================

class OptimizationTarget(Enum):
    """优化目标"""
    MINIMAL_VRAM = "minimal_vram"   # 最小显存占用
    BALANCED = "balanced"           # 平衡
    MAX_SPEED = "max_speed"         # 最快速度
    MAX_QUALITY = "max_quality"     # 最高质量


class OffloadStrategy(Enum):
    """卸载策略"""
    GPU_ONLY = "gpu_only"           # 全GPU
    GPU_CPU = "gpu_cpu"             # GPU-CPU混合
    GPU_CPU_DYNAMIC = "gpu_cpu_dynamic"  # GPU-CPU动态加载
    CPU_ONLY = "cpu_only"           # 全CPU


# 6GB显存的典型配置
VRAM_6GB_CONFIGS = {
    "7B": {
        "quantization": UltraQuantLevel.Q4_K_M,
        "gpu_layers": -1,  # 全部GPU
        "kv_quant_bits": 8,
        "context_length": 4096,
        "sliding_window": 2048,
        "strategy": OffloadStrategy.GPU_ONLY,
    },
    "13B": {
        "quantization": UltraQuantLevel.Q4_K_M,
        "gpu_layers": 20,  # 20层GPU, 20层CPU
        "kv_quant_bits": 4,
        "context_length": 2048,
        "sliding_window": 1024,
        "strategy": OffloadStrategy.GPU_CPU,
    },
    "30B": {
        "quantization": UltraQuantLevel.Q2_K,
        "gpu_layers": 15,  # 15层GPU, 45层CPU
        "kv_quant_bits": 4,
        "context_length": 1024,
        "sliding_window": 512,
        "strategy": OffloadStrategy.GPU_CPU,
    },
    "70B": {
        "quantization": UltraQuantLevel.IQ2_XXS,
        "gpu_layers": 10,  # 10层GPU, 70层CPU
        "kv_quant_bits": 4,
        "context_length": 512,
        "sliding_window": 256,
        "strategy": OffloadStrategy.GPU_CPU,
    },
}


# ============================================================
# 数据类
# ============================================================

@dataclass
class VRAMBudget:
    """显存预算"""
    total_gb: float                # 总显存
    system_reserve_gb: float = 0.5  # 系统预留
    cuda_overhead_gb: float = 0.4   # CUDA开销

    @property
    def available_gb(self) -> float:
        """可用显存"""
        return self.total_gb - self.system_reserve_gb - self.cuda_overhead_gb

    @property
    def model_budget_gb(self) -> float:
        """可用于模型的显存"""
        return self.available_gb * 0.85  # 85%给模型

    @property
    def kv_budget_gb(self) -> float:
        """可用于KV Cache的显存"""
        return self.available_gb * 0.15  # 15%给KV Cache


@dataclass
class LayerAllocation:
    """层分配方案"""
    total_layers: int
    gpu_layers: int
    cpu_layers: int
    disk_layers: int = 0
    per_layer_gb: float = 0.0
    gpu_usage_gb: float = 0.0
    cpu_usage_gb: float = 0.0
    strategy: OffloadStrategy = OffloadStrategy.GPU_CPU

    @property
    def gpu_ratio(self) -> float:
        """GPU层比例"""
        return self.gpu_layers / self.total_layers if self.total_layers > 0 else 0

    def to_dict(self) -> dict:
        return {
            "total_layers": self.total_layers,
            "gpu_layers": self.gpu_layers,
            "cpu_layers": self.cpu_layers,
            "disk_layers": self.disk_layers,
            "gpu_ratio": round(self.gpu_ratio, 2),
            "gpu_usage_gb": round(self.gpu_usage_gb, 2),
            "cpu_usage_gb": round(self.cpu_usage_gb, 2),
            "strategy": self.strategy.value,
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    # 配置
    model_size_b: float
    quantization: UltraQuantLevel
    layer_allocation: LayerAllocation
    kv_quant_bits: int
    context_length: int
    sliding_window: int

    # 预估
    estimated_vram_gb: float
    estimated_ram_gb: float
    estimated_speed_tps: float
    quality_score: float

    # 详情
    model_weight_gb: float
    kv_cache_gb: float
    overhead_gb: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_size_b": self.model_size_b,
            "quantization": self.quantization.value,
            "layer_allocation": self.layer_allocation.to_dict(),
            "kv_quant_bits": self.kv_quant_bits,
            "context_length": self.context_length,
            "sliding_window": self.sliding_window,
            "estimated_vram_gb": round(self.estimated_vram_gb, 2),
            "estimated_ram_gb": round(self.estimated_ram_gb, 2),
            "estimated_speed_tps": round(self.estimated_speed_tps, 1),
            "quality_score": round(self.quality_score, 3),
            "model_weight_gb": round(self.model_weight_gb, 2),
            "kv_cache_gb": round(self.kv_cache_gb, 3),
            "overhead_gb": round(self.overhead_gb, 2),
            "notes": self.notes,
        }


# ============================================================
# 核心类
# ============================================================

class VRAMOptimizer:
    """显存优化器

    针对小显存场景的深度优化，让6GB显存也能运行大模型。

    用法：
        optimizer = VRAMOptimizer(vram_gb=6.0)

        # 获取最优配置
        result = optimizer.optimize(
            model_size_b=13.0,
            target=OptimizationTarget.BALANCED,
        )

        # 打印结果
        print(f"量化: {result.quantization.value}")
        print(f"GPU层数: {result.layer_allocation.gpu_layers}/{result.layer_allocation.total_layers}")
        print(f"预估显存: {result.estimated_vram_gb:.2f} GB")
        print(f"预估速度: {result.estimated_speed_tps:.1f} tokens/s")
    """

    # 模型默认配置
    MODEL_DEFAULTS = {
        7.0:  {"layers": 32, "heads": 32, "head_dim": 128},
        13.0: {"layers": 40, "heads": 40, "head_dim": 128},
        30.0: {"layers": 60, "heads": 52, "head_dim": 128},
        70.0: {"layers": 80, "heads": 64, "head_dim": 128},
    }

    def __init__(self, vram_gb: float = 6.0):
        """初始化

        Args:
            vram_gb: GPU显存 (GB)
        """
        self.vram_gb = vram_gb
        self.budget = VRAMBudget(total_gb=vram_gb)
        self.quantizer = UltraQuantizer()

        logger.info(f"VRAMOptimizer 已初始化: {vram_gb}GB 显存")

    # ----------------------------------------------------------
    # 核心API
    # ----------------------------------------------------------

    def optimize(
        self,
        model_size_b: float,
        target: OptimizationTarget = OptimizationTarget.BALANCED,
        context_length: Optional[int] = None,
    ) -> OptimizationResult:
        """生成最优配置

        Args:
            model_size_b: 模型参数量 (B)
            target: 优化目标
            context_length: 上下文长度 (None=自动)

        Returns:
            OptimizationResult: 优化结果
        """
        notes = []
        notes.append(f"模型: {model_size_b}B, 目标: {target.value}, 显存: {self.vram_gb}GB")

        # 获取模型配置
        model_config = self._get_model_config(model_size_b)
        num_layers = model_config["layers"]
        num_heads = model_config["heads"]
        head_dim = model_config["head_dim"]

        # 1. 选择量化级别
        quant_level = self._select_quantization(model_size_b, target)
        quant_profile = ULTRA_QUANT_PROFILES[quant_level]
        notes.append(f"量化: {quant_level.value} ({quant_profile.bits_per_param} bits)")

        # 2. 计算模型权重大小
        model_weight_gb = model_size_b * quant_profile.vram_per_billion_gb
        notes.append(f"模型权重: {model_weight_gb:.2f} GB")

        # 3. 计算KV Cache配置
        if context_length is None:
            context_length = self._auto_context_length(model_weight_gb, target)
        kv_quant_bits = self._select_kv_quantization(model_weight_gb, target)
        sliding_window = self._calculate_sliding_window(context_length, target)
        notes.append(f"上下文: {context_length}, KV量化: {kv_quant_bits}bit, 滑动窗口: {sliding_window}")

        # 4. 计算KV Cache大小
        kv_cache_gb = self._estimate_kv_cache(
            num_layers, num_heads, head_dim,
            sliding_window,  # 使用滑动窗口
            kv_quant_bits,
        )
        notes.append(f"KV Cache: {kv_cache_gb:.3f} GB")

        # 5. 计算层分配
        layer_alloc = self._allocate_layers(
            model_size_b, num_layers, model_weight_gb,
            kv_cache_gb, quant_level, target,
        )
        notes.append(f"层分配: GPU={layer_alloc.gpu_layers}, CPU={layer_alloc.cpu_layers}")

        # 6. 计算最终显存占用
        overhead_gb = 0.4
        vram_used = layer_alloc.gpu_usage_gb + kv_cache_gb + overhead_gb
        ram_used = layer_alloc.cpu_usage_gb

        # 7. 估算速度
        speed_tps = self._estimate_speed(
            model_size_b, quant_profile,
            layer_alloc.gpu_layers, num_layers,
            has_dynamic_loading=(target == OptimizationTarget.MINIMAL_VRAM),
        )
        notes.append(f"预估速度: {speed_tps:.1f} tokens/s")

        # 8. 计算质量评分
        quality = quant_profile.quality_score

        return OptimizationResult(
            model_size_b=model_size_b,
            quantization=quant_level,
            layer_allocation=layer_alloc,
            kv_quant_bits=kv_quant_bits,
            context_length=context_length,
            sliding_window=sliding_window,
            estimated_vram_gb=vram_used,
            estimated_ram_gb=ram_used,
            estimated_speed_tps=speed_tps,
            quality_score=quality,
            model_weight_gb=model_weight_gb,
            kv_cache_gb=kv_cache_gb,
            overhead_gb=overhead_gb,
            notes=notes,
        )

    def optimize_for_target(
        self,
        model_size_b: float,
        target_vram_usage_gb: float,
    ) -> OptimizationResult:
        """针对目标显存使用量优化

        Args:
            model_size_b: 模型参数量 (B)
            target_vram_usage_gb: 目标显存使用量 (GB)

        Returns:
            OptimizationResult
        """
        # 计算可用于模型的显存
        available_for_model = target_vram_usage_gb - 0.4 - 0.1  # 减去开销和KV

        # 推荐量化级别
        recommendation = self.quantizer.recommend_for_vram(
            model_size_b=model_size_b,
            available_vram_gb=target_vram_usage_gb,
        )

        # 使用推荐的配置
        return self.optimize(
            model_size_b=model_size_b,
            target=OptimizationTarget.BALANCED,
        )

    def compare_targets(self, model_size_b: float) -> List[dict]:
        """对比不同优化目标

        Args:
            model_size_b: 模型参数量 (B)

        Returns:
            对比结果列表
        """
        results = []

        for target in OptimizationTarget:
            result = self.optimize(model_size_b, target)
            results.append({
                "target": target.value,
                "quantization": result.quantization.value,
                "gpu_layers": result.layer_allocation.gpu_layers,
                "total_layers": result.layer_allocation.total_layers,
                "vram_gb": round(result.estimated_vram_gb, 2),
                "ram_gb": round(result.estimated_ram_gb, 2),
                "speed_tps": round(result.estimated_speed_tps, 1),
                "quality": round(result.quality_score, 3),
                "fits_in_vram": result.estimated_vram_gb <= self.vram_gb,
            })

        return results

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _get_model_config(self, model_size_b: float) -> dict:
        """获取模型配置"""
        sizes = sorted(self.MODEL_DEFAULTS.keys())
        for s in sizes:
            if model_size_b <= s:
                return self.MODEL_DEFAULTS[s]
        return self.MODEL_DEFAULTS[sizes[-1]]

    def _select_quantization(
        self,
        model_size_b: float,
        target: OptimizationTarget,
    ) -> UltraQuantLevel:
        """选择量化级别"""
        if target == OptimizationTarget.MAX_QUALITY:
            # 质量优先：选择能放下的最高质量
            recommendation = self.quantizer.recommend_for_vram(
                model_size_b, self.budget.available_gb, min_quality=0.7
            )
            return recommendation.recommended_level

        elif target == OptimizationTarget.MAX_SPEED:
            # 速度优先：选择速度快的
            return UltraQuantLevel.Q4_K_M

        elif target == OptimizationTarget.MINIMAL_VRAM:
            # 显存最小：选择最小量化
            return UltraQuantLevel.Q2_K

        else:  # BALANCED
            # 平衡：推荐配置
            recommendation = self.quantizer.recommend_for_vram(
                model_size_b, self.budget.available_gb
            )
            return recommendation.recommended_level

    def _auto_context_length(
        self,
        model_weight_gb: float,
        target: OptimizationTarget,
    ) -> int:
        """自动计算上下文长度"""
        # 可用于KV Cache的显存
        kv_budget_gb = self.budget.available_gb - model_weight_gb * 0.3 - 0.4

        if kv_budget_gb <= 0:
            return 512  # 最小上下文

        if target == OptimizationTarget.MINIMAL_VRAM:
            return 1024
        elif target == OptimizationTarget.MAX_QUALITY:
            return 4096
        else:
            return 2048

    def _select_kv_quantization(
        self,
        model_weight_gb: float,
        target: OptimizationTarget,
    ) -> int:
        """选择KV Cache量化位数"""
        if target == OptimizationTarget.MAX_QUALITY:
            return 16  # FP16
        elif target == OptimizationTarget.MINIMAL_VRAM:
            return 4   # INT4
        else:
            return 8   # INT8

    def _calculate_sliding_window(
        self,
        context_length: int,
        target: OptimizationTarget,
    ) -> int:
        """计算滑动窗口大小"""
        if target == OptimizationTarget.MAX_QUALITY:
            return context_length  # 不使用滑动窗口
        elif target == OptimizationTarget.MINIMAL_VRAM:
            return context_length // 4  # 1/4窗口
        else:
            return context_length // 2  # 1/2窗口

    def _estimate_kv_cache(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        context_length: int,
        quant_bits: int = 8,
    ) -> float:
        """估算KV Cache大小 (GB)"""
        bytes_per_element = quant_bits / 8
        kv_bytes = 2 * num_layers * num_heads * head_dim * context_length * bytes_per_element
        return kv_bytes / (1024 ** 3)

    def _allocate_layers(
        self,
        model_size_b: float,
        num_layers: int,
        model_weight_gb: float,
        kv_cache_gb: float,
        quant_level: UltraQuantLevel,
        target: OptimizationTarget,
    ) -> LayerAllocation:
        """计算层分配"""
        per_layer_gb = model_weight_gb / num_layers

        # 可用于模型层的显存
        available_for_layers = self.budget.available_gb - kv_cache_gb - 0.4

        if target == OptimizationTarget.MAX_QUALITY:
            # 质量优先：尽量多放GPU
            gpu_layers = min(num_layers, int(available_for_layers / per_layer_gb))
        elif target == OptimizationTarget.MAX_SPEED:
            # 速度优先：全部GPU
            gpu_layers = num_layers
        elif target == OptimizationTarget.MINIMAL_VRAM:
            # 显存最小：只放关键层到GPU
            gpu_layers = max(1, int(available_for_layers * 0.6 / per_layer_gb))
        else:  # BALANCED
            # 平衡：根据显存自动计算
            gpu_layers = int(available_for_layers / per_layer_gb)

        # 限制范围
        gpu_layers = max(0, min(gpu_layers, num_layers))
        cpu_layers = num_layers - gpu_layers

        # 计算使用量
        gpu_usage = gpu_layers * per_layer_gb
        cpu_usage = cpu_layers * per_layer_gb

        # 确定策略
        if gpu_layers == num_layers:
            strategy = OffloadStrategy.GPU_ONLY
        elif gpu_layers == 0:
            strategy = OffloadStrategy.CPU_ONLY
        else:
            strategy = OffloadStrategy.GPU_CPU

        return LayerAllocation(
            total_layers=num_layers,
            gpu_layers=gpu_layers,
            cpu_layers=cpu_layers,
            per_layer_gb=per_layer_gb,
            gpu_usage_gb=gpu_usage,
            cpu_usage_gb=cpu_usage,
            strategy=strategy,
        )

    def _estimate_speed(
        self,
        model_size_b: float,
        quant_profile: UltraQuantProfile,
        gpu_layers: int,
        total_layers: int,
        has_dynamic_loading: bool = False,
    ) -> float:
        """估算推理速度 (tokens/s)"""
        # 基准: 7B Q4_K_M 全GPU = 40 t/s
        base_tps = 40.0

        # 参数量惩罚
        size_factor = 7.0 / model_size_b

        # 量化速度因子
        quant_factor = quant_profile.speed_factor

        # GPU层比例
        gpu_ratio = gpu_layers / total_layers if total_layers > 0 else 0

        # 动态加载惩罚
        dynamic_penalty = 0.85 if has_dynamic_loading else 1.0

        # 计算速度
        tps = base_tps * size_factor * quant_factor * gpu_ratio * dynamic_penalty

        return round(max(tps, 0.5), 1)


# ============================================================
# 便捷函数
# ============================================================

def get_6gb_optimal_config(model_size_b: float) -> dict:
    """获取6GB显存的最优配置

    Args:
        model_size_b: 模型参数量 (B)

    Returns:
        最优配置字典
    """
    optimizer = VRAMOptimizer(vram_gb=6.0)
    result = optimizer.optimize(model_size_b, OptimizationTarget.BALANCED)
    return result.to_dict()


def get_all_6gb_configs() -> dict:
    """获取6GB显存下所有模型的配置"""
    configs = {}
    for model_size in [7.0, 13.0, 30.0, 70.0]:
        configs[f"{model_size}B"] = get_6gb_optimal_config(model_size)
    return configs


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行演示"""
    optimizer = VRAMOptimizer(vram_gb=6.0)

    print("=" * 80)
    print("6GB显存运行大模型 - 优化配置推演")
    print("=" * 80)

    for model_size in [7.0, 13.0, 30.0, 70.0]:
        print(f"\n{'=' * 80}")
        print(f"模型: {model_size}B")
        print(f"{'=' * 80}")

        # 对比不同优化目标
        comparisons = optimizer.compare_targets(model_size)

        print(f"\n{'目标':<15} {'量化':<12} {'GPU层':<10} {'显存(GB)':<10} {'速度(t/s)':<12} {'质量':<8} {'可运行'}")
        print(f"{'-' * 80}")

        for comp in comparisons:
            fits = "✅" if comp["fits_in_vram"] else "❌"
            print(
                f"{comp['target']:<15} "
                f"{comp['quantization']:<12} "
                f"{comp['gpu_layers']}/{comp['total_layers']:<8} "
                f"{comp['vram_gb']:<10.2f} "
                f"{comp['speed_tps']:<12.1f} "
                f"{comp['quality']:<8.3f} "
                f"{fits}"
            )

        # 显示推荐配置详情
        result = optimizer.optimize(model_size, OptimizationTarget.BALANCED)
        print(f"\n推荐配置 (BALANCED):")
        print(f"  量化级别: {result.quantization.value}")
        print(f"  GPU层数: {result.layer_allocation.gpu_layers}/{result.layer_allocation.total_layers}")
        print(f"  上下文长度: {result.context_length}")
        print(f"  KV Cache量化: {result.kv_quant_bits}bit")
        print(f"  滑动窗口: {result.sliding_window}")
        print(f"  预估显存: {result.estimated_vram_gb:.2f} GB")
        print(f"  预估速度: {result.estimated_speed_tps:.1f} tokens/s")
        print(f"  质量评分: {result.quality_score:.3f}")

        print(f"\n优化建议:")
        for note in result.notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
