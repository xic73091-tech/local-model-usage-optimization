"""
极低精度量化模块

支持 Q2_K, IQ2_XXS, IQ3_XXS 等极低精度量化格式，
让6GB显存也能运行13B甚至更大的模型。

核心技术：
- Q2_K: 2-bit量化，每参数仅0.31GB
- IQ2_XXS: 极限2-bit，每参数0.26GB
- IQ3_XXS: 3-bit极限，每参数0.37GB
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 枚举与常量
# ============================================================

class UltraQuantLevel(Enum):
    """极低精度量化级别"""
    Q2_K = "Q2_K"           # 2.9 bits - 标准2-bit
    IQ2_XXS = "IQ2_XXS"     # 2.1 bits - 极限2-bit (需要imatrix)
    IQ2_XS = "IQ2_XS"       # 2.3 bits
    IQ2_S = "IQ2_S"         # 2.5 bits
    IQ3_XXS = "IQ3_XXS"     # 3.1 bits - 极限3-bit
    IQ3_S = "IQ3_S"         # 3.4 bits
    IQ4_XS = "IQ4_XS"       # 4.3 bits - 高效4-bit
    Q3_K_S = "Q3_K_S"       # 3.5 bits - 标准3-bit小
    Q3_K_M = "Q3_K_M"       # 3.9 bits - 标准3-bit中
    Q4_K_M = "Q4_K_M"       # 4.8 bits - 标准4-bit (平衡点)


@dataclass
class UltraQuantProfile:
    """极低精度量化画像"""
    level: UltraQuantLevel
    bits_per_param: float
    vram_per_billion_gb: float  # GB per billion params
    quality_score: float        # 0-1, 1=无损
    speed_factor: float         # 相对于Q4_K_M的速度 (>1更快)
    description: str
    requires_imatrix: bool      # 是否需要importance matrix
    min_llama_cpp_version: str  # 最低llama.cpp版本要求


# 量化级别注册表
ULTRA_QUANT_PROFILES: Dict[UltraQuantLevel, UltraQuantProfile] = {
    # 极低精度 (2-bit)
    UltraQuantLevel.IQ2_XXS: UltraQuantProfile(
        level=UltraQuantLevel.IQ2_XXS,
        bits_per_param=2.06,
        vram_per_billion_gb=0.26,
        quality_score=0.45,
        speed_factor=1.10,
        description="极限2-bit量化，显存最小，质量损失大",
        requires_imatrix=True,
        min_llama_cpp_version="b2916",
    ),
    UltraQuantLevel.IQ2_XS: UltraQuantProfile(
        level=UltraQuantLevel.IQ2_XS,
        bits_per_param=2.31,
        vram_per_billion_gb=0.29,
        quality_score=0.50,
        speed_factor=1.10,
        description="极限2-bit量化，略好于XXS",
        requires_imatrix=True,
        min_llama_cpp_version="b2916",
    ),
    UltraQuantLevel.IQ2_S: UltraQuantProfile(
        level=UltraQuantLevel.IQ2_S,
        bits_per_param=2.5,
        vram_per_billion_gb=0.31,
        quality_score=0.52,
        speed_factor=1.10,
        description="极限2-bit量化，质量略好",
        requires_imatrix=True,
        min_llama_cpp_version="b2916",
    ),
    UltraQuantLevel.Q2_K: UltraQuantProfile(
        level=UltraQuantLevel.Q2_K,
        bits_per_param=2.9,
        vram_per_billion_gb=0.31,
        quality_score=0.55,
        speed_factor=1.20,
        description="标准2-bit量化，不需要imatrix",
        requires_imatrix=False,
        min_llama_cpp_version="b1000",
    ),

    # 低精度 (3-bit)
    UltraQuantLevel.IQ3_XXS: UltraQuantProfile(
        level=UltraQuantLevel.IQ3_XXS,
        bits_per_param=3.06,
        vram_per_billion_gb=0.37,
        quality_score=0.60,
        speed_factor=1.15,
        description="极限3-bit量化，平衡选择",
        requires_imatrix=True,
        min_llama_cpp_version="b2916",
    ),
    UltraQuantLevel.IQ3_S: UltraQuantProfile(
        level=UltraQuantLevel.IQ3_S,
        bits_per_param=3.4,
        vram_per_billion_gb=0.41,
        quality_score=0.65,
        speed_factor=1.12,
        description="极限3-bit量化，质量更好",
        requires_imatrix=True,
        min_llama_cpp_version="b2916",
    ),
    UltraQuantLevel.Q3_K_S: UltraQuantProfile(
        level=UltraQuantLevel.Q3_K_S,
        bits_per_param=3.5,
        vram_per_billion_gb=0.38,
        quality_score=0.62,
        speed_factor=1.18,
        description="标准3-bit小尺寸",
        requires_imatrix=False,
        min_llama_cpp_version="b1000",
    ),
    UltraQuantLevel.Q3_K_M: UltraQuantProfile(
        level=UltraQuantLevel.Q3_K_M,
        bits_per_param=3.9,
        vram_per_billion_gb=0.44,
        quality_score=0.70,
        speed_factor=1.15,
        description="标准3-bit中尺寸，推荐",
        requires_imatrix=False,
        min_llama_cpp_version="b1000",
    ),

    # 中等精度 (4-bit)
    UltraQuantLevel.IQ4_XS: UltraQuantProfile(
        level=UltraQuantLevel.IQ4_XS,
        bits_per_param=4.3,
        vram_per_billion_gb=0.50,
        quality_score=0.78,
        speed_factor=1.12,
        description="高效4-bit，比Q4_K更小",
        requires_imatrix=True,
        min_llama_cpp_version="b2916",
    ),
    UltraQuantLevel.Q4_K_M: UltraQuantProfile(
        level=UltraQuantLevel.Q4_K_M,
        bits_per_param=4.8,
        vram_per_billion_gb=0.56,
        quality_score=0.85,
        speed_factor=1.00,  # 基准
        description="标准4-bit，平衡推荐",
        requires_imatrix=False,
        min_llama_cpp_version="b1000",
    ),
}


# ============================================================
# 数据类
# ============================================================

@dataclass
class VRAMEstimate:
    """显存估算结果"""
    model_weight_gb: float
    kv_cache_gb: float
    overhead_gb: float
    activation_gb: float
    total_gb: float
    available_gb: float
    fits_in_vram: bool
    remaining_gb: float

    def to_dict(self) -> dict:
        return {
            "model_weight_gb": round(self.model_weight_gb, 2),
            "kv_cache_gb": round(self.kv_cache_gb, 3),
            "overhead_gb": round(self.overhead_gb, 2),
            "activation_gb": round(self.activation_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "fits_in_vram": self.fits_in_vram,
            "remaining_gb": round(self.remaining_gb, 2),
        }


@dataclass
class QuantRecommendation:
    """量化推荐结果"""
    recommended_level: UltraQuantLevel
    profile: UltraQuantProfile
    vram_estimate: VRAMEstimate
    alternative_levels: List[UltraQuantLevel]
    notes: List[str]

    def to_dict(self) -> dict:
        return {
            "recommended_level": self.recommended_level.value,
            "profile": {
                "bits": self.profile.bits_per_param,
                "vram_per_billion_gb": self.profile.vram_per_billion_gb,
                "quality_score": self.profile.quality_score,
                "description": self.profile.description,
            },
            "vram_estimate": self.vram_estimate.to_dict(),
            "alternatives": [l.value for l in self.alternative_levels],
            "notes": self.notes,
        }


# ============================================================
# 核心类
# ============================================================

class UltraQuantizer:
    """极低精度量化管理器

    核心功能：
    1. 根据显存推荐最优量化级别
    2. 估算各量化级别的显存需求
    3. 对比不同量化方案

    用法：
        quantizer = UltraQuantizer()

        # 推荐量化级别
        recommendation = quantizer.recommend_for_vram(
            model_size_b=13.0,
            available_vram_gb=6.0,
        )

        # 估算显存
        estimate = quantizer.estimate_vram(
            model_size_b=13.0,
            level=UltraQuantLevel.Q4_K_M,
            context_length=2048,
        )
    """

    # 模型默认参数配置
    MODEL_DEFAULTS = {
        0.5:  {"layers": 24,  "heads": 8,   "head_dim": 64},
        1.0:  {"layers": 22,  "heads": 16,  "head_dim": 64},
        1.5:  {"layers": 24,  "heads": 16,  "head_dim": 96},
        2.0:  {"layers": 24,  "heads": 16,  "head_dim": 128},
        3.0:  {"layers": 32,  "heads": 32,  "head_dim": 128},
        7.0:  {"layers": 32,  "heads": 32,  "head_dim": 128},
        8.0:  {"layers": 32,  "heads": 32,  "head_dim": 128},
        13.0: {"layers": 40,  "heads": 40,  "head_dim": 128},
        14.0: {"layers": 40,  "heads": 40,  "head_dim": 128},
        30.0: {"layers": 60,  "heads": 52,  "head_dim": 128},
        34.0: {"layers": 60,  "heads": 52,  "head_dim": 128},
        65.0: {"layers": 80,  "heads": 64,  "head_dim": 128},
        70.0: {"layers": 80,  "heads": 64,  "head_dim": 128},
    }

    def __init__(self):
        """初始化"""
        self._profiles = ULTRA_QUANT_PROFILES
        logger.info("UltraQuantizer 已初始化，支持 %d 种量化级别", len(self._profiles))

    # ----------------------------------------------------------
    # 核心API
    # ----------------------------------------------------------

    def recommend_for_vram(
        self,
        model_size_b: float,
        available_vram_gb: float,
        min_quality: float = 0.0,
        context_length: int = 2048,
    ) -> QuantRecommendation:
        """根据显存推荐最优量化级别

        Args:
            model_size_b: 模型参数量 (B)
            available_vram_gb: 可用显存 (GB)
            min_quality: 最低质量要求 (0-1)
            context_length: 上下文长度

        Returns:
            QuantRecommendation: 推荐结果
        """
        notes = []
        notes.append(f"模型: {model_size_b}B, 可用显存: {available_vram_gb}GB, 上下文: {context_length}")

        # 获取模型配置
        model_config = self._get_model_config(model_size_b)

        # 计算运行时开销
        overhead_gb = 0.4  # CUDA context等
        activation_gb = model_size_b * 0.01  # 激活值

        # 计算KV Cache
        kv_cache_gb = self._estimate_kv_cache(
            model_config["layers"],
            model_config["heads"],
            model_config["head_dim"],
            context_length,
            quant_bits=8,  # KV Cache用INT8
        )

        # 可用于模型权重的显存
        available_for_model = available_vram_gb - overhead_gb - activation_gb - kv_cache_gb

        if available_for_model <= 0:
            notes.append("警告: 显存不足以运行任何配置，建议使用GPU-CPU混合卸载")
            available_for_model = available_vram_gb * 0.6  # 使用60%显存

        # 按质量从高到低遍历，找到第一个能放下的
        sorted_levels = sorted(
            self._profiles.items(),
            key=lambda x: x[1].quality_score,
            reverse=True,
        )

        recommended = None
        alternatives = []

        for level, profile in sorted_levels:
            # 检查质量要求
            if profile.quality_score < min_quality:
                continue

            # 计算显存需求
            vram_needed = model_size_b * profile.vram_per_billion_gb

            if vram_needed <= available_for_model:
                if recommended is None:
                    recommended = level
                    notes.append(f"推荐: {level.value} ({profile.bits_per_param} bits)")
                    notes.append(f"  模型权重: {vram_needed:.2f} GB")
                    notes.append(f"  质量评分: {profile.quality_score:.2f}")
                else:
                    alternatives.append(level)
            else:
                alternatives.append(level)

        # 如果没有找到合适的，推荐最小的
        if recommended is None:
            min_level = min(self._profiles.items(), key=lambda x: x[1].vram_per_billion_gb)
            recommended = min_level[0]
            notes.append(f"警告: 显存紧张，推荐最小量化: {recommended.value}")

        # 生成估算
        profile = self._profiles[recommended]
        model_weight_gb = model_size_b * profile.vram_per_billion_gb
        total_gb = model_weight_gb + kv_cache_gb + overhead_gb + activation_gb

        vram_estimate = VRAMEstimate(
            model_weight_gb=model_weight_gb,
            kv_cache_gb=kv_cache_gb,
            overhead_gb=overhead_gb,
            activation_gb=activation_gb,
            total_gb=total_gb,
            available_gb=available_vram_gb,
            fits_in_vram=total_gb <= available_vram_gb,
            remaining_gb=available_vram_gb - total_gb,
        )

        return QuantRecommendation(
            recommended_level=recommended,
            profile=profile,
            vram_estimate=vram_estimate,
            alternative_levels=alternatives[:3],  # 最多3个备选
            notes=notes,
        )

    def estimate_vram(
        self,
        model_size_b: float,
        level: UltraQuantLevel,
        context_length: int = 2048,
        batch_size: int = 1,
    ) -> VRAMEstimate:
        """估算指定量化级别的显存需求

        Args:
            model_size_b: 模型参数量 (B)
            level: 量化级别
            context_length: 上下文长度
            batch_size: 批大小

        Returns:
            VRAMEstimate: 显存估算结果
        """
        profile = self._profiles[level]
        model_config = self._get_model_config(model_size_b)

        # 模型权重
        model_weight_gb = model_size_b * profile.vram_per_billion_gb

        # KV Cache
        kv_cache_gb = self._estimate_kv_cache(
            model_config["layers"],
            model_config["heads"],
            model_config["head_dim"],
            context_length * batch_size,
            quant_bits=8,
        )

        # 运行时开销
        overhead_gb = 0.4

        # 激活值
        activation_gb = model_size_b * 0.01

        # 总计
        total_gb = model_weight_gb + kv_cache_gb + overhead_gb + activation_gb

        return VRAMEstimate(
            model_weight_gb=model_weight_gb,
            kv_cache_gb=kv_cache_gb,
            overhead_gb=overhead_gb,
            activation_gb=activation_gb,
            total_gb=total_gb,
            available_gb=0.0,  # 未知
            fits_in_vram=False,  # 未知
            remaining_gb=0.0,
        )

    def compare_levels(
        self,
        model_size_b: float,
        available_vram_gb: float,
        context_length: int = 2048,
    ) -> List[dict]:
        """对比所有量化级别

        Args:
            model_size_b: 模型参数量 (B)
            available_vram_gb: 可用显存 (GB)
            context_length: 上下文长度

        Returns:
            对比结果列表
        """
        results = []

        for level, profile in self._profiles.items():
            estimate = self.estimate_vram(model_size_b, level, context_length)

            results.append({
                "level": level.value,
                "bits": profile.bits_per_param,
                "model_weight_gb": round(estimate.model_weight_gb, 2),
                "total_gb": round(estimate.total_gb, 2),
                "fits": estimate.total_gb <= available_vram_gb,
                "quality": profile.quality_score,
                "speed": profile.speed_factor,
                "description": profile.description,
                "requires_imatrix": profile.requires_imatrix,
            })

        # 按是否能放下、然后按质量排序
        results.sort(key=lambda x: (not x["fits"], -x["quality"]))

        return results

    def get_optimal_config_for_6gb(self, model_size_b: float) -> dict:
        """获取6GB显存的最优配置

        Args:
            model_size_b: 模型参数量 (B)

        Returns:
            最优配置字典
        """
        recommendation = self.recommend_for_vram(
            model_size_b=model_size_b,
            available_vram_gb=5.5,  # 留0.5GB给系统
            min_quality=0.0,
            context_length=2048,
        )

        # 计算GPU-CPU分配
        model_config = self._get_model_config(model_size_b)
        per_layer_gb = model_size_b * recommendation.profile.vram_per_billion_gb / model_config["layers"]
        gpu_layers = int((5.5 - 0.4 - 0.2) / per_layer_gb)  # 预留开销和KV
        gpu_layers = max(0, min(gpu_layers, model_config["layers"]))

        return {
            "model_size_b": model_size_b,
            "quantization": recommendation.recommended_level.value,
            "bits": recommendation.profile.bits_per_param,
            "gpu_layers": gpu_layers,
            "total_layers": model_config["layers"],
            "cpu_layers": model_config["layers"] - gpu_layers,
            "context_length": 2048,
            "kv_cache_bits": 8,
            "estimated_vram_gb": round(recommendation.vram_estimate.total_gb, 2),
            "estimated_tps": self._estimate_speed(model_size_b, recommendation.profile, gpu_layers, model_config["layers"]),
            "quality_score": recommendation.profile.quality_score,
            "notes": recommendation.notes,
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _get_model_config(self, model_size_b: float) -> dict:
        """获取模型默认配置"""
        # 找到最接近的配置
        sizes = sorted(self.MODEL_DEFAULTS.keys())
        for i, s in enumerate(sizes):
            if model_size_b <= s:
                return self.MODEL_DEFAULTS[s]
            if i == len(sizes) - 1:
                return self.MODEL_DEFAULTS[s]

        # 默认配置
        return {"layers": 32, "heads": 32, "head_dim": 128}

    def _estimate_kv_cache(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        total_tokens: int,
        quant_bits: int = 8,
    ) -> float:
        """估算KV Cache显存 (GB)

        公式: 2(K+V) × 层数 × 头数 × 头维度 × 序列长度 × 精度
        """
        # 每个元素字节数
        bytes_per_element = quant_bits / 8

        # KV Cache字节数
        kv_bytes = 2 * num_layers * num_heads * head_dim * total_tokens * bytes_per_element

        # 转换为GB
        kv_gb = kv_bytes / (1024 ** 3)

        return kv_gb

    def _estimate_speed(
        self,
        model_size_b: float,
        profile: UltraQuantProfile,
        gpu_layers: int,
        total_layers: int,
    ) -> float:
        """估算推理速度 (tokens/s)"""
        # 基准: 7B Q4_K_M 全GPU = 40 t/s
        base_tps = 40.0

        # 参数量惩罚
        size_factor = 7.0 / model_size_b

        # 量化速度因子
        quant_factor = profile.speed_factor

        # GPU层比例
        gpu_ratio = gpu_layers / total_layers if total_layers > 0 else 0

        # 计算速度
        tps = base_tps * size_factor * quant_factor * gpu_ratio

        return round(max(tps, 0.5), 1)


# ============================================================
# 便捷函数
# ============================================================

def quick_recommend(model_size_b: float, vram_gb: float = 6.0) -> dict:
    """快速推荐量化配置

    Args:
        model_size_b: 模型参数量 (B)
        vram_gb: 可用显存 (GB)

    Returns:
        推荐配置字典
    """
    quantizer = UltraQuantizer()
    recommendation = quantizer.recommend_for_vram(model_size_b, vram_gb)
    return recommendation.to_dict()


def compare_all_quantizations(model_size_b: float, vram_gb: float = 6.0) -> List[dict]:
    """对比所有量化级别

    Args:
        model_size_b: 模型参数量 (B)
        vram_gb: 可用显存 (GB)

    Returns:
        对比结果列表
    """
    quantizer = UltraQuantizer()
    return quantizer.compare_levels(model_size_b, vram_gb)


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行演示"""
    import json

    quantizer = UltraQuantizer()

    print("=" * 70)
    print("6GB显存运行大模型 - 量化级别对比")
    print("=" * 70)

    # 测试不同模型大小
    for model_size in [7.0, 13.0, 30.0, 70.0]:
        print(f"\n{'=' * 70}")
        print(f"模型: {model_size}B")
        print(f"{'=' * 70}")

        recommendation = quantizer.recommend_for_vram(
            model_size_b=model_size,
            available_vram_gb=6.0,
            context_length=2048,
        )

        print(f"\n推荐量化: {recommendation.recommended_level.value}")
        print(f"  有效位数: {recommendation.profile.bits_per_param} bits")
        print(f"  每B参数显存: {recommendation.profile.vram_per_billion_gb} GB")
        print(f"  质量评分: {recommendation.profile.quality_score:.2f}")
        print(f"  描述: {recommendation.description}")

        print(f"\n显存估算:")
        vram = recommendation.vram_estimate
        print(f"  模型权重: {vram.model_weight_gb:.2f} GB")
        print(f"  KV Cache: {vram.kv_cache_gb:.3f} GB")
        print(f"  运行时开销: {vram.overhead_gb:.2f} GB")
        print(f"  总计: {vram.total_gb:.2f} GB")
        print(f"  可用: {vram.available_gb:.2f} GB")
        print(f"  是否可运行: {'✅ 是' if vram.fits_in_vram else '❌ 否'}")

        print(f"\n优化建议:")
        for note in recommendation.notes:
            print(f"  - {note}")

        # 显示所有可选级别
        print(f"\n所有量化级别对比:")
        comparisons = quantizer.compare_levels(model_size, 6.0)
        print(f"  {'级别':<12} {'位数':<8} {'显存(GB)':<10} {'可运行':<8} {'质量':<8} {'描述'}")
        print(f"  {'-'*70}")
        for comp in comparisons[:5]:
            fits = "✅" if comp["fits"] else "❌"
            print(f"  {comp['level']:<12} {comp['bits']:<8.1f} {comp['total_gb']:<10.2f} {fits:<8} {comp['quality']:<8.2f} {comp['description']}")


if __name__ == "__main__":
    main()
