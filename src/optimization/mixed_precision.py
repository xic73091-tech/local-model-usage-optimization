"""
混合精度量化模块

实现关键层高精度、非关键层低精度的量化策略，在质量和速度间取得更好平衡。

原理:
- 注意力层(Attention): 使用较高精度(如Q4_K_M)，因为对质量影响大
- FFN层: 使用较低精度(如Q2_K)，因为对质量影响较小
- Embedding层: 使用较高精度，因为是输入输出的关键路径

优势:
- 比全模型低精度量化质量更好
- 比全模型高精度量化速度更快
- 显存占用介于两者之间

预期效果:
- 相比Q4_K_M: 显存减少20-30%，速度提升10-20%
- 相比Q2_K: 质量提升显著，显存增加10-20%
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 量化级别定义
# ============================================================

class QuantLevel(Enum):
    """量化级别"""
    Q2_K = "q2_k"
    Q3_K_S = "q3_k_s"
    Q3_K_M = "q3_k_m"
    Q4_0 = "q4_0"
    Q4_K_S = "q4_k_s"
    Q4_K_M = "q4_k_m"
    Q5_K_S = "q5_k_s"
    Q5_K_M = "q5_k_m"
    Q6_K = "q6_k"
    Q8_0 = "q8_0"
    FP16 = "fp16"


# 量化级别的属性
_QUANT_PROPERTIES = {
    QuantLevel.Q2_K: {"bits": 2.0, "quality": 0.45, "speed_factor": 1.30, "bytes_per_param": 0.25},
    QuantLevel.Q3_K_S: {"bits": 3.0, "quality": 0.55, "speed_factor": 1.25, "bytes_per_param": 0.375},
    QuantLevel.Q3_K_M: {"bits": 3.0, "quality": 0.60, "speed_factor": 1.20, "bytes_per_param": 0.4375},
    QuantLevel.Q4_0: {"bits": 4.0, "quality": 0.65, "speed_factor": 1.20, "bytes_per_param": 0.5},
    QuantLevel.Q4_K_S: {"bits": 4.0, "quality": 0.70, "speed_factor": 1.15, "bytes_per_param": 0.5625},
    QuantLevel.Q4_K_M: {"bits": 4.0, "quality": 0.75, "speed_factor": 1.10, "bytes_per_param": 0.625},
    QuantLevel.Q5_K_S: {"bits": 5.0, "quality": 0.80, "speed_factor": 1.05, "bytes_per_param": 0.6875},
    QuantLevel.Q5_K_M: {"bits": 5.0, "quality": 0.85, "speed_factor": 1.00, "bytes_per_param": 0.75},
    QuantLevel.Q6_K: {"bits": 6.0, "quality": 0.90, "speed_factor": 0.95, "bytes_per_param": 0.875},
    QuantLevel.Q8_0: {"bits": 8.0, "quality": 0.95, "speed_factor": 0.85, "bytes_per_param": 1.0},
    QuantLevel.FP16: {"bits": 16.0, "quality": 1.0, "speed_factor": 0.50, "bytes_per_param": 2.0},
}


# ============================================================
# 层类型定义
# ============================================================

class LayerType(Enum):
    """层类型"""
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    FFN = "ffn"
    NORM = "norm"
    OUTPUT = "output"


# 层类型对质量的影响权重
_LAYER_QUALITY_WEIGHT = {
    LayerType.EMBEDDING: 0.15,  # 对质量影响大
    LayerType.ATTENTION: 0.35,  # 对质量影响最大
    LayerType.FFN: 0.40,        # 对质量影响中等
    LayerType.NORM: 0.05,       # 对质量影响小
    LayerType.OUTPUT: 0.05,     # 对质量影响小
}


# ============================================================
# 配置和数据类
# ============================================================

@dataclass
class MixedPrecisionConfig:
    """混合精度量化配置"""
    # 各层类型的量化级别
    embedding_quant: QuantLevel = QuantLevel.Q4_K_M
    attention_quant: QuantLevel = QuantLevel.Q4_K_M
    ffn_quant: QuantLevel = QuantLevel.Q3_K_M
    norm_quant: QuantLevel = QuantLevel.FP16  # 归一化层通常不量化
    output_quant: QuantLevel = QuantLevel.Q4_K_M

    # 是否启用自动优化
    auto_optimize: bool = True

    # 目标显存占用（GB），用于自动优化
    target_vram_gb: Optional[float] = None

    # 目标质量分数（0-1），用于自动优化
    target_quality: float = 0.7

    # 模型参数量（B），用于自动优化
    model_size_b: float = 7.0


@dataclass
class LayerQuantInfo:
    """层量化信息"""
    layer_type: LayerType
    quant_level: QuantLevel
    bits: float
    quality: float
    speed_factor: float
    bytes_per_param: float
    param_count: int = 0  # 该层的参数数量
    size_gb: float = 0.0  # 该层的显存占用


@dataclass
class MixedPrecisionResult:
    """混合精度量化结果"""
    # 各层的量化配置
    layer_configs: List[LayerQuantInfo]

    # 总体统计
    total_bits: float
    total_quality: float
    total_speed_factor: float
    total_size_gb: float

    # 与标准量化的对比
    compared_to_q4_k_m: Dict[str, float]
    compared_to_q2_k: Dict[str, float]


# ============================================================
# 混合精度量化器
# ============================================================

class MixedPrecisionQuantizer:
    """混合精度量化器"""

    # 典型Transformer模型的层分布
    _TYPICAL_LAYER_DISTRIBUTION = {
        "embedding": 0.05,   # 5% 参数
        "attention": 0.30,   # 30% 参数
        "ffn": 0.55,         # 55% 参数
        "norm": 0.02,        # 2% 参数
        "output": 0.08,      # 8% 参数
    }

    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        """初始化混合精度量化器

        Args:
            config: 配置
        """
        self.config = config or MixedPrecisionConfig()

    def quantize(self, model_size_b: Optional[float] = None) -> MixedPrecisionResult:
        """执行混合精度量化

        Args:
            model_size_b: 模型参数量（B），覆盖配置中的值

        Returns:
            MixedPrecisionResult: 量化结果
        """
        size_b = model_size_b or self.config.model_size_b

        # 如果启用自动优化，计算最优配置
        if self.config.auto_optimize:
            config = self._auto_optimize(size_b)
        else:
            config = self.config

        # 计算各层的量化信息
        layer_configs = self._calculate_layer_configs(size_b, config)

        # 计算总体统计
        total_bits = self._calculate_total_bits(layer_configs)
        total_quality = self._calculate_total_quality(layer_configs)
        total_speed_factor = self._calculate_total_speed_factor(layer_configs)
        total_size_gb = self._calculate_total_size(layer_configs)

        # 与标准量化对比
        compared_to_q4_k_m = self._compare_to_standard(
            layer_configs, size_b, QuantLevel.Q4_K_M
        )
        compared_to_q2_k = self._compare_to_standard(
            layer_configs, size_b, QuantLevel.Q2_K
        )

        return MixedPrecisionResult(
            layer_configs=layer_configs,
            total_bits=total_bits,
            total_quality=total_quality,
            total_speed_factor=total_speed_factor,
            total_size_gb=total_size_gb,
            compared_to_q4_k_m=compared_to_q4_k_m,
            compared_to_q2_k=compared_to_q2_k,
        )

    def _auto_optimize(self, model_size_b: float) -> MixedPrecisionConfig:
        """自动优化配置

        根据目标显存和质量，自动选择最优的量化级别。

        Args:
            model_size_b: 模型参数量

        Returns:
            MixedPrecisionConfig: 优化后的配置
        """
        config = MixedPrecisionConfig()

        # 如果有显存目标，根据显存约束优化
        if config.target_vram_gb is not None:
            # 计算每B参数的显存预算
            vram_per_b = config.target_vram_gb / model_size_b

            # 根据显存预算选择量化级别
            if vram_per_b >= 0.8:
                config.attention_quant = QuantLevel.Q5_K_M
                config.ffn_quant = QuantLevel.Q4_K_M
            elif vram_per_b >= 0.6:
                config.attention_quant = QuantLevel.Q4_K_M
                config.ffn_quant = QuantLevel.Q3_K_M
            elif vram_per_b >= 0.4:
                config.attention_quant = QuantLevel.Q4_K_S
                config.ffn_quant = QuantLevel.Q2_K
            else:
                config.attention_quant = QuantLevel.Q3_K_M
                config.ffn_quant = QuantLevel.Q2_K
        else:
            # 根据质量目标优化
            if config.target_quality >= 0.8:
                config.attention_quant = QuantLevel.Q5_K_M
                config.ffn_quant = QuantLevel.Q4_K_M
            elif config.target_quality >= 0.7:
                config.attention_quant = QuantLevel.Q4_K_M
                config.ffn_quant = QuantLevel.Q3_K_M
            elif config.target_quality >= 0.6:
                config.attention_quant = QuantLevel.Q4_K_S
                config.ffn_quant = QuantLevel.Q2_K
            else:
                config.attention_quant = QuantLevel.Q3_K_M
                config.ffn_quant = QuantLevel.Q2_K

        # Embedding和Output层使用与注意力层相同的精度
        config.embedding_quant = config.attention_quant
        config.output_quant = config.attention_quant

        return config

    def _calculate_layer_configs(
        self, model_size_b: float, config: MixedPrecisionConfig
    ) -> List[LayerQuantInfo]:
        """计算各层的量化配置

        Args:
            model_size_b: 模型参数量
            config: 配置

        Returns:
            List[LayerQuantInfo]: 层量化信息列表
        """
        # 总参数数量
        total_params = int(model_size_b * 1e9)

        layer_configs = []

        # 各层类型的参数分布
        layer_types = [
            (LayerType.EMBEDDING, config.embedding_quant, "embedding"),
            (LayerType.ATTENTION, config.attention_quant, "attention"),
            (LayerType.FFN, config.ffn_quant, "ffn"),
            (LayerType.NORM, config.norm_quant, "norm"),
            (LayerType.OUTPUT, config.output_quant, "output"),
        ]

        for layer_type, quant_level, dist_key in layer_types:
            # 计算该层类型的参数数量
            ratio = self._TYPICAL_LAYER_DISTRIBUTION[dist_key]
            param_count = int(total_params * ratio)

            # 获取量化属性
            props = _QUANT_PROPERTIES[quant_level]

            # 计算显存占用
            size_gb = param_count * props["bytes_per_param"] / (1024 ** 3)

            layer_configs.append(LayerQuantInfo(
                layer_type=layer_type,
                quant_level=quant_level,
                bits=props["bits"],
                quality=props["quality"],
                speed_factor=props["speed_factor"],
                bytes_per_param=props["bytes_per_param"],
                param_count=param_count,
                size_gb=size_gb,
            ))

        return layer_configs

    def _calculate_total_bits(self, layer_configs: List[LayerQuantInfo]) -> float:
        """计算总体平均位数"""
        total_params = sum(lc.param_count for lc in layer_configs)
        if total_params == 0:
            return 0.0
        weighted_bits = sum(lc.bits * lc.param_count for lc in layer_configs)
        return weighted_bits / total_params

    def _calculate_total_quality(self, layer_configs: List[LayerQuantInfo]) -> float:
        """计算总体质量分数

        使用加权平均，考虑各层类型对质量的影响。
        """
        total_weight = 0.0
        weighted_quality = 0.0

        for lc in layer_configs:
            weight = _LAYER_QUALITY_WEIGHT.get(lc.layer_type, 0.1)
            weighted_quality += lc.quality * weight * lc.param_count
            total_weight += weight * lc.param_count

        return weighted_quality / total_weight if total_weight > 0 else 0.0

    def _calculate_total_speed_factor(self, layer_configs: List[LayerQuantInfo]) -> float:
        """计算总体速度因子"""
        total_params = sum(lc.param_count for lc in layer_configs)
        if total_params == 0:
            return 1.0
        weighted_speed = sum(lc.speed_factor * lc.param_count for lc in layer_configs)
        return weighted_speed / total_params

    def _calculate_total_size(self, layer_configs: List[LayerQuantInfo]) -> float:
        """计算总显存占用"""
        return sum(lc.size_gb for lc in layer_configs)

    def _compare_to_standard(
        self,
        layer_configs: List[LayerQuantInfo],
        model_size_b: float,
        standard_quant: QuantLevel,
    ) -> Dict[str, float]:
        """与标准量化对比

        Args:
            layer_configs: 混合精度配置
            model_size_b: 模型参数量
            standard_quant: 标准量化级别

        Returns:
            Dict: 对比结果
        """
        # 混合精度的结果
        mixed_size = self._calculate_total_size(layer_configs)
        mixed_quality = self._calculate_total_quality(layer_configs)
        mixed_speed = self._calculate_total_speed_factor(layer_configs)

        # 标准量化的结果
        standard_props = _QUANT_PROPERTIES[standard_quant]
        standard_size = model_size_b * standard_props["bytes_per_param"]
        standard_quality = standard_props["quality"]
        standard_speed = standard_props["speed_factor"]

        return {
            "size_diff_gb": mixed_size - standard_size,
            "size_ratio": mixed_size / standard_size if standard_size > 0 else 1.0,
            "quality_diff": mixed_quality - standard_quality,
            "quality_ratio": mixed_quality / standard_quality if standard_quality > 0 else 1.0,
            "speed_diff": mixed_speed - standard_speed,
            "speed_ratio": mixed_speed / standard_speed if standard_speed > 0 else 1.0,
        }

    def get_recommendation(self, model_size_b: float, vram_gb: float) -> MixedPrecisionConfig:
        """获取推荐配置

        Args:
            model_size_b: 模型参数量
            vram_gb: 可用显存

        Returns:
            MixedPrecisionConfig: 推荐配置
        """
        # 计算显存预算
        available_vram = vram_gb - 0.4  # 减去开销
        vram_per_b = available_vram / model_size_b

        config = MixedPrecisionConfig()
        config.model_size_b = model_size_b
        config.target_vram_gb = available_vram

        # 根据显存预算选择配置
        if vram_per_b >= 1.0:
            # 显存充足，优先质量
            config.attention_quant = QuantLevel.Q5_K_M
            config.ffn_quant = QuantLevel.Q4_K_M
            config.embedding_quant = QuantLevel.Q5_K_M
            config.output_quant = QuantLevel.Q5_K_M
        elif vram_per_b >= 0.7:
            # 显存适中，平衡质量和速度
            config.attention_quant = QuantLevel.Q4_K_M
            config.ffn_quant = QuantLevel.Q3_K_M
            config.embedding_quant = QuantLevel.Q4_K_M
            config.output_quant = QuantLevel.Q4_K_M
        elif vram_per_b >= 0.5:
            # 显存紧张，优先速度
            config.attention_quant = QuantLevel.Q4_K_S
            config.ffn_quant = QuantLevel.Q2_K
            config.embedding_quant = QuantLevel.Q4_K_S
            config.output_quant = QuantLevel.Q4_K_S
        else:
            # 显存非常紧张，最大压缩
            config.attention_quant = QuantLevel.Q3_K_M
            config.ffn_quant = QuantLevel.Q2_K
            config.embedding_quant = QuantLevel.Q3_K_M
            config.output_quant = QuantLevel.Q3_K_M

        return config


# ============================================================
# 便捷函数
# ============================================================

def get_mixed_precision_config(
    model_size_b: float,
    vram_gb: float,
    target_quality: float = 0.7,
) -> MixedPrecisionResult:
    """获取混合精度量化配置

    Args:
        model_size_b: 模型参数量
        vram_gb: 可用显存
        target_quality: 目标质量

    Returns:
        MixedPrecisionResult: 量化结果
    """
    quantizer = MixedPrecisionQuantizer()
    config = quantizer.get_recommendation(model_size_b, vram_gb)
    config.target_quality = target_quality
    quantizer.config = config
    return quantizer.quantize(model_size_b)


def compare_quantization_strategies(
    model_size_b: float,
    vram_gb: float,
) -> Dict[str, Any]:
    """对比不同量化策略

    Args:
        model_size_b: 模型参数量
        vram_gb: 可用显存

    Returns:
        Dict: 对比结果
    """
    quantizer = MixedPrecisionQuantizer()

    # 混合精度
    mixed_result = quantizer.quantize(model_size_b)

    # 标准量化
    q4_result = _calculate_standard_quant(model_size_b, QuantLevel.Q4_K_M)
    q2_result = _calculate_standard_quant(model_size_b, QuantLevel.Q2_K)

    return {
        "mixed_precision": {
            "avg_bits": mixed_result.total_bits,
            "quality": mixed_result.total_quality,
            "speed_factor": mixed_result.total_speed_factor,
            "size_gb": mixed_result.total_size_gb,
        },
        "q4_k_m": q4_result,
        "q2_k": q2_result,
        "recommendation": "mixed_precision" if mixed_result.total_quality > 0.65 else "q4_k_m",
    }


def _calculate_standard_quant(model_size_b: float, quant_level: QuantLevel) -> Dict[str, float]:
    """计算标准量化的属性"""
    props = _QUANT_PROPERTIES[quant_level]
    return {
        "avg_bits": props["bits"],
        "quality": props["quality"],
        "speed_factor": props["speed_factor"],
        "size_gb": model_size_b * props["bytes_per_param"],
    }
