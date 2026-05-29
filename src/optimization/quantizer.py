"""
量化策略模块
提供模型量化级别管理、显存估算、智能推荐和对比分析。
支持 GGUF、GPTQ、AWQ、bitsandbytes 等主流量化格式。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举与常量
# ---------------------------------------------------------------------------

class QuantFormat(Enum):
    """量化格式"""
    GGUF = "gguf"
    GPTQ = "gptq"
    AWQ = "awq"
    BNB = "bitsandbytes"
    FP16 = "fp16"


class TaskType(Enum):
    """任务类型，用于智能量化推荐"""
    GENERAL = "general"         # 通用对话
    CODE = "code"               # 代码生成/补全
    CREATIVE = "creative"       # 创意写作
    ANALYSIS = "analysis"       # 分析推理
    TRANSLATION = "translation" # 翻译
    CHAT = "chat"               # 聊天


# 任务类型对质量的最低要求 (quality_score 下限)
_TASK_QUALITY_FLOOR: Dict[str, float] = {
    TaskType.CODE.value:       0.80,  # 代码需要高精度
    TaskType.ANALYSIS.value:   0.80,  # 分析推理需要高精度
    TaskType.CREATIVE.value:   0.70,  # 创意写作中等要求
    TaskType.TRANSLATION.value: 0.75, # 翻译中高要求
    TaskType.CHAT.value:       0.60,  # 聊天容忍度高
    TaskType.GENERAL.value:    0.65,  # 通用默认
}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantizationProfile:
    """单个量化级别的完整画像

    Attributes:
        format:              量化格式 (gguf / gptq / awq / bitsandbytes / fp16)
        name:                量化级别名称 (如 Q4_K_M, gptq-4bit)
        bits:                有效位数
        vram_per_billion_params: 每 1B 参数所需显存 (GB)
        quality_score:       质量得分 0-1 (1 = 无损)
        speed_score:         推理速度得分 0-1 (1 = 最快，受量化位宽和格式影响)
        description:         简要描述
    """
    format: str
    name: str
    bits: int
    vram_per_billion_params: float  # GB per billion params
    quality_score: float            # 0-1
    speed_score: float              # 0-1
    description: str


# ---------------------------------------------------------------------------
# 量化级别注册表
# ---------------------------------------------------------------------------

# GGUF 系列 (llama.cpp / ctransformers)
_GGUF_PROFILES: Dict[str, QuantizationProfile] = {
    "Q2_K": QuantizationProfile(
        format="gguf", name="Q2_K", bits=2,
        vram_per_billion_params=0.35, quality_score=0.55, speed_score=0.95,
        description="最低质量，最小体积；适合显存极度受限场景",
    ),
    "Q3_K_S": QuantizationProfile(
        format="gguf", name="Q3_K_S", bits=3,
        vram_per_billion_params=0.38, quality_score=0.62, speed_score=0.92,
        description="低质量小体积，比 Q2_K 略好",
    ),
    "Q3_K_M": QuantizationProfile(
        format="gguf", name="Q3_K_M", bits=3,
        vram_per_billion_params=0.44, quality_score=0.70, speed_score=0.88,
        description="低质量，小体积",
    ),
    "Q3_K_L": QuantizationProfile(
        format="gguf", name="Q3_K_L", bits=3,
        vram_per_billion_params=0.48, quality_score=0.74, speed_score=0.85,
        description="3-bit 大尺寸变体，质量稍好",
    ),
    "Q4_0": QuantizationProfile(
        format="gguf", name="Q4_0", bits=4,
        vram_per_billion_params=0.50, quality_score=0.78, speed_score=0.85,
        description="基础 4-bit 量化，兼容性最佳",
    ),
    "Q4_K_S": QuantizationProfile(
        format="gguf", name="Q4_K_S", bits=4,
        vram_per_billion_params=0.53, quality_score=0.82, speed_score=0.82,
        description="4-bit K-quant 小尺寸，性价比不错",
    ),
    "Q4_K_M": QuantizationProfile(
        format="gguf", name="Q4_K_M", bits=4,
        vram_per_billion_params=0.56, quality_score=0.85, speed_score=0.80,
        description="平衡选择，质量和体积的最佳折中",
    ),
    "Q5_0": QuantizationProfile(
        format="gguf", name="Q5_0", bits=5,
        vram_per_billion_params=0.62, quality_score=0.88, speed_score=0.75,
        description="基础 5-bit 量化",
    ),
    "Q5_K_S": QuantizationProfile(
        format="gguf", name="Q5_K_S", bits=5,
        vram_per_billion_params=0.66, quality_score=0.90, speed_score=0.73,
        description="5-bit K-quant 小尺寸",
    ),
    "Q5_K_M": QuantizationProfile(
        format="gguf", name="Q5_K_M", bits=5,
        vram_per_billion_params=0.70, quality_score=0.92, speed_score=0.70,
        description="高质量，推荐用于代码生成",
    ),
    "Q6_K": QuantizationProfile(
        format="gguf", name="Q6_K", bits=6,
        vram_per_billion_params=0.85, quality_score=0.96, speed_score=0.60,
        description="接近原始质量，适合对精度敏感的任务",
    ),
    "Q8_0": QuantizationProfile(
        format="gguf", name="Q8_0", bits=8,
        vram_per_billion_params=1.00, quality_score=0.99, speed_score=0.50,
        description="几乎无损，显存占用较高",
    ),
}

# GPTQ 系列 (GPU 推理，适合 transformers)
_GPTQ_PROFILES: Dict[str, QuantizationProfile] = {
    "gptq-2bit": QuantizationProfile(
        format="gptq", name="gptq-2bit", bits=2,
        vram_per_billion_params=0.38, quality_score=0.50, speed_score=0.90,
        description="GPTQ 2-bit，质量损失明显，仅用于实验",
    ),
    "gptq-3bit": QuantizationProfile(
        format="gptq", name="gptq-3bit", bits=3,
        vram_per_billion_params=0.48, quality_score=0.65, speed_score=0.85,
        description="GPTQ 3-bit，低质量低显存",
    ),
    "gptq-4bit": QuantizationProfile(
        format="gptq", name="gptq-4bit", bits=4,
        vram_per_billion_params=0.58, quality_score=0.83, speed_score=0.82,
        description="GPTQ 4-bit，主流 GPU 量化方案",
    ),
    "gptq-8bit": QuantizationProfile(
        format="gptq", name="gptq-8bit", bits=8,
        vram_per_billion_params=1.05, quality_score=0.98, speed_score=0.55,
        description="GPTQ 8-bit，几乎无损但显存较大",
    ),
}

# AWQ 系列 (GPU 推理，激活感知量化)
_AWQ_PROFILES: Dict[str, QuantizationProfile] = {
    "awq-4bit": QuantizationProfile(
        format="awq", name="awq-4bit", bits=4,
        vram_per_billion_params=0.56, quality_score=0.87, speed_score=0.85,
        description="AWQ 4-bit，激活感知量化，同位宽质量优于 GPTQ",
    ),
}

# bitsandbytes 系列 (HuggingFace 生态，动态量化)
_BNB_PROFILES: Dict[str, QuantizationProfile] = {
    "bnb-8bit": QuantizationProfile(
        format="bitsandbytes", name="bnb-8bit", bits=8,
        vram_per_billion_params=1.10, quality_score=0.97, speed_score=0.45,
        description="bitsandbytes 8-bit NF4，易用但推理较慢",
    ),
    "bnb-4bit": QuantizationProfile(
        format="bitsandbytes", name="bnb-4bit", bits=4,
        vram_per_billion_params=0.60, quality_score=0.80, speed_score=0.40,
        description="bitsandbytes 4-bit NF4，显存友好但速度最慢",
    ),
}

# FP16 基准 (无量化)
_FP16_PROFILE: QuantizationProfile = QuantizationProfile(
    format="fp16", name="fp16", bits=16,
    vram_per_billion_params=2.00, quality_score=1.00, speed_score=1.00,
    description="FP16 无量化，质量最高但显存翻倍",
)

# 合并所有量化配置
ALL_PROFILES: Dict[str, QuantizationProfile] = {
    "FP16": _FP16_PROFILE,
    **_GGUF_PROFILES,
    **_GPTQ_PROFILES,
    **_AWQ_PROFILES,
    **_BNB_PROFILES,
}


# ---------------------------------------------------------------------------
# 量化管理器
# ---------------------------------------------------------------------------

class QuantizationManager:
    """量化策略管理器

    核心能力:
    1. 显存需求估算 — 基于参数量和量化类型计算所需显存
    2. 智能量化推荐 — 根据可用显存、任务类型推荐最优量化级别
    3. 量化对比表   — 列出所有支持的量化级别的显存、质量、速度指标
    4. 多格式管理   — 统一管理 GGUF / GPTQ / AWQ / bitsandbytes

    用法示例::

        mgr = QuantizationManager()
        vram = mgr.estimate_vram(7, "Q4_K_M")          # 7B 模型 Q4_K_M 显存
        best = mgr.recommend_quantization(7, 8.0, "code")  # 8GB 显存，代码任务
        table = mgr.get_comparison_table()               # 完整对比表
    """

    # 兼容旧接口的快速索引
    QUANT_PROFILES: Dict[str, QuantizationProfile] = ALL_PROFILES

    # GGUF 量化级别的显存占用排序 (从低到高，用于推荐算法)
    _GGUF_QUALITY_ORDER: List[str] = [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_K_S", "Q4_K_M",
        "Q5_0", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0",
    ]

    # ------------------------------------------------------------------
    # 显存估算
    # ------------------------------------------------------------------

    def estimate_vram(self, model_size_b: float, quant_type: str) -> float:
        """估算指定模型在指定量化级别下的显存需求 (GB)

        公式: vram = model_size_b * vram_per_billion_params + overhead

        其中 overhead 为 KV cache / 激活值等固定开销 (约 0.5 GB 基础 + 每 B 参数 0.1 GB)。

        :param model_size_b: 模型参数量 (单位: 十亿，如 7 表示 7B)
        :param quant_type:   量化类型名称 (如 Q4_K_M, gptq-4bit, awq-4bit)
        :return: 预估显存需求 (GB)
        :raises KeyError: 未知量化类型
        """
        profile = self._get_profile(quant_type)
        base_vram = model_size_b * profile.vram_per_billion_params
        # KV cache 和运行时开销估算
        overhead = 0.5 + model_size_b * 0.08
        return round(base_vram + overhead, 2)

    def estimate_vram_breakdown(
        self, model_size_b: float, quant_type: str
    ) -> Dict[str, float]:
        """显存需求详细分项估算

        :return: {"weights": ..., "kv_cache": ..., "overhead": ..., "total": ...}
        """
        profile = self._get_profile(quant_type)
        weights = model_size_b * profile.vram_per_billion_params
        kv_cache = model_size_b * 0.08
        overhead = 0.5
        total = round(weights + kv_cache + overhead, 2)
        return {
            "weights": round(weights, 2),
            "kv_cache": round(kv_cache, 2),
            "overhead": round(overhead, 2),
            "total": total,
        }

    # ------------------------------------------------------------------
    # 智能推荐
    # ------------------------------------------------------------------

    def recommend_quantization(
        self,
        model_size_b: float,
        available_vram_gb: float,
        task_type: str = "general",
    ) -> str:
        """根据可用显存和任务类型推荐最优量化级别

        算法:
        1. 过滤掉显存超出可用量的量化级别 (预留 5% 余量)
        2. 过滤掉不满足任务质量要求的量化级别
        3. 按综合得分排序: quality * 0.6 + speed * 0.2 + memory_efficiency * 0.2
        4. 返回得分最高的量化级别

        :param model_size_b:      模型参数量 (B)
        :param available_vram_gb: 可用显存 (GB)
        :param task_type:         任务类型 (general / code / creative / analysis / translation / chat)
        :return: 推荐的量化类型名称
        :raises ValueError: 显存不足以运行该模型的任何量化级别
        """
        effective_vram = available_vram_gb * 0.95  # 预留 5% 余量
        quality_floor = _TASK_QUALITY_FLOOR.get(task_type, 0.65)

        candidates: List[Tuple[float, str, QuantizationProfile]] = []

        for name, profile in self.QUANT_PROFILES.items():
            vram_needed = self.estimate_vram(model_size_b, name)

            # 显存约束
            if vram_needed > effective_vram:
                continue

            # 质量约束
            if profile.quality_score < quality_floor:
                continue

            # 综合得分
            mem_efficiency = max(0, 1.0 - vram_needed / available_vram_gb)
            score = (
                profile.quality_score * 0.6
                + profile.speed_score * 0.2
                + mem_efficiency * 0.2
            )
            candidates.append((score, name, profile))

        if not candidates:
            # 降级策略: 忽略质量约束，选最小显存的
            fallback = self._find_min_vram_quant(model_size_b, effective_vram)
            if fallback:
                logger.warning(
                    "显存不足，降级推荐 %s (任务 %s 质量要求无法满足)",
                    fallback, task_type,
                )
                return fallback
            raise ValueError(
                f"显存 {available_vram_gb}GB 不足以运行 {model_size_b}B 模型的任何量化级别"
            )

        candidates.sort(key=lambda c: c[0], reverse=True)
        best_name = candidates[0][1]
        logger.info(
            "推荐量化: %s (模型=%sB, 显存=%sGB, 任务=%s, 得分=%.3f)",
            best_name, model_size_b, available_vram_gb, task_type, candidates[0][0],
        )
        return best_name

    def recommend_for_format(
        self,
        model_size_b: float,
        available_vram_gb: float,
        preferred_format: str,
        task_type: str = "general",
    ) -> Optional[str]:
        """在指定格式内推荐最优量化级别

        :param model_size_b:      模型参数量 (B)
        :param available_vram_gb: 可用显存 (GB)
        :param preferred_format:  偏好格式 (gguf / gptq / awq / bitsandbytes)
        :param task_type:         任务类型
        :return: 推荐的量化名称，无合适选项时返回 None
        """
        effective_vram = available_vram_gb * 0.95
        quality_floor = _TASK_QUALITY_FLOOR.get(task_type, 0.65)

        candidates: List[Tuple[float, str]] = []
        for name, profile in self.QUANT_PROFILES.items():
            if profile.format != preferred_format:
                continue
            vram_needed = self.estimate_vram(model_size_b, name)
            if vram_needed > effective_vram:
                continue
            if profile.quality_score < quality_floor:
                continue
            mem_eff = max(0, 1.0 - vram_needed / available_vram_gb)
            score = profile.quality_score * 0.6 + profile.speed_score * 0.2 + mem_eff * 0.2
            candidates.append((score, name))

        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    # ------------------------------------------------------------------
    # 对比分析
    # ------------------------------------------------------------------

    def get_comparison_table(self) -> List[Dict[str, Any]]:
        """获取所有量化级别的对比表

        :return: 每个元素为一个字典，包含:
            - name:           量化名称
            - format:         量化格式
            - bits:           位数
            - vram_per_b:     每 B 参数显存 (GB)
            - quality_score:  质量得分 (0-1)
            - speed_score:    速度得分 (0-1)
            - vram_7b:        7B 模型显存需求 (GB)
            - vram_13b:       13B 模型显存需求 (GB)
            - description:    描述
        """
        table: List[Dict[str, Any]] = []
        for name, profile in self.QUANT_PROFILES.items():
            table.append({
                "name": name,
                "format": profile.format,
                "bits": profile.bits,
                "vram_per_b": profile.vram_per_billion_params,
                "quality_score": profile.quality_score,
                "speed_score": profile.speed_score,
                "vram_7b": self.estimate_vram(7.0, name),
                "vram_13b": self.estimate_vram(13.0, name),
                "description": profile.description,
            })
        # 按质量得分降序排列
        table.sort(key=lambda x: x["quality_score"], reverse=True)
        return table

    def get_format_comparison(self) -> Dict[str, List[Dict[str, Any]]]:
        """按格式分组的对比数据

        :return: {format_name: [profile_dicts]}
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for name, profile in self.QUANT_PROFILES.items():
            fmt = profile.format
            if fmt not in grouped:
                grouped[fmt] = []
            grouped[fmt].append({
                "name": name,
                "bits": profile.bits,
                "vram_per_b": profile.vram_per_billion_params,
                "quality_score": profile.quality_score,
                "speed_score": profile.speed_score,
                "description": profile.description,
            })
        for fmt in grouped:
            grouped[fmt].sort(key=lambda x: x["quality_score"], reverse=True)
        return grouped

    def compare_quantizations(
        self, quant_types: List[str], model_size_b: float
    ) -> List[Dict[str, Any]]:
        """对比指定几种量化级别在同一模型上的表现

        :param quant_types:  要对比的量化名称列表
        :param model_size_b: 模型参数量 (B)
        :return: 对比结果列表
        """
        results: List[Dict[str, Any]] = []
        for name in quant_types:
            profile = self._get_profile(name)
            results.append({
                "name": name,
                "format": profile.format,
                "bits": profile.bits,
                "vram_gb": self.estimate_vram(model_size_b, name),
                "quality_score": profile.quality_score,
                "speed_score": profile.speed_score,
                "composite_score": round(
                    profile.quality_score * 0.6 + profile.speed_score * 0.4, 3
                ),
                "description": profile.description,
            })
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_profile(self, quant_type: str) -> Optional[QuantizationProfile]:
        """获取量化配置，不存在返回 None"""
        return self.QUANT_PROFILES.get(quant_type)

    def list_quant_types(self, format_filter: Optional[str] = None) -> List[str]:
        """列出所有支持的量化类型名称

        :param format_filter: 可选格式过滤 (gguf / gptq / awq / bitsandbytes)
        """
        if format_filter:
            return [
                name for name, p in self.QUANT_PROFILES.items()
                if p.format == format_filter
            ]
        return list(self.QUANT_PROFILES.keys())

    def list_formats(self) -> List[str]:
        """列出所有支持的量化格式"""
        return list({p.format for p in self.QUANT_PROFILES.values()})

    def get_min_vram_quant(
        self, model_size_b: float, available_vram_gb: float
    ) -> Optional[str]:
        """在显存约束下找到质量最高的量化级别

        :return: 量化名称，无合适选项返回 None
        """
        best: Optional[Tuple[float, str]] = None
        for name, profile in self.QUANT_PROFILES.items():
            vram = self.estimate_vram(model_size_b, name)
            if vram > available_vram_gb * 0.95:
                continue
            if best is None or profile.quality_score > best[0]:
                best = (profile.quality_score, name)
        return best[1] if best else None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_profile(self, quant_type: str) -> QuantizationProfile:
        """获取量化配置，不存在时抛出 KeyError"""
        profile = self.QUANT_PROFILES.get(quant_type)
        if profile is None:
            available = ", ".join(sorted(self.QUANT_PROFILES.keys()))
            raise KeyError(
                f"未知量化类型 '{quant_type}'，可选: {available}"
            )
        return profile

    def _find_min_vram_quant(
        self, model_size_b: float, max_vram_gb: float
    ) -> Optional[str]:
        """在显存约束下找到占用最小的量化级别 (降级用)"""
        best: Optional[Tuple[float, str]] = None
        for name in self.QUANT_PROFILES:
            vram = self.estimate_vram(model_size_b, name)
            if vram > max_vram_gb:
                continue
            if best is None or vram < best[0]:
                best = (vram, name)
        return best[1] if best else None
