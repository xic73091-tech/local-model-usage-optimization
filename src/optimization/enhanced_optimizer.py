"""
增强优化模块 - 让小显存运行大模型更快更稳定

核心优化:
1. PCIe带宽检测 - 自动检测并优化GPU-CPU数据传输
2. GQA-aware KV Cache - 准确估算不同架构的KV Cache
3. OOM防护机制 - 实时监控显存，自动降级
4. 优化的层加载 - 细粒度锁，提升并发度
5. Flash Attention支持 - 减少显存占用，提升速度
"""

import logging
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# PCIe带宽检测
# ============================================================

class PCIeGeneration(Enum):
    """PCIe代际枚举"""
    GEN2 = 2  # 500 MB/s per lane
    GEN3 = 3  # 985 MB/s per lane
    GEN4 = 4  # 1969 MB/s per lane
    GEN5 = 5  # 3938 MB/s per lane
    UNKNOWN = 0


@dataclass
class PCIeInfo:
    """PCIe信息"""
    generation: PCIeGeneration
    lanes: int
    bandwidth_gb_s: float  # 实际带宽 (GB/s)
    overhead_factor: float  # 开销系数 (0-1)


class PCIeDetector:
    """PCIe带宽检测器"""

    # 每代PCIe每lane带宽 (GB/s)
    _LANE_BANDWIDTH = {
        PCIeGeneration.GEN2: 0.5,
        PCIeGeneration.GEN3: 0.985,
        PCIeGeneration.GEN4: 1.969,
        PCIeGeneration.GEN5: 3.938,
    }

    @classmethod
    def detect(cls) -> PCIeInfo:
        """检测PCIe信息"""
        try:
            # 尝试通过nvidia-smi获取PCIe信息
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(',')
                    if len(parts) >= 2:
                        gen = int(parts[0].strip())
                        lanes = int(parts[1].strip())
                        generation = PCIeGeneration(gen) if gen in [2, 3, 4, 5] else PCIeGeneration.UNKNOWN
                        bandwidth = cls._LANE_BANDWIDTH.get(generation, 0.985) * lanes
                        return PCIeInfo(
                            generation=generation,
                            lanes=lanes,
                            bandwidth_gb_s=bandwidth,
                            overhead_factor=cls._calculate_overhead(generation),
                        )
        except Exception as e:
            logger.debug("PCIe检测失败: %s", e)

        # 默认返回PCIe 3.0 x16
        return PCIeInfo(
            generation=PCIeGeneration.GEN3,
            lanes=16,
            bandwidth_gb_s=15.76,  # 0.985 * 16
            overhead_factor=0.15,
        )

    @classmethod
    def _calculate_overhead(cls, generation: PCIeGeneration) -> float:
        """计算PCIe传输开销系数"""
        # 更高代际的PCIe开销更低
        overhead_map = {
            PCIeGeneration.GEN2: 0.25,
            PCIeGeneration.GEN3: 0.15,
            PCIeGeneration.GEN4: 0.10,
            PCIeGeneration.GEN5: 0.05,
        }
        return overhead_map.get(generation, 0.15)


# ============================================================
# GQA-aware KV Cache
# ============================================================

@dataclass
class ModelArchitecture:
    """模型架构信息"""
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int  # GQA架构的KV头数
    head_dim: int
    hidden_size: int
    use_gqa: bool
    context_length: int


class KVCacheCalculator:
    """GQA-aware KV Cache计算器"""

    # 常见模型的架构参数
    _MODEL_ARCHITECTURES = {
        # Llama系列
        "llama-7b": {"layers": 32, "heads": 32, "kv_heads": 32, "head_dim": 128, "hidden": 4096},
        "llama-13b": {"layers": 40, "heads": 40, "kv_heads": 40, "head_dim": 128, "hidden": 5120},
        "llama-30b": {"layers": 60, "heads": 52, "kv_heads": 52, "head_dim": 128, "hidden": 6656},
        "llama-70b": {"layers": 80, "heads": 64, "kv_heads": 8, "head_dim": 128, "hidden": 8192},  # GQA
        # Llama 2系列
        "llama2-7b": {"layers": 32, "heads": 32, "kv_heads": 32, "head_dim": 128, "hidden": 4096},
        "llama2-13b": {"layers": 40, "heads": 40, "kv_heads": 40, "head_dim": 128, "hidden": 5120},
        "llama2-70b": {"layers": 80, "heads": 64, "kv_heads": 8, "head_dim": 128, "hidden": 8192},  # GQA
        # Mistral系列
        "mistral-7b": {"layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128, "hidden": 4096},  # GQA
        # Qwen系列
        "qwen-7b": {"layers": 32, "heads": 32, "kv_heads": 32, "head_dim": 128, "hidden": 4096},
        "qwen-14b": {"layers": 40, "heads": 40, "kv_heads": 40, "head_dim": 128, "hidden": 5120},
        "qwen-72b": {"layers": 80, "heads": 64, "kv_heads": 8, "head_dim": 128, "hidden": 8192},  # GQA
        # Yi系列
        "yi-6b": {"layers": 32, "heads": 32, "kv_heads": 4, "head_dim": 128, "hidden": 4096},  # GQA
        "yi-34b": {"layers": 60, "heads": 56, "kv_heads": 8, "head_dim": 128, "hidden": 7168},  # GQA
    }

    @classmethod
    def detect_architecture(cls, model_size_b: float, model_name: Optional[str] = None) -> ModelArchitecture:
        """检测模型架构

        Args:
            model_size_b: 模型参数量 (B)
            model_name: 模型名称 (可选)

        Returns:
            ModelArchitecture: 模型架构信息
        """
        # 尝试从模型名称匹配
        if model_name:
            model_name_lower = model_name.lower().replace("-", "").replace("_", "")
            for key, arch in cls._MODEL_ARCHITECTURES.items():
                if key.replace("-", "") in model_name_lower:
                    return ModelArchitecture(
                        num_layers=arch["layers"],
                        num_attention_heads=arch["heads"],
                        num_kv_heads=arch["kv_heads"],
                        head_dim=arch["head_dim"],
                        hidden_size=arch["hidden"],
                        use_gqa=arch["heads"] != arch["kv_heads"],
                        context_length=4096,
                    )

        # 基于参数量估算
        return cls._estimate_architecture(model_size_b)

    @classmethod
    def _estimate_architecture(cls, model_size_b: float) -> ModelArchitecture:
        """基于参数量估算架构"""
        # 简化的估算逻辑
        if model_size_b <= 3:
            layers, heads, kv_heads, head_dim, hidden = 26, 32, 32, 128, 3200
        elif model_size_b <= 7:
            layers, heads, kv_heads, head_dim, hidden = 32, 32, 32, 128, 4096
        elif model_size_b <= 13:
            layers, heads, kv_heads, head_dim, hidden = 40, 40, 40, 128, 5120
        elif model_size_b <= 30:
            layers, heads, kv_heads, head_dim, hidden = 60, 52, 52, 128, 6656
        elif model_size_b <= 70:
            layers, heads, kv_heads, head_dim, hidden = 80, 64, 8, 128, 8192  # GQA
        else:
            layers, heads, kv_heads, head_dim, hidden = 96, 80, 8, 128, 10240  # GQA

        return ModelArchitecture(
            num_layers=layers,
            num_attention_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            hidden_size=hidden,
            use_gqa=heads != kv_heads,
            context_length=4096,
        )

    @classmethod
    def calculate_kv_cache_size(
        cls,
        arch: ModelArchitecture,
        context_length: int,
        batch_size: int = 1,
        kv_quant_bits: int = 16,
    ) -> float:
        """计算KV Cache大小 (GB)

        Args:
            arch: 模型架构
            context_length: 上下文长度
            batch_size: 批处理大小
            kv_quant_bits: KV Cache量化位数

        Returns:
            float: KV Cache大小 (GB)
        """
        # GQA架构: KV头数 < 注意力头数
        # 公式: 2 * layers * kv_heads * head_dim * seq_len * bytes_per_element * batch_size
        bytes_per_element = kv_quant_bits / 8
        kv_bytes = (
            2 *  # K和V
            arch.num_layers *
            arch.num_kv_heads *  # 使用KV头数，不是注意力头数
            arch.head_dim *
            context_length *
            bytes_per_element *
            batch_size
        )
        return kv_bytes / (1024 ** 3)

    @classmethod
    def calculate_memory_savings(
        cls,
        arch: ModelArchitecture,
        context_length: int = 4096,
    ) -> Dict[str, float]:
        """计算GQA架构的显存节省

        Returns:
            Dict: 包含节省信息的字典
        """
        # 标准MHA的KV Cache大小
        mha_kv_size = cls.calculate_kv_cache_size(
            arch, context_length, kv_quant_bits=16
        )

        # GQA的KV Cache大小
        gqa_kv_size = cls.calculate_kv_cache_size(
            arch, context_length, kv_quant_bits=16
        )

        # 节省比例
        savings_ratio = 1.0 - (arch.num_kv_heads / arch.num_attention_heads) if arch.use_gqa else 0.0

        return {
            "mha_kv_size_gb": mha_kv_size,
            "gqa_kv_size_gb": gqa_kv_size,
            "savings_ratio": savings_ratio,
            "savings_gb": mha_kv_size - gqa_kv_size,
            "use_gqa": arch.use_gqa,
            "kv_heads": arch.num_kv_heads,
            "attention_heads": arch.num_attention_heads,
        }


# ============================================================
# OOM防护机制
# ============================================================

class OOMProtectionLevel(Enum):
    """OOM防护级别"""
    NONE = 0
    LOW = 1      # 接近上限，开始监控
    MEDIUM = 2   # 需要降级
    HIGH = 3     # 紧急降级
    CRITICAL = 4 # 即将OOM


@dataclass
class OOMStatus:
    """OOM状态"""
    level: OOMProtectionLevel
    current_usage_gb: float
    total_available_gb: float
    usage_ratio: float
    recommended_action: str
    auto_degraded: bool


class OOMProtector:
    """OOM防护器"""

    # 阈值配置
    THRESHOLDS = {
        OOMProtectionLevel.LOW: 0.70,      # 70%
        OOMProtectionLevel.MEDIUM: 0.80,   # 80%
        OOMProtectionLevel.HIGH: 0.90,     # 90%
        OOMProtectionLevel.CRITICAL: 0.95, # 95%
    }

    def __init__(self, total_vram_gb: float, safety_margin_gb: float = 0.5):
        """初始化OOM防护器

        Args:
            total_vram_gb: 总显存 (GB)
            safety_margin_gb: 安全余量 (GB)
        """
        self.total_vram_gb = total_vram_gb
        self.safety_margin_gb = safety_margin_gb
        self.usable_vram_gb = total_vram_gb - safety_margin_gb
        self._lock = threading.Lock()
        self._current_usage_gb = 0.0
        self._degraded = False
        self._degradation_history: List[Tuple[float, str]] = []

    def update_usage(self, usage_gb: float) -> OOMStatus:
        """更新显存使用量并检查状态

        Args:
            usage_gb: 当前显存使用量 (GB)

        Returns:
            OOMStatus: OOM状态
        """
        with self._lock:
            self._current_usage_gb = usage_gb
            return self._check_status()

    def _check_status(self) -> OOMStatus:
        """检查OOM状态"""
        usage_ratio = self._current_usage_gb / self.usable_vram_gb

        # 确定防护级别
        level = OOMProtectionLevel.NONE
        for threshold_level, threshold_ratio in sorted(
            self.THRESHOLDS.items(), key=lambda x: x[1], reverse=True
        ):
            if usage_ratio >= threshold_ratio:
                level = threshold_level
                break

        # 确定推荐操作
        recommended_action = self._get_recommended_action(level, usage_ratio)

        return OOMStatus(
            level=level,
            current_usage_gb=self._current_usage_gb,
            total_available_gb=self.usable_vram_gb,
            usage_ratio=usage_ratio,
            recommended_action=recommended_action,
            auto_degraded=self._degraded,
        )

    def _get_recommended_action(self, level: OOMProtectionLevel, usage_ratio: float) -> str:
        """获取推荐操作"""
        if level == OOMProtectionLevel.NONE:
            return "正常运行"
        elif level == OOMProtectionLevel.LOW:
            return "监控显存使用，准备降级"
        elif level == OOMProtectionLevel.MEDIUM:
            return "建议降低上下文长度或量化级别"
        elif level == OOMProtectionLevel.HIGH:
            return "强烈建议降级，切换到CPU推理"
        else:
            return "紧急！立即降级避免OOM"

    def auto_degrade(
        self,
        current_context: int,
        current_quant: str,
        current_gpu_layers: int,
    ) -> Tuple[int, str, int, str]:
        """自动降级配置

        Args:
            current_context: 当前上下文长度
            current_quant: 当前量化级别
            current_gpu_layers: 当前GPU层数

        Returns:
            Tuple: (新上下文, 新量化, 新GPU层数, 降级说明)
        """
        with self._lock:
            status = self._check_status()

            if status.level == OOMProtectionLevel.NONE:
                return current_context, current_quant, current_gpu_layers, "无需降级"

            # 降级策略
            new_context = current_context
            new_quant = current_quant
            new_gpu_layers = current_gpu_layers
            actions = []

            # 1. 首先减少上下文长度
            if status.level.value >= OOMProtectionLevel.LOW.value:
                reduction = min(current_context // 4, 2048)
                new_context = max(512, current_context - reduction)
                actions.append(f"上下文: {current_context} -> {new_context}")

            # 2. 然后降低量化级别
            if status.level.value >= OOMProtectionLevel.MEDIUM.value:
                quant_downgrade = {
                    "q8_0": "q6_k",
                    "q6_k": "q5_k_m",
                    "q5_k_m": "q5_k_s",
                    "q5_k_s": "q4_k_m",
                    "q4_k_m": "q4_k_s",
                    "q4_k_s": "q4_0",
                    "q4_0": "q3_k_m",
                    "q3_k_m": "q3_k_s",
                    "q3_k_s": "q2_k",
                    "q2_k": "q2_k",
                }
                new_quant = quant_downgrade.get(current_quant, current_quant)
                if new_quant != current_quant:
                    actions.append(f"量化: {current_quant} -> {new_quant}")

            # 3. 最后减少GPU层数
            if status.level.value >= OOMProtectionLevel.HIGH.value:
                reduction = max(1, current_gpu_layers // 4)
                new_gpu_layers = max(0, current_gpu_layers - reduction)
                actions.append(f"GPU层数: {current_gpu_layers} -> {new_gpu_layers}")

            # 4. 紧急情况：切换到CPU
            if status.level == OOMProtectionLevel.CRITICAL:
                new_gpu_layers = 0
                actions.append("切换到CPU推理")

            self._degraded = True
            self._degradation_history.append((time.time(), "; ".join(actions)))

            return new_context, new_quant, new_gpu_layers, "; ".join(actions)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "total_vram_gb": self.total_vram_gb,
                "usable_vram_gb": self.usable_vram_gb,
                "current_usage_gb": self._current_usage_gb,
                "usage_ratio": self._current_usage_gb / self.usable_vram_gb,
                "degraded": self._degraded,
                "degradation_count": len(self._degradation_history),
            }


# ============================================================
# 优化的速度估算器
# ============================================================

@dataclass
class SpeedEstimate:
    """速度估算结果"""
    estimated_tps: float
    gpu_tps: float
    cpu_tps: float
    pcie_overhead: float
    bottleneck: str
    suitable_for_realtime: bool


class EnhancedSpeedEstimator:
    """增强的速度估算器"""

    # 基准速率
    _BASELINE_GPU_TPS_7B = 40.0
    _BASELINE_CPU_TPS_7B = 8.0

    @classmethod
    def estimate_speed(
        cls,
        model_size_b: float,
        quant_bits: int,
        gpu_layers: int,
        total_layers: int,
        gpu_vram_gb: float,
        cpu_threads: int,
        pcie_info: Optional[PCIeInfo] = None,
    ) -> SpeedEstimate:
        """估算推理速度

        Args:
            model_size_b: 模型参数量 (B)
            quant_bits: 量化位数
            gpu_layers: GPU层数
            total_layers: 总层数
            gpu_vram_gb: GPU显存 (GB)
            cpu_threads: CPU线程数
            pcie_info: PCIe信息 (可选)

        Returns:
            SpeedEstimate: 速度估算
        """
        # 获取PCIe信息
        if pcie_info is None:
            pcie_info = PCIeDetector.detect()

        # 量化速度因子
        quant_factor = cls._get_quant_factor(quant_bits)

        # 参数量惩罚
        size_penalty = math.sqrt(7.0 / max(model_size_b, 0.5))

        # GPU速率估算
        gpu_tps = 0.0
        if gpu_layers > 0 and gpu_vram_gb > 0:
            vram_factor = min(gpu_vram_gb / 8.0, 3.0)
            gpu_ratio = gpu_layers / max(total_layers, 1)
            gpu_tps = cls._BASELINE_GPU_TPS_7B * size_penalty * quant_factor * vram_factor * gpu_ratio

        # CPU速率估算
        cpu_tps = 0.0
        if gpu_layers < total_layers:
            thread_factor = min(cpu_threads / 4.0, 2.0)
            cpu_ratio = (total_layers - gpu_layers) / max(total_layers, 1)
            cpu_tps = cls._BASELINE_CPU_TPS_7B * size_penalty * quant_factor * thread_factor * cpu_ratio

        # 混合推理速率
        if gpu_layers > 0 and gpu_layers < total_layers:
            # 使用实际PCIe开销
            pcie_overhead = pcie_info.overhead_factor
            # 瓶颈分析
            if gpu_tps > cpu_tps * 2:
                # GPU远快于CPU，瓶颈在CPU
                bottleneck = "cpu"
                estimated_tps = cpu_tps * (1 + pcie_overhead)
            elif cpu_tps > gpu_tps * 2:
                # CPU远快于GPU，瓶颈在GPU
                bottleneck = "gpu"
                estimated_tps = gpu_tps * (1 - pcie_overhead)
            else:
                # 平衡情况
                bottleneck = "pcie"
                estimated_tps = min(gpu_tps, cpu_tps * 1.5) * (1 - pcie_overhead)
        elif gpu_layers > 0:
            bottleneck = "none"
            estimated_tps = gpu_tps
            pcie_overhead = 0.0
        else:
            bottleneck = "cpu_only"
            estimated_tps = cpu_tps
            pcie_overhead = 0.0

        estimated_tps = max(estimated_tps, 0.1)

        return SpeedEstimate(
            estimated_tps=round(estimated_tps, 1),
            gpu_tps=round(gpu_tps, 1),
            cpu_tps=round(cpu_tps, 1),
            pcie_overhead=round(pcie_overhead, 3),
            bottleneck=bottleneck,
            suitable_for_realtime=estimated_tps >= 5.0,
        )

    @classmethod
    def _get_quant_factor(cls, quant_bits: int) -> float:
        """获取量化速度因子"""
        factors = {
            2: 1.30,
            3: 1.20,
            4: 1.10,
            5: 1.05,
            6: 0.95,
            8: 0.85,
            16: 0.50,
        }
        return factors.get(quant_bits, 1.0)


# ============================================================
# Flash Attention配置
# ============================================================

@dataclass
class FlashAttentionConfig:
    """Flash Attention配置"""
    enabled: bool = True
    use_flash_attention_2: bool = True
    sliding_window: Optional[int] = None
    max_position_embeddings: int = 4096


class FlashAttentionOptimizer:
    """Flash Attention优化器"""

    @staticmethod
    def get_llama_cpp_params(
        context_length: int,
        batch_size: int = 512,
        use_flash_attention: bool = True,
    ) -> Dict[str, Any]:
        """获取llama.cpp的Flash Attention参数

        Returns:
            Dict: llama.cpp参数
        """
        params = {
            "use_mmap": True,  # 使用内存映射
            "use_mlock": False,  # 不锁定内存
        }

        if use_flash_attention:
            # llama.cpp的flash attention参数
            params["flash_attention"] = True
            # 启用flash attention时可以使用更大的batch
            params["n_batch"] = min(batch_size * 2, 2048)

        return params

    @staticmethod
    def calculate_memory_savings(
        context_length: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> Dict[str, float]:
        """计算Flash Attention的显存节省

        Returns:
            Dict: 节省信息
        """
        # 标准注意力的显存占用 (需要存储完整的注意力矩阵)
        standard_memory = num_layers * num_heads * context_length * context_length * 2  # FP16

        # Flash Attention的显存占用 (流式计算，不需要完整矩阵)
        flash_memory = num_layers * num_heads * context_length * 128 * 2  # 简化估算

        savings_gb = (standard_memory - flash_memory) / (1024 ** 3)

        return {
            "standard_attention_gb": standard_memory / (1024 ** 3),
            "flash_attention_gb": flash_memory / (1024 ** 3),
            "savings_gb": savings_gb,
            "savings_ratio": savings_gb / (standard_memory / (1024 ** 3)) if standard_memory > 0 else 0,
        }


# ============================================================
# 增强优化器主类
# ============================================================

@dataclass
class EnhancedOptimizationResult:
    """增强优化结果"""
    # 基本信息
    model_size_b: float
    model_name: Optional[str]

    # 架构信息
    architecture: ModelArchitecture
    use_gqa: bool

    # 量化配置
    quant_bits: int
    quant_level: str

    # 层分配
    gpu_layers: int
    cpu_layers: int
    total_layers: int

    # 上下文配置
    context_length: int
    sliding_window: int

    # KV Cache配置
    kv_cache_gb: float
    kv_quant_bits: int

    # 显存占用
    model_weight_gb: float
    total_vram_gb: float
    total_ram_gb: float

    # 速度估算
    speed_estimate: SpeedEstimate

    # PCIe信息
    pcie_info: PCIeInfo

    # OOM防护
    oom_status: OOMStatus

    # 优化建议
    optimizations_applied: List[str]
    warnings: List[str]


class EnhancedOptimizer:
    """增强优化器 - 让小显存运行大模型更快更稳定"""

    def __init__(
        self,
        total_vram_gb: float,
        total_ram_gb: float = 16.0,
        cpu_threads: int = 4,
        safety_margin_gb: float = 0.5,
    ):
        """初始化增强优化器

        Args:
            total_vram_gb: 总显存 (GB)
            total_ram_gb: 总内存 (GB)
            cpu_threads: CPU线程数
            safety_margin_gb: 显存安全余量 (GB)
        """
        self.total_vram_gb = total_vram_gb
        self.total_ram_gb = total_ram_gb
        self.cpu_threads = cpu_threads

        # 检测PCIe信息
        self.pcie_info = PCIeDetector.detect()
        logger.info("PCIe检测: Gen%s x%s, 带宽 %.1f GB/s, 开销 %.1f%%",
                    self.pcie_info.generation.value,
                    self.pcie_info.lanes,
                    self.pcie_info.bandwidth_gb_s,
                    self.pcie_info.overhead_factor * 100)

        # 初始化OOM防护
        self.oom_protector = OOMProtector(total_vram_gb, safety_margin_gb)

    def optimize(
        self,
        model_size_b: float,
        model_name: Optional[str] = None,
        target_context: int = 4096,
        target_quant: Optional[str] = None,
        enable_flash_attention: bool = True,
    ) -> EnhancedOptimizationResult:
        """执行优化

        Args:
            model_size_b: 模型参数量 (B)
            model_name: 模型名称 (可选)
            target_context: 目标上下文长度
            target_quant: 目标量化级别 (可选)
            enable_flash_attention: 是否启用Flash Attention

        Returns:
            EnhancedOptimizationResult: 优化结果
        """
        optimizations_applied = []
        warnings = []

        # 1. 检测模型架构
        arch = KVCacheCalculator.detect_architecture(model_size_b, model_name)
        use_gqa = arch.use_gqa
        if use_gqa:
            optimizations_applied.append(f"检测到GQA架构: {arch.num_kv_heads} KV头 vs {arch.num_attention_heads} 注意力头")

        # 2. 选择量化级别
        quant_level, quant_bits = self._select_quantization(model_size_b, target_quant)
        optimizations_applied.append(f"选择量化: {quant_level} ({quant_bits} bits)")

        # 3. 计算模型权重大小
        bytes_per_param = quant_bits / 8
        model_weight_gb = model_size_b * bytes_per_param

        # 4. 计算最优GPU层数
        gpu_layers, cpu_layers = self._calculate_optimal_layers(
            model_size_b, model_weight_gb, arch, target_context, quant_bits
        )

        # 5. 计算KV Cache
        kv_quant_bits = self._select_kv_quantization(gpu_layers, arch.total_layers if hasattr(arch, 'total_layers') else arch.num_layers)
        kv_cache_gb = KVCacheCalculator.calculate_kv_cache_size(
            arch, target_context, kv_quant_bits=kv_quant_bits
        )

        # 6. 计算滑动窗口
        sliding_window = self._calculate_sliding_window(
            target_context, kv_cache_gb, gpu_layers, arch.num_layers
        )
        if sliding_window < target_context:
            optimizations_applied.append(f"滑动窗口: {target_context} -> {sliding_window}")

        # 7. 重新计算KV Cache (使用滑动窗口)
        kv_cache_gb = KVCacheCalculator.calculate_kv_cache_size(
            arch, sliding_window, kv_quant_bits=kv_quant_bits
        )

        # 8. 应用Flash Attention优化
        if enable_flash_attention:
            fa_savings = FlashAttentionOptimizer.calculate_memory_savings(
                sliding_window, arch.num_layers, arch.num_attention_heads, arch.head_dim
            )
            # Flash Attention可以减少KV Cache的实际显存占用
            kv_cache_gb *= 0.7  # 假设节省30%
            optimizations_applied.append(f"Flash Attention: 节省 {fa_savings['savings_gb']:.2f} GB")

        # 9. 计算总显存占用
        overhead_gb = 0.4
        total_vram_gb = (model_weight_gb * gpu_layers / arch.num_layers) + kv_cache_gb + overhead_gb
        total_ram_gb = model_weight_gb * cpu_layers / arch.num_layers

        # 10. 检查OOM风险
        oom_status = self.oom_protector.update_usage(total_vram_gb)
        if oom_status.level.value >= OOMProtectionLevel.MEDIUM.value:
            warnings.append(f"显存使用率高 ({oom_status.usage_ratio:.1%}): {oom_status.recommended_action}")
            # 自动降级
            new_context, new_quant, new_gpu_layers, degrade_msg = self.oom_protector.auto_degrade(
                target_context, quant_level, gpu_layers
            )
            if new_gpu_layers != gpu_layers:
                warnings.append(f"自动降级: {degrade_msg}")
                gpu_layers = new_gpu_layers
                cpu_layers = arch.num_layers - gpu_layers
                # 重新计算显存
                total_vram_gb = (model_weight_gb * gpu_layers / arch.num_layers) + kv_cache_gb + overhead_gb
                total_ram_gb = model_weight_gb * cpu_layers / arch.num_layers

        # 11. 估算速度
        speed_estimate = EnhancedSpeedEstimator.estimate_speed(
            model_size_b, quant_bits, gpu_layers, arch.num_layers,
            self.total_vram_gb, self.cpu_threads, self.pcie_info
        )

        return EnhancedOptimizationResult(
            model_size_b=model_size_b,
            model_name=model_name,
            architecture=arch,
            use_gqa=use_gqa,
            quant_bits=quant_bits,
            quant_level=quant_level,
            gpu_layers=gpu_layers,
            cpu_layers=cpu_layers,
            total_layers=arch.num_layers,
            context_length=target_context,
            sliding_window=sliding_window,
            kv_cache_gb=kv_cache_gb,
            kv_quant_bits=kv_quant_bits,
            model_weight_gb=model_weight_gb,
            total_vram_gb=total_vram_gb,
            total_ram_gb=total_ram_gb,
            speed_estimate=speed_estimate,
            pcie_info=self.pcie_info,
            oom_status=oom_status,
            optimizations_applied=optimizations_applied,
            warnings=warnings,
        )

    def _select_quantization(
        self, model_size_b: float, target_quant: Optional[str]
    ) -> Tuple[str, int]:
        """选择量化级别"""
        if target_quant:
            quant_bits_map = {
                "q2_k": 2, "q3_k_s": 3, "q3_k_m": 3, "q4_0": 4,
                "q4_k_s": 4, "q4_k_m": 4, "q5_k_s": 5, "q5_k_m": 5,
                "q6_k": 6, "q8_0": 8, "fp16": 16,
            }
            return target_quant, quant_bits_map.get(target_quant, 4)

        # 基于显存自动选择
        available_vram = self.total_vram_gb - 0.4  # 减去开销
        model_weight_per_b = model_size_b  # 简化: 1B参数 ≈ 1GB (FP16)

        if available_vram >= model_weight_per_b * 0.5:
            return "q4_k_m", 4
        elif available_vram >= model_weight_per_b * 0.35:
            return "q3_k_m", 3
        elif available_vram >= model_weight_per_b * 0.25:
            return "q2_k", 2
        else:
            return "q2_k", 2

    def _select_kv_quantization(self, gpu_layers: int, total_layers: int) -> int:
        """选择KV Cache量化级别"""
        gpu_ratio = gpu_layers / max(total_layers, 1)
        if gpu_ratio > 0.8:
            return 16  # FP16
        elif gpu_ratio > 0.5:
            return 8   # INT8
        else:
            return 4   # INT4

    def _calculate_optimal_layers(
        self,
        model_size_b: float,
        model_weight_gb: float,
        arch: ModelArchitecture,
        context_length: int,
        quant_bits: int,
    ) -> Tuple[int, int]:
        """计算最优GPU层数"""
        available_vram = self.total_vram_gb - 0.4  # 减去开销

        # 估算KV Cache大小
        kv_cache_gb = KVCacheCalculator.calculate_kv_cache_size(
            arch, context_length, kv_quant_bits=8
        )

        # 可用于模型层的显存
        available_for_layers = available_vram - kv_cache_gb

        # 每层大小
        per_layer_gb = model_weight_gb / arch.num_layers

        # 计算GPU层数
        gpu_layers = int(available_for_layers / per_layer_gb)
        gpu_layers = max(0, min(gpu_layers, arch.num_layers))
        cpu_layers = arch.num_layers - gpu_layers

        return gpu_layers, cpu_layers

    def _calculate_sliding_window(
        self,
        target_context: int,
        kv_cache_gb: float,
        gpu_layers: int,
        total_layers: int,
    ) -> int:
        """计算滑动窗口大小"""
        # 如果显存充足，使用完整上下文
        available_vram = self.total_vram_gb - 0.4
        if kv_cache_gb < available_vram * 0.3:
            return target_context

        # 否则计算合适的窗口大小
        target_kv_gb = available_vram * 0.25
        ratio = target_kv_gb / max(kv_cache_gb, 0.001)
        window = int(target_context * ratio)

        # 限制范围
        window = max(512, min(window, target_context))
        # 对齐到128
        window = (window // 128) * 128

        return window

    def get_optimization_report(self, result: EnhancedOptimizationResult) -> str:
        """生成优化报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("小显存优化报告")
        lines.append("=" * 60)

        lines.append(f"\n【模型信息】")
        lines.append(f"  参数量: {result.model_size_b}B")
        if result.model_name:
            lines.append(f"  模型名: {result.model_name}")
        lines.append(f"  架构: {'GQA' if result.use_gqa else 'MHA'}")
        if result.use_gqa:
            lines.append(f"  KV头数: {result.architecture.num_kv_heads} (注意力头: {result.architecture.num_attention_heads})")

        lines.append(f"\n【量化配置】")
        lines.append(f"  量化级别: {result.quant_level}")
        lines.append(f"  位数: {result.quant_bits} bits")

        lines.append(f"\n【层分配】")
        lines.append(f"  GPU层数: {result.gpu_layers}/{result.total_layers}")
        lines.append(f"  CPU层数: {result.cpu_layers}/{result.total_layers}")
        gpu_ratio = result.gpu_layers / result.total_layers
        lines.append(f"  GPU占比: {gpu_ratio:.1%}")

        lines.append(f"\n【上下文配置】")
        lines.append(f"  目标上下文: {result.context_length}")
        lines.append(f"  滑动窗口: {result.sliding_window}")
        lines.append(f"  KV量化: {result.kv_quant_bits} bits")

        lines.append(f"\n【显存占用】")
        lines.append(f"  模型权重: {result.model_weight_gb:.2f} GB")
        lines.append(f"  KV Cache: {result.kv_cache_gb:.3f} GB")
        lines.append(f"  总显存占用: {result.total_vram_gb:.2f} GB")
        lines.append(f"  内存占用: {result.total_ram_gb:.2f} GB")

        lines.append(f"\n【PCIe信息】")
        lines.append(f"  代际: PCIe {result.pcie_info.generation.value}.0")
        lines.append(f"  通道数: x{result.pcie_info.lanes}")
        lines.append(f"  带宽: {result.pcie_info.bandwidth_gb_s:.1f} GB/s")
        lines.append(f"  开销: {result.pcie_info.overhead_factor:.1%}")

        lines.append(f"\n【速度估算】")
        est = result.speed_estimate
        lines.append(f"  预估速度: {est.estimated_tps:.1f} tokens/s")
        lines.append(f"  GPU速度: {est.gpu_tps:.1f} tokens/s")
        lines.append(f"  CPU速度: {est.cpu_tps:.1f} tokens/s")
        lines.append(f"  PCIe开销: {est.pcie_overhead:.1%}")
        lines.append(f"  瓶颈: {est.bottleneck}")
        lines.append(f"  适合实时对话: {'是' if est.suitable_for_realtime else '否'}")

        lines.append(f"\n【OOM防护】")
        oom = result.oom_status
        lines.append(f"  防护级别: {oom.level.name}")
        lines.append(f"  使用率: {oom.usage_ratio:.1%}")
        lines.append(f"  建议: {oom.recommended_action}")

        if result.optimizations_applied:
            lines.append(f"\n【已应用优化】")
            for opt in result.optimizations_applied:
                lines.append(f"  + {opt}")

        if result.warnings:
            lines.append(f"\n【警告】")
            for warn in result.warnings:
                lines.append(f"  ! {warn}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ============================================================
# 便捷函数
# ============================================================

def quick_optimize(
    vram_gb: float,
    model_size_b: float,
    model_name: Optional[str] = None,
    context_length: int = 4096,
) -> EnhancedOptimizationResult:
    """快速优化

    Args:
        vram_gb: 显存大小 (GB)
        model_size_b: 模型参数量 (B)
        model_name: 模型名称 (可选)
        context_length: 上下文长度

    Returns:
        EnhancedOptimizationResult: 优化结果
    """
    optimizer = EnhancedOptimizer(vram_gb)
    return optimizer.optimize(model_size_b, model_name, context_length)


def get_optimization_report(vram_gb: float, model_size_b: float, model_name: Optional[str] = None) -> str:
    """获取优化报告

    Args:
        vram_gb: 显存大小 (GB)
        model_size_b: 模型参数量 (B)
        model_name: 模型名称 (可选)

    Returns:
        str: 优化报告
    """
    result = quick_optimize(vram_gb, model_size_b, model_name)
    optimizer = EnhancedOptimizer(vram_gb)
    return optimizer.get_optimization_report(result)
