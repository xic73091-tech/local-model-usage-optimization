"""
全方位优化协调器

整合所有优化模块，提供统一的优化接口。

新增优化维度:
1. 稀疏注意力 - 只计算重要token的注意力
2. 分块预填充 - 将长prompt分块处理
3. 激活检查点 - 用计算换内存
4. 连续批处理 - 动态批处理提升吞吐
5. 算子融合 - 减少内存访问
6. Token合并 - 减少序列长度

核心目标: 小显存运行大模型更快更稳
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================

class OptimizationDimension(Enum):
    """优化维度"""
    QUANTIZATION = "quantization"       # 量化
    OFFLOADING = "offloading"           # 卸载
    KV_CACHE = "kv_cache"              # KV Cache
    DYNAMIC_LOADING = "dynamic_loading" # 动态加载
    PIPELINE = "pipeline"              # 流水线并行
    SPECULATIVE = "speculative"        # 推测解码
    ATTENTION = "attention"            # 注意力优化
    PREFILL = "prefill"                # 预填充优化
    ACTIVATION = "activation"          # 激活优化
    BATCHING = "batching"              # 批处理优化
    COMPILATION = "compilation"        # 编译优化


class OptimizationLevel(Enum):
    """优化级别"""
    CONSERVATIVE = 0   # 保守: 只使用稳定优化
    BALANCED = 1       # 平衡: 稳定+实验性优化
    AGGRESSIVE = 2     # 激进: 所有优化
    CUSTOM = 3         # 自定义


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class AttentionOptConfig:
    """注意力优化配置"""
    # 稀疏注意力
    sparse_attention: bool = True
    sparse_ratio: float = 0.5  # 只保留50%的token

    # 滑动窗口
    sliding_window: bool = True
    window_size: int = 2048

    # Flash Attention
    flash_attention: bool = True

    # 分组查询注意力 (GQA)
    gqa_enabled: bool = True
    num_kv_heads: int = 8  # KV头数 (少于Q头数)


@dataclass
class PrefillOptConfig:
    """预填充优化配置"""
    # 分块预填充
    chunked_prefill: bool = True
    chunk_size: int = 512

    # Token合并
    token_merging: bool = True
    merge_ratio: float = 0.25  # 合并25%的token

    # 前缀缓存
    prefix_caching: bool = True


@dataclass
class ActivationOptConfig:
    """激活优化配置"""
    # 激活检查点
    activation_checkpointing: bool = True
    checkpoint_ratio: float = 0.5  # 检查50%的层

    # 激活重计算
    selective_recompute: bool = True
    recompute_attention: bool = True

    # 激活量化
    activation_quantization: bool = True
    activation_bits: int = 8


@dataclass
class BatchingOptConfig:
    """批处理优化配置"""
    # 连续批处理
    continuous_batching: bool = True
    max_batch_size: int = 8

    # 动态批处理
    dynamic_batching: bool = True
    max_wait_ms: float = 100.0

    # 优先级调度
    priority_scheduling: bool = True


@dataclass
class CompilationOptConfig:
    """编译优化配置"""
    # torch.compile
    torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # default, reduce-overhead, max-autotune

    # ONNX优化
    onnx_optimization: bool = False

    # 图优化
    graph_optimization: bool = True
    fusion_enabled: bool = True


@dataclass
class ComprehensiveConfig:
    """全方位优化配置"""
    # 优化级别
    level: OptimizationLevel = OptimizationLevel.BALANCED

    # 各维度配置
    attention: AttentionOptConfig = field(default_factory=AttentionOptConfig)
    prefill: PrefillOptConfig = field(default_factory=PrefillOptConfig)
    activation: ActivationOptConfig = field(default_factory=ActivationOptConfig)
    batching: BatchingOptConfig = field(default_factory=BatchingOptConfig)
    compilation: CompilationOptConfig = field(default_factory=CompilationOptConfig)

    # 硬件约束
    vram_budget_gb: float = 6.0
    target_speed_tps: float = 10.0
    target_quality: float = 0.7

    def apply_level(self, level: OptimizationLevel) -> None:
        """应用优化级别预设"""
        self.level = level

        if level == OptimizationLevel.CONSERVATIVE:
            # 保守: 只使用稳定优化
            self.attention.sparse_attention = False
            self.attention.sliding_window = False
            self.prefill.token_merging = False
            self.activation.activation_checkpointing = False
            self.activation.selective_recompute = False
            self.compilation.torch_compile = False

        elif level == OptimizationLevel.BALANCED:
            # 平衡: 稳定+部分实验性
            self.attention.sparse_attention = True
            self.attention.sparse_ratio = 0.7  # 保留70%
            self.prefill.token_merging = True
            self.prefill.merge_ratio = 0.15
            self.activation.activation_checkpointing = True
            self.activation.checkpoint_ratio = 0.3
            self.compilation.torch_compile = True
            self.compilation.compile_mode = "reduce-overhead"

        elif level == OptimizationLevel.AGGRESSIVE:
            # 激进: 所有优化
            self.attention.sparse_attention = True
            self.attention.sparse_ratio = 0.3  # 只保留30%
            self.attention.sliding_window = True
            self.attention.window_size = 1024
            self.prefill.token_merging = True
            self.prefill.merge_ratio = 0.3
            self.activation.activation_checkpointing = True
            self.activation.checkpoint_ratio = 0.5
            self.activation.selective_recompute = True
            self.activation.activation_quantization = True
            self.activation.activation_bits = 4
            self.compilation.torch_compile = True
            self.compilation.compile_mode = "max-autotune"


# ============================================================
# 优化效果估算
# ============================================================

@dataclass
class OptimizationEstimate:
    """优化效果估算"""
    # 内存节省
    memory_saved_gb: float = 0.0
    memory_saved_ratio: float = 0.0

    # 速度提升
    speedup_ratio: float = 1.0
    estimated_tps: float = 0.0

    # 质量影响
    quality_impact: float = 0.0  # 负值表示质量下降

    # 各维度贡献
    dimension_contributions: Dict[str, float] = field(default_factory=dict)

    # 风险评估
    stability_risk: float = 0.0  # 0-1, 越高越不稳定
    compatibility_risk: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_saved_gb": round(self.memory_saved_gb, 2),
            "memory_saved_ratio": round(self.memory_saved_ratio, 3),
            "speedup_ratio": round(self.speedup_ratio, 2),
            "estimated_tps": round(self.estimated_tps, 1),
            "quality_impact": round(self.quality_impact, 3),
            "stability_risk": round(self.stability_risk, 3),
            "compatibility_risk": round(self.compatibility_risk, 3),
            "dimension_contributions": {
                k: round(v, 3) for k, v in self.dimension_contributions.items()
            },
        }


# ============================================================
# 稀疏注意力优化器
# ============================================================

class SparseAttentionOptimizer:
    """稀疏注意力优化器

    只计算重要token的注意力，减少计算量和内存。
    """

    def __init__(self, sparse_ratio: float = 0.5):
        """初始化稀疏注意力优化器

        Args:
            sparse_ratio: 保留的token比例 (0.5 = 保留50%)
        """
        self.sparse_ratio = sparse_ratio

        # 统计
        self._total_tokens = 0
        self._selected_tokens = 0

    def select_important_tokens(
        self,
        attention_scores: Any,
        seq_len: int,
    ) -> List[int]:
        """选择重要的token

        基于注意力分数选择最重要的token。

        Args:
            attention_scores: 注意力分数
            seq_len: 序列长度

        Returns:
            List[int]: 选中的token索引
        """
        # 计算要保留的token数
        keep_count = max(1, int(seq_len * self.sparse_ratio))

        # 模拟选择过程
        # 实际实现会基于注意力分数排序
        selected = list(range(keep_count))

        self._total_tokens += seq_len
        self._selected_tokens += keep_count

        return selected

    def get_savings(self) -> Dict[str, Any]:
        """获取节省信息"""
        if self._total_tokens == 0:
            return {"total_tokens": 0, "selected_tokens": 0, "savings_ratio": 0.0}

        return {
            "total_tokens": self._total_tokens,
            "selected_tokens": self._selected_tokens,
            "savings_ratio": 1.0 - (self._selected_tokens / self._total_tokens),
        }


# ============================================================
# 分块预填充优化器
# ============================================================

class ChunkedPrefillOptimizer:
    """分块预填充优化器

    将长prompt分块处理，减少峰值内存。
    """

    def __init__(self, chunk_size: int = 512):
        """初始化分块预填充优化器

        Args:
            chunk_size: 分块大小
        """
        self.chunk_size = chunk_size

        # 统计
        self._total_prefills = 0
        self._total_chunks = 0

    def split_prompt(self, prompt_length: int) -> List[Tuple[int, int]]:
        """将prompt分成多个块

        Args:
            prompt_length: prompt长度

        Returns:
            List[Tuple[int, int]]: 每个块的(start, end)索引
        """
        chunks = []
        for start in range(0, prompt_length, self.chunk_size):
            end = min(start + self.chunk_size, prompt_length)
            chunks.append((start, end))

        self._total_prefills += 1
        self._total_chunks += len(chunks)

        return chunks

    def estimate_memory_savings(self, prompt_length: int) -> float:
        """估算内存节省比例

        Args:
            prompt_length: prompt长度

        Returns:
            float: 节省比例 (0-1)
        """
        if prompt_length <= self.chunk_size:
            return 0.0

        # 分块后峰值内存 = 单块大小 / 原始大小
        return 1.0 - (self.chunk_size / prompt_length)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_prefills": self._total_prefills,
            "total_chunks": self._total_chunks,
            "avg_chunks_per_prefill": (
                self._total_chunks / max(self._total_prefills, 1)
            ),
        }


# ============================================================
# 激活检查点优化器
# ============================================================

class ActivationCheckpointOptimizer:
    """激活检查点优化器

    通过重新计算激活值来节省内存。
    """

    def __init__(self, checkpoint_ratio: float = 0.5):
        """初始化激活检查点优化器

        Args:
            checkpoint_ratio: 检查点比例 (0.5 = 50%的层使用检查点)
        """
        self.checkpoint_ratio = checkpoint_ratio

        # 统计
        self._total_layers = 0
        self._checkpointed_layers = 0

    def select_checkpoint_layers(self, num_layers: int) -> List[int]:
        """选择使用检查点的层

        Args:
            num_layers: 总层数

        Returns:
            List[int]: 使用检查点的层索引
        """
        checkpoint_count = max(1, int(num_layers * self.checkpoint_ratio))

        # 均匀分布检查点
        step = num_layers / checkpoint_count
        selected = [int(i * step) for i in range(checkpoint_count)]

        self._total_layers += num_layers
        self._checkpointed_layers += len(selected)

        return selected

    def estimate_memory_savings(self, num_layers: int) -> float:
        """估算内存节省比例

        Args:
            num_layers: 总层数

        Returns:
            float: 节省比例 (0-1)
        """
        # 检查点可以节省约 checkpoint_ratio 的激活内存
        # 但会增加约 checkpoint_ratio/2 的计算时间
        return self.checkpoint_ratio * 0.4  # 假设激活占总内存的40%

    def estimate_compute_overhead(self) -> float:
        """估算计算开销比例

        Returns:
            float: 计算开销比例 (0-1)
        """
        # 检查点层需要重新计算，增加约 checkpoint_ratio/2 的计算
        return self.checkpoint_ratio * 0.3

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_layers": self._total_layers,
            "checkpointed_layers": self._checkpointed_layers,
            "checkpoint_ratio": self.checkpoint_ratio,
        }


# ============================================================
# Token合并优化器
# ============================================================

class TokenMergingOptimizer:
    """Token合并优化器

    合并相似的token，减少序列长度。
    """

    def __init__(self, merge_ratio: float = 0.25):
        """初始化Token合并优化器

        Args:
            merge_ratio: 合并比例 (0.25 = 合并25%的token)
        """
        self.merge_ratio = merge_ratio

        # 统计
        self._total_tokens = 0
        self._merged_tokens = 0

    def merge_tokens(
        self,
        token_embeddings: Any,
        seq_len: int,
    ) -> Tuple[Any, int]:
        """合并相似的token

        Args:
            token_embeddings: token嵌入
            seq_len: 序列长度

        Returns:
            Tuple: (合并后的嵌入, 新序列长度)
        """
        # 计算要合并的token数
        merge_count = max(0, int(seq_len * self.merge_ratio))
        new_seq_len = seq_len - merge_count

        self._total_tokens += seq_len
        self._merged_tokens += merge_count

        # 实际实现会基于相似度合并token
        # 这里返回模拟结果
        merged_embeddings = token_embeddings  # 简化

        return merged_embeddings, new_seq_len

    def estimate_speedup(self) -> float:
        """估算加速比

        Returns:
            float: 加速比
        """
        # 注意力计算复杂度 O(n^2)，减少序列长度可以显著加速
        remaining_ratio = 1.0 - self.merge_ratio
        # 加速比 = 1 / remaining_ratio^2
        return 1.0 / (remaining_ratio ** 2)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self._total_tokens == 0:
            return {"total_tokens": 0, "merged_tokens": 0, "merge_ratio": 0.0}

        return {
            "total_tokens": self._total_tokens,
            "merged_tokens": self._merged_tokens,
            "merge_ratio": self._merged_tokens / self._total_tokens,
        }


# ============================================================
# 连续批处理器
# ============================================================

class ContinuousBatcher:
    """连续批处理器

    动态合并请求，提升吞吐量。
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: float = 100.0,
    ):
        """初始化连续批处理器

        Args:
            max_batch_size: 最大批处理大小
            max_wait_ms: 最大等待时间 (毫秒)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # 请求队列
        self._pending_requests: List[Dict[str, Any]] = []
        self._batch_count = 0
        self._total_requests = 0

    def add_request(self, request: Dict[str, Any]) -> None:
        """添加请求到队列

        Args:
            request: 请求数据
        """
        self._pending_requests.append(request)
        self._total_requests += 1

    def get_next_batch(self) -> List[Dict[str, Any]]:
        """获取下一批请求

        Returns:
            List[Dict]: 批请求
        """
        if not self._pending_requests:
            return []

        # 取出最多 max_batch_size 个请求
        batch_size = min(self.max_batch_size, len(self._pending_requests))
        batch = self._pending_requests[:batch_size]
        self._pending_requests = self._pending_requests[batch_size:]

        self._batch_count += 1

        return batch

    def estimate_throughput_improvement(self) -> float:
        """估算吞吐量提升

        Returns:
            float: 提升比例
        """
        # 连续批处理可以显著提升吞吐量
        # 特别是对于短请求
        return 1.5 + (self.max_batch_size - 1) * 0.3

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self._total_requests,
            "batch_count": self._batch_count,
            "avg_batch_size": (
                self._total_requests / max(self._batch_count, 1)
            ),
            "pending_requests": len(self._pending_requests),
        }


# ============================================================
# 全方位优化协调器
# ============================================================

class ComprehensiveOptimizer:
    """全方位优化协调器

    整合所有优化维度，提供统一的优化接口。
    """

    def __init__(self, config: Optional[ComprehensiveConfig] = None):
        """初始化全方位优化协调器

        Args:
            config: 配置
        """
        self.config = config or ComprehensiveConfig()

        # 初始化各优化器
        self._sparse_attention = SparseAttentionOptimizer(
            self.config.attention.sparse_ratio
        )
        self._chunked_prefill = ChunkedPrefillOptimizer(
            self.config.prefill.chunk_size
        )
        self._activation_checkpoint = ActivationCheckpointOptimizer(
            self.config.activation.checkpoint_ratio
        )
        self._token_merging = TokenMergingOptimizer(
            self.config.prefill.merge_ratio
        )
        self._continuous_batcher = ContinuousBatcher(
            self.config.batching.max_batch_size,
            self.config.batching.max_wait_ms,
        )

        # 统计
        self._optimization_count = 0
        self._total_memory_saved_gb = 0.0
        self._total_speedup = 1.0

    def analyze_optimization_potential(
        self,
        model_size_b: float,
        vram_gb: float,
        seq_len: int = 2048,
        num_layers: int = 32,
    ) -> OptimizationEstimate:
        """分析优化潜力

        Args:
            model_size_b: 模型参数量 (B)
            vram_gb: 可用显存 (GB)
            seq_len: 序列长度
            num_layers: 层数

        Returns:
            OptimizationEstimate: 优化效果估算
        """
        estimate = OptimizationEstimate()

        # 基础内存占用 (FP16)
        base_memory_gb = model_size_b * 2  # 2 bytes per param

        # KV Cache内存 (FP16)
        kv_cache_gb = (seq_len * num_layers * 128 * 32 * 2 * 2) / (1024**3)

        # 各维度优化贡献
        contributions = {}

        # 1. 稀疏注意力
        if self.config.attention.sparse_attention:
            sparse_savings = kv_cache_gb * (1 - self.config.attention.sparse_ratio)
            estimate.memory_saved_gb += sparse_savings
            contributions["sparse_attention"] = sparse_savings

        # 2. 滑动窗口
        if self.config.attention.sliding_window:
            window_ratio = self.config.attention.window_size / seq_len
            window_savings = kv_cache_gb * (1 - window_ratio) * 0.5
            estimate.memory_saved_gb += window_savings
            contributions["sliding_window"] = window_savings

        # 3. Flash Attention
        if self.config.attention.flash_attention:
            # Flash Attention减少注意力内存，但不减少KV Cache
            flash_savings = 0.5  # 固定节省
            contributions["flash_attention"] = flash_savings

        # 4. GQA
        if self.config.attention.gqa_enabled:
            gqa_ratio = self.config.attention.num_kv_heads / 32
            gqa_savings = kv_cache_gb * (1 - gqa_ratio)
            estimate.memory_saved_gb += gqa_savings
            contributions["gqa"] = gqa_savings

        # 5. 分块预填充
        if self.config.prefill.chunked_prefill:
            prefill_savings = self._chunked_prefill.estimate_memory_savings(seq_len)
            contributions["chunked_prefill"] = prefill_savings * 0.3

        # 6. Token合并
        if self.config.prefill.token_merging:
            merge_speedup = self._token_merging.estimate_speedup()
            estimate.speedup_ratio *= merge_speedup
            contributions["token_merging"] = merge_speedup - 1.0

        # 7. 激活检查点
        if self.config.activation.activation_checkpointing:
            checkpoint_savings = self._activation_checkpoint.estimate_memory_savings(num_layers)
            checkpoint_memory = base_memory_gb * checkpoint_savings
            estimate.memory_saved_gb += checkpoint_memory
            contributions["activation_checkpoint"] = checkpoint_memory

            # 计算开销
            compute_overhead = self._activation_checkpoint.estimate_compute_overhead()
            estimate.speedup_ratio *= (1 - compute_overhead)

        # 8. 连续批处理
        if self.config.batching.continuous_batching:
            batch_speedup = self._continuous_batcher.estimate_throughput_improvement()
            contributions["continuous_batching"] = batch_speedup - 1.0

        # 计算总内存
        total_memory = base_memory_gb + kv_cache_gb
        estimate.memory_saved_ratio = estimate.memory_saved_gb / total_memory

        # 估算速度
        estimate.estimated_tps = 20.0 * estimate.speedup_ratio  # 假设基线20 tps

        # 质量影响
        quality_impact = 0.0
        if self.config.attention.sparse_attention:
            quality_impact -= (1 - self.config.attention.sparse_ratio) * 0.1
        if self.config.prefill.token_merging:
            quality_impact -= self.config.prefill.merge_ratio * 0.05
        if self.config.activation.activation_quantization:
            quality_impact -= (16 - self.config.activation.activation_bits) / 16 * 0.02
        estimate.quality_impact = quality_impact

        # 风险评估
        if self.config.level == OptimizationLevel.AGGRESSIVE:
            estimate.stability_risk = 0.3
            estimate.compatibility_risk = 0.2
        elif self.config.level == OptimizationLevel.BALANCED:
            estimate.stability_risk = 0.1
            estimate.compatibility_risk = 0.05
        else:
            estimate.stability_risk = 0.02
            estimate.compatibility_risk = 0.01

        estimate.dimension_contributions = contributions

        return estimate

    def get_optimization_plan(
        self,
        model_size_b: float,
        vram_gb: float,
    ) -> Dict[str, Any]:
        """获取优化计划

        Args:
            model_size_b: 模型参数量 (B)
            vram_gb: 可用显存 (GB)

        Returns:
            Dict: 优化计划
        """
        plan = {
            "model_size_b": model_size_b,
            "vram_gb": vram_gb,
            "optimization_level": self.config.level.value,
            "enabled_optimizations": [],
            "disabled_optimizations": [],
            "recommendations": [],
        }

        # 检查哪些优化应该启用
        if self.config.attention.sparse_attention:
            plan["enabled_optimizations"].append({
                "name": "稀疏注意力",
                "config": f"保留{self.config.attention.sparse_ratio*100:.0f}%的token",
                "expected_savings": "30-50% KV Cache",
            })

        if self.config.attention.sliding_window:
            plan["enabled_optimizations"].append({
                "name": "滑动窗口注意力",
                "config": f"窗口大小={self.config.attention.window_size}",
                "expected_savings": "减少长序列内存",
            })

        if self.config.attention.flash_attention:
            plan["enabled_optimizations"].append({
                "name": "Flash Attention",
                "config": "启用",
                "expected_savings": "减少90%+注意力内存",
            })

        if self.config.attention.gqa_enabled:
            plan["enabled_optimizations"].append({
                "name": "分组查询注意力 (GQA)",
                "config": f"KV头数={self.config.attention.num_kv_heads}",
                "expected_savings": "减少75% KV Cache",
            })

        if self.config.prefill.chunked_prefill:
            plan["enabled_optimizations"].append({
                "name": "分块预填充",
                "config": f"块大小={self.config.prefill.chunk_size}",
                "expected_savings": "减少峰值内存",
            })

        if self.config.prefill.token_merging:
            plan["enabled_optimizations"].append({
                "name": "Token合并",
                "config": f"合并比例={self.config.prefill.merge_ratio*100:.0f}%",
                "expected_savings": f"加速{self._token_merging.estimate_speedup():.1f}x",
            })

        if self.config.activation.activation_checkpointing:
            plan["enabled_optimizations"].append({
                "name": "激活检查点",
                "config": f"检查{self.config.activation.checkpoint_ratio*100:.0f}%的层",
                "expected_savings": "减少激活内存",
            })

        if self.config.batching.continuous_batching:
            plan["enabled_optimizations"].append({
                "name": "连续批处理",
                "config": f"最大批次={self.config.batching.max_batch_size}",
                "expected_savings": f"吞吐提升{self._continuous_batcher.estimate_throughput_improvement():.1f}x",
            })

        # 生成建议
        if vram_gb <= 4:
            plan["recommendations"].append(
                "显存极小，建议使用AGGRESSIVE级别优化，配合Q2_K/Q3_K_M量化"
            )
        elif vram_gb <= 6:
            plan["recommendations"].append(
                "显存较小，建议使用BALANCED级别优化，配合Q4_K_M量化"
            )
        elif vram_gb <= 8:
            plan["recommendations"].append(
                "显存中等，建议使用BALANCED级别优化，配合Q4_K_M/Q5_K_M量化"
            )
        else:
            plan["recommendations"].append(
                "显存充足，可使用CONSERVATIVE级别优化，配合高质量量化"
            )

        return plan

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "optimization_count": self._optimization_count,
            "total_memory_saved_gb": round(self._total_memory_saved_gb, 2),
            "total_speedup": round(self._total_speedup, 2),
            "sparse_attention": self._sparse_attention.get_savings(),
            "chunked_prefill": self._chunked_prefill.get_stats(),
            "activation_checkpoint": self._activation_checkpoint.get_stats(),
            "token_merging": self._token_merging.get_stats(),
            "continuous_batcher": self._continuous_batcher.get_stats(),
        }


# ============================================================
# 便捷函数
# ============================================================

def create_comprehensive_optimizer(
    level: str = "balanced",
    vram_gb: float = 6.0,
) -> ComprehensiveOptimizer:
    """创建全方位优化器

    Args:
        level: 优化级别 ("conservative", "balanced", "aggressive")
        vram_gb: 可用显存 (GB)

    Returns:
        ComprehensiveOptimizer: 优化器实例
    """
    config = ComprehensiveConfig()
    config.vram_budget_gb = vram_gb

    level_enum = OptimizationLevel[level.upper()]
    config.apply_level(level_enum)

    return ComprehensiveOptimizer(config)


def quick_optimization_analysis(
    model_size_b: float,
    vram_gb: float,
    level: str = "balanced",
) -> Dict[str, Any]:
    """快速优化分析

    Args:
        model_size_b: 模型参数量 (B)
        vram_gb: 可用显存 (GB)
        level: 优化级别

    Returns:
        Dict: 分析结果
    """
    optimizer = create_comprehensive_optimizer(level, vram_gb)

    # 分析优化潜力
    estimate = optimizer.analyze_optimization_potential(
        model_size_b=model_size_b,
        vram_gb=vram_gb,
    )

    # 获取优化计划
    plan = optimizer.get_optimization_plan(
        model_size_b=model_size_b,
        vram_gb=vram_gb,
    )

    return {
        "estimate": estimate.to_dict(),
        "plan": plan,
        "stats": optimizer.get_stats(),
    }
