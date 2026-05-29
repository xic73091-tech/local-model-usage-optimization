"""
智能优化调度模块

提供基于任务类型、优先级、资源约束的模型选择与请求调度。
支持量化级别自动选择、模型热切换和动态层加载。

子模块：
- scheduler:   推理请求调度与优先级队列
- offloader:   GPU-CPU 智能卸载策略
- kv_cache:    KV Cache 优化（分页注意力、前缀共享）
- dynamic_loader: 动态层加载与 LRU 淘汰
- memory_optimizer: 综合内存优化
- quantizer:   量化策略管理与智能推荐
- ultra_quantizer: 极低精度量化（Q2_K/IQ2_XXS 等）
- vram_optimizer:   小显存深度优化
- multi_vram_optimizer: 多显存配置对比优化
"""

# ---------------------------------------------------------------------------
# scheduler - 推理调度
# ---------------------------------------------------------------------------
from .scheduler import (
    InferenceRequest,
    InferenceResult,
    InferenceScheduler,
    ModelProfile,
    Priority,
    QuantizationLevel,
    SchedulerStats,
    TaskType,
)

# ---------------------------------------------------------------------------
# offloader - 卸载策略
# ---------------------------------------------------------------------------
from .offloader import (
    MemoryEstimate,
    ModelOffloader,
    OffloadConfig,
    OffloadReport,
    OffloadStrategy as OffloadStrategyOffloader,
    PerformanceEstimate,
)

# ---------------------------------------------------------------------------
# kv_cache - KV Cache 优化
# ---------------------------------------------------------------------------
from .kv_cache import (
    CacheEntry,
    CacheStats,
    EvictionPolicy,
    KVCacheConfig,
    KVCacheOptimizer,
    PagedAttentionManager,
    PrefixShareManager,
)

# ---------------------------------------------------------------------------
# dynamic_loader - 动态层加载
# ---------------------------------------------------------------------------
try:
    from .dynamic_loader import (
        DynamicConfig as DynamicLoaderConfig,
        DynamicLayerLoader,
        EvictionStrategy,
        LayerLocation,
        LayerManager,
        LayerStats,
        LayerState,
    )
    _DYNAMIC_LOADER_AVAILABLE = True
except (ImportError, AttributeError):
    _DYNAMIC_LOADER_AVAILABLE = False

# ---------------------------------------------------------------------------
# memory_optimizer - 综合内存优化
# ---------------------------------------------------------------------------
from .memory_optimizer import (
    DynamicConfig as MemoryDynamicConfig,
    HardwareProfile,
    MemoryOptimizer,
    OptimizationProfile,
    OptimizationResult,
    QuantizationConfig,
)

# ---------------------------------------------------------------------------
# quantizer - 量化策略管理
# ---------------------------------------------------------------------------
from .quantizer import (
    QuantFormat,
    QuantizationManager,
    QuantizationProfile,
)

# ---------------------------------------------------------------------------
# ultra_quantizer - 极低精度量化
# ---------------------------------------------------------------------------
from .ultra_quantizer import (
    QuantRecommendation,
    UltraQuantLevel,
    UltraQuantProfile,
    UltraQuantizer,
    VRAMEstimate as UltraVRAMEstimate,
)

# ---------------------------------------------------------------------------
# vram_optimizer - 小显存优化
# ---------------------------------------------------------------------------
from .vram_optimizer import (
    LayerAllocation,
    OptimizationTarget,
    VRAMBudget,
    VRAMOptimizer,
    OffloadStrategy as OffloadStrategyVRAM,
)

# ---------------------------------------------------------------------------
# multi_vram_optimizer - 多显存配置对比
# ---------------------------------------------------------------------------
from .multi_vram_optimizer import (
    ComparisonMatrix,
    MultiVRAMOptimizer,
    VRAMProfile,
)

# ---------------------------------------------------------------------------
# comprehensive_optimizer - 全方位优化
# ---------------------------------------------------------------------------
from .comprehensive_optimizer import (
    ComprehensiveConfig,
    ComprehensiveOptimizer,
    OptimizationLevel,
    create_comprehensive_optimizer,
    quick_optimization_analysis,
)

# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------
__all__ = [
    # scheduler
    "InferenceScheduler",
    "InferenceRequest",
    "InferenceResult",
    "ModelProfile",
    "SchedulerStats",
    "TaskType",
    "Priority",
    "QuantizationLevel",
    # offloader
    "ModelOffloader",
    "OffloadConfig",
    "OffloadReport",
    "MemoryEstimate",
    "PerformanceEstimate",
    # kv_cache
    "KVCacheConfig",
    "KVCacheOptimizer",
    "CacheEntry",
    "CacheStats",
    "EvictionPolicy",
    "PagedAttentionManager",
    "PrefixShareManager",
    # memory_optimizer
    "MemoryOptimizer",
    "OptimizationProfile",
    "HardwareProfile",
    "OptimizationResult",
    "QuantizationConfig",
    "MemoryDynamicConfig",
    # quantizer
    "QuantizationManager",
    "QuantFormat",
    "QuantizationProfile",
    # ultra_quantizer
    "UltraQuantizer",
    "UltraQuantLevel",
    "UltraQuantProfile",
    "QuantRecommendation",
    "UltraVRAMEstimate",
    # vram_optimizer
    "VRAMOptimizer",
    "OptimizationTarget",
    "VRAMBudget",
    "LayerAllocation",
    # multi_vram_optimizer
    "MultiVRAMOptimizer",
    "VRAMProfile",
    "ComparisonMatrix",
    # comprehensive_optimizer
    "ComprehensiveOptimizer",
    "ComprehensiveConfig",
    "OptimizationLevel",
    "create_comprehensive_optimizer",
    "quick_optimization_analysis",
]

# dynamic_loader 可选导出
if _DYNAMIC_LOADER_AVAILABLE:
    __all__.extend([
        "DynamicLayerLoader",
        "DynamicLoaderConfig",
        "EvictionStrategy",
        "LayerLocation",
        "LayerManager",
        "LayerStats",
        "LayerState",
    ])
