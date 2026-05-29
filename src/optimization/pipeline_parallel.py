"""
层间流水线并行模块

实现GPU计算时CPU预加载下一层，隐藏数据传输延迟。

原理:
- 当第N层在GPU上计算时，第N+1层从CPU预加载到GPU
- 使用双缓冲区实现计算和传输的重叠
- 预测接下来需要的层，提前加载

优势:
- 隐藏CPU-GPU数据传输延迟
- 提升混合推理的吞吐量
- 特别适合小显存场景（需要频繁在CPU和GPU间切换）

预期提升: 20-40%的混合推理速度
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 配置和数据类
# ============================================================

class PipelineStrategy(Enum):
    """流水线策略"""
    NONE = "none"              # 不使用流水线
    SINGLE_BUFFER = "single"   # 单缓冲
    DOUBLE_BUFFER = "double"   # 双缓冲
    TRIPLE_BUFFER = "triple"   # 三缓冲


@dataclass
class PipelineConfig:
    """流水线并行配置"""
    # 流水线策略
    strategy: PipelineStrategy = PipelineStrategy.DOUBLE_BUFFER

    # 预加载层数
    prefetch_layers: int = 2

    # 是否启用预测性预加载
    enable_predictive_prefetch: bool = True

    # 预测窗口大小
    prediction_window: int = 5

    # 缓冲区大小（MB）
    buffer_size_mb: int = 512

    # 是否启用异步传输
    enable_async_transfer: bool = True

    # 传输超时（秒）
    transfer_timeout: float = 5.0


@dataclass
class LayerBuffer:
    """层缓冲区"""
    layer_idx: int
    data: Any
    size_mb: float
    is_ready: bool = False
    is_transferring: bool = False
    transfer_start: float = 0.0


@dataclass
class PipelineStats:
    """流水线统计"""
    total_layers_processed: int = 0
    total_prefetch_hits: int = 0
    total_prefetch_misses: int = 0
    total_transfer_time_ms: float = 0.0
    total_compute_time_ms: float = 0.0
    avg_overlap_ratio: float = 0.0


# ============================================================
# 层访问模式预测器
# ============================================================

class AccessPatternPredictor:
    """层访问模式预测器

    基于历史访问模式预测接下来需要的层。
    """

    def __init__(self, window_size: int = 10):
        """初始化访问模式预测器

        Args:
            window_size: 窗口大小
        """
        self.window_size = window_size
        self._access_history: deque = deque(maxlen=window_size * 2)
        self._pattern_cache: Dict[Tuple[int, ...], List[int]] = {}

    def record_access(self, layer_idx: int) -> None:
        """记录层访问

        Args:
            layer_idx: 层索引
        """
        self._access_history.append(layer_idx)

        # 更新模式缓存
        if len(self._access_history) >= 3:
            # 使用最近的3个访问作为模式
            pattern = tuple(list(self._access_history)[-3:])
            if pattern not in self._pattern_cache:
                self._pattern_cache[pattern] = []
            self._pattern_cache[pattern].append(layer_idx)

    def predict_next(self, count: int = 2) -> List[int]:
        """预测接下来需要的层

        Args:
            count: 预测数量

        Returns:
            List[int]: 预测的层索引列表
        """
        if len(self._access_history) < 2:
            return []

        # 使用最近的访问模式
        recent = tuple(list(self._access_history)[-2:])

        # 查找匹配的模式
        predictions = []
        for pattern, next_layers in self._pattern_cache.items():
            if pattern[:2] == recent:
                # 统计最可能的下一层
                from collections import Counter
                counter = Counter(next_layers)
                most_common = counter.most_common(count)
                predictions.extend([layer for layer, _ in most_common])

        # 如果没有匹配的模式，使用简单的线性预测
        if not predictions:
            last_layer = self._access_history[-1]
            # 假设层是顺序访问的
            predictions = [last_layer + i + 1 for i in range(count)]

        return predictions[:count]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "history_size": len(self._access_history),
            "pattern_count": len(self._pattern_cache),
        }


# ============================================================
# 流水线并行管理器
# ============================================================

class PipelineParallelManager:
    """流水线并行管理器

    管理层的预加载和计算重叠。
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """初始化流水线并行管理器

        Args:
            config: 配置
        """
        self.config = config or PipelineConfig()

        # 初始化预测器
        self._predictor = AccessPatternPredictor(
            window_size=self.config.prediction_window
        )

        # 缓冲区
        self._buffers: Dict[int, LayerBuffer] = {}
        self._buffer_queue: asyncio.Queue = asyncio.Queue(maxsize=3)

        # 统计
        self._stats = PipelineStats()

        # 锁
        self._buffer_lock = asyncio.Lock()
        self._transfer_lock = asyncio.Lock()

        # 当前计算的层
        self._current_layer: Optional[int] = None

        # 预加载任务
        self._prefetch_tasks: Dict[int, asyncio.Task] = {}

    @property
    def stats(self) -> PipelineStats:
        """获取统计信息"""
        return self._stats

    async def prepare_layer(
        self,
        layer_idx: int,
        load_fn: Callable[[int], Any],
    ) -> Optional[LayerBuffer]:
        """准备层数据

        如果层已在缓冲区中，直接返回；否则加载并返回。

        Args:
            layer_idx: 层索引
            load_fn: 加载函数

        Returns:
            Optional[LayerBuffer]: 层缓冲区
        """
        start_time = time.time()

        # 检查缓冲区
        if layer_idx in self._buffers:
            buffer = self._buffers[layer_idx]
            if buffer.is_ready:
                # 缓冲区命中
                self._stats.total_prefetch_hits += 1
                return buffer

        # 缓冲区未命中
        self._stats.total_prefetch_misses += 1

        # 加载层
        buffer = await self._load_layer(layer_idx, load_fn)

        # 记录访问模式
        self._predictor.record_access(layer_idx)

        # 触发预测性预加载
        if self.config.enable_predictive_prefetch:
            await self._trigger_predictive_prefetch(layer_idx, load_fn)

        # 更新统计
        self._stats.total_layers_processed += 1
        self._stats.total_compute_time_ms += (time.time() - start_time) * 1000

        return buffer

    async def _load_layer(
        self,
        layer_idx: int,
        load_fn: Callable[[int], Any],
    ) -> LayerBuffer:
        """加载层到缓冲区

        Args:
            layer_idx: 层索引
            load_fn: 加载函数

        Returns:
            LayerBuffer: 层缓冲区
        """
        async with self._buffer_lock:
            # 创建缓冲区
            buffer = LayerBuffer(
                layer_idx=layer_idx,
                data=None,
                size_mb=self.config.buffer_size_mb,
            )

            # 标记为传输中
            buffer.is_transferring = True
            buffer.transfer_start = time.time()

            # 加载数据
            try:
                data = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, load_fn, layer_idx),
                    timeout=self.config.transfer_timeout,
                )
                buffer.data = data
                buffer.is_ready = True
            except asyncio.TimeoutError:
                logger.warning("层 %d 加载超时", layer_idx)
            except Exception as e:
                logger.error("层 %d 加载失败: %s", layer_idx, e)
            finally:
                buffer.is_transferring = False
                self._stats.total_transfer_time_ms += (time.time() - buffer.transfer_start) * 1000

            # 存储到缓冲区
            self._buffers[layer_idx] = buffer

            return buffer

    async def _trigger_predictive_prefetch(
        self,
        current_layer: int,
        load_fn: Callable[[int], Any],
    ) -> None:
        """触发预测性预加载

        Args:
            current_layer: 当前层索引
            load_fn: 加载函数
        """
        # 预测接下来需要的层
        predicted_layers = self._predictor.predict_next(self.config.prefetch_layers)

        # 预加载预测的层
        for layer_idx in predicted_layers:
            if layer_idx not in self._buffers and layer_idx not in self._prefetch_tasks:
                # 创建预加载任务
                task = asyncio.create_task(
                    self._prefetch_layer(layer_idx, load_fn),
                    name=f"prefetch-{layer_idx}",
                )
                self._prefetch_tasks[layer_idx] = task

    async def _prefetch_layer(
        self,
        layer_idx: int,
        load_fn: Callable[[int], Any],
    ) -> None:
        """预加载层

        Args:
            layer_idx: 层索引
            load_fn: 加载函数
        """
        try:
            await self._load_layer(layer_idx, load_fn)
        finally:
            # 清理任务引用
            self._prefetch_tasks.pop(layer_idx, None)

    async def release_layer(self, layer_idx: int) -> None:
        """释放层缓冲区

        Args:
            layer_idx: 层索引
        """
        async with self._buffer_lock:
            self._buffers.pop(layer_idx, None)

    async def cleanup(self) -> None:
        """清理所有缓冲区"""
        async with self._buffer_lock:
            # 取消所有预加载任务
            for task in self._prefetch_tasks.values():
                task.cancel()
            self._prefetch_tasks.clear()

            # 清空缓冲区
            self._buffers.clear()

    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计"""
        return {
            "buffer_count": len(self._buffers),
            "prefetch_tasks": len(self._prefetch_tasks),
            "prefetch_hit_rate": (
                self._stats.total_prefetch_hits /
                max(self._stats.total_prefetch_hits + self._stats.total_prefetch_misses, 1)
            ),
            "avg_transfer_time_ms": (
                self._stats.total_transfer_time_ms /
                max(self._stats.total_layers_processed, 1)
            ),
        }


# ============================================================
# 双缓冲流水线
# ============================================================

class DoubleBufferPipeline:
    """双缓冲流水线

    实现计算和传输的完全重叠。
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """初始化双缓冲流水线

        Args:
            config: 配置
        """
        self.config = config or PipelineConfig()

        # 双缓冲区
        self._buffer_a: Optional[LayerBuffer] = None
        self._buffer_b: Optional[LayerBuffer] = None

        # 当前使用的缓冲区
        self._current_buffer = "a"

        # 流水线状态
        self._is_computing = False
        self._is_transferring = False

        # 统计
        self._stats = PipelineStats()

    @property
    def stats(self) -> PipelineStats:
        """获取统计信息"""
        return self._stats

    async def process_layer(
        self,
        layer_idx: int,
        load_fn: Callable[[int], Any],
        compute_fn: Callable[[Any], Any],
    ) -> Any:
        """处理层

        使用双缓冲实现计算和传输的重叠。

        Args:
            layer_idx: 层索引
            load_fn: 加载函数
            compute_fn: 计算函数

        Returns:
            Any: 计算结果
        """
        start_time = time.time()

        # 获取当前缓冲区
        current_buffer = self._get_current_buffer()

        # 如果当前缓冲区没有数据，先加载
        if current_buffer is None or current_buffer.layer_idx != layer_idx:
            current_buffer = await self._load_to_buffer(layer_idx, load_fn)

        # 切换到另一个缓冲区
        self._switch_buffer()

        # 异步预加载下一层到另一个缓冲区
        next_layer = layer_idx + 1
        asyncio.create_task(
            self._preload_to_buffer(next_layer, load_fn)
        )

        # 计算当前层
        self._is_computing = True
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, compute_fn, current_buffer.data
            )
        finally:
            self._is_computing = False

        # 更新统计
        self._stats.total_layers_processed += 1
        self._stats.total_compute_time_ms += (time.time() - start_time) * 1000

        return result

    def _get_current_buffer(self) -> Optional[LayerBuffer]:
        """获取当前缓冲区"""
        if self._current_buffer == "a":
            return self._buffer_a
        else:
            return self._buffer_b

    def _switch_buffer(self) -> None:
        """切换缓冲区"""
        if self._current_buffer == "a":
            self._current_buffer = "b"
        else:
            self._current_buffer = "a"

    async def _load_to_buffer(
        self,
        layer_idx: int,
        load_fn: Callable[[int], Any],
    ) -> LayerBuffer:
        """加载层到当前缓冲区

        Args:
            layer_idx: 层索引
            load_fn: 加载函数

        Returns:
            LayerBuffer: 层缓冲区
        """
        buffer = LayerBuffer(
            layer_idx=layer_idx,
            data=None,
            size_mb=self.config.buffer_size_mb,
        )

        # 加载数据
        buffer.is_transferring = True
        buffer.transfer_start = time.time()

        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None, load_fn, layer_idx
            )
            buffer.data = data
            buffer.is_ready = True
        finally:
            buffer.is_transferring = False
            self._stats.total_transfer_time_ms += (time.time() - buffer.transfer_start) * 1000

        # 存储到当前缓冲区
        if self._current_buffer == "a":
            self._buffer_a = buffer
        else:
            self._buffer_b = buffer

        return buffer

    async def _preload_to_buffer(
        self,
        layer_idx: int,
        load_fn: Callable[[int], Any],
    ) -> None:
        """预加载层到另一个缓冲区

        Args:
            layer_idx: 层索引
            load_fn: 加载函数
        """
        # 确定目标缓冲区
        if self._current_buffer == "a":
            target = "b"
        else:
            target = "a"

        # 加载数据
        buffer = LayerBuffer(
            layer_idx=layer_idx,
            data=None,
            size_mb=self.config.buffer_size_mb,
        )

        buffer.is_transferring = True
        buffer.transfer_start = time.time()

        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None, load_fn, layer_idx
            )
            buffer.data = data
            buffer.is_ready = True
        finally:
            buffer.is_transferring = False

        # 存储到目标缓冲区
        if target == "a":
            self._buffer_a = buffer
        else:
            self._buffer_b = buffer

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "layers_processed": self._stats.total_layers_processed,
            "avg_compute_time_ms": (
                self._stats.total_compute_time_ms /
                max(self._stats.total_layers_processed, 1)
            ),
            "avg_transfer_time_ms": (
                self._stats.total_transfer_time_ms /
                max(self._stats.total_layers_processed, 1)
            ),
            "current_buffer": self._current_buffer,
        }


# ============================================================
# 便捷函数
# ============================================================

def create_pipeline_manager(
    strategy: str = "double",
    prefetch_layers: int = 2,
    enable_predictive: bool = True,
) -> PipelineParallelManager:
    """创建流水线管理器

    Args:
        strategy: 策略 ("none", "single", "double", "triple")
        prefetch_layers: 预加载层数
        enable_predictive: 是否启用预测性预加载

    Returns:
        PipelineParallelManager: 管理器实例
    """
    strategy_enum = PipelineStrategy(strategy)
    config = PipelineConfig(
        strategy=strategy_enum,
        prefetch_layers=prefetch_layers,
        enable_predictive_prefetch=enable_predictive,
    )
    return PipelineParallelManager(config)


def create_double_buffer_pipeline(
    buffer_size_mb: int = 512,
) -> DoubleBufferPipeline:
    """创建双缓冲流水线

    Args:
        buffer_size_mb: 缓冲区大小（MB）

    Returns:
        DoubleBufferPipeline: 流水线实例
    """
    config = PipelineConfig(
        strategy=PipelineStrategy.DOUBLE_BUFFER,
        buffer_size_mb=buffer_size_mb,
    )
    return DoubleBufferPipeline(config)
