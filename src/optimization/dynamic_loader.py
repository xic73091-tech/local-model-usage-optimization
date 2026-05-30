"""
动态层加载模块

实现按需加载模型层，而非一次性加载整个模型。
核心策略:
- LRU淘汰: 最久未使用的层卸载到CPU/磁盘
- 预测预取: 根据生成模式预测下一层并预加载
- 流水线加载: 一层在GPU计算时，下一层从CPU预加载
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------

class LayerLocation(Enum):
    """层所在位置"""
    GPU = "gpu"           # GPU显存中
    CPU = "cpu"           # CPU内存中
    DISK = "disk"         # 磁盘缓存中
    UNLOADED = "unloaded" # 未加载


class EvictionStrategy(Enum):
    """层淘汰策略"""
    LRU = "lru"           # 最久未使用
    LFU = "lfu"           # 最不常用
    ADAPTIVE = "adaptive" # 自适应策略


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------

@dataclass
class DynamicConfig:
    """动态层加载配置

    Attributes:
        max_gpu_layers: GPU最大驻留层数
        max_cpu_layers: CPU最大驻留层数
        prefetch_enabled: 是否启用预测预取
        prefetch_count: 预取层数
        swap_enabled: 是否启用层交换
        swap_directory: 磁盘交换目录
        layer_access_threshold: 层访问N次后提升优先级
        eviction_strategy: 淘汰策略
        pipeline_enabled: 是否启用流水线加载
        batch_load_size: 批量加载层数
        async_io_enabled: 是否启用异步IO
    """
    max_gpu_layers: int = 20
    max_cpu_layers: int = 40
    prefetch_enabled: bool = True
    prefetch_count: int = 2
    swap_enabled: bool = True
    swap_directory: str = "./layer_cache"
    layer_access_threshold: int = 3
    eviction_strategy: EvictionStrategy = EvictionStrategy.LRU
    pipeline_enabled: bool = True
    batch_load_size: int = 4
    async_io_enabled: bool = True


@dataclass
class LayerState:
    """单层状态信息

    Tracks position, access patterns, and load timing for each layer.
    """
    layer_idx: int
    location: LayerLocation = LayerLocation.UNLOADED
    size_bytes: int = 0
    access_count: int = 0
    last_access_time: float = 0.0
    load_time_ms: float = 0.0
    priority_score: float = 0.0
    # 预测相关
    next_access_probability: float = 0.0
    access_pattern: List[float] = field(default_factory=list)

    @property
    def is_on_gpu(self) -> bool:
        return self.location == LayerLocation.GPU

    @property
    def is_loaded(self) -> bool:
        return self.location in (LayerLocation.GPU, LayerLocation.CPU)


@dataclass
class LayerStats:
    """层访问统计信息"""
    total_loads: int = 0
    total_unloads: int = 0
    gpu_hits: int = 0
    cpu_hits: int = 0
    disk_hits: int = 0
    cache_misses: int = 0
    total_load_time_ms: float = 0.0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    eviction_count: int = 0

    @property
    def hit_rate(self) -> float:
        """GPU命中率"""
        total = self.gpu_hits + self.cpu_hits + self.disk_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.gpu_hits / total

    @property
    def avg_load_time_ms(self) -> float:
        """平均加载时间"""
        if self.total_loads == 0:
            return 0.0
        return self.total_load_time_ms / self.total_loads

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_loads": self.total_loads,
            "total_unloads": self.total_unloads,
            "gpu_hits": self.gpu_hits,
            "cpu_hits": self.cpu_hits,
            "disk_hits": self.disk_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "avg_load_time_ms": round(self.avg_load_time_ms, 2),
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "eviction_count": self.eviction_count,
        }


# ---------------------------------------------------------------------------
# 层管理器
# ---------------------------------------------------------------------------

class LayerManager:
    """层位置与生命周期管理器

    职责:
    - 跟踪每层的位置 (GPU/CPU/Disk)
    - 管理层的加载/卸载决策
    - 统计层访问频率与模式
    - 实现LRU/LFU淘汰策略
    """

    def __init__(self, total_layers: int, config: DynamicConfig):
        self._total_layers = total_layers
        self._config = config
        self._stats = LayerStats()

        # 层状态表
        self._layers: Dict[int, LayerState] = {}
        for i in range(total_layers):
            self._layers[i] = LayerState(layer_idx=i)

        # LRU有序字典: 最近访问的在末尾
        self._gpu_lru: OrderedDict[int, float] = OrderedDict()
        self._cpu_lru: OrderedDict[int, float] = OrderedDict()

        # 访问频率表 (用于LFU)
        self._access_freq: Dict[int, int] = {i: 0 for i in range(total_layers)}

        # 访问模式记录 (用于预测)
        self._access_history: List[int] = []
        self._max_history = 1000

        # 同步锁
        self._lock = asyncio.Lock()

    @property
    def total_layers(self) -> int:
        return self._total_layers

    @property
    def gpu_layer_count(self) -> int:
        """当前GPU上的层数"""
        return len(self._gpu_lru)

    @property
    def cpu_layer_count(self) -> int:
        """当前CPU上的层数"""
        return len(self._cpu_lru)

    @property
    def stats(self) -> LayerStats:
        return self._stats

    def get_layer_state(self, layer_idx: int) -> Optional[LayerState]:
        """获取层状态"""
        return self._layers.get(layer_idx)

    def get_gpu_layers(self) -> List[int]:
        """获取GPU上的所有层索引"""
        return list(self._gpu_lru.keys())

    def get_cpu_layers(self) -> List[int]:
        """获取CPU上的所有层索引"""
        return list(self._cpu_lru.keys())

    async def touch_layer(self, layer_idx: int) -> LayerLocation:
        """访问层并更新统计

        Returns:
            层当前位置
        """
        async with self._lock:
            state = self._layers.get(layer_idx)
            if state is None:
                self._stats.cache_misses += 0
                return LayerLocation.UNLOADED

            now = time.time()
            state.access_count += 1
            state.last_access_time = now
            self._access_freq[layer_idx] = self._access_freq.get(layer_idx, 0) + 1

            # 记录访问历史
            self._access_history.append(layer_idx)
            if len(self._access_history) > self._max_history:
                self._access_history = self._access_history[-self._max_history:]

            # 更新LRU
            if state.location == LayerLocation.GPU:
                self._gpu_lru.move_to_end(layer_idx)
                self._stats.gpu_hits += 1
            elif state.location == LayerLocation.CPU:
                self._cpu_lru.move_to_end(layer_idx)
                self._stats.cpu_hits += 1
            elif state.location == LayerLocation.DISK:
                self._stats.disk_hits += 1
            else:
                self._stats.cache_misses += 1

            return state.location

    async def register_gpu_layer(self, layer_idx: int, size_bytes: int = 0, load_time_ms: float = 0.0):
        """注册层到GPU"""
        async with self._lock:
            state = self._layers[layer_idx]
            # 从原位置移除
            self._remove_from_location(layer_idx, state.location)

            state.location = LayerLocation.GPU
            state.size_bytes = size_bytes
            state.load_time_ms = load_time_ms
            state.last_access_time = time.time()
            self._gpu_lru[layer_idx] = state.last_access_time
            self._stats.total_loads += 1
            self._stats.total_load_time_ms += load_time_ms

    async def register_cpu_layer(self, layer_idx: int, size_bytes: int = 0):
        """注册层到CPU"""
        async with self._lock:
            state = self._layers[layer_idx]
            self._remove_from_location(layer_idx, state.location)

            state.location = LayerLocation.CPU
            state.size_bytes = size_bytes
            state.last_access_time = time.time()
            self._cpu_lru[layer_idx] = state.last_access_time

    async def register_disk_layer(self, layer_idx: int):
        """注册层到磁盘"""
        async with self._lock:
            state = self._layers[layer_idx]
            self._remove_from_location(layer_idx, state.location)
            state.location = LayerLocation.DISK

    async def mark_unloaded(self, layer_idx: int):
        """标记层为未加载"""
        async with self._lock:
            state = self._layers[layer_idx]
            self._remove_from_location(layer_idx, state.location)
            state.location = LayerLocation.UNLOADED
            self._stats.total_unloads += 1

    def _remove_from_location(self, layer_idx: int, location: LayerLocation):
        """从指定位置移除层记录"""
        if location == LayerLocation.GPU:
            self._gpu_lru.pop(layer_idx, None)
        elif location == LayerLocation.CPU:
            self._cpu_lru.pop(layer_idx, None)

    async def select_eviction_candidates(self, count: int = 1) -> List[int]:
        """选择淘汰候选层

        根据配置的淘汰策略选择最应该被淘汰的GPU层。

        Args:
            count: 需要淘汰的层数

        Returns:
            候选淘汰的层索引列表
        """
        async with self._lock:
            if not self._gpu_lru:
                return []

            strategy = self._config.eviction_strategy

            if strategy == EvictionStrategy.LRU:
                # LRU: 选择最久未访问的层
                candidates = list(self._gpu_lru.keys())[:count]
            elif strategy == EvictionStrategy.LFU:
                # LFU: 选择访问频率最低的层
                gpu_layers = list(self._gpu_lru.keys())
                gpu_layers.sort(key=lambda idx: self._access_freq.get(idx, 0))
                candidates = gpu_layers[:count]
            else:
                # ADAPTIVE: 综合考虑访问频率和最近访问时间
                now = time.time()
                gpu_layers = list(self._gpu_lru.keys())

                def _score(idx: int) -> float:
                    state = self._layers[idx]
                    freq = self._access_freq.get(idx, 0)
                    recency = now - state.last_access_time
                    # 频率低 + 时间久 -> 分数高 -> 更应该被淘汰
                    threshold = self._config.layer_access_threshold
                    if freq >= threshold:
                        return float('inf')  # 高频层不淘汰
                    return recency / (freq + 1)

                gpu_layers.sort(key=_score, reverse=True)
                candidates = gpu_layers[:count]

            self._stats.eviction_count += len(candidates)
            return candidates

    def get_layer_stats(self) -> Dict[str, Any]:
        """获取各层访问统计"""
        layer_info = {}
        for idx, state in self._layers.items():
            layer_info[idx] = {
                "location": state.location.value,
                "access_count": state.access_count,
                "last_access_time": state.last_access_time,
                "size_bytes": state.size_bytes,
                "load_time_ms": state.load_time_ms,
            }
        return {
            "total_layers": self._total_layers,
            "gpu_count": self.gpu_layer_count,
            "cpu_count": self.cpu_layer_count,
            "stats": self._stats.to_dict(),
            "layers": layer_info,
        }

    def predict_next_layers(self, current_layer: int, count: int = 2) -> List[int]:
        """预测接下来可能需要的层

        基于访问历史的马尔可夫链预测:
        - 统计从 current_layer 出发后各层的转移概率
        - 返回概率最高的 count 个层

        Args:
            current_layer: 当前正在使用的层
            count: 预测层数

        Returns:
            预测的下一批层索引
        """
        if len(self._access_history) < 10:
            # 历史不足，使用顺序预测
            return [
                min(current_layer + i + 1, self._total_layers - 1)
                for i in range(count)
            ]

        # 统计转移频率
        transition_counts: Dict[int, int] = {}
        history = self._access_history
        for i in range(len(history) - 1):
            if history[i] == current_layer:
                next_layer = history[i + 1]
                transition_counts[next_layer] = transition_counts.get(next_layer, 0) + 1

        if not transition_counts:
            # 无转移记录，使用顺序预测
            return [
                min(current_layer + i + 1, self._total_layers - 1)
                for i in range(count)
            ]

        # 按转移概率排序
        sorted_transitions = sorted(
            transition_counts.items(), key=lambda x: x[1], reverse=True
        )
        predicted = [idx for idx, _ in sorted_transitions[:count]]

        # 更新预测概率
        total = sum(transition_counts.values())
        for idx, cnt in transition_counts.items():
            if idx in self._layers:
                self._layers[idx].next_access_probability = cnt / total

        return predicted


# ---------------------------------------------------------------------------
# 动态层加载器
# ---------------------------------------------------------------------------

# 层数据加载回调类型
LayerLoaderCallback = Callable[[int], Any]
LayerUnloaderCallback = Callable[[int, str], Any]


class DynamicLayerLoader:
    """动态层加载器

    核心思想: 不一次性加载整个模型，按需加载层。
    支持:
    - LRU淘汰: 最久未使用的层卸载到CPU/磁盘
    - 预测预取: 根据生成模式预测下一层并预加载
    - 流水线加载: 一层在GPU计算时，下一层从CPU加载

    Usage:
        config = DynamicConfig(max_gpu_layers=20, prefetch_enabled=True)
        loader = DynamicLayerLoader(model_path, config)
        await loader.initialize()
        await loader.load_layer(0)  # 按需加载第0层
        stats = loader.get_layer_stats()
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[DynamicConfig] = None,
        layer_loader: Optional[LayerLoaderCallback] = None,
        layer_unloader: Optional[LayerUnloaderCallback] = None,
        total_layers: Optional[int] = None,
    ):
        """初始化动态层加载器

        Args:
            model_path: 模型路径
            config: 动态加载配置
            layer_loader: 自定义层加载回调 (layer_idx) -> layer_data
            layer_unloader: 自定义层卸载回调 (layer_idx, target) -> None
            total_layers: 模型总层数 (若为None则从模型元数据读取)
        """
        self._model_path = Path(model_path)
        self._config = config or DynamicConfig()

        # 回调函数
        self._layer_loader = layer_loader or self._default_layer_loader
        self._layer_unloader = layer_unloader or self._default_layer_unloader

        # 模型信息
        self._total_layers = total_layers or 0
        self._layer_sizes: Dict[int, int] = {}

        # 层管理器
        self._manager: Optional[LayerManager] = None

        # 预取相关
        self._prefetch_tasks: Dict[int, asyncio.Task] = {}
        self._prefetch_cancel: Set[int] = set()

        # 流水线相关
        self._pipeline_queue: asyncio.Queue = asyncio.Queue()
        self._pipeline_task: Optional[asyncio.Task] = None

        # 磁盘缓存目录
        self._cache_dir = Path(self._config.swap_directory)

        # 状态
        self._initialized = False
        # 优化: 使用细粒度锁，每层一个锁，提升并发度
        self._layer_locks: Dict[int, asyncio.Lock] = {}
        self._gpu_space_lock = asyncio.Lock()  # GPU空间管理锁
        self._loading_lock = asyncio.Lock()  # 保留用于向后兼容

        # 统计
        self._access_pattern_buffer: List[Tuple[int, float]] = []

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def total_layers(self) -> int:
        return self._total_layers

    @property
    def config(self) -> DynamicConfig:
        return self._config

    def _get_layer_lock(self, layer_idx: int) -> asyncio.Lock:
        """获取层的锁（细粒度锁，每层一个）

        Args:
            layer_idx: 层索引

        Returns:
            asyncio.Lock: 该层的锁
        """
        if layer_idx not in self._layer_locks:
            self._layer_locks[layer_idx] = asyncio.Lock()
        return self._layer_locks[layer_idx]

    # ------------------------------------------------------------------
    # 初始化与清理
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """初始化加载器

        创建缓存目录，读取模型元数据，初始化层管理器。
        """
        if self._initialized:
            return

        # 创建缓存目录 (安全验证: 确保路径在项目目录内)
        if self._config.swap_enabled:
            # Security: Resolve to absolute path and verify it's within project root
            try:
                resolved = self._cache_dir.resolve()
                project_root = Path(__file__).resolve().parent.parent.parent
                if not str(resolved).startswith(str(project_root)):
                    logger.error(
                        "SECURITY: swap_directory '%s' is outside project root '%s'. "
                        "Refusing to create cache directory.",
                        resolved, project_root
                    )
                    raise ValueError(
                        f"swap_directory must be within project root: {project_root}"
                    )
            except (OSError, ValueError) as e:
                logger.error("Invalid swap_directory: %s", e)
                raise

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("磁盘缓存目录: %s", self._cache_dir)

        # 如果未指定总层数，尝试从模型推断
        if self._total_layers == 0:
            self._total_layers = self._infer_total_layers()

        # 初始化层管理器
        self._manager = LayerManager(self._total_layers, self._config)

        # 启动流水线消费者
        if self._config.pipeline_enabled:
            self._pipeline_task = asyncio.create_task(self._pipeline_consumer())

        self._initialized = True
        logger.info(
            "动态层加载器已初始化  layers=%d  max_gpu=%d  prefetch=%s  swap=%s",
            self._total_layers,
            self._config.max_gpu_layers,
            self._config.prefetch_enabled,
            self._config.swap_enabled,
        )

    async def shutdown(self) -> None:
        """关闭加载器，取消所有后台任务"""
        # 取消预取任务
        for task in self._prefetch_tasks.values():
            task.cancel()
        self._prefetch_tasks.clear()

        # 停止流水线
        if self._pipeline_task:
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        logger.info("动态层加载器已关闭")

    def _infer_total_layers(self) -> int:
        """从模型路径推断总层数

        尝试从模型文件名或目录结构推断层数。
        默认返回32层 (常见7B模型配置)。
        """
        # 尝试从文件名解析
        name = self._model_path.stem.lower()
        import re
        match = re.search(r'(\d+)l', name)
        if match:
            return int(match.group(1))

        # 默认值
        logger.warning("无法推断模型层数，使用默认值32")
        return 32

    # ------------------------------------------------------------------
    # 核心加载/卸载接口
    # ------------------------------------------------------------------

    async def load_layer(self, layer_idx: int) -> None:
        """按需加载指定层到GPU

        加载流程:
        1. 检查层是否已在GPU上 -> 直接返回
        2. 检查层是否在CPU上 -> 从CPU迁移到GPU
        3. 检查层是否在磁盘上 -> 从磁盘加载到GPU
        4. GPU空间不足时 -> 淘汰最久未使用的层
        5. 触发预取 (如果启用)

        Args:
            layer_idx: 层索引
        """
        if not self._initialized:
            await self.initialize()

        assert self._manager is not None

        # 记录访问模式
        self._access_pattern_buffer.append((layer_idx, time.time()))

        # 触摸层并获取当前位置
        location = await self._manager.touch_layer(layer_idx)

        if location == LayerLocation.GPU:
            # 已在GPU上，无需操作
            return

        # 优化: 使用细粒度锁，每层一个锁，提升并发度
        layer_lock = self._get_layer_lock(layer_idx)

        # GPU空间管理使用单独的锁
        async with self._gpu_space_lock:
            await self._ensure_gpu_space()

        # 层加载使用层级别的锁
        async with layer_lock:
            # 根据当前位置选择加载路径
            if location == LayerLocation.CPU:
                await self._load_from_cpu_to_gpu(layer_idx)
            elif location == LayerLocation.DISK:
                await self._load_from_disk_to_gpu(layer_idx)
            else:
                await self._load_from_source_to_gpu(layer_idx)

        # 触发预取（不需要锁，可以并发）
        if self._config.prefetch_enabled:
            await self._trigger_prefetch(layer_idx)

    async def unload_layer(self, layer_idx: int, target: str = "cpu") -> None:
        """卸载指定层

        Args:
            layer_idx: 层索引
            target: 卸载目标 ("cpu" 或 "disk")
        """
        if not self._initialized:
            await self.initialize()

        assert self._manager is not None

        state = self._manager.get_layer_state(layer_idx)
        if state is None or not state.is_loaded:
            return

        if target == "cpu":
            await self._unload_to_cpu(layer_idx)
        elif target == "disk":
            await self._unload_to_disk(layer_idx)
        else:
            logger.warning("未知卸载目标: %s", target)

    async def prefetch_next_layers(self, current_layer: int, count: int = 2) -> None:
        """预取接下来可能需要的层

        使用预测算法确定最可能需要的层并预加载到CPU。

        Args:
            current_layer: 当前层索引
            count: 预取层数
        """
        if not self._initialized:
            await self.initialize()

        assert self._manager is not None

        predicted_layers = self._manager.predict_next_layers(current_layer, count)

        for layer_idx in predicted_layers:
            if layer_idx == current_layer:
                continue

            state = self._manager.get_layer_state(layer_idx)
            if state is None:
                continue

            # 如果层未加载，预取到CPU
            if state.location == LayerLocation.UNLOADED:
                if layer_idx not in self._prefetch_tasks:
                    task = asyncio.create_task(
                        self._prefetch_to_cpu(layer_idx),
                        name=f"prefetch-{layer_idx}",
                    )
                    self._prefetch_tasks[layer_idx] = task

    # ------------------------------------------------------------------
    # GPU空间管理
    # ------------------------------------------------------------------

    async def _ensure_gpu_space(self) -> None:
        """确保GPU有空间容纳新层

        如果GPU层已满，淘汰最久未使用的层到CPU。
        """
        assert self._manager is not None

        while self._manager.gpu_layer_count >= self._config.max_gpu_layers:
            candidates = await self._manager.select_eviction_candidates(1)
            if not candidates:
                logger.error("无法淘汰GPU层，无候选层")
                break

            for layer_idx in candidates:
                if self._config.swap_enabled:
                    await self._unload_to_cpu(layer_idx)
                else:
                    await self._manager.mark_unloaded(layer_idx)
                logger.debug("淘汰GPU层 %d -> CPU", layer_idx)

    # ------------------------------------------------------------------
    # 层迁移实现
    # ------------------------------------------------------------------

    async def _load_from_cpu_to_gpu(self, layer_idx: int) -> None:
        """从CPU迁移到GPU"""
        logger.debug("层 %d: CPU -> GPU", layer_idx)
        t0 = time.time()

        try:
            await self._layer_loader(layer_idx)
            elapsed_ms = (time.time() - t0) * 1000
            await self._manager.register_gpu_layer(
                layer_idx,
                size_bytes=self._layer_sizes.get(layer_idx, 0),
                load_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error("层 %d CPU->GPU 加载失败: %s", layer_idx, e)
            raise

    async def _load_from_disk_to_gpu(self, layer_idx: int) -> None:
        """从磁盘加载到GPU (经过CPU中转)"""
        logger.debug("层 %d: Disk -> CPU -> GPU", layer_idx)
        t0 = time.time()

        try:
            # 先加载到CPU
            if self._config.swap_enabled:
                cache_path = self._get_cache_path(layer_idx)
                if cache_path.exists():
                    # 异步读取磁盘缓存
                    await self._async_read_cache(layer_idx, cache_path)

            # 再加载到GPU
            await self._layer_loader(layer_idx)
            elapsed_ms = (time.time() - t0) * 1000
            await self._manager.register_gpu_layer(
                layer_idx,
                size_bytes=self._layer_sizes.get(layer_idx, 0),
                load_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error("层 %d Disk->GPU 加载失败: %s", layer_idx, e)
            raise

    async def _load_from_source_to_gpu(self, layer_idx: int) -> None:
        """从模型源文件加载到GPU"""
        logger.debug("层 %d: Source -> GPU", layer_idx)
        t0 = time.time()

        try:
            await self._layer_loader(layer_idx)
            elapsed_ms = (time.time() - t0) * 1000
            await self._manager.register_gpu_layer(
                layer_idx,
                size_bytes=self._layer_sizes.get(layer_idx, 0),
                load_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error("层 %d Source->GPU 加载失败: %s", layer_idx, e)
            raise

    async def _unload_to_cpu(self, layer_idx: int) -> None:
        """卸载层到CPU"""
        assert self._manager is not None

        try:
            await self._layer_unloader(layer_idx, "cpu")
            await self._manager.register_cpu_layer(
                layer_idx,
                size_bytes=self._layer_sizes.get(layer_idx, 0),
            )
        except Exception as e:
            logger.error("层 %d 卸载到CPU失败: %s", layer_idx, e)
            raise

    async def _unload_to_disk(self, layer_idx: int) -> None:
        """卸载层到磁盘"""
        assert self._manager is not None

        if not self._config.swap_enabled:
            await self._manager.mark_unloaded(layer_idx)
            return

        try:
            # 写入磁盘缓存
            cache_path = self._get_cache_path(layer_idx)
            await self._async_write_cache(layer_idx, cache_path)
            await self._layer_unloader(layer_idx, "disk")
            await self._manager.register_disk_layer(layer_idx)
        except Exception as e:
            logger.error("层 %d 卸载到磁盘失败: %s", layer_idx, e)
            raise

    # ------------------------------------------------------------------
    # 预取实现
    # ------------------------------------------------------------------

    async def _trigger_prefetch(self, current_layer: int) -> None:
        """触发预取"""
        if not self._config.prefetch_enabled:
            return

        assert self._manager is not None

        # 取消不再需要的预取任务
        predicted = self._manager.predict_next_layers(
            current_layer, self._config.prefetch_count
        )
        for idx in list(self._prefetch_tasks.keys()):
            if idx not in predicted and idx != current_layer:
                task = self._prefetch_tasks.pop(idx, None)
                if task and not task.done():
                    task.cancel()

        # 启动新的预取
        await self.prefetch_next_layers(current_layer, self._config.prefetch_count)

    async def _prefetch_to_cpu(self, layer_idx: int) -> None:
        """预取层到CPU (后台任务)"""
        assert self._manager is not None

        try:
            state = self._manager.get_layer_state(layer_idx)
            if state is None or state.is_loaded:
                return

            logger.debug("预取层 %d 到CPU", layer_idx)

            if self._config.swap_enabled:
                cache_path = self._get_cache_path(layer_idx)
                if cache_path.exists():
                    await self._async_read_cache(layer_idx, cache_path)
                    await self._manager.register_cpu_layer(layer_idx)
                    self._manager.stats.prefetch_hits += 1
                    return

            # 从源加载到CPU (不经过GPU)
            await self._layer_loader(layer_idx)
            await self._manager.register_cpu_layer(
                layer_idx,
                size_bytes=self._layer_sizes.get(layer_idx, 0),
            )
            self._manager.stats.prefetch_hits += 1

        except asyncio.CancelledError:
            logger.debug("预取层 %d 已取消", layer_idx)
        except Exception as e:
            logger.warning("预取层 %d 失败: %s", layer_idx, e)
            self._manager.stats.prefetch_misses += 1
        finally:
            self._prefetch_tasks.pop(layer_idx, None)

    # ------------------------------------------------------------------
    # 流水线加载
    # ------------------------------------------------------------------

    async def _pipeline_consumer(self) -> None:
        """流水线消费者: 从队列取出层并预加载"""
        assert self._manager is not None

        while True:
            try:
                layer_idx = await self._pipeline_queue.get()
                state = self._manager.get_layer_state(layer_idx)
                if state and not state.is_loaded:
                    logger.debug("流水线加载层 %d", layer_idx)
                    await self._prefetch_to_cpu(layer_idx)
                self._pipeline_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("流水线处理异常: %s", e)

    async def enqueue_pipeline(self, layer_idx: int) -> None:
        """将层加入流水线预加载队列

        Args:
            layer_idx: 需要预加载的层索引
        """
        if self._config.pipeline_enabled and self._pipeline_task:
            await self._pipeline_queue.put(layer_idx)

    async def enqueue_batch_pipeline(self, layer_indices: List[int]) -> None:
        """批量加入流水线

        Args:
            layer_indices: 层索引列表
        """
        for idx in layer_indices:
            await self.enqueue_pipeline(idx)

    # ------------------------------------------------------------------
    # 磁盘缓存IO
    # ------------------------------------------------------------------

    def _get_cache_path(self, layer_idx: int) -> Path:
        """获取层的磁盘缓存路径"""
        return self._cache_dir / f"layer_{layer_idx}.bin"

    async def _async_write_cache(self, layer_idx: int, path: Path) -> None:
        """异步写入层数据到磁盘"""
        if not self._config.async_io_enabled:
            # 同步写入
            path.write_bytes(b'\x00' * self._layer_sizes.get(layer_idx, 0))
            return

        loop = asyncio.get_event_loop()
        size = self._layer_sizes.get(layer_idx, 0)
        await loop.run_in_executor(None, path.write_bytes, b'\x00' * size)

    async def _async_read_cache(self, layer_idx: int, path: Path) -> None:
        """异步从磁盘读取层数据"""
        if not self._config.async_io_enabled:
            data = path.read_bytes()
            return data

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, path.read_bytes)

    # ------------------------------------------------------------------
    # 统计与查询
    # ------------------------------------------------------------------

    def get_layer_stats(self) -> Dict[str, Any]:
        """获取各层访问统计"""
        if self._manager is None:
            return {"error": "加载器未初始化"}

        return self._manager.get_layer_stats()

    def get_gpu_utilization(self) -> float:
        """GPU层利用率 (0.0 ~ 1.0)"""
        if self._manager is None:
            return 0.0
        if self._config.max_gpu_layers == 0:
            return 0.0
        return self._manager.gpu_layer_count / self._config.max_gpu_layers

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        if self._manager is None:
            return {}

        gpu_layers = self._manager.get_gpu_layers()
        cpu_layers = self._manager.get_cpu_layers()

        gpu_bytes = sum(self._layer_sizes.get(idx, 0) for idx in gpu_layers)
        cpu_bytes = sum(self._layer_sizes.get(idx, 0) for idx in cpu_layers)

        return {
            "gpu_layers": len(gpu_layers),
            "gpu_memory_bytes": gpu_bytes,
            "gpu_memory_mb": round(gpu_bytes / (1024 * 1024), 2),
            "cpu_layers": len(cpu_layers),
            "cpu_memory_bytes": cpu_bytes,
            "cpu_memory_mb": round(cpu_bytes / (1024 * 1024), 2),
            "max_gpu_layers": self._config.max_gpu_layers,
            "gpu_utilization": round(self.get_gpu_utilization(), 4),
            "total_layers": self._total_layers,
        }

    def get_prefetch_status(self) -> Dict[str, Any]:
        """获取预取状态"""
        return {
            "prefetch_enabled": self._config.prefetch_enabled,
            "active_prefetch_tasks": len(self._prefetch_tasks),
            "prefetch_task_layers": list(self._prefetch_tasks.keys()),
            "pipeline_enabled": self._config.pipeline_enabled,
            "pipeline_queue_size": self._pipeline_queue.qsize(),
        }

    # ------------------------------------------------------------------
    # 默认回调
    # ------------------------------------------------------------------

    @staticmethod
    async def _default_layer_loader(layer_idx: int) -> None:
        """默认层加载回调 (无实际操作，需外部替换)"""
        logger.debug("[默认加载器] 加载层 %d", layer_idx)
        await asyncio.sleep(0)

    @staticmethod
    async def _default_layer_unloader(layer_idx: int, target: str) -> None:
        """默认层卸载回调 (无实际操作，需外部替换)"""
        logger.debug("[默认卸载器] 卸载层 %d -> %s", layer_idx, target)
        await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # 批量操作
    # ------------------------------------------------------------------

    async def load_layers_batch(self, layer_indices: List[int]) -> None:
        """批量加载层

        按顺序加载，但会尽量利用流水线并行。

        Args:
            layer_indices: 需要加载的层索引列表
        """
        if not layer_indices:
            return

        if self._config.pipeline_enabled:
            # 使用流水线: 先加载第一层，同时预取后续层
            first, *rest = layer_indices

            # 预取后续层到CPU
            if rest:
                await self.enqueue_batch_pipeline(rest[:self._config.batch_load_size])

            # 加载第一层
            await self.load_layer(first)

            # 等待流水线完成并提升到GPU
            for idx in rest:
                await self.load_layer(idx)
        else:
            # 顺序加载
            for idx in layer_indices:
                await self.load_layer(idx)

    async def warmup(self, layer_indices: Optional[List[int]] = None) -> None:
        """预热: 预加载指定层

        Args:
            layer_indices: 需要预热的层，None则预热前max_gpu_layers层
        """
        if not self._initialized:
            await self.initialize()

        if layer_indices is None:
            layer_indices = list(range(min(self._config.max_gpu_layers, self._total_layers)))

        logger.info("预热加载 %d 层", len(layer_indices))
        await self.load_layers_batch(layer_indices)
        logger.info("预热完成")
