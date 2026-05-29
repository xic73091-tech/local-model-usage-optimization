"""
智能推理调度器
实现基于任务类型、优先级、资源约束的模型选择与请求调度。
支持量化级别自动选择和模型热切换。
"""

import asyncio
import heapq
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------

class TaskType(Enum):
    """推理任务类型"""
    CHAT = "chat"               # 对话聊天
    CODE = "code"               # 代码生成/补全
    CREATIVE = "creative"       # 创意写作
    ANALYSIS = "analysis"       # 分析推理
    TRANSLATION = "translation" # 翻译
    EMBEDDING = "embedding"     # 向量嵌入


class Priority(IntEnum):
    """请求优先级，数值越小优先级越高"""
    URGENT = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class QuantizationLevel(Enum):
    """量化级别，按质量从低到高排列"""
    Q2_K = "q2_k"
    Q3_K_S = "q3_k_s"
    Q3_K_M = "q3_k_m"
    Q4_0 = "q4_0"
    Q4_K_S = "q4_k_s"
    Q4_K_M = "q4_k_m"
    Q5_K_M = "q5_k_m"
    Q5_K_S = "q5_k_s"
    Q6_K = "q6_k"
    Q8_0 = "q8_0"
    FP16 = "fp16"


# 量化级别排序 (从低质量到高质量)
_QUANT_ORDER: List[QuantizationLevel] = [
    QuantizationLevel.Q2_K,
    QuantizationLevel.Q3_K_S,
    QuantizationLevel.Q3_K_M,
    QuantizationLevel.Q4_0,
    QuantizationLevel.Q4_K_S,
    QuantizationLevel.Q4_K_M,
    QuantizationLevel.Q5_K_S,
    QuantizationLevel.Q5_K_M,
    QuantizationLevel.Q6_K,
    QuantizationLevel.Q8_0,
    QuantizationLevel.FP16,
]

# 各量化级别相对于 FP16 的参数大小比例 (以7B模型为基准估算)
_QUANT_SIZE_RATIO: Dict[QuantizationLevel, float] = {
    QuantizationLevel.Q2_K:   0.15,
    QuantizationLevel.Q3_K_S: 0.19,
    QuantizationLevel.Q3_K_M: 0.20,
    QuantizationLevel.Q4_0:   0.22,
    QuantizationLevel.Q4_K_S: 0.24,
    QuantizationLevel.Q4_K_M: 0.25,
    QuantizationLevel.Q5_K_S: 0.27,
    QuantizationLevel.Q5_K_M: 0.28,
    QuantizationLevel.Q6_K:   0.31,
    QuantizationLevel.Q8_0:   0.38,
    QuantizationLevel.FP16:   0.50,
}


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------

@dataclass(order=True)
class InferenceRequest:
    """推理请求，支持优先级堆比较。

    比较规则: 优先级数值越小越优先; 同优先级按创建时间先到先服务。
    """
    priority: Priority
    created_at: float = field(default_factory=time.time, compare=True)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], compare=False)
    task_type: TaskType = field(default=TaskType.CHAT, compare=False)
    prompt: str = field(default="", compare=False)
    max_tokens: int = field(default=512, compare=False)
    temperature: float = field(default=0.7, compare=False)
    top_p: float = field(default=0.9, compare=False)
    stop: Optional[List[str]] = field(default=None, compare=False)
    callback: Optional[Callable[["InferenceResult"], Awaitable[None]]] = field(
        default=None, compare=False
    )
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    # 调度器内部使用
    _submitted: bool = field(default=False, compare=False, repr=False)
    _cancelled: bool = field(default=False, compare=False, repr=False)


@dataclass
class InferenceResult:
    """推理结果"""
    request_id: str
    text: str = ""
    tokens_generated: int = 0
    generation_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    model_id: str = ""
    error: Optional[str] = None
    finish_reason: str = "stop"  # stop | length | error | cancelled


@dataclass
class ModelProfile:
    """模型性能画像

    用于智能模型选择，记录模型在各任务类型下的性能特征。
    """
    model_id: str
    model_path: str
    # 基础能力
    supported_tasks: Set[TaskType] = field(default_factory=lambda: set(TaskType))
    # 资源需求
    parameter_billions: float = 7.0    # 参数量 (B)
    memory_mb: float = 0.0             # 模型显存/内存占用(MB) - 当前量化级别下
    requires_gpu: bool = False         # 是否需要GPU
    min_gpu_memory_mb: float = 0.0     # 最低GPU显存要求
    # 当前量化级别
    current_quantization: QuantizationLevel = field(default=QuantizationLevel.Q4_K_M)
    # 各量化级别下的显存占用(MB)，用于智能选择
    quant_memory_map: Dict[QuantizationLevel, float] = field(default_factory=dict)
    # 性能指标(运行时更新)
    avg_tokens_per_second: float = 0.0
    avg_ttft_ms: float = 0.0         # 首token延迟
    # 任务专项评分 0-100
    task_scores: Dict[TaskType, float] = field(default_factory=dict)
    # 状态
    is_loaded: bool = False
    current_load: int = 0            # 当前并发请求数
    max_concurrency: int = 1         # 该模型最大并发数
    load_time_seconds: float = 0.0   # 模型加载耗时
    # 统计
    total_requests: int = 0
    total_tokens: int = 0
    total_time_s: float = 0.0
    error_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """平均每请求延迟(ms)"""
        if self.total_requests == 0:
            return 0.0
        return (self.total_time_s / self.total_requests) * 1000

    @property
    def error_rate(self) -> float:
        """错误率"""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    @property
    def is_available(self) -> bool:
        """是否可接受新请求"""
        return self.is_loaded and self.current_load < self.max_concurrency

    def get_memory_for_quant(self, quant: QuantizationLevel) -> float:
        """获取指定量化级别的显存占用(MB)"""
        if quant in self.quant_memory_map:
            return self.quant_memory_map[quant]
        # 估算: 基于参数量和量化比例
        return self.parameter_billions * 1000 * _QUANT_SIZE_RATIO[quant]

    def update_stats(self, tokens: int, time_s: float, is_error: bool = False):
        """更新性能统计"""
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_time_s += time_s
        if is_error:
            self.error_count += 1
        # 指数移动平均更新速率
        if tokens > 0 and time_s > 0:
            tps = tokens / time_s
            alpha = 0.3
            self.avg_tokens_per_second = alpha * tps + (1 - alpha) * self.avg_tokens_per_second


@dataclass
class SchedulerStats:
    """调度器统计信息"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    cancelled_requests: int = 0
    total_tokens_generated: int = 0
    total_processing_time_s: float = 0.0
    queue_peak_size: int = 0
    avg_queue_wait_ms: float = 0.0
    model_switches: int = 0
    # 按任务类型统计
    requests_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # 按优先级统计
    requests_by_priority: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # 模型使用统计
    requests_by_model: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def avg_processing_time_ms(self) -> float:
        if self.completed_requests == 0:
            return 0.0
        return (self.total_processing_time_s / self.completed_requests) * 1000

    @property
    def throughput_tokens_per_second(self) -> float:
        if self.total_processing_time_s == 0:
            return 0.0
        return self.total_tokens_generated / self.total_processing_time_s

    @property
    def success_rate(self) -> float:
        """请求成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.completed_requests / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "cancelled_requests": self.cancelled_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "throughput_tokens_per_second": round(self.throughput_tokens_per_second, 2),
            "queue_peak_size": self.queue_peak_size,
            "avg_queue_wait_ms": round(self.avg_queue_wait_ms, 2),
            "model_switches": self.model_switches,
            "success_rate": round(self.success_rate, 4),
            "requests_by_type": dict(self.requests_by_type),
            "requests_by_priority": dict(self.requests_by_priority),
            "requests_by_model": dict(self.requests_by_model),
        }


# ---------------------------------------------------------------------------
# 推理后端协议
# ---------------------------------------------------------------------------

class InferenceBackendProtocol:
    """推理后端需满足的最小协议。

    实际项目中可替换为 InferenceBackend 基类实例。
    """

    def generate(self, prompt: str, **kwargs) -> Any:
        raise NotImplementedError

    async def generate_stream(self, prompt: str, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 智能调度器
# ---------------------------------------------------------------------------

# 推断完成回调类型
DoneCallback = Optional[Callable[[InferenceResult], Awaitable[None]]]


class InferenceScheduler:
    """智能推理调度器

    核心能力:
    1. 基于任务类型、速度、内存约束的智能模型选择
    2. 基于可用显存自动选择最优量化级别
    3. 模型热切换 (卸载旧模型 → 加载新模型)
    4. 优先级堆请求队列
    5. 并发控制
    6. 异步处理与回调
    7. 性能统计
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        max_global_concurrency: int = 8,
        available_vram_mb: float = 8192.0,
        model_select_strategy: str = "balanced",  # balanced | speed | quality | memory
        model_loader: Optional[Callable[[str, QuantizationLevel], Awaitable[None]]] = None,
        model_unloader: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self._max_queue_size = max_queue_size
        self._max_global_concurrency = max_global_concurrency
        self._available_vram_mb = available_vram_mb
        self._model_select_strategy = model_select_strategy

        # 模型注册表
        self._models: Dict[str, ModelProfile] = {}
        # 后端实例 (model_id -> backend)
        self._backends: Dict[str, Any] = {}

        # 当前已加载模型
        self._active_model_id: Optional[str] = None
        self._active_quant: Optional[QuantizationLevel] = None

        # 热切换回调
        self._model_loader = model_loader or self._default_model_loader
        self._model_unloader = model_unloader or self._default_model_unloader

        # 请求队列 (优先级堆)
        self._queue: List[InferenceRequest] = []
        self._queue_lock = asyncio.Lock()

        # 进行中的请求
        self._in_flight: Dict[str, asyncio.Task] = {}
        self._in_flight_lock = asyncio.Lock()

        # 统计
        self._stats = SchedulerStats()
        self._queue_wait_times: List[float] = []

        # 调度循环控制
        self._running = False
        self._dispatch_task: Optional[asyncio.Task] = None
        self._dispatch_event = asyncio.Event()

        # 全局并发信号量
        self._semaphore: Optional[asyncio.Semaphore] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self):
        """启动调度器"""
        if self._running:
            return
        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_global_concurrency)
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info(
            "调度器已启动  max_queue=%d  max_concurrency=%d  vram=%dMB  strategy=%s",
            self._max_queue_size,
            self._max_global_concurrency,
            self._available_vram_mb,
            self._model_select_strategy,
        )

    async def stop(self, timeout: float = 10.0):
        """停止调度器，等待进行中任务完成"""
        self._running = False
        self._dispatch_event.set()  # 唤醒调度循环
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await asyncio.wait_for(self._dispatch_task, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        # 取消所有进行中的任务
        async with self._in_flight_lock:
            for task in self._in_flight.values():
                task.cancel()
            self._in_flight.clear()
        logger.info("调度器已停止")

    # ------------------------------------------------------------------
    # 模型注册
    # ------------------------------------------------------------------

    def register_model(self, profile: ModelProfile, backend: Any) -> None:
        """注册模型及其后端"""
        self._models[profile.model_id] = profile
        self._backends[profile.model_id] = backend
        logger.info("已注册模型: %s  tasks=%s", profile.model_id, profile.supported_tasks)

    def unregister_model(self, model_id: str) -> None:
        """注销模型"""
        self._models.pop(model_id, None)
        self._backends.pop(model_id, None)

    def get_model_profile(self, model_id: str) -> Optional[ModelProfile]:
        return self._models.get(model_id)

    # ------------------------------------------------------------------
    # 资源更新
    # ------------------------------------------------------------------

    def update_available_vram(self, vram_mb: float) -> None:
        """更新可用显存(MB)"""
        self._available_vram_mb = vram_mb
        logger.info("可用显存更新为 %.0f MB", vram_mb)

    # ------------------------------------------------------------------
    # 请求提交
    # ------------------------------------------------------------------

    async def submit(self, request: InferenceRequest) -> str:
        """提交推理请求，返回 request_id

        Raises:
            RuntimeError: 队列已满
        """
        async with self._queue_lock:
            if len(self._queue) >= self._max_queue_size:
                raise RuntimeError(f"请求队列已满({self._max_queue_size})")
            heapq.heappush(self._queue, request)
            self._stats.total_requests += 1
            self._stats.requests_by_type[request.task_type.value] += 1
            self._stats.requests_by_priority[request.priority.name] += 1
            if len(self._queue) > self._stats.queue_peak_size:
                self._stats.queue_peak_size = len(self._queue)
        # 通知调度循环
        self._dispatch_event.set()
        logger.debug("请求已入队: %s  priority=%s", request.request_id, request.priority.name)
        return request.request_id

    async def submit_and_wait(
        self,
        request: InferenceRequest,
        timeout: Optional[float] = None,
    ) -> InferenceResult:
        """提交请求并同步等待结果"""
        result_future: asyncio.Future[InferenceResult] = asyncio.get_event_loop().create_future()

        async def _on_done(res: InferenceResult):
            if not result_future.done():
                result_future.set_result(res)

        request.callback = _on_done
        await self.submit(request)
        return await asyncio.wait_for(result_future, timeout=timeout)

    async def cancel(self, request_id: str) -> bool:
        """取消请求(仅对尚未开始的请求有效)"""
        async with self._queue_lock:
            for req in self._queue:
                if req.request_id == request_id:
                    req._cancelled = True
                    self._stats.cancelled_requests += 1
                    return True
        # 检查进行中的任务
        async with self._in_flight_lock:
            task = self._in_flight.get(request_id)
            if task and not task.done():
                task.cancel()
                self._stats.cancelled_requests += 1
                return True
        return False

    # ------------------------------------------------------------------
    # 量化级别选择
    # ------------------------------------------------------------------

    def _select_quantization(
        self, profile: ModelProfile, available_vram_mb: Optional[float] = None
    ) -> Optional[QuantizationLevel]:
        """基于可用显存选择最优量化级别

        策略: 在显存约束下选择最高质量的量化级别。
        如果模型已注册 quant_memory_map 则使用精确值，否则基于参数量估算。

        :param profile: 模型画像
        :param available_vram_mb: 可用显存(MB)，None 则使用调度器全局值
        :return: 最优量化级别，若显存不足则返回 None
        """
        vram = available_vram_mb if available_vram_mb is not None else self._available_vram_mb

        # 预留 10% 显存余量
        effective_vram = vram * 0.9

        best: Optional[QuantizationLevel] = None
        for quant in reversed(_QUANT_ORDER):  # 从高质量往低质量遍历
            mem_needed = profile.get_memory_for_quant(quant)
            if mem_needed <= effective_vram:
                best = quant
                break

        return best

    # ------------------------------------------------------------------
    # 智能模型选择
    # ------------------------------------------------------------------

    def _select_model(self, request: InferenceRequest) -> Optional[tuple[ModelProfile, QuantizationLevel]]:
        """基于任务类型、速度和内存约束选择最优模型及量化级别

        策略:
        - balanced: 综合评分 = 任务匹配度 * 0.4 + 速度 * 0.3 + 负载均衡 * 0.2 + 内存效率 * 0.1
        - speed:    优先高 tokens/s
        - quality:  优先高任务评分
        - memory:   优先低内存占用

        :return: (ModelProfile, QuantizationLevel) 或 None
        """
        candidates: List[tuple[float, ModelProfile, QuantizationLevel]] = []
        strategy = self._model_select_strategy

        for profile in self._models.values():
            if request.task_type not in profile.supported_tasks:
                continue

            # 选择最优量化级别
            quant = self._select_quantization(profile)
            if quant is None:
                continue  # 显存不足

            mem_needed = profile.get_memory_for_quant(quant)

            # 计算综合评分
            task_score = profile.task_scores.get(request.task_type, 50) / 100.0
            speed_score = min(profile.avg_tokens_per_second / 50.0, 1.0) if profile.avg_tokens_per_second > 0 else 0.5
            load_score = 1.0 - (profile.current_load / profile.max_concurrency) if profile.max_concurrency > 0 else 0.0
            mem_score = max(0, 1.0 - mem_needed / self._available_vram_mb)

            # 量化质量因子 (越高质量评分越高)
            quant_idx = _QUANT_ORDER.index(quant)
            quant_quality = quant_idx / (len(_QUANT_ORDER) - 1)

            if strategy == "speed":
                score = speed_score * 0.6 + quant_quality * 0.2 + load_score * 0.2
            elif strategy == "quality":
                score = task_score * 0.5 + quant_quality * 0.3 + speed_score * 0.2
            elif strategy == "memory":
                score = mem_score * 0.5 + quant_quality * 0.3 + task_score * 0.2
            else:  # balanced
                score = task_score * 0.3 + speed_score * 0.25 + load_score * 0.2 + quant_quality * 0.15 + mem_score * 0.1

            # 已加载模型加分 (避免不必要的热切换)
            if profile.is_loaded and profile.model_id == self._active_model_id:
                score += 0.1

            candidates.append((score, profile, quant))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0], reverse=True)
        _, best_model, best_quant = candidates[0]
        return best_model, best_quant

    # ------------------------------------------------------------------
    # 模型热切换
    # ------------------------------------------------------------------

    async def _ensure_model_loaded(
        self, profile: ModelProfile, quant: QuantizationLevel
    ) -> None:
        """确保目标模型已加载，必要时执行热切换

        切换流程:
        1. 卸载旧模型 (如果不同)
        2. 加载新模型 (指定量化级别)
        3. 更新调度器状态
        """
        # 已经是目标模型且已加载
        if (
            self._active_model_id == profile.model_id
            and self._active_quant == quant
            and profile.is_loaded
        ):
            return

        # 步骤 1: 卸载旧模型
        if self._active_model_id and self._active_model_id != profile.model_id:
            old_id = self._active_model_id
            old_profile = self._models.get(old_id)
            logger.info("热切换: 卸载旧模型 %s", old_id)
            try:
                await self._model_unloader(old_id)
            except Exception as e:
                logger.warning("卸载模型 %s 失败: %s", old_id, e)
            if old_profile:
                old_profile.is_loaded = False
            self._stats.model_switches += 1

        # 步骤 2: 加载新模型
        if not profile.is_loaded or self._active_quant != quant:
            logger.info(
                "热切换: 加载模型 %s (量化=%s, 预估显存=%.0fMB)",
                profile.model_id, quant.value, profile.get_memory_for_quant(quant)
            )
            t0 = time.time()
            await self._model_loader(profile.model_id, quant)
            load_time = time.time() - t0
            profile.load_time_seconds = load_time
            profile.is_loaded = True
            logger.info("模型 %s 加载完成 (%.1fs)", profile.model_id, load_time)

        # 步骤 3: 更新状态
        self._active_model_id = profile.model_id
        self._active_quant = quant
        profile.current_quantization = quant
        # 更新 memory_mb 为当前量化级别的实际占用
        profile.memory_mb = profile.get_memory_for_quant(quant)

    # ------------------------------------------------------------------
    # 默认回调
    # ------------------------------------------------------------------

    @staticmethod
    async def _default_model_loader(model_id: str, quant: QuantizationLevel) -> None:
        """默认模型加载回调 (无实际操作，需外部替换)"""
        logger.info("[默认加载器] 加载 %s / %s", model_id, quant.value)
        await asyncio.sleep(0)

    @staticmethod
    async def _default_model_unloader(model_id: str) -> None:
        """默认模型卸载回调 (无实际操作，需外部替换)"""
        logger.info("[默认卸载器] 卸载 %s", model_id)
        await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # 调度循环
    # ------------------------------------------------------------------

    async def _dispatch_loop(self):
        """主调度循环：从队列取出请求并分派到模型"""
        while self._running:
            # 等待有新请求或停止信号
            await self._dispatch_event.wait()
            self._dispatch_event.clear()
            if not self._running:
                break

            # 持续消费队列
            while True:
                request = await self._dequeue()
                if request is None:
                    break
                if request._cancelled:
                    continue

                selection = self._select_model(request)
                if selection is None:
                    # 无可用模型，重新入队或标记失败
                    logger.warning("无可用模型，请求 %s 等待重试", request.request_id)
                    await asyncio.sleep(0.5)
                    async with self._queue_lock:
                        heapq.heappush(self._queue, request)
                    self._dispatch_event.set()
                    break

                model, quant = selection

                # 执行热切换 (如需要)
                try:
                    await self._ensure_model_loaded(model, quant)
                except Exception as e:
                    logger.error("模型加载失败 %s: %s", model.model_id, e)
                    self._stats.failed_requests += 1
                    if request.callback:
                        result = InferenceResult(
                            request_id=request.request_id,
                            error=f"模型加载失败: {e}",
                            finish_reason="error",
                        )
                        try:
                            await request.callback(result)
                        except Exception:
                            pass
                    continue

                # 分派任务
                task = asyncio.create_task(
                    self._execute_request(request, model),
                    name=f"infer-{request.request_id}",
                )
                async with self._in_flight_lock:
                    self._in_flight[request.request_id] = task
                task.add_done_callback(
                    lambda t, rid=request.request_id: self._on_task_done(rid, t)
                )

    async def _dequeue(self) -> Optional[InferenceRequest]:
        """从优先级堆取出最高优先级请求"""
        async with self._queue_lock:
            while self._queue:
                req = heapq.heappop(self._queue)
                if not req._cancelled:
                    return req
        return None

    def _on_task_done(self, request_id: str, task: asyncio.Task):
        """任务完成回调清理"""
        self._in_flight.pop(request_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error("推理任务异常 %s: %s", request_id, exc)

    # ------------------------------------------------------------------
    # 请求执行
    # ------------------------------------------------------------------

    async def _execute_request(self, request: InferenceRequest, model: ModelProfile):
        """执行单个推理请求"""
        assert self._semaphore is not None
        async with self._semaphore:
            model.current_load += 1
            wait_ms = (time.time() - request.created_at) * 1000
            self._queue_wait_times.append(wait_ms)
            # 滑动窗口保留最近1000条
            if len(self._queue_wait_times) > 1000:
                self._queue_wait_times = self._queue_wait_times[-1000:]
            self._stats.avg_queue_wait_ms = (
                sum(self._queue_wait_times) / len(self._queue_wait_times)
            )

            result: Optional[InferenceResult] = None
            try:
                start = time.time()
                backend = self._backends.get(model.model_id)
                if backend is None:
                    raise RuntimeError(f"模型 {model.model_id} 未注册后端")

                # 在线程池中执行同步推理
                gen_result = await asyncio.to_thread(
                    backend.generate,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                )
                elapsed = time.time() - start

                # 适配 GenerationResult 或裸字符串
                if hasattr(gen_result, "text"):
                    text = gen_result.text
                    tokens = gen_result.tokens_generated
                    tps = gen_result.tokens_per_second
                    finish = gen_result.finish_reason
                else:
                    text = str(gen_result)
                    tokens = 0
                    tps = 0.0
                    finish = "stop"

                result = InferenceResult(
                    request_id=request.request_id,
                    text=text,
                    tokens_generated=tokens,
                    generation_time_ms=elapsed * 1000,
                    tokens_per_second=tps,
                    model_id=model.model_id,
                    finish_reason=finish,
                )

                # 更新统计
                model.update_stats(tokens, elapsed)
                self._stats.completed_requests += 1
                self._stats.total_tokens_generated += tokens
                self._stats.total_processing_time_s += elapsed
                self._stats.requests_by_model[model.model_id] += 1

            except asyncio.CancelledError:
                result = InferenceResult(
                    request_id=request.request_id,
                    model_id=model.model_id,
                    finish_reason="cancelled",
                )
            except Exception as exc:
                # Security: Don't log full stack traces in production
                # Use exc_info only in debug mode
                if logger.isEnabledFor(logging.DEBUG):
                    logger.error("推理失败 %s: %s", request.request_id, exc, exc_info=True)
                else:
                    logger.error("推理失败 %s: %s", request.request_id, type(exc).__name__)
                model.update_stats(0, 0, is_error=True)
                self._stats.failed_requests += 1
                result = InferenceResult(
                    request_id=request.request_id,
                    model_id=model.model_id,
                    error=str(exc),
                    finish_reason="error",
                )
            finally:
                model.current_load = max(0, model.current_load - 1)

            # 触发回调
            if result and request.callback:
                try:
                    await request.callback(result)
                except Exception as cb_exc:
                    logger.error("回调执行失败 %s: %s", request.request_id, cb_exc)

            return result

    # ------------------------------------------------------------------
    # 统计与查询
    # ------------------------------------------------------------------

    def get_stats(self) -> SchedulerStats:
        """获取调度器统计"""
        return self._stats

    def get_queue_size(self) -> int:
        """当前队列长度"""
        return len(self._queue)

    def get_in_flight_count(self) -> int:
        """当前进行中请求数"""
        return len(self._in_flight)

    def get_model_profiles(self) -> Dict[str, ModelProfile]:
        """获取所有模型画像"""
        return dict(self._models)

    def get_active_model(self) -> Optional[str]:
        """获取当前活跃模型 ID"""
        return self._active_model_id

    def reset_stats(self):
        """重置统计"""
        self._stats = SchedulerStats()
        self._queue_wait_times.clear()
