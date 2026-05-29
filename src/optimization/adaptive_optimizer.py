"""
运行时自适应优化模块

根据实际推理负载动态调整配置，实现最优性能。

功能:
1. 实时监控推理性能（tokens/s, 延迟, 显存使用）
2. 自动调整配置（量化级别, 上下文长度, GPU层数）
3. 学习最优配置（基于历史数据）
4. 预测性优化（提前调整避免性能下降）

优势:
- 无需手动调优
- 适应不同的工作负载
- 避免OOM和性能抖动
- 持续学习改进
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 配置和数据类
# ============================================================

class OptimizationGoal(Enum):
    """优化目标"""
    SPEED = "speed"          # 优先速度
    QUALITY = "quality"      # 优先质量
    BALANCED = "balanced"    # 平衡
    MEMORY = "memory"        # 优先显存


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    tokens_per_second: float
    latency_ms: float
    vram_usage_gb: float
    ram_usage_gb: float
    gpu_utilization: float
    context_length: int
    batch_size: int
    quant_level: str
    gpu_layers: int


@dataclass
class AdaptiveConfig:
    """自适应优化配置"""
    # 优化目标
    goal: OptimizationGoal = OptimizationGoal.BALANCED

    # 监控间隔（秒）
    monitor_interval: float = 1.0

    # 历史窗口大小
    history_size: int = 100

    # 性能阈值
    min_tokens_per_second: float = 5.0
    max_latency_ms: float = 500.0
    max_vram_usage_ratio: float = 0.9

    # 调整步长
    quant_step: int = 1  # 量化级别调整步长
    context_step: int = 512  # 上下文长度调整步长
    gpu_layers_step: int = 2  # GPU层数调整步长

    # 是否启用学习
    enable_learning: bool = True

    # 配置文件路径
    config_path: Optional[str] = None


@dataclass
class OptimizationState:
    """优化状态"""
    # 当前配置
    quant_level: str = "q4_k_m"
    context_length: int = 4096
    gpu_layers: int = 32
    batch_size: int = 512

    # 性能统计
    avg_tokens_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    avg_vram_usage: float = 0.0

    # 调整历史
    adjustments: List[Dict[str, Any]] = field(default_factory=list)

    # 最后调整时间
    last_adjustment_time: float = 0.0


# ============================================================
# 性能监控器
# ============================================================

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, history_size: int = 100):
        """初始化性能监控器

        Args:
            history_size: 历史记录大小
        """
        self.history_size = history_size
        self._history: deque = deque(maxlen=history_size)
        self._current_metrics: Optional[PerformanceMetrics] = None

    def record(self, metrics: PerformanceMetrics) -> None:
        """记录性能指标

        Args:
            metrics: 性能指标
        """
        self._history.append(metrics)
        self._current_metrics = metrics

    def get_current(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        return self._current_metrics

    def get_average(self, window: int = 10) -> Dict[str, float]:
        """获取平均性能指标

        Args:
            window: 计算窗口大小

        Returns:
            Dict: 平均指标
        """
        if not self._history:
            return {}

        recent = list(self._history)[-window:]
        return {
            "tokens_per_second": sum(m.tokens_per_second for m in recent) / len(recent),
            "latency_ms": sum(m.latency_ms for m in recent) / len(recent),
            "vram_usage_gb": sum(m.vram_usage_gb for m in recent) / len(recent),
            "gpu_utilization": sum(m.gpu_utilization for m in recent) / len(recent),
        }

    def get_trend(self, metric: str, window: int = 20) -> str:
        """获取指标趋势

        Args:
            metric: 指标名称
            window: 计算窗口

        Returns:
            str: 趋势 ("increasing", "decreasing", "stable")
        """
        if len(self._history) < window:
            return "stable"

        recent = list(self._history)[-window:]
        values = [getattr(m, metric, 0) for m in recent]

        # 简单线性回归
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # 根据斜率判断趋势
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def get_history(self, count: Optional[int] = None) -> List[PerformanceMetrics]:
        """获取历史记录

        Args:
            count: 记录数量

        Returns:
            List[PerformanceMetrics]: 历史记录
        """
        if count is None:
            return list(self._history)
        return list(self._history)[-count:]


# ============================================================
# 自适应优化器
# ============================================================

class AdaptiveOptimizer:
    """自适应优化器

    根据实时性能指标动态调整配置。
    """

    # 量化级别列表（从低到高）
    QUANT_LEVELS = [
        "q2_k", "q3_k_s", "q3_k_m", "q4_0", "q4_k_s",
        "q4_k_m", "q5_k_s", "q5_k_m", "q6_k", "q8_0"
    ]

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """初始化自适应优化器

        Args:
            config: 配置
        """
        self.config = config or AdaptiveConfig()

        # 初始化性能监控器
        self._monitor = PerformanceMonitor(self.config.history_size)

        # 当前状态
        self._state = OptimizationState()

        # 学习到的最优配置
        self._learned_configs: Dict[str, OptimizationState] = {}

        # 加载学习到的配置
        if self.config.config_path:
            self._load_learned_configs()

        # 调整锁
        self._adjustment_lock = asyncio.Lock()

    @property
    def state(self) -> OptimizationState:
        """获取当前状态"""
        return self._state

    @property
    def monitor(self) -> PerformanceMonitor:
        """获取性能监控器"""
        return self._monitor

    async def record_performance(
        self,
        tokens_per_second: float,
        latency_ms: float,
        vram_usage_gb: float,
        ram_usage_gb: float = 0.0,
        gpu_utilization: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """记录性能并可能触发调整

        Args:
            tokens_per_second: 生成速度
            latency_ms: 延迟
            vram_usage_gb: 显存使用
            ram_usage_gb: 内存使用
            gpu_utilization: GPU利用率

        Returns:
            Optional[Dict]: 如果触发了调整，返回调整信息
        """
        # 记录性能指标
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            tokens_per_second=tokens_per_second,
            latency_ms=latency_ms,
            vram_usage_gb=vram_usage_gb,
            ram_usage_gb=ram_usage_gb,
            gpu_utilization=gpu_utilization,
            context_length=self._state.context_length,
            batch_size=self._state.batch_size,
            quant_level=self._state.quant_level,
            gpu_layers=self._state.gpu_layers,
        )
        self._monitor.record(metrics)

        # 更新统计
        avg = self._monitor.get_average(10)
        self._state.avg_tokens_per_second = avg.get("tokens_per_second", 0)
        self._state.avg_latency_ms = avg.get("latency_ms", 0)
        self._state.avg_vram_usage = avg.get("vram_usage_gb", 0)

        # 检查是否需要调整
        adjustment = await self._check_and_adjust()
        return adjustment

    async def _check_and_adjust(self) -> Optional[Dict[str, Any]]:
        """检查并调整配置

        Returns:
            Optional[Dict]: 调整信息
        """
        async with self._adjustment_lock:
            # 检查调整间隔
            time_since_last = time.time() - self._state.last_adjustment_time
            if time_since_last < 5.0:  # 至少5秒间隔
                return None

            # 分析性能
            adjustment = self._analyze_performance()

            if adjustment:
                # 应用调整
                self._apply_adjustment(adjustment)
                self._state.last_adjustment_time = time.time()

                # 记录调整历史
                self._state.adjustments.append({
                    "timestamp": time.time(),
                    "adjustment": adjustment,
                    "performance_before": {
                        "tps": self._state.avg_tokens_per_second,
                        "latency": self._state.avg_latency_ms,
                        "vram": self._state.avg_vram_usage,
                    },
                })

                # 学习最优配置
                if self.config.enable_learning:
                    self._learn_from_adjustment()

                logger.info("自适应调整: %s", adjustment)

            return adjustment

    def _analyze_performance(self) -> Optional[Dict[str, Any]]:
        """分析性能，决定是否需要调整

        Returns:
            Optional[Dict]: 调整建议
        """
        metrics = self._monitor.get_current()
        if metrics is None:
            return None

        avg = self._monitor.get_average(10)
        if not avg:
            return None

        adjustments = {}

        # 检查速度
        if avg["tokens_per_second"] < self.config.min_tokens_per_second:
            # 速度太慢，需要优化
            adjustments["reason"] = "speed_too_low"

            # 尝试降低量化级别
            current_idx = self.QUANT_LEVELS.index(self._state.quant_level)
            if current_idx > 0:
                new_idx = max(0, current_idx - self.config.quant_step)
                adjustments["quant_level"] = self.QUANT_LEVELS[new_idx]

            # 尝试减少上下文长度
            if self._state.context_length > 1024:
                adjustments["context_length"] = max(
                    1024,
                    self._state.context_length - self.config.context_step
                )

        # 检查延迟
        elif avg["latency_ms"] > self.config.max_latency_ms:
            # 延迟太高，需要优化
            adjustments["reason"] = "latency_too_high"

            # 尝试减少GPU层数
            if self._state.gpu_layers > 0:
                adjustments["gpu_layers"] = max(
                    0,
                    self._state.gpu_layers - self.config.gpu_layers_step
                )

        # 检查显存使用
        elif avg["vram_usage_gb"] > self.config.max_vram_usage_ratio * 8.0:  # 假设8GB显存
            # 显存使用过高，需要降级
            adjustments["reason"] = "vram_too_high"

            # 降低量化级别
            current_idx = self.QUANT_LEVELS.index(self._state.quant_level)
            if current_idx > 0:
                adjustments["quant_level"] = self.QUANT_LEVELS[current_idx - 1]

            # 减少上下文长度
            adjustments["context_length"] = max(
                512,
                self._state.context_length - self.config.context_step
            )

        # 检查是否可以升级
        elif (avg["tokens_per_second"] > self.config.min_tokens_per_second * 2 and
              avg["vram_usage_gb"] < self.config.max_vram_usage_ratio * 6.0):
            # 性能良好，可以尝试提升质量
            adjustments["reason"] = "can_improve_quality"

            # 提升量化级别
            current_idx = self.QUANT_LEVELS.index(self._state.quant_level)
            if current_idx < len(self.QUANT_LEVELS) - 1:
                new_idx = min(
                    len(self.QUANT_LEVELS) - 1,
                    current_idx + self.config.quant_step
                )
                adjustments["quant_level"] = self.QUANT_LEVELS[new_idx]

        return adjustments if adjustments else None

    def _apply_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """应用调整

        Args:
            adjustment: 调整配置
        """
        if "quant_level" in adjustment:
            self._state.quant_level = adjustment["quant_level"]

        if "context_length" in adjustment:
            self._state.context_length = adjustment["context_length"]

        if "gpu_layers" in adjustment:
            self._state.gpu_layers = adjustment["gpu_layers"]

        if "batch_size" in adjustment:
            self._state.batch_size = adjustment["batch_size"]

    def _learn_from_adjustment(self) -> None:
        """从调整中学习

        记录成功的配置，以便未来使用。
        """
        # 创建配置键
        config_key = f"{self._state.quant_level}_{self._state.context_length}_{self._state.gpu_layers}"

        # 检查调整是否成功
        avg = self._monitor.get_average(5)
        if avg and avg["tokens_per_second"] >= self.config.min_tokens_per_second:
            # 记录成功的配置
            self._learned_configs[config_key] = OptimizationState(
                quant_level=self._state.quant_level,
                context_length=self._state.context_length,
                gpu_layers=self._state.gpu_layers,
                batch_size=self._state.batch_size,
                avg_tokens_per_second=avg["tokens_per_second"],
                avg_latency_ms=avg["latency_ms"],
                avg_vram_usage=avg["vram_usage_gb"],
            )

            # 保存学习到的配置
            if self.config.config_path:
                self._save_learned_configs()

    def get_optimal_config(self, workload_type: str = "default") -> Optional[OptimizationState]:
        """获取最优配置

        Args:
            workload_type: 工作负载类型

        Returns:
            Optional[OptimizationState]: 最优配置
        """
        if workload_type in self._learned_configs:
            return self._learned_configs[workload_type]

        # 返回默认配置
        return None

    def _load_learned_configs(self) -> None:
        """加载学习到的配置"""
        if not self.config.config_path:
            return

        path = Path(self.config.config_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    for key, config_data in data.items():
                        self._learned_configs[key] = OptimizationState(**config_data)
                logger.info("加载学习配置: %d 条", len(self._learned_configs))
            except Exception as e:
                logger.warning("加载学习配置失败: %s", e)

    def _save_learned_configs(self) -> None:
        """保存学习到的配置"""
        if not self.config.config_path:
            return

        path = Path(self.config.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {}
            for key, state in self._learned_configs.items():
                data[key] = {
                    "quant_level": state.quant_level,
                    "context_length": state.context_length,
                    "gpu_layers": state.gpu_layers,
                    "batch_size": state.batch_size,
                    "avg_tokens_per_second": state.avg_tokens_per_second,
                    "avg_latency_ms": state.avg_latency_ms,
                    "avg_vram_usage": state.avg_vram_usage,
                }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info("保存学习配置: %d 条", len(data))
        except Exception as e:
            logger.warning("保存学习配置失败: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "current_state": {
                "quant_level": self._state.quant_level,
                "context_length": self._state.context_length,
                "gpu_layers": self._state.gpu_layers,
                "avg_tps": self._state.avg_tokens_per_second,
                "avg_latency": self._state.avg_latency_ms,
                "avg_vram": self._state.avg_vram_usage,
            },
            "adjustment_count": len(self._state.adjustments),
            "learned_configs_count": len(self._learned_configs),
            "monitor_stats": {
                "history_size": len(self._monitor._history),
                "current_metrics": self._monitor.get_current(),
            },
        }

    def get_recommendation(self) -> Dict[str, Any]:
        """获取优化建议

        Returns:
            Dict: 优化建议
        """
        avg = self._monitor.get_average(10)
        if not avg:
            return {"status": "insufficient_data"}

        recommendations = []

        # 速度建议
        if avg["tokens_per_second"] < self.config.min_tokens_per_second:
            recommendations.append({
                "type": "speed",
                "priority": "high",
                "suggestion": "降低量化级别或减少上下文长度以提升速度",
                "current_tps": avg["tokens_per_second"],
                "target_tps": self.config.min_tokens_per_second,
            })

        # 延迟建议
        if avg["latency_ms"] > self.config.max_latency_ms:
            recommendations.append({
                "type": "latency",
                "priority": "medium",
                "suggestion": "减少GPU层数或批处理大小以降低延迟",
                "current_latency": avg["latency_ms"],
                "target_latency": self.config.max_latency_ms,
            })

        # 显存建议
        if avg["vram_usage_gb"] > self.config.max_vram_usage_ratio * 8.0:
            recommendations.append({
                "type": "memory",
                "priority": "high",
                "suggestion": "降低量化级别或减少上下文长度以减少显存使用",
                "current_vram": avg["vram_usage_gb"],
                "target_vram": self.config.max_vram_usage_ratio * 8.0,
            })

        # 质量建议
        if (avg["tokens_per_second"] > self.config.min_tokens_per_second * 2 and
            avg["vram_usage_gb"] < self.config.max_vram_usage_ratio * 6.0):
            recommendations.append({
                "type": "quality",
                "priority": "low",
                "suggestion": "可以提升量化级别以获得更好的质量",
                "current_quant": self._state.quant_level,
            })

        return {
            "status": "ok",
            "recommendations": recommendations,
            "current_performance": avg,
        }


# ============================================================
# 便捷函数
# ============================================================

def create_adaptive_optimizer(
    goal: str = "balanced",
    min_tps: float = 5.0,
    max_latency_ms: float = 500.0,
    config_path: Optional[str] = None,
) -> AdaptiveOptimizer:
    """创建自适应优化器

    Args:
        goal: 优化目标 ("speed", "quality", "balanced", "memory")
        min_tps: 最小tokens/s
        max_latency_ms: 最大延迟
        config_path: 配置文件路径

    Returns:
        AdaptiveOptimizer: 优化器实例
    """
    goal_enum = OptimizationGoal(goal)
    config = AdaptiveConfig(
        goal=goal_enum,
        min_tokens_per_second=min_tps,
        max_latency_ms=max_latency_ms,
        config_path=config_path,
    )
    return AdaptiveOptimizer(config)
