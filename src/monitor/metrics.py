"""
Performance metrics collection module.
Collects CPU, memory, and GPU metrics with historical storage.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    tokens_per_second: float = 0.0
    ttft_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": round(self.memory_used_gb, 2),
            "memory_total_gb": round(self.memory_total_gb, 2),
            "gpu_percent": self.gpu_percent,
            "gpu_memory_percent": self.gpu_memory_percent,
            "gpu_memory_used_gb": round(self.gpu_memory_used_gb, 2),
            "gpu_memory_total_gb": round(self.gpu_memory_total_gb, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "ttft_ms": round(self.ttft_ms, 2),
        }


class MetricsCollector:
    """Collects and stores performance metrics."""

    def __init__(self, history_size: int = 300, collect_interval: float = 1.0):
        self._history: deque[PerformanceMetrics] = deque(maxlen=history_size)
        self._collect_interval = collect_interval
        self._collecting = False
        self._task: Optional[asyncio.Task] = None
        self._latest = PerformanceMetrics()

    def collect_once(self) -> PerformanceMetrics:
        """Collect current metrics synchronously."""
        mem = psutil.virtual_memory()
        metrics = PerformanceMetrics(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=mem.percent,
            memory_used_gb=mem.used / (1024 ** 3),
            memory_total_gb=mem.total / (1024 ** 3),
        )

        if HAS_TORCH and torch.cuda.is_available():
            try:
                metrics.gpu_percent = torch.cuda.utilization()
                gpu_mem = torch.cuda.mem_get_info()
                metrics.gpu_memory_used_gb = (gpu_mem[1] - gpu_mem[0]) / (1024 ** 3)
                metrics.gpu_memory_total_gb = gpu_mem[1] / (1024 ** 3)
                metrics.gpu_memory_percent = (
                    metrics.gpu_memory_used_gb / metrics.gpu_memory_total_gb * 100
                    if metrics.gpu_memory_total_gb > 0 else 0.0
                )
            except Exception:
                pass

        self._latest = metrics
        return metrics

    async def _collect_loop(self):
        """Background collection loop."""
        while self._collecting:
            metrics = await asyncio.to_thread(self.collect_once)
            self._history.append(metrics)
            await asyncio.sleep(self._collect_interval)

    async def start(self):
        """Start async collection."""
        if self._collecting:
            return
        self._collecting = True
        self._task = asyncio.create_task(self._collect_loop())

    async def stop(self):
        """Stop async collection."""
        self._collecting = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def get_current(self) -> PerformanceMetrics:
        """Get latest metrics."""
        return self._latest

    def get_history(self, count: Optional[int] = None) -> list[PerformanceMetrics]:
        """Get historical metrics."""
        if count is None:
            return list(self._history)
        return list(self._history)[-count:]

    def get_average(self, count: Optional[int] = None) -> PerformanceMetrics:
        """Get average of recent metrics."""
        history = self.get_history(count)
        if not history:
            return PerformanceMetrics()

        n = len(history)
        return PerformanceMetrics(
            cpu_percent=sum(m.cpu_percent for m in history) / n,
            memory_percent=sum(m.memory_percent for m in history) / n,
            memory_used_gb=sum(m.memory_used_gb for m in history) / n,
            memory_total_gb=history[-1].memory_total_gb,
            gpu_percent=sum(m.gpu_percent for m in history) / n,
            gpu_memory_percent=sum(m.gpu_memory_percent for m in history) / n,
            gpu_memory_used_gb=sum(m.gpu_memory_used_gb for m in history) / n,
            gpu_memory_total_gb=history[-1].gpu_memory_total_gb,
            tokens_per_second=sum(m.tokens_per_second for m in history) / n,
            ttft_ms=sum(m.ttft_ms for m in history) / n,
        )

    def update_inference_metrics(self, tokens_per_second: float, ttft_ms: float):
        """Update inference-related metrics on the latest snapshot."""
        self._latest.tokens_per_second = tokens_per_second
        self._latest.ttft_ms = ttft_ms
