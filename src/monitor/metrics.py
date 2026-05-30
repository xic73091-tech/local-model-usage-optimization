"""
Performance metrics collection module.
Collects CPU, memory, and GPU metrics with historical storage.
Supports NVIDIA (torch.cuda / pynvml), AMD (rocm-smi), and Apple Silicon.
"""

import asyncio
import os
import platform
import re
import shutil
import subprocess
import sys
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

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


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

        # GPU metrics: 按优先级尝试 pynvml → torch.cuda → rocm-smi → Apple
        self._collect_gpu_metrics(metrics)

        self._latest = metrics
        return metrics

    def _collect_gpu_metrics(self, metrics: PerformanceMetrics) -> None:
        """收集GPU指标，自动适配NVIDIA/AMD/Apple。"""
        # --- NVIDIA: pynvml (轻量，无需torch) ---
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_percent = util.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics.gpu_memory_total_gb = mem_info.total / (1024 ** 3)
                metrics.gpu_memory_used_gb = mem_info.used / (1024 ** 3)
                metrics.gpu_memory_percent = (
                    metrics.gpu_memory_used_gb / metrics.gpu_memory_total_gb * 100
                    if metrics.gpu_memory_total_gb > 0 else 0.0
                )
                pynvml.nvmlShutdown()
                return
            except Exception:
                pass

        # --- NVIDIA: torch.cuda 回退 ---
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
                return
            except Exception:
                pass

        # --- AMD: rocm-smi ---
        rocm_smi = shutil.which("rocm-smi")
        if rocm_smi:
            try:
                result = subprocess.run(
                    [rocm_smi, "--showmeminfo", "vram", "--showuse"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    total, used = 0.0, 0.0
                    for line in result.stdout.splitlines():
                        ll = line.lower()
                        if "total" in ll:
                            m = re.search(r"([\d.]+)\s*(MB|GB|MiB|GiB)", line, re.IGNORECASE)
                            if m:
                                val = float(m.group(1))
                                if "gb" in m.group(2).lower():
                                    val *= 1024
                                total = val
                        elif "used" in ll:
                            m = re.search(r"([\d.]+)\s*(MB|GB|MiB|GiB)", line, re.IGNORECASE)
                            if m:
                                val = float(m.group(1))
                                if "gb" in m.group(2).lower():
                                    val *= 1024
                                used = val
                    if total > 0:
                        metrics.gpu_memory_total_gb = total / 1024
                        metrics.gpu_memory_used_gb = used / 1024
                        metrics.gpu_memory_percent = (used / total * 100) if total > 0 else 0.0
                        # GPU利用率需要通过 rocm-smi 的其他参数获取
                        try:
                            util_result = subprocess.run(
                                [rocm_smi, "--showuse"],
                                capture_output=True, text=True, timeout=10,
                            )
                            for line in util_result.stdout.splitlines():
                                if "gpu use" in line.lower() or "gpu %" in line.lower():
                                    m = re.search(r"(\d+)%", line)
                                    if m:
                                        metrics.gpu_percent = float(m.group(1))
                        except Exception:
                            pass
                        return
            except Exception:
                pass

        # --- AMD Windows: PowerShell CIM (基本显存信息，无实时利用率) ---
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-CimInstance Win32_VideoController | Where-Object {"
                     "$_.Name -match 'AMD|Radeon'} | "
                     "Select-Object -First 1 Name,AdapterRAM | ConvertTo-Json"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    import json
                    data = json.loads(result.stdout)
                    if isinstance(data, dict) and data.get("Name"):
                        adapter_ram = data.get("AdapterRAM", 0)
                        if adapter_ram and adapter_ram > 0:
                            metrics.gpu_memory_total_gb = adapter_ram / (1024 ** 3)
                            # Windows 无法通过标准API获取AMD实时显存使用量
                            # 使用 psutil 系统内存使用率作为近似
                            metrics.gpu_memory_percent = psutil.virtual_memory().percent
                            metrics.gpu_percent = 0.0  # 无实时利用率数据
                            return
            except Exception:
                pass

        # --- Apple Silicon ---
        if sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                # 统一内存: 从 sysctl 获取总内存
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.stdout.strip().isdigit():
                    total_bytes = int(result.stdout.strip())
                    metrics.gpu_memory_total_gb = total_bytes / (1024 ** 3)

                    # 通过 vm_stat 获取已用内存近似值
                    vm_result = subprocess.run(
                        ["vm_stat"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if vm_result.returncode == 0:
                        page_size = 16384
                        free_pages, active_pages, wired_pages = 0, 0, 0
                        for line in vm_result.stdout.splitlines():
                            if "page size of" in line.lower():
                                m_match = re.search(r"(\d+)\s+bytes", line)
                                if m_match:
                                    page_size = int(m_match.group(1))
                            elif "Pages free" in line:
                                m_match = re.search(r"(\d+)", line)
                                if m_match:
                                    free_pages = int(m_match.group(1))
                            elif "Pages active" in line:
                                m_match = re.search(r"(\d+)", line)
                                if m_match:
                                    active_pages = int(m_match.group(1))
                            elif "Pages wired" in line:
                                m_match = re.search(r"(\d+)", line)
                                if m_match:
                                    wired_pages = int(m_match.group(1))
                        used_bytes = (active_pages + wired_pages) * page_size
                        metrics.gpu_memory_used_gb = used_bytes / (1024 ** 3)
                        metrics.gpu_memory_percent = (
                            metrics.gpu_memory_used_gb / metrics.gpu_memory_total_gb * 100
                            if metrics.gpu_memory_total_gb > 0 else 0.0
                        )
                        # Apple 没有标准API获取GPU利用率，使用内存利用率近似
                        metrics.gpu_percent = metrics.gpu_memory_percent
                        return
            except Exception:
                pass

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
