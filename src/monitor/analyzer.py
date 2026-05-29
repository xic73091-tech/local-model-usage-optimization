"""
Performance analyzer module.
Identifies bottlenecks and generates optimization suggestions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .metrics import MetricsCollector, PerformanceMetrics


class BottleneckType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    GPU_MEMORY = "gpu_memory"
    NONE = "none"


@dataclass
class Bottleneck:
    """Identified performance bottleneck."""
    type: BottleneckType
    severity: float  # 0.0 ~ 1.0
    current_value: float
    threshold: float
    description: str


@dataclass
class Suggestion:
    """Optimization suggestion."""
    priority: int  # 1=highest
    category: str
    title: str
    description: str
    expected_impact: str


@dataclass
class AnalysisResult:
    """Structured analysis result."""
    bottlenecks: list[Bottleneck]
    suggestions: list[Suggestion]
    summary: str

    def to_dict(self) -> dict:
        return {
            "bottlenecks": [
                {
                    "type": b.type.value,
                    "severity": round(b.severity, 2),
                    "current_value": round(b.current_value, 2),
                    "threshold": b.threshold,
                    "description": b.description,
                }
                for b in self.bottlenecks
            ],
            "suggestions": [
                {
                    "priority": s.priority,
                    "category": s.category,
                    "title": s.title,
                    "description": s.description,
                    "expected_impact": s.expected_impact,
                }
                for s in self.suggestions
            ],
            "summary": self.summary,
        }


class PerformanceAnalyzer:
    """Analyzes performance metrics and generates optimization suggestions."""

    # Thresholds
    CPU_HIGH = 85.0
    MEMORY_HIGH = 80.0
    MEMORY_CRITICAL = 90.0
    GPU_HIGH = 90.0
    GPU_MEMORY_HIGH = 80.0
    GPU_MEMORY_CRITICAL = 90.0
    TPS_LOW = 10.0
    TTFT_HIGH = 2000.0  # ms

    def __init__(self, collector: MetricsCollector):
        self._collector = collector

    def analyze(self, sample_count: int = 60) -> AnalysisResult:
        """Run full performance analysis."""
        metrics = self._collector.get_average(sample_count or None)
        current = self._collector.get_current()

        bottlenecks = self._detect_bottlenecks(metrics, current)
        suggestions = self._generate_suggestions(bottlenecks, metrics)
        summary = self._build_summary(bottlenecks, metrics)

        return AnalysisResult(
            bottlenecks=bottlenecks,
            suggestions=suggestions,
            summary=summary,
        )

    def _detect_bottlenecks(
        self, avg: PerformanceMetrics, current: PerformanceMetrics
    ) -> list[Bottleneck]:
        bottlenecks = []

        # CPU
        if avg.cpu_percent > self.CPU_HIGH:
            severity = min((avg.cpu_percent - self.CPU_HIGH) / (100 - self.CPU_HIGH), 1.0)
            bottlenecks.append(Bottleneck(
                type=BottleneckType.CPU,
                severity=severity,
                current_value=avg.cpu_percent,
                threshold=self.CPU_HIGH,
                description=f"CPU usage avg {avg.cpu_percent:.1f}% exceeds {self.CPU_HIGH}%",
            ))

        # Memory
        if avg.memory_percent > self.MEMORY_HIGH:
            severity = min((avg.memory_percent - self.MEMORY_HIGH) / (100 - self.MEMORY_HIGH), 1.0)
            bottlenecks.append(Bottleneck(
                type=BottleneckType.MEMORY,
                severity=severity,
                current_value=avg.memory_percent,
                threshold=self.MEMORY_HIGH,
                description=(
                    f"Memory usage avg {avg.memory_percent:.1f}% "
                    f"({avg.memory_used_gb:.1f}/{avg.memory_total_gb:.1f} GB)"
                ),
            ))

        # GPU utilization
        if avg.gpu_percent > self.GPU_HIGH:
            severity = min((avg.gpu_percent - self.GPU_HIGH) / (100 - self.GPU_HIGH), 1.0)
            bottlenecks.append(Bottleneck(
                type=BottleneckType.GPU,
                severity=severity,
                current_value=avg.gpu_percent,
                threshold=self.GPU_HIGH,
                description=f"GPU utilization avg {avg.gpu_percent:.1f}% exceeds {self.GPU_HIGH}%",
            ))

        # GPU memory
        if avg.gpu_memory_percent > self.GPU_MEMORY_HIGH:
            severity = min(
                (avg.gpu_memory_percent - self.GPU_MEMORY_HIGH) / (100 - self.GPU_MEMORY_HIGH), 1.0
            )
            bottlenecks.append(Bottleneck(
                type=BottleneckType.GPU_MEMORY,
                severity=severity,
                current_value=avg.gpu_memory_percent,
                threshold=self.GPU_MEMORY_HIGH,
                description=(
                    f"GPU memory avg {avg.gpu_memory_percent:.1f}% "
                    f"({avg.gpu_memory_used_gb:.1f}/{avg.gpu_memory_total_gb:.1f} GB)"
                ),
            ))

        # Inference speed
        if 0 < avg.tokens_per_second < self.TPS_LOW:
            severity = min((self.TPS_LOW - avg.tokens_per_second) / self.TPS_LOW, 1.0)
            bottlenecks.append(Bottleneck(
                type=BottleneckType.GPU,  # usually GPU-bound
                severity=severity,
                current_value=avg.tokens_per_second,
                threshold=self.TPS_LOW,
                description=f"Token throughput {avg.tokens_per_second:.1f} t/s below {self.TPS_LOW} t/s",
            ))

        # TTFT
        if avg.ttft_ms > self.TTFT_HIGH:
            severity = min((avg.ttft_ms - self.TTFT_HIGH) / self.TTFT_HIGH, 1.0)
            bottlenecks.append(Bottleneck(
                type=BottleneckType.GPU,
                severity=severity,
                current_value=avg.ttft_ms,
                threshold=self.TTFT_HIGH,
                description=f"TTFT {avg.ttft_ms:.0f}ms exceeds {self.TTFT_HIGH:.0f}ms",
            ))

        bottlenecks.sort(key=lambda b: b.severity, reverse=True)
        return bottlenecks

    def _generate_suggestions(
        self, bottlenecks: list[Bottleneck], metrics: PerformanceMetrics
    ) -> list[Suggestion]:
        suggestions = []
        types = {b.type for b in bottlenecks}

        if BottleneckType.GPU_MEMORY in types:
            suggestions.append(Suggestion(
                priority=1,
                category="model",
                title="Enable quantization",
                description="Use GPTQ/AWQ 4-bit quantization to reduce VRAM footprint by ~60%.",
                expected_impact="VRAM reduction 50-70%, slight quality trade-off",
            ))
            suggestions.append(Suggestion(
                priority=2,
                category="model",
                title="Reduce context length",
                description="Shorter context windows reduce KV-cache VRAM significantly.",
                expected_impact="VRAM reduction proportional to context cut",
            ))

        if BottleneckType.MEMORY in types:
            suggestions.append(Suggestion(
                priority=1,
                category="offloading",
                title="Enable CPU offloading",
                description="Offload model layers to CPU/RAM when VRAM is insufficient (e.g. llama.cpp --n-gpu-layers).",
                expected_impact="Reduces VRAM usage at cost of slower inference",
            ))
            suggestions.append(Suggestion(
                priority=2,
                category="offloading",
                title="Use disk offloading",
                description="Enable memory-mapped model loading to reduce RAM pressure on large models.",
                expected_impact="Lower RAM footprint, slight latency increase",
            ))

        if BottleneckType.GPU in types:
            if metrics.tokens_per_second > 0:
                suggestions.append(Suggestion(
                    priority=1,
                    category="inference",
                    title="Use flash attention",
                    description="Flash Attention 2/3 improves throughput and reduces memory.",
                    expected_impact="10-30% throughput improvement",
                ))
            suggestions.append(Suggestion(
                priority=2,
                category="inference",
                title="Optimize batch size",
                description="Tune batch size to maximize GPU utilization without OOM.",
                expected_impact="Better GPU utilization",
            ))

        if BottleneckType.CPU in types:
            suggestions.append(Suggestion(
                priority=2,
                category="system",
                title="Offload to GPU",
                description="Move more computation to GPU; use GPU offloading for model layers.",
                expected_impact="CPU relief, faster inference",
            ))
            suggestions.append(Suggestion(
                priority=3,
                category="system",
                title="Reduce background processes",
                description="Close unnecessary applications to free CPU cores.",
                expected_impact="5-15% CPU freed",
            ))

        if metrics.ttft_ms > self.TTFT_HIGH:
            suggestions.append(Suggestion(
                priority=1,
                category="inference",
                title="Use speculative decoding",
                description="Draft model predicts tokens ahead, reducing TTFT.",
                expected_impact="20-50% TTFT reduction",
            ))

        if 0 < metrics.tokens_per_second < self.TPS_LOW:
            suggestions.append(Suggestion(
                priority=1,
                category="speed",
                title="Use smaller quantized model",
                description="Switch to a smaller model or higher quantization level (e.g. Q4_K_M) for faster generation.",
                expected_impact="2-5x throughput improvement with quality trade-off",
            ))
            suggestions.append(Suggestion(
                priority=2,
                category="speed",
                title="Increase GPU layers offload",
                description="Offload more layers to GPU (--n-gpu-layers) to maximize hardware utilization.",
                expected_impact="Significant speed boost if VRAM allows",
            ))

        if not bottlenecks:
            suggestions.append(Suggestion(
                priority=3,
                category="general",
                title="System healthy",
                description="No bottlenecks detected. Consider pushing batch size or context length.",
                expected_impact="Potential throughput increase",
            ))

        suggestions.sort(key=lambda s: s.priority)
        return suggestions

    def _build_summary(
        self, bottlenecks: list[Bottleneck], metrics: PerformanceMetrics
    ) -> str:
        if not bottlenecks:
            return (
                f"System healthy. CPU {metrics.cpu_percent:.0f}%, "
                f"Memory {metrics.memory_percent:.0f}%, "
                f"GPU {metrics.gpu_percent:.0f}%, "
                f"GPU-mem {metrics.gpu_memory_percent:.0f}%, "
                f"{metrics.tokens_per_second:.1f} t/s, "
                f"TTFT {metrics.ttft_ms:.0f}ms."
            )

        primary = bottlenecks[0]
        return (
            f"Primary bottleneck: {primary.type.value} ({primary.severity:.0%} severity). "
            f"CPU {metrics.cpu_percent:.0f}%, Memory {metrics.memory_percent:.0f}%, "
            f"GPU {metrics.gpu_percent:.0f}%, GPU-mem {metrics.gpu_memory_percent:.0f}%, "
            f"{metrics.tokens_per_second:.1f} t/s, TTFT {metrics.ttft_ms:.0f}ms."
        )
