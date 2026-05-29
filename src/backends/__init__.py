"""
推理后端模块

提供统一的推理后端抽象和具体实现。
当前支持 llama.cpp 后端，可扩展 vLLM、ExLlamaV2 等。
"""

from .base import (
    BackendType,
    GenerationResult,
    InferenceBackend,
    InferenceConfig,
    MemoryUsage,
)
from .llama_cpp import LlamaCppBackend

__all__ = [
    # base
    "InferenceBackend",
    "InferenceConfig",
    "GenerationResult",
    "MemoryUsage",
    "BackendType",
    # llama_cpp
    "LlamaCppBackend",
]
