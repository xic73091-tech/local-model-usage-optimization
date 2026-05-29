"""
本地模型使用优化 - 核心模块

提供硬件检测、模型管理和配置管理功能。
"""

from .config import (
    AppConfig,
    ModelConfig,
    ServerConfig,
    MemoryOptimizationConfig,
    OffloadConfigData,
    KVCacheConfigData,
    DynamicConfigData,
    default_config,
    generate_sample_yaml,
    load_config,
)
from .hardware_detector import (
    CPUInfo,
    DiskSpeedInfo,
    GPUInfo,
    GPUVendor,
    HardwareDetector,
    HardwareProfile,
    MemoryInfo,
    ModelQuant,
    ModelRecommendation,
)
from .model_manager import (
    ModelFormat,
    ModelInfo,
    ModelManager,
)

__all__ = [
    # config
    "AppConfig",
    "ModelConfig",
    "ServerConfig",
    "MemoryOptimizationConfig",
    "OffloadConfigData",
    "KVCacheConfigData",
    "DynamicConfigData",
    "load_config",
    "default_config",
    "generate_sample_yaml",
    # hardware_detector
    "HardwareDetector",
    "HardwareProfile",
    "GPUInfo",
    "GPUVendor",
    "CPUInfo",
    "MemoryInfo",
    "DiskSpeedInfo",
    "ModelQuant",
    "ModelRecommendation",
    # model_manager
    "ModelManager",
    "ModelInfo",
    "ModelFormat",
]
