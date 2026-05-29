"""配置管理模块

支持:
- YAML配置文件加载
- 环境变量覆盖 (前缀: LMOPT_)
- 默认配置
- 配置验证
- 内存优化子配置 (整合 offloader / kv_cache / dynamic_loader)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

from ..optimization.offloader import OffloadConfig, OffloadStrategy
from ..optimization.kv_cache import KVCacheConfig
from ..optimization.dynamic_loader import (
    DynamicConfig as _DynamicLoaderConfig,
    EvictionStrategy,
)


# ---------------------------------------------------------------------------
# 默认值常量
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, dict[str, Any]] = {
    "model": {
        "model_path": "",
        "context_length": 4096,
        "gpu_layers": 0,
        "threads": 4,
        "batch_size": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "mlock": False,
        # 内存优化相关配置
        "use_mmap": True,
        "rope_freq_base": 10000.0,
        "rope_freq_scale": 1.0,
        "verbose": False,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8000,
        "workers": 1,
    },
    "app": {
        "models_dir": "models",
        "log_level": "INFO",
        "log_file": "logs/app.log",
        "monitor_enabled": True,
        "monitor_interval": 10,
    },
    "memory_optimization": {
        "quantization": "q4_k_m",
        "offload": {
            "gpu_layers": -1,
            "cpu_threads": 4,
            "disk_offload": False,
            "swap_space_gb": 0.0,
            "prefetch_layers": 2,
            "strategy": "gpu_cpu",
            "context_length": 4096,
            "batch_size": 512,
        },
        "kv_cache": {
            "cache_bits": 16,
            "max_cache_size_gb": 2.0,
            "eviction_policy": "attention_score",
            "prefix_sharing": True,
            "compression_ratio": 0.5,
            "page_size": 64,
            "num_pages": 0,
        },
        "dynamic": {
            "max_gpu_layers": 20,
            "max_cpu_layers": 40,
            "prefetch_enabled": True,
            "prefetch_count": 2,
            "swap_enabled": True,
            "swap_directory": "./layer_cache",
            "layer_access_threshold": 3,
            "eviction_strategy": "lru",
            "pipeline_enabled": True,
            "batch_load_size": 4,
            "async_io_enabled": True,
        },
    },
}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """模型推理参数

    Attributes:
        model_path:       模型文件路径
        context_length:   上下文窗口大小
        gpu_layers:       卸载到GPU的层数 (-1=全部, 0=纯CPU)
        threads:          CPU推理线程数
        batch_size:       提示词处理批大小
        temperature:      采样温度
        top_p:            nucleus sampling 概率阈值
        top_k:            top-k 采样保留数
        repeat_penalty:   重复惩罚系数
        mlock:            是否将模型锁定在内存中
        use_mmap:         是否使用内存映射加载模型
        rope_freq_base:   RoPE 基础频率
        rope_freq_scale:  RoPE 频率缩放系数
        verbose:          是否启用 llama.cpp 详细日志
    """

    model_path: str = ""
    context_length: int = 4096
    gpu_layers: int = 0
    threads: int = 4
    batch_size: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    mlock: bool = False
    # 内存优化相关
    use_mmap: bool = True
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0
    verbose: bool = False

    # ---- 验证 ----
    def validate(self) -> list[str]:
        """返回错误消息列表, 空列表表示合法."""
        errors: list[str] = []
        if self.context_length < 1:
            errors.append("context_length 必须 >= 1")
        if self.gpu_layers < -1:
            errors.append("gpu_layers 必须 >= -1 (-1 表示全部GPU)")
        if self.threads < 1:
            errors.append("threads 必须 >= 1")
        if self.batch_size < 1:
            errors.append("batch_size 必须 >= 1")
        if not (0.0 <= self.temperature <= 2.0):
            errors.append("temperature 必须在 [0.0, 2.0] 范围内")
        if not (0.0 <= self.top_p <= 1.0):
            errors.append("top_p 必须在 [0.0, 1.0] 范围内")
        if self.top_k < 0:
            errors.append("top_k 必须 >= 0")
        if self.repeat_penalty < 1.0:
            errors.append("repeat_penalty 必须 >= 1.0")
        if self.rope_freq_base <= 0:
            errors.append("rope_freq_base 必须 > 0")
        if self.rope_freq_scale <= 0:
            errors.append("rope_freq_scale 必须 > 0")
        return errors


@dataclass
class OffloadConfigData:
    """卸载配置 (YAML 可序列化版本)

    与 optimization.offloader.OffloadConfig 对应, 但使用字符串表示策略枚举,
    以便 YAML 加载/保存。加载完成后通过 to_offload_config() 转换为实际类型。
    """

    gpu_layers: int = -1
    cpu_threads: int = 4
    disk_offload: bool = False
    swap_space_gb: float = 0.0
    prefetch_layers: int = 2
    strategy: str = "gpu_cpu"
    context_length: int = 4096
    batch_size: int = 512

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.gpu_layers < -1:
            errors.append("offload.gpu_layers 必须 >= -1")
        if self.cpu_threads < 1:
            errors.append("offload.cpu_threads 必须 >= 1")
        if self.swap_space_gb < 0:
            errors.append("offload.swap_space_gb 必须 >= 0")
        if self.prefetch_layers < 0:
            errors.append("offload.prefetch_layers 必须 >= 0")
        valid_strategies = {"gpu_only", "gpu_cpu", "gpu_cpu_disk", "cpu_only"}
        if self.strategy not in valid_strategies:
            errors.append(
                f"offload.strategy 必须是以下之一: {valid_strategies}"
            )
        if self.context_length < 1:
            errors.append("offload.context_length 必须 >= 1")
        if self.batch_size < 1:
            errors.append("offload.batch_size 必须 >= 1")
        return errors

    def to_offload_config(self) -> OffloadConfig:
        """转换为 optimization.offloader.OffloadConfig"""
        strategy_map = {
            "gpu_only": OffloadStrategy.GPU_ONLY,
            "gpu_cpu": OffloadStrategy.GPU_CPU,
            "gpu_cpu_disk": OffloadStrategy.GPU_CPU_DISK,
            "cpu_only": OffloadStrategy.CPU_ONLY,
        }
        return OffloadConfig(
            gpu_layers=self.gpu_layers,
            cpu_threads=self.cpu_threads,
            disk_offload=self.disk_offload,
            swap_space_gb=self.swap_space_gb,
            prefetch_layers=self.prefetch_layers,
            strategy=strategy_map.get(self.strategy, OffloadStrategy.GPU_CPU),
            context_length=self.context_length,
            batch_size=self.batch_size,
        )

    @classmethod
    def from_offload_config(cls, config: OffloadConfig) -> "OffloadConfigData":
        """从 OffloadConfig 构造"""
        return cls(
            gpu_layers=config.gpu_layers,
            cpu_threads=config.cpu_threads,
            disk_offload=config.disk_offload,
            swap_space_gb=config.swap_space_gb,
            prefetch_layers=config.prefetch_layers,
            strategy=config.strategy.value,
            context_length=config.context_length,
            batch_size=config.batch_size,
        )


@dataclass
class KVCacheConfigData:
    """KV Cache 配置 (YAML 可序列化版本)

    与 optimization.kv_cache.KVCacheConfig 对应。
    """

    cache_bits: int = 16
    max_cache_size_gb: float = 2.0
    eviction_policy: str = "attention_score"
    prefix_sharing: bool = True
    compression_ratio: float = 0.5
    page_size: int = 64
    num_pages: int = 0

    def validate(self) -> list[str]:
        errors: list[str] = []
        valid_bits = {4, 8, 16}
        if self.cache_bits not in valid_bits:
            errors.append(
                f"kv_cache.cache_bits 必须是以下之一: {valid_bits}"
            )
        if self.max_cache_size_gb <= 0:
            errors.append("kv_cache.max_cache_size_gb 必须 > 0")
        valid_policies = {"lru", "lfu", "attention_score"}
        if self.eviction_policy not in valid_policies:
            errors.append(
                f"kv_cache.eviction_policy 必须是以下之一: {valid_policies}"
            )
        if not (0.0 < self.compression_ratio <= 1.0):
            errors.append("kv_cache.compression_ratio 必须在 (0.0, 1.0] 范围内")
        if self.page_size < 1:
            errors.append("kv_cache.page_size 必须 >= 1")
        if self.num_pages < 0:
            errors.append("kv_cache.num_pages 必须 >= 0")
        return errors

    def to_kv_cache_config(self) -> KVCacheConfig:
        """转换为 optimization.kv_cache.KVCacheConfig"""
        return KVCacheConfig(
            cache_bits=self.cache_bits,
            max_cache_size_gb=self.max_cache_size_gb,
            eviction_policy=self.eviction_policy,
            prefix_sharing=self.prefix_sharing,
            compression_ratio=self.compression_ratio,
            page_size=self.page_size,
            num_pages=self.num_pages,
        )

    @classmethod
    def from_kv_cache_config(cls, config: KVCacheConfig) -> "KVCacheConfigData":
        """从 KVCacheConfig 构造"""
        return cls(
            cache_bits=config.cache_bits,
            max_cache_size_gb=config.max_cache_size_gb,
            eviction_policy=config.eviction_policy,
            prefix_sharing=config.prefix_sharing,
            compression_ratio=config.compression_ratio,
            page_size=config.page_size,
            num_pages=config.num_pages,
        )


@dataclass
class DynamicConfigData:
    """动态层加载配置 (YAML 可序列化版本)

    与 optimization.dynamic_loader.DynamicConfig 对应, 但使用字符串表示枚举。
    """

    max_gpu_layers: int = 20
    max_cpu_layers: int = 40
    prefetch_enabled: bool = True
    prefetch_count: int = 2
    swap_enabled: bool = True
    swap_directory: str = "./layer_cache"
    layer_access_threshold: int = 3
    eviction_strategy: str = "lru"
    pipeline_enabled: bool = True
    batch_load_size: int = 4
    async_io_enabled: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.max_gpu_layers < 0:
            errors.append("dynamic.max_gpu_layers 必须 >= 0")
        if self.max_cpu_layers < 0:
            errors.append("dynamic.max_cpu_layers 必须 >= 0")
        if self.prefetch_count < 0:
            errors.append("dynamic.prefetch_count 必须 >= 0")
        if self.layer_access_threshold < 1:
            errors.append("dynamic.layer_access_threshold 必须 >= 1")
        valid_strategies = {"lru", "lfu", "adaptive"}
        if self.eviction_strategy not in valid_strategies:
            errors.append(
                f"dynamic.eviction_strategy 必须是以下之一: {valid_strategies}"
            )
        if self.batch_load_size < 1:
            errors.append("dynamic.batch_load_size 必须 >= 1")
        return errors

    def to_dynamic_config(self) -> _DynamicLoaderConfig:
        """转换为 optimization.dynamic_loader.DynamicConfig"""
        strategy_map = {
            "lru": EvictionStrategy.LRU,
            "lfu": EvictionStrategy.LFU,
            "adaptive": EvictionStrategy.ADAPTIVE,
        }
        return _DynamicLoaderConfig(
            max_gpu_layers=self.max_gpu_layers,
            max_cpu_layers=self.max_cpu_layers,
            prefetch_enabled=self.prefetch_enabled,
            prefetch_count=self.prefetch_count,
            swap_enabled=self.swap_enabled,
            swap_directory=self.swap_directory,
            layer_access_threshold=self.layer_access_threshold,
            eviction_strategy=strategy_map.get(
                self.eviction_strategy, EvictionStrategy.LRU
            ),
            pipeline_enabled=self.pipeline_enabled,
            batch_load_size=self.batch_load_size,
            async_io_enabled=self.async_io_enabled,
        )

    @classmethod
    def from_dynamic_config(cls, config: _DynamicLoaderConfig) -> "DynamicConfigData":
        """从 DynamicConfig 构造"""
        return cls(
            max_gpu_layers=config.max_gpu_layers,
            max_cpu_layers=config.max_cpu_layers,
            prefetch_enabled=config.prefetch_enabled,
            prefetch_count=config.prefetch_count,
            swap_enabled=config.swap_enabled,
            swap_directory=config.swap_directory,
            layer_access_threshold=config.layer_access_threshold,
            eviction_strategy=config.eviction_strategy.value,
            pipeline_enabled=config.pipeline_enabled,
            batch_load_size=config.batch_load_size,
            async_io_enabled=config.async_io_enabled,
        )


@dataclass
class MemoryOptimizationConfig:
    """内存优化配置

    整合卸载策略、KV Cache 优化、动态层加载和量化级别配置。

    Attributes:
        offload_config:    模型卸载配置 (GPU/CPU/Disk 分配)
        kv_cache_config:   KV Cache 优化配置 (分页/量化/压缩)
        dynamic_config:    动态层加载配置 (LRU淘汰/预测预取)
        quantization:      量化级别 (如 q4_k_m, q8_0, fp16)
    """

    offload_config: OffloadConfigData = field(default_factory=OffloadConfigData)
    kv_cache_config: KVCacheConfigData = field(default_factory=KVCacheConfigData)
    dynamic_config: DynamicConfigData = field(default_factory=DynamicConfigData)
    quantization: str = "q4_k_m"

    # 有效量化级别
    _VALID_QUANTIZATIONS: set[str] = field(
        default_factory=lambda: {
            "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
            "q4_0", "q4_k_s", "q4_k_m",
            "q5_0", "q5_k_s", "q5_k_m",
            "q6_k", "q8_0", "fp16",
            "gptq-2bit", "gptq-3bit", "gptq-4bit", "gptq-8bit",
            "awq-4bit",
            "bnb-8bit", "bnb-4bit",
        },
        repr=False,
    )

    def validate(self) -> list[str]:
        """返回错误消息列表, 空列表表示合法."""
        errors: list[str] = []
        errors.extend(self.offload_config.validate())
        errors.extend(self.kv_cache_config.validate())
        errors.extend(self.dynamic_config.validate())

        quant_lower = self.quantization.lower().replace("-", "_").replace(" ", "_")
        # 也接受大写形式
        quant_normalized = self.quantization.upper().replace("-", "_").replace(" ", "_")
        valid_upper = {q.upper() for q in self._VALID_QUANTIZATIONS}
        if quant_lower not in self._VALID_QUANTIZATIONS and quant_normalized not in valid_upper:
            errors.append(
                f"quantization '{self.quantization}' 不是有效量化级别, "
                f"可选: {sorted(self._VALID_QUANTIZATIONS)}"
            )
        return errors

    def get_offload_config(self) -> OffloadConfig:
        """获取 optimization.offloader.OffloadConfig 实例"""
        return self.offload_config.to_offload_config()

    def get_kv_cache_config(self) -> KVCacheConfig:
        """获取 optimization.kv_cache.KVCacheConfig 实例"""
        return self.kv_cache_config.to_kv_cache_config()

    def get_dynamic_config(self) -> _DynamicLoaderConfig:
        """获取 optimization.dynamic_loader.DynamicConfig 实例"""
        return self.dynamic_config.to_dynamic_config()


@dataclass
class ServerConfig:
    """HTTP 服务配置

    Attributes:
        host:     监听地址
        port:     监听端口
        workers:  uvicorn worker 数量
    """

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not (0 < self.port <= 65535):
            errors.append("port 必须在 1-65535 范围内")
        if self.workers < 1:
            errors.append("workers 必须 >= 1")
        return errors


@dataclass
class AppConfig:
    """应用主配置

    Attributes:
        model:                模型推理配置
        server:               HTTP 服务配置
        memory_optimization:  内存优化配置
        models_dir:           模型文件目录
        log_level:            日志级别
        log_file:             日志文件路径
        monitor_enabled:      是否启用监控
        monitor_interval:     监控采集间隔 (秒)
    """

    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    memory_optimization: MemoryOptimizationConfig = field(
        default_factory=MemoryOptimizationConfig
    )

    # 顶层应用配置
    models_dir: str = "models"
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    monitor_enabled: bool = True
    monitor_interval: int = 10  # 秒

    # ---- 校验 ----
    def validate(self) -> list[str]:
        """合并所有子配置的验证结果."""
        errors: list[str] = []
        errors.extend(self.model.validate())
        errors.extend(self.server.validate())
        errors.extend(self.memory_optimization.validate())

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            errors.append(f"log_level 必须是以下之一: {valid_levels}")
        if self.monitor_interval < 1:
            errors.append("monitor_interval 必须 >= 1")
        return errors


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    """递归合并 override 到 base, 返回新字典."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(cfg_dict: dict[str, Any], prefix: str = "LMOPT_") -> dict[str, Any]:
    """读取环境变量并覆盖对应配置键.

    环境变量命名规则:
        LMOPT_MODEL__TEMPERATURE          -> model.temperature
        LMOPT_SERVER__PORT                -> server.port
        LMOPT_MEMORY_OPTIMIZATION__QUANTIZATION -> memory_optimization.quantization
        LMOPT_MEMORY_OPTIMIZATION__OFFLOAD__GPU_LAYERS -> memory_optimization.offload.gpu_layers
        LMOPT_LOG_LEVEL                   -> log_level (顶层)
    双下划线 "__" 映射为嵌套层级.
    """
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        target = cfg_dict
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        leaf_key = parts[-1]
        # 简单类型推断
        if isinstance(target.get(leaf_key), bool):
            target[leaf_key] = value.lower() in ("1", "true", "yes")
        elif isinstance(target.get(leaf_key), int):
            try:
                target[leaf_key] = int(value)
            except ValueError:
                pass
        elif isinstance(target.get(leaf_key), float):
            try:
                target[leaf_key] = float(value)
            except ValueError:
                pass
        else:
            target[leaf_key] = value
    return cfg_dict


def _build_memory_opt_config(data: dict[str, Any]) -> MemoryOptimizationConfig:
    """从字典构造 MemoryOptimizationConfig (支持嵌套)."""
    offload_data = data.get("offload", {})
    kv_data = data.get("kv_cache", {})
    dynamic_data = data.get("dynamic", {})

    offload = OffloadConfigData(**{
        k: v for k, v in offload_data.items()
        if k in {f.name for f in fields(OffloadConfigData)}
    })
    kv = KVCacheConfigData(**{
        k: v for k, v in kv_data.items()
        if k in {f.name for f in fields(KVCacheConfigData)}
    })
    dynamic = DynamicConfigData(**{
        k: v for k, v in dynamic_data.items()
        if k in {f.name for f in fields(DynamicConfigData)}
    })

    return MemoryOptimizationConfig(
        offload_config=offload,
        kv_cache_config=kv,
        dynamic_config=dynamic,
        quantization=data.get("quantization", "q4_k_m"),
    )


# ---------------------------------------------------------------------------
# 公共加载 API
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """加载并返回 AppConfig 实例.

    加载顺序 (后者覆盖前者):
        1. 内置默认值
        2. YAML 配置文件 (如果提供)
        3. 环境变量 (前缀 LMOPT_)

    Args:
        config_path: YAML 配置文件路径, None 则仅使用默认值+环境变量.

    Raises:
        FileNotFoundError: 指定的配置文件不存在.
        ValueError: 配置验证失败.
    """
    # 1) 默认值
    cfg_dict: dict[str, Any] = {
        "model": _DEFAULTS["model"].copy(),
        "server": _DEFAULTS["server"].copy(),
        "app": _DEFAULTS["app"].copy(),
        "memory_optimization": _deep_update({}, _DEFAULTS["memory_optimization"]),
    }

    # 2) YAML 文件
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        # "model" / "server" / "memory_optimization" 嵌套, 其余放 "app"
        if "model" in file_cfg:
            cfg_dict["model"] = _deep_update(cfg_dict["model"], file_cfg.pop("model"))
        if "server" in file_cfg:
            cfg_dict["server"] = _deep_update(cfg_dict["server"], file_cfg.pop("server"))
        if "memory_optimization" in file_cfg:
            cfg_dict["memory_optimization"] = _deep_update(
                cfg_dict["memory_optimization"], file_cfg.pop("memory_optimization")
            )
        if file_cfg:
            cfg_dict["app"] = _deep_update(cfg_dict["app"], file_cfg)

    # 3) 环境变量覆盖
    cfg_dict = _apply_env_overrides(cfg_dict)

    # 4) 构造 dataclass
    model_cfg = ModelConfig(**{
        k: v for k, v in cfg_dict["model"].items()
        if k in {f.name for f in fields(ModelConfig)}
    })
    server_cfg = ServerConfig(**{
        k: v for k, v in cfg_dict["server"].items()
        if k in {f.name for f in fields(ServerConfig)}
    })
    memory_opt_cfg = _build_memory_opt_config(cfg_dict["memory_optimization"])

    app_cfg = AppConfig(
        model=model_cfg,
        server=server_cfg,
        memory_optimization=memory_opt_cfg,
        **{
            k: v for k, v in cfg_dict["app"].items()
            if k in {f.name for f in fields(AppConfig)} and k not in ("model", "server", "memory_optimization")
        },
    )

    # 5) 验证
    errors = app_cfg.validate()
    if errors:
        raise ValueError("配置验证失败:\n  - " + "\n  - ".join(errors))

    return app_cfg


def default_config() -> AppConfig:
    """返回纯默认配置 (不读文件, 不读环境变量)."""
    return AppConfig(
        model=ModelConfig(**{
            k: v for k, v in _DEFAULTS["model"].items()
            if k in {f.name for f in fields(ModelConfig)}
        }),
        server=ServerConfig(**{
            k: v for k, v in _DEFAULTS["server"].items()
            if k in {f.name for f in fields(ServerConfig)}
        }),
        memory_optimization=_build_memory_opt_config(_DEFAULTS["memory_optimization"]),
        **_DEFAULTS["app"],
    )


def generate_sample_yaml(dest: str | Path = "config.yaml") -> Path:
    """生成示例 YAML 配置文件, 返回路径."""
    sample = {
        "model": _DEFAULTS["model"],
        "server": _DEFAULTS["server"],
        "memory_optimization": _DEFAULTS["memory_optimization"],
        **_DEFAULTS["app"],
    }
    path = Path(dest)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(sample, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return path


if __name__ == "__main__":
    # 快速演示
    from dataclasses import asdict

    print("=== 默认配置 ===")
    cfg = default_config()
    print(asdict(cfg))

    print("\n=== 内存优化配置 ===")
    mem_cfg = cfg.memory_optimization
    print(f"  量化级别: {mem_cfg.quantization}")
    print(f"  卸载策略: {mem_cfg.offload_config.strategy}")
    print(f"  KV Cache: {mem_cfg.kv_cache_config.cache_bits}bit")
    print(f"  动态加载: max_gpu={mem_cfg.dynamic_config.max_gpu_layers}")

    print("\n=== 转换为优化模块类型 ===")
    offload = mem_cfg.get_offload_config()
    print(f"  OffloadConfig: strategy={offload.strategy.value}")
    kv = mem_cfg.get_kv_cache_config()
    print(f"  KVCacheConfig: bits={kv.cache_bits}")
    dyn = mem_cfg.get_dynamic_config()
    print(f"  DynamicConfig: strategy={dyn.eviction_strategy.value}")

    print("\n=== 生成示例 YAML ===")
    p = generate_sample_yaml("config_sample.yaml")
    print(f"已写入: {p}")

    print("\n=== 从 YAML 加载 ===")
    cfg2 = load_config(p)
    print(asdict(cfg2))
