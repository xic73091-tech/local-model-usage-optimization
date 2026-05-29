"""
模型管理模块

负责本地模型的扫描、元数据解析、显存估算、下载和格式转换。
支持 GGUF / SafeTensors / ONNX 三种格式。
"""

from __future__ import annotations

import enum
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# 常量 / 配置
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# 每种量化级别对应的 **每参数字节数**
_QUANT_BYTES_PER_PARAM: Dict[str, float] = {
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "Q8_0": 1.0,
    "Q6_K": 0.75,
    "Q5_K_M": 0.625,
    "Q5_K_S": 0.625,
    "Q5_0": 0.625,
    "Q4_K_M": 0.5,
    "Q4_K_S": 0.5,
    "Q4_0": 0.5,
    "Q3_K_L": 0.375,
    "Q3_K_M": 0.375,
    "Q3_K_S": 0.375,
    "Q2_K": 0.25,
    "IQ4_XS": 0.4375,
    "IQ3_XXS": 0.3125,
    "IQ2_XXS": 0.1875,
}

# 显存开销系数：模型权重 + KV Cache + 运行时开销
_VRAM_OVERHEAD_FACTOR = 1.2

# 安全阈值：推荐时留出的显存余量比例
_VRAM_SAFETY_MARGIN = 0.15

# HuggingFace 文件名中常见的参数量模式
_PARAM_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s*[Bb]", re.IGNORECASE),   # 7B, 13B, 70B
    re.compile(r"(\d+\.?\d*)\s*[Mm]", re.IGNORECASE),   # 110M
    re.compile(r"(\d+\.?\d*)B(?:illion)?", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# 枚举 / 数据类
# ---------------------------------------------------------------------------

class ModelFormat(enum.Enum):
    """模型文件格式"""
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    PYTORCH = "pytorch"          # .bin / .pt / .pth
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "ModelFormat":
        ext = ext.lower().lstrip(".")
        mapping = {
            "gguf": cls.GGUF,
            "ggml": cls.GGUF,           # 旧版 GGML 也归入 GGUF
            "safetensors": cls.SAFETENSORS,
            "onnx": cls.ONNX,
            "bin": cls.PYTORCH,
            "pt": cls.PYTORCH,
            "pth": cls.PYTORCH,
        }
        return mapping.get(ext, cls.UNKNOWN)


@dataclass
class ModelInfo:
    """单个模型文件的元数据"""
    name: str                                   # 人类可读名称 (目录名或文件名)
    path: str                                   # 绝对路径
    format: ModelFormat                         # 文件格式
    size_bytes: int = 0                         # 文件大小 (字节)
    parameter_count: Optional[int] = None       # 参数量 (个)
    quantization: Optional[str] = None          # 量化级别, e.g. "Q4_K_M"
    architecture: Optional[str] = None          # 架构, e.g. "llama"
    vocab_size: Optional[int] = None            # 词表大小
    context_length: Optional[int] = None        # 最大上下文长度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 原始元数据

    # --- 便捷属性 ----------------------------------------------------------

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def parameter_count_b(self) -> Optional[float]:
        """参数量 (十亿)"""
        if self.parameter_count is None:
            return None
        return self.parameter_count / 1e9

    def __repr__(self) -> str:
        param_str = f"{self.parameter_count_b:.1f}B" if self.parameter_count_b else "N/A"
        return (
            f"ModelInfo({self.name!r}, fmt={self.format.value}, "
            f"params={param_str}, quant={self.quantization or 'N/A'}, "
            f"size={self.size_gb:.2f}GB)"
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["format"] = self.format.value
        d["size_gb"] = round(self.size_gb, 3)
        if self.parameter_count_b is not None:
            d["parameter_count_b"] = round(self.parameter_count_b, 2)
        return d


# ---------------------------------------------------------------------------
# 文件解析工具函数
# ---------------------------------------------------------------------------

def _parse_gguf_metadata(filepath: str) -> Dict[str, Any]:
    """
    解析 GGUF 文件的元数据头 (GGUF v2/v3)。

    GGUF 格式参考: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    """
    metadata: Dict[str, Any] = {}
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return metadata

            version = struct.unpack("<I", f.read(4))[0]
            metadata["gguf_version"] = version

            if version >= 2:
                tensor_count = struct.unpack("<Q", f.read(8))[0]
                metadata["tensor_count"] = tensor_count
                kv_count = struct.unpack("<Q", f.read(8))[0]
            else:
                tensor_count = struct.unpack("<I", f.read(4))[0]
                metadata["tensor_count"] = tensor_count
                kv_count = struct.unpack("<I", f.read(4))[0]

            # 读取 key-value 对
            _GGUF_TYPE_READERS = {
                0: lambda f: struct.unpack("<B", f.read(1))[0],       # UINT8
                1: lambda f: struct.unpack("<b", f.read(1))[0],       # INT8
                2: lambda f: struct.unpack("<H", f.read(2))[0],       # UINT16
                3: lambda f: struct.unpack("<h", f.read(2))[0],       # INT16
                4: lambda f: struct.unpack("<I", f.read(4))[0],       # UINT32
                5: lambda f: struct.unpack("<i", f.read(4))[0],       # INT32
                6: lambda f: struct.unpack("<f", f.read(4))[0],       # FLOAT32
                7: lambda f: struct.unpack("?", f.read(1))[0],        # BOOL
                8: lambda f: _read_gguf_string(f),                    # STRING
                10: lambda f: struct.unpack("<Q", f.read(8))[0],      # UINT64
                11: lambda f: struct.unpack("<q", f.read(8))[0],      # INT64
                12: lambda f: struct.unpack("<d", f.read(8))[0],      # FLOAT64
            }

            for _ in range(min(kv_count, 200)):  # 限制读取数量防止损坏文件
                try:
                    key = _read_gguf_string(f)
                    value_type = struct.unpack("<I", f.read(4))[0]

                    if value_type in _GGUF_TYPE_READERS:
                        metadata[key] = _GGUF_TYPE_READERS[value_type](f)
                    elif value_type == 9:  # ARRAY
                        arr_type = struct.unpack("<I", f.read(4))[0]
                        arr_len = struct.unpack("<Q", f.read(8))[0]
                        reader = _GGUF_TYPE_READERS.get(arr_type)
                        if reader:
                            metadata[key] = [reader(f) for _ in range(min(arr_len, 100))]
                        else:
                            break  # 未知数组类型, 停止解析
                    else:
                        break
                except (struct.error, OSError):
                    break
    except (OSError, PermissionError) as exc:
        logger.warning("无法读取 GGUF 文件 %s: %s", filepath, exc)

    return metadata


def _read_gguf_string(f) -> str:
    """读取 GGUF 格式的字符串 (length-prefixed UTF-8)"""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _parse_safetensors_metadata(filepath: str) -> Dict[str, Any]:
    """
    解析 SafeTensors 文件的元数据头。

    SafeTensors 头部为 JSON 格式: {"__metadata__": {...}, "tensor_name": {...}, ...}
    第 8 字节为头部 JSON 长度 (小端 uint64)。
    """
    metadata: Dict[str, Any] = {}
    try:
        with open(filepath, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return metadata
            header_size = struct.unpack("<Q", header_size_bytes)[0]

            # 限制读取量，防止损坏文件
            if header_size > 50 * 1024 * 1024:  # 50MB 上限
                return metadata

            header_json = f.read(header_size).decode("utf-8", errors="replace")
            header = json.loads(header_json)

            if "__metadata__" in header:
                metadata.update(header["__metadata__"])

            # 统计张量信息
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            metadata["_tensor_count"] = len(tensors)
            metadata["_tensor_names"] = list(tensors.keys())[:20]
    except (OSError, json.JSONDecodeError, PermissionError) as exc:
        logger.warning("无法解析 SafeTensors 文件 %s: %s", filepath, exc)

    return metadata


def _extract_quantization(filename: str, metadata: Dict[str, Any]) -> Optional[str]:
    """从文件名或元数据中提取量化级别"""
    # 从文件名提取
    fname_upper = filename.upper()
    for quant in sorted(_QUANT_BYTES_PER_PARAM.keys(), key=len, reverse=True):
        if quant in fname_upper:
            return quant

    # 从元数据提取
    for key in ("general.file_type", "quantization", "quantization_version"):
        val = metadata.get(key)
        if isinstance(val, str) and val.upper() in _QUANT_BYTES_PER_PARAM:
            return val.upper()
        # GGUF file_type 是整数, 需要映射
        if key == "general.file_type" and isinstance(val, int):
            _FILE_TYPE_MAP = {
                0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
                7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
                11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
                15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
            }
            return _FILE_TYPE_MAP.get(val)

    return None


def _extract_parameter_count(
    metadata: Dict[str, Any], filename: str, size_bytes: int, fmt: ModelFormat
) -> Optional[int]:
    """从元数据 / 文件名 / 文件大小推断参数量"""

    # 1) 尝试从元数据中获取
    for key in (
        "general.parameter_count",
        "n_params",
        "param_count",
        "num_parameters",
    ):
        val = metadata.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)

    # 2) 尝试从张量信息推断 (SafeTensors)
    tensor_count = metadata.get("_tensor_count")
    if fmt == ModelFormat.SAFETENSORS and tensor_count:
        # 如果有完整头部信息可计算精确参数量，这里用近似
        pass

    # 3) 从文件名提取
    for pattern in _PARAM_PATTERNS:
        match = pattern.search(filename)
        if match:
            value = float(match.group(1))
            if "M" in filename.upper():
                return int(value * 1e6)
            else:
                return int(value * 1e9)

    # 4) 从文件大小反推 (非常粗略的估算)
    if size_bytes > 0:
        # 假设平均每参数 2 字节 (FP16)
        estimated = size_bytes / 2
        # 只在文件足够大时才给出估算
        if estimated > 1e8:  # > 100M 参数
            return int(estimated)

    return None


def _extract_architecture(metadata: Dict[str, Any]) -> Optional[str]:
    """从元数据中提取模型架构"""
    for key in ("general.architecture", "model_type", "architecture"):
        val = metadata.get(key)
        if isinstance(val, str):
            return val.lower()
    return None


def _extract_context_length(metadata: Dict[str, Any], arch: Optional[str]) -> Optional[int]:
    """从元数据中提取最大上下文长度"""
    # 直接的 context length 字段
    for key in (
        "general.context_length",
        "max_position_embeddings",
        "model.context_length",
        "n_ctx",
    ):
        val = metadata.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)

    # 架构相关的 key
    if arch:
        for suffix in (".context_length", ".context_length"):
            val = metadata.get(f"{arch}{suffix}")
            if isinstance(val, (int, float)) and val > 0:
                return int(val)

    return None


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class ModelManager:
    """
    模型管理器

    功能:
    - 扫描指定目录中的模型文件
    - 解析模型元数据
    - 估算显存需求
    - 从 HuggingFace 下载模型
    - 格式转换 (PyTorch -> GGUF)
    - 智能推荐模型
    """

    def __init__(
        self,
        models_dir: str | Path,
        index_path: Optional[str | Path] = None,
        hf_mirror: Optional[str] = None,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = Path(index_path) if index_path else self.models_dir / "index.json"
        self.hf_mirror = hf_mirror  # e.g. "https://hf-mirror.com"

        # 内存中的模型索引 {model_name: ModelInfo}
        self._index: Dict[str, ModelInfo] = {}

        # 加载已有索引
        self._load_index()

    # -----------------------------------------------------------------------
    # 索引持久化
    # -----------------------------------------------------------------------

    def _load_index(self) -> None:
        """从 index.json 加载模型索引"""
        if not self.index_path.exists():
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, item in data.items():
                fmt = ModelFormat(item.get("format", "unknown"))
                self._index[name] = ModelInfo(
                    name=item["name"],
                    path=item["path"],
                    format=fmt,
                    size_bytes=item.get("size_bytes", 0),
                    parameter_count=item.get("parameter_count"),
                    quantization=item.get("quantization"),
                    architecture=item.get("architecture"),
                    vocab_size=item.get("vocab_size"),
                    context_length=item.get("context_length"),
                    metadata=item.get("metadata", {}),
                )
            logger.info("已从 %s 加载 %d 条模型索引", self.index_path, len(self._index))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("加载模型索引失败: %s", exc)

    def _save_index(self) -> None:
        """将模型索引写入 index.json"""
        data = {name: info.to_dict() for name, info in self._index.items()}
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("模型索引已保存到 %s (%d 条)", self.index_path, len(data))
        except OSError as exc:
            logger.error("保存模型索引失败: %s", exc)

    # -----------------------------------------------------------------------
    # 扫描 & 解析
    # -----------------------------------------------------------------------

    def scan_models(self, deep: bool = True) -> List[ModelInfo]:
        """
        扫描 models_dir 下的所有模型文件。

        Args:
            deep: True 时解析文件元数据 (较慢), False 时仅收集基本信息。

        Returns:
            发现的模型列表
        """
        supported_exts = {".gguf", ".ggml", ".safetensors", ".onnx", ".bin", ".pt", ".pth"}
        found: List[ModelInfo] = []

        for root, _dirs, files in os.walk(self.models_dir):
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in supported_exts:
                    continue

                fpath = os.path.join(root, fname)
                try:
                    stat = os.stat(fpath)
                except OSError:
                    continue

                fmt = ModelFormat.from_extension(ext)
                metadata: Dict[str, Any] = {}
                quantization: Optional[str] = None
                parameter_count: Optional[int] = None
                architecture: Optional[str] = None
                vocab_size: Optional[int] = None
                context_length: Optional[int] = None

                if deep:
                    if fmt == ModelFormat.GGUF:
                        metadata = _parse_gguf_metadata(fpath)
                    elif fmt == ModelFormat.SAFETENSORS:
                        metadata = _parse_safetensors_metadata(fpath)

                    quantization = _extract_quantization(fname, metadata)
                    parameter_count = _extract_parameter_count(metadata, fname, stat.st_size, fmt)
                    architecture = _extract_architecture(metadata)
                    context_length = _extract_context_length(metadata, architecture)

                    # vocab_size
                    for key in ("tokenizer.vocab_size", "vocab_size"):
                        v = metadata.get(key)
                        if isinstance(v, (int, float)) and v > 0:
                            vocab_size = int(v)
                            break

                # 模型名称: 优先使用目录名, 否则文件名 (去掉扩展名)
                rel = os.path.relpath(fpath, self.models_dir)
                parts = Path(rel).parts
                name = parts[0] if len(parts) > 1 else Path(fname).stem

                info = ModelInfo(
                    name=name,
                    path=fpath,
                    format=fmt,
                    size_bytes=stat.st_size,
                    parameter_count=parameter_count,
                    quantization=quantization,
                    architecture=architecture,
                    vocab_size=vocab_size,
                    context_length=context_length,
                    metadata=metadata,
                )
                found.append(info)

        # 更新索引
        for info in found:
            self._index[info.name] = info
        self._save_index()

        logger.info("扫描完成: 发现 %d 个模型文件", len(found))
        return found

    def list_models(self) -> List[ModelInfo]:
        """列出索引中的所有模型"""
        return list(self._index.values())

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """按名称获取模型信息"""
        return self._index.get(name)

    def remove_from_index(self, name: str) -> bool:
        """从索引中移除模型记录 (不删除文件)"""
        if name in self._index:
            del self._index[name]
            self._save_index()
            return True
        return False

    # -----------------------------------------------------------------------
    # 显存估算
    # -----------------------------------------------------------------------

    def estimate_vram_requirement(
        self,
        model_info: ModelInfo,
        context_length: Optional[int] = None,
        batch_size: int = 1,
    ) -> float:
        """
        预估模型运行所需的 GPU 显存 (GB)。

        计算公式:
            vram = (参数量 × 每参数字节数) × 开销系数 + KV Cache

        Args:
            model_info: 模型信息
            context_length: 推理时使用的上下文长度, 默认使用模型最大值或 4096
            batch_size: 批处理大小

        Returns:
            预估显存需求 (GB)
        """
        params = model_info.parameter_count
        quant = model_info.quantization

        if params is None or params <= 0:
            # 无法估算时返回文件大小的 1.5 倍作为保守估计
            logger.warning("模型 %s 缺少参数量信息, 使用文件大小粗略估算", model_info.name)
            return model_info.size_gb * 1.5

        # 每参数字节数
        if quant and quant in _QUANT_BYTES_PER_PARAM:
            bytes_per_param = _QUANT_BYTES_PER_PARAM[quant]
        elif model_info.format == ModelFormat.GGUF:
            # GGUF 默认假设 Q4_K_M
            bytes_per_param = 0.5
        elif model_info.format == ModelFormat.SAFETENSORS:
            # SafeTensors 默认 FP16
            bytes_per_param = 2.0
        elif model_info.format == ModelFormat.ONNX:
            bytes_per_param = 2.0
        else:
            bytes_per_param = 2.0

        # 权重显存
        weight_vram_gb = (params * bytes_per_param) / (1024 ** 3)

        # KV Cache 估算
        # 每 token 的 KV cache ≈ 2 × num_layers × hidden_dim × bytes_per_element
        # 简化估算: KV cache ≈ 参数量 × 0.02 × context_length / 2048 (经验公式)
        ctx = context_length or model_info.context_length or 4096
        # 使用更精确的估算: KV cache 约为模型大小的 5% × (ctx / 2048)
        kv_cache_gb = weight_vram_gb * 0.05 * (ctx / 2048) * batch_size

        total_vram = (weight_vram_gb + kv_cache_gb) * _VRAM_OVERHEAD_FACTOR

        logger.debug(
            "显存估算 %s: 权重=%.2fGB, KV Cache=%.2fGB, 总计=%.2fGB",
            model_info.name, weight_vram_gb, kv_cache_gb, total_vram,
        )

        return round(total_vram, 2)

    # -----------------------------------------------------------------------
    # HuggingFace 下载
    # -----------------------------------------------------------------------

    def download_model(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        local_dir_name: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Path:
        """
        从 HuggingFace 下载模型文件。

        Args:
            repo_id: 仓库 ID, e.g. "meta-llama/Llama-2-7b-chat-hf"
            filename: 指定下载的文件名, None 则下载整个仓库
            revision: 分支/tag/commit
            local_dir_name: 本地保存目录名, 默认使用 repo_id 的最后一段
            token: HuggingFace 访问令牌

        Returns:
            下载后的本地路径
        """
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise ImportError(
                "请安装 huggingface_hub: pip install huggingface_hub"
            )

        save_name = local_dir_name or repo_id.split("/")[-1]
        # Security: sanitize directory name to prevent path traversal
        save_name = Path(save_name).name  # strips any path components like ../
        if not save_name or save_name in (".", ".."):
            raise ValueError(f"Invalid directory name: {save_name}")
        local_dir = self.models_dir / save_name
        local_dir.mkdir(parents=True, exist_ok=True)

        # 构建下载参数
        kwargs: Dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir": str(local_dir),
            "local_dir_use_symlinks": False,
        }
        if revision:
            kwargs["revision"] = revision
        if token:
            kwargs["token"] = token
        elif os.environ.get("HF_TOKEN"):
            kwargs["token"] = os.environ["HF_TOKEN"]

        if self.hf_mirror:
            kwargs["endpoint"] = self.hf_mirror

        if filename:
            kwargs["filename"] = filename
            logger.info("正在从 %s 下载 %s ...", repo_id, filename)
            result = hf_hub_download(**kwargs)
            logger.info("下载完成: %s", result)
        else:
            logger.info("正在从 %s 下载整个仓库 ...", repo_id)
            result = snapshot_download(**kwargs)
            logger.info("下载完成: %s", result)

        # 下载后重新扫描
        self.scan_models(deep=True)

        return Path(result)

    # -----------------------------------------------------------------------
    # 格式转换
    # -----------------------------------------------------------------------

    def convert_to_gguf(
        self,
        model_path: str | Path,
        output_path: Optional[str | Path] = None,
        quantization: str = "Q4_K_M",
        llama_cpp_path: Optional[str | Path] = None,
    ) -> Path:
        """
        将 PyTorch / SafeTensors 模型转换为 GGUF 格式。

        依赖 llama.cpp 的 convert 脚本。

        Args:
            model_path: 源模型路径 (目录或文件)
            output_path: 输出路径, 默认在同目录下生成 .gguf
            quantization: 目标量化级别
            llama_cpp_path: llama.cpp 仓库路径, 需包含 convert 脚本

        Returns:
            输出的 GGUF 文件路径
        """
        model_path = Path(model_path)

        # 查找 llama.cpp 的 convert 脚本
        convert_script = self._find_convert_script(llama_cpp_path)
        if convert_script is None:
            raise FileNotFoundError(
                "未找到 llama.cpp 的 convert 脚本。"
                "请提供 llama_cpp_path 参数或确保 llama.cpp 在 PATH 中。"
            )

        # 确定输入目录
        if model_path.is_file():
            input_dir = model_path.parent
        else:
            input_dir = model_path

        # 确定输出路径
        if output_path is None:
            output_path = input_dir / f"{input_dir.name}-{quantization}.gguf"
        output_path = Path(output_path)

        logger.info("正在转换 %s -> %s (量化: %s)", input_dir, output_path, quantization)

        # 第 1 步: 转换为 F16 GGUF
        f16_output = output_path.parent / f"{output_path.stem}-f16.gguf"
        convert_cmd = [
            sys.executable, str(convert_script),
            str(input_dir),
            "--outfile", str(f16_output),
            "--outtype", "f16",
        ]

        try:
            result = subprocess.run(
                convert_cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )
            if result.returncode != 0:
                logger.error("转换失败: %s", result.stderr)
                raise RuntimeError(f"模型转换失败: {result.stderr}")
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到 Python 解释器或转换脚本: {convert_script}")

        # 第 2 步: 量化
        if quantization.upper() != "F16":
            quantize_bin = self._find_quantize_bin(llama_cpp_path)
            if quantize_bin:
                quant_cmd = [
                    str(quantize_bin),
                    str(f16_output),
                    str(output_path),
                    quantization.upper(),
                ]
                result = subprocess.run(
                    quant_cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    check=False,
                )
                if result.returncode == 0:
                    # 删除中间的 F16 文件
                    f16_output.unlink(missing_ok=True)
                else:
                    logger.warning("量化失败, 保留 F16 版本: %s", result.stderr)
                    output_path = f16_output
            else:
                logger.warning("未找到 quantize 二进制, 保留 F16 版本")
                output_path = f16_output
        else:
            output_path = f16_output

        logger.info("转换完成: %s", output_path)

        # 更新索引
        self.scan_models(deep=True)

        return output_path

    @staticmethod
    def _find_convert_script(llama_cpp_path: Optional[str | Path]) -> Optional[Path]:
        """查找 llama.cpp 的 convert_hf_to_gguf.py 脚本"""
        candidates = []

        if llama_cpp_path:
            base = Path(llama_cpp_path)
            candidates.extend([
                base / "convert_hf_to_gguf.py",
                base / "convert_hf_to_gguf_update.py",
                base / "convert" / "convert_hf_to_gguf.py",
            ])

        # 在 PATH 中搜索
        for p in os.environ.get("PATH", "").split(os.pathsep):
            candidates.append(Path(p) / "convert_hf_to_gguf.py")

        # 常见安装位置
        home = Path.home()
        candidates.extend([
            home / "llama.cpp" / "convert_hf_to_gguf.py",
            home / "tools" / "llama.cpp" / "convert_hf_to_gguf.py",
            Path("/usr/local/lib/llama.cpp/convert_hf_to_gguf.py"),
        ])

        for c in candidates:
            if c.exists():
                return c
        return None

    @staticmethod
    def _find_quantize_bin(llama_cpp_path: Optional[str | Path]) -> Optional[Path]:
        """查找 llama.cpp 的 quantize 可执行文件"""
        candidates = []

        if llama_cpp_path:
            base = Path(llama_cpp_path)
            candidates.extend([
                base / "build" / "bin" / "llama-quantize",
                base / "build" / "bin" / "llama-quantize.exe",
                base / "build" / "bin" / "Release" / "llama-quantize.exe",
                base / "quantize",
                base / "quantize.exe",
            ])

        # 在 PATH 中搜索
        for p in os.environ.get("PATH", "").split(os.pathsep):
            for name in ("llama-quantize", "quantize"):
                candidates.append(Path(p) / name)
                candidates.append(Path(p) / f"{name}.exe")

        for c in candidates:
            if c.exists():
                return c
        return None

    # -----------------------------------------------------------------------
    # 模型推荐
    # -----------------------------------------------------------------------

    def recommend_model(
        self,
        task: str = "general",
        available_vram_gb: float = 8.0,
        preferred_format: Optional[ModelFormat] = None,
        min_quality: str = "Q4_K_M",
        top_k: int = 3,
    ) -> List[Tuple[ModelInfo, float]]:
        """
        根据任务和硬件显存推荐最佳模型。

        Args:
            task: 任务类型 ("general", "coding", "chat", "translation", "summarization")
            available_vram_gb: 可用 GPU 显存 (GB)
            preferred_format: 偏好格式
            min_quality: 最低量化质量 (越高越好)
            top_k: 返回前 K 个推荐

        Returns:
            推荐列表 [(ModelInfo, 匹配分数), ...] 按分数降序
        """
        if not self._index:
            logger.warning("模型索引为空, 请先调用 scan_models()")
            return []

        # 量化级别优先级 (越高越靠前)
        quant_rank = {q: i for i, q in enumerate(_QUANT_BYTES_PER_PARAM.keys())}
        min_quant_rank = quant_rank.get(min_quality, 0)

        scored: List[Tuple[ModelInfo, float]] = []

        for info in self._index.values():
            # 格式过滤
            if preferred_format and info.format != preferred_format:
                continue

            # 量化质量过滤
            if info.quantization:
                q_rank = quant_rank.get(info.quantization, 0)
                if q_rank < min_quant_rank:
                    continue

            # 估算显存需求
            vram_needed = self.estimate_vram_requirement(info)
            if vram_needed > available_vram_gb:
                continue

            # 计算匹配分数
            score = self._compute_recommendation_score(
                info, task, available_vram_gb, vram_needed,
            )
            scored.append((info, score))

        # 按分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _compute_recommendation_score(
        info: ModelInfo,
        task: str,
        available_vram_gb: float,
        vram_needed: float,
    ) -> float:
        """
        计算模型推荐分数。

        考虑因素:
        1. 参数量 (越大越好, 但有边际递减)
        2. 显存利用率 (不能太低浪费, 也不能太高危险)
        3. 量化质量
        4. 任务匹配度
        """
        score = 0.0

        # 1) 参数量分数 (0-40 分)
        if info.parameter_count_b:
            # 对数缩放, 70B 满分
            import math
            param_score = min(40, 40 * math.log10(max(info.parameter_count_b, 0.1)) / math.log10(70))
            score += max(0, param_score)
        else:
            score += 10  # 未知参数量给基础分

        # 2) 显存利用率分数 (0-25 分)
        # 理想情况: 使用 70-90% 的可用显存
        utilization = vram_needed / available_vram_gb if available_vram_gb > 0 else 0
        if 0.7 <= utilization <= 0.9:
            score += 25
        elif 0.5 <= utilization < 0.7:
            score += 20
        elif 0.9 < utilization <= 0.95:
            score += 18
        elif 0.3 <= utilization < 0.5:
            score += 15
        else:
            score += 5

        # 3) 量化质量分数 (0-20 分)
        if info.quantization:
            quant_rank = {q: i for i, q in enumerate(_QUANT_BYTES_PER_PARAM.keys())}
            rank = quant_rank.get(info.quantization, 0)
            max_rank = len(quant_rank)
            score += 20 * (rank / max_rank)
        else:
            score += 8

        # 4) 任务匹配度 (0-15 分)
        task_bonus = 0.0
        arch = (info.architecture or "").lower()
        name_lower = info.name.lower()

        task_keywords = {
            "coding": ["code", "coder", "starcoder", "deepseek-coder", "codellama", "codeqwen", "qwen-coder"],
            "chat": ["chat", "instruct", "assistant", "dialog"],
            "translation": ["translate", "nllb", "marian", "opus-mt"],
            "summarization": ["summar", "bart", "pegasus"],
            "math": ["math", "wizard", "metamath"],
            "general": [],
        }

        keywords = task_keywords.get(task, [])
        if keywords:
            for kw in keywords:
                if kw in name_lower or kw in arch:
                    task_bonus = 15
                    break
        else:
            task_bonus = 10  # general 任务给基础分

        score += task_bonus

        return round(score, 2)

    # -----------------------------------------------------------------------
    # 辅助方法
    # -----------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """获取模型库摘要信息"""
        models = list(self._index.values())
        if not models:
            return {"total": 0, "total_size_gb": 0, "by_format": {}, "by_quantization": {}}

        by_format: Dict[str, int] = {}
        by_quant: Dict[str, int] = {}
        total_size = 0

        for m in models:
            by_format[m.format.value] = by_format.get(m.format.value, 0) + 1
            if m.quantization:
                by_quant[m.quantization] = by_quant.get(m.quantization, 0) + 1
            total_size += m.size_bytes

        return {
            "total": len(models),
            "total_size_gb": round(total_size / (1024 ** 3), 2),
            "by_format": by_format,
            "by_quantization": by_quant,
            "models_dir": str(self.models_dir),
        }

    def find_model_file(self, name: str) -> Optional[Path]:
        """根据名称查找模型文件路径"""
        info = self._index.get(name)
        if info:
            p = Path(info.path)
            if p.exists():
                return p
        return None

    def delete_model(self, name: str, remove_files: bool = False) -> bool:
        """
        从索引中移除模型, 可选删除文件。

        Args:
            name: 模型名称
            remove_files: 是否同时删除文件

        Returns:
            是否成功
        """
        info = self._index.get(name)
        if not info:
            return False

        if remove_files:
            p = Path(info.path)
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
                logger.info("已删除模型文件: %s", p)
            except OSError as exc:
                logger.error("删除文件失败: %s", exc)
                return False

        del self._index[name]
        self._save_index()
        return True
