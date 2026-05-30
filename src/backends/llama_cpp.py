"""llama.cpp推理后端实现"""

import asyncio
import gc
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from typing import AsyncIterator, Dict, Iterator, List, Optional

from .base import (
    BackendType,
    GenerationResult,
    InferenceBackend,
    InferenceConfig,
    MemoryUsage,
)


def _detect_gpu_vendor() -> str:
    """
    检测GPU厂商，返回 'nvidia' / 'amd' / 'apple' / 'intel' / 'unknown'。
    """
    # NVIDIA: nvidia-smi 存在即认为有NVIDIA GPU
    if shutil.which("nvidia-smi") or any(
        os.path.exists(p) for p in [
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            "/usr/bin/nvidia-smi",
            "/usr/local/bin/nvidia-smi",
        ]
    ):
        return "nvidia"

    # Apple Silicon
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "apple"

    # AMD: Linux 有 rocm-smi，Windows 通过 WMI 检测
    if shutil.which("rocm-smi"):
        return "amd"
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_VideoController | Where-Object {"
                 "$_.Name -match 'AMD|Radeon'} | Select-Object -First 1 Name"],
                capture_output=True, text=True, timeout=10,
            )
            if "AMD" in result.stdout or "Radeon" in result.stdout:
                return "amd"
        except Exception:
            pass

    # Intel (Windows)
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_VideoController | Where-Object {"
                 "$_.Name -match 'Intel.*Arc|Intel.*Graphics'} | Select-Object -First 1 Name"],
                capture_output=True, text=True, timeout=10,
            )
            if "Intel" in result.stdout:
                return "intel"
        except Exception:
            pass

    # Intel (Linux: lspci)
    if sys.platform == "linux":
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                ll = line.lower()
                if ("vga" in ll or "display" in ll or "3d" in ll) and "intel" in ll:
                    return "intel"
        except Exception:
            pass

    return "unknown"


def _detect_gpu_vram_mb() -> float:
    """
    检测可用GPU显存(MB)。
    按优先级尝试: NVIDIA → AMD → Apple → 返回0(CPU-only)。
    """
    vendor = _detect_gpu_vendor()

    # --- NVIDIA: nvidia-smi ---
    if vendor == "nvidia":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                values = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
                return float(max(values)) if values else 0.0
        except Exception:
            pass

    # --- AMD Linux: rocm-smi ---
    if vendor == "amd":
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                import re
                total, used = 0, 0
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
                    return total - used
        except Exception:
            pass

        # AMD Windows: 通过 PowerShell 获取显存信息
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-CimInstance Win32_VideoController | Where-Object {"
                     "$_.Name -match 'AMD|Radeon'} | "
                     "Select-Object -First 1 AdapterRAM | ConvertTo-Json"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    import json
                    data = json.loads(result.stdout)
                    adapter_ram = data.get("AdapterRAM", 0)
                    if adapter_ram > 0:
                        return adapter_ram / (1024 * 1024)
            except Exception:
                pass

    # --- Apple Silicon: 统一内存 ---
    if vendor == "apple":
        try:
            # 获取总内存
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip().isdigit():
                total_bytes = int(result.stdout.strip())
                total_mb = total_bytes // (1024 * 1024)

                # 通过 vm_stat 获取可用内存
                vm_result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True, text=True, timeout=5,
                )
                if vm_result.returncode == 0:
                    import re
                    page_size = 16384  # Apple Silicon 默认 16KB page
                    free_pages = 0
                    speculative_pages = 0
                    inactive_pages = 0
                    for line in vm_result.stdout.splitlines():
                        if "page size of" in line.lower():
                            m = re.search(r"(\d+)\s+bytes", line)
                            if m:
                                page_size = int(m.group(1))
                        elif "Pages free" in line:
                            m = re.search(r"(\d+)", line)
                            if m:
                                free_pages = int(m.group(1))
                        elif "Pages speculative" in line:
                            m = re.search(r"(\d+)", line)
                            if m:
                                speculative_pages = int(m.group(1))
                        elif "Pages inactive" in line:
                            m = re.search(r"(\d+)", line)
                            if m:
                                inactive_pages = int(m.group(1))
                    # 可用内存 = free + speculative + inactive (inactive可被系统回收)
                    available_mb = (free_pages + speculative_pages + inactive_pages) * page_size // (1024 * 1024)
                    return float(available_mb)

                # vm_stat 失败则返回总内存的70%（保守估计）
                return float(total_mb) * 0.70
        except Exception:
            pass

    return 0.0


def _estimate_model_size_mb(model_path: str) -> float:
    """通过文件大小估算模型占用(MB)。"""
    try:
        return os.path.getsize(model_path) / (1024 * 1024)
    except OSError:
        return 0.0


def _guess_n_layers(model_path: str) -> int:
    """
    从GGUF元数据中读取模型层数。
    解析失败返回0。
    """
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return 0
            # version (u32)
            f.read(4)
            # n_tensors (u64)
            f.read(8)
            # n_metadata_kv (u64)
            n_kv = struct.unpack("<Q", f.read(8))[0]

            for _ in range(n_kv):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]

                if key.endswith(".block_count") or key.endswith(".n_layers"):
                    val = struct.unpack("<I", f.read(4))[0]
                    return val
                else:
                    _skip_gguf_value(f, vtype)
    except Exception:
        pass
    return 0


def _skip_gguf_value(f, vtype: int, depth: int = 0) -> None:
    """跳过GGUF元数据值。"""
    # Security: Limit recursion depth to prevent stack overflow from malicious files
    MAX_DEPTH = 10
    if depth > MAX_DEPTH:
        raise ValueError(f"GGUF嵌套深度超过限制 ({MAX_DEPTH})")

    # Security: Limit array length to prevent DoS
    MAX_ARRAY_LEN = 10000

    _GGUF_TYPE_SIZES = {
        0: 1, 1: 1, 2: 2, 3: 2,   # uint8, int8, uint16, int16
        4: 4, 5: 4, 6: 4, 7: 8,   # uint32, int32, float32, float64
    }
    if vtype in _GGUF_TYPE_SIZES:
        f.read(_GGUF_TYPE_SIZES[vtype])
    elif vtype == 8:  # string
        slen = struct.unpack("<Q", f.read(8))[0]
        # Security: Limit string length
        if slen > 1_000_000:  # 1MB max
            raise ValueError(f"GGUF字符串长度异常: {slen}")
        f.read(slen)
    elif vtype == 9:  # array
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        # Security: Limit array length
        if arr_len > MAX_ARRAY_LEN:
            raise ValueError(f"GGUF数组长度超过限制: {arr_len}")
        for _ in range(arr_len):
            _skip_gguf_value(f, arr_type, depth + 1)
    else:
        raise ValueError(f"未知的GGUF值类型: {vtype}")


def _estimate_layers_memory_mb(model_path: str) -> float:
    """估算每层内存占用(MB)。"""
    model_size = _estimate_model_size_mb(model_path)
    n_layers = _guess_n_layers(model_path)
    if n_layers > 0 and model_size > 0:
        return model_size / n_layers
    return 120.0  # 保守估计


class LlamaCppBackend(InferenceBackend):
    """
    基于llama-cpp-python的推理后端实现。
    支持GPU自动卸载、流式生成、embedding、内存监控和自动配置。
    """

    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self._model = None
        self._model_size_mb: float = 0.0
        self._backend_type = BackendType.LLAMA_CPP

    # ------------------------------------------------------------------
    # 模型生命周期
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """加载GGUF模型文件"""
        if self._is_loaded:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python未安装。请执行: pip install llama-cpp-python"
            )

        if not os.path.exists(self._config.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self._config.model_path}")

        try:
            params = {
                "model_path": self._config.model_path,
                "n_ctx": self._config.n_ctx,
                "n_batch": self._config.n_batch,
                "n_gpu_layers": self._config.n_gpu_layers,
                "rope_freq_base": self._config.rope_freq_base,
                "rope_freq_scale": self._config.rope_freq_scale,
                "use_mmap": self._config.use_mmap,
                "use_mlock": self._config.use_mlock,
                "verbose": self._config.verbose,
            }

            if self._config.n_threads > 0:
                params["n_threads"] = self._config.n_threads

            # 合并额外参数
            params.update(self._config.extra_params)

            self._model = Llama(**params)
            self._model_size_mb = _estimate_model_size_mb(self._config.model_path)
            self._is_loaded = True

        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")

    def unload_model(self) -> None:
        """卸载模型释放内存"""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            self._model_size_mb = 0.0
            gc.collect()

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """确保模型已加载"""
        if not self._is_loaded or self._model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        """同步生成文本"""
        self._ensure_loaded()

        try:
            start_time = time.perf_counter()

            gen_params = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "echo": False,
            }

            if stop is not None:
                gen_params["stop"] = stop
            if seed is not None:
                gen_params["seed"] = seed
            gen_params.update(kwargs)

            output = self._model.create_completion(**gen_params)

            end_time = time.perf_counter()
            total_duration_ms = (end_time - start_time) * 1000

            text = output["choices"][0]["text"]
            finish_reason = output["choices"][0].get("finish_reason", "stop")
            usage = output.get("usage", {})
            tokens_generated = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)

            tokens_per_second = 0.0
            if total_duration_ms > 0 and tokens_generated > 0:
                tokens_per_second = tokens_generated / (total_duration_ms / 1000)

            return GenerationResult(
                text=text,
                tokens_generated=tokens_generated,
                prompt_tokens=prompt_tokens,
                tokens_per_second=tokens_per_second,
                finish_reason=finish_reason,
                total_duration_ms=total_duration_ms,
            )

        except Exception as e:
            raise RuntimeError(f"生成失败: {e}")

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """同步流式生成文本"""
        self._ensure_loaded()

        gen_params = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": True,
            "echo": False,
        }

        if stop is not None:
            gen_params["stop"] = stop
        if seed is not None:
            gen_params["seed"] = seed
        gen_params.update(kwargs)

        try:
            stream = self._model.create_completion(**gen_params)
            for chunk in stream:
                text = chunk["choices"][0]["text"]
                if text:
                    yield text
        except Exception as e:
            raise RuntimeError(f"流式生成失败: {e}")

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """异步流式生成文本"""
        self._ensure_loaded()

        gen_params = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": True,
            "echo": False,
        }

        if stop is not None:
            gen_params["stop"] = stop
        if seed is not None:
            gen_params["seed"] = seed
        gen_params.update(kwargs)

        loop = asyncio.get_event_loop()

        def _create_stream():
            return self._model.create_completion(**gen_params)

        stream = await loop.run_in_executor(None, _create_stream)

        def _next_chunk():
            try:
                return next(stream)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, _next_chunk)
            if chunk is None:
                break
            text = chunk["choices"][0]["text"]
            if text:
                yield text
                await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self, text: str) -> List[float]:
        """获取文本的embedding向量"""
        self._ensure_loaded()

        try:
            embedding = self._model.embed(text)
            if isinstance(embedding, list):
                if len(embedding) > 0 and isinstance(embedding[0], list):
                    return embedding[0]
                return embedding
            return []
        except Exception as e:
            raise RuntimeError(f"获取embedding失败: {e}")

    # ------------------------------------------------------------------
    # 内存监控
    # ------------------------------------------------------------------

    def get_memory_usage(self) -> MemoryUsage:
        """获取当前内存使用情况"""
        model_size = _estimate_model_size_mb(self._config.model_path)
        vram_total = _detect_gpu_vram_mb()

        vram_used = 0.0
        ram_used = 0.0

        if self._is_loaded and self._config.n_gpu_layers != 0:
            n_layers = _guess_n_layers(self._config.model_path) or 32
            if self._config.n_gpu_layers == -1:
                # 全部卸载到GPU
                vram_used = model_size
            else:
                fraction = min(self._config.n_gpu_layers / n_layers, 1.0)
                vram_used = model_size * fraction
            ram_used = max(model_size - vram_used, 0.0)
        elif self._is_loaded:
            ram_used = model_size

        # KV缓存估算: 每层每1k上下文约0.5MB
        n_layers = _guess_n_layers(self._config.model_path) or 32
        kv_cache = n_layers * 0.5 * (self._config.n_ctx / 1024)

        return MemoryUsage(
            vram_used_mb=round(vram_used, 1),
            vram_total_mb=round(vram_total, 1),
            ram_used_mb=round(ram_used, 1),
            model_size_mb=round(model_size, 1),
            kv_cache_mb=round(kv_cache, 1),
        )

    # ------------------------------------------------------------------
    # 最优配置计算
    # ------------------------------------------------------------------

    @staticmethod
    def get_optimal_config(
        model_path: str,
        vram_budget_mb: Optional[float] = None
    ) -> InferenceConfig:
        """
        根据模型和硬件条件计算最优配置。

        自动检测可用显存，计算可卸载的GPU层数，
        预留KV缓存空间，返回最优InferenceConfig。

        Args:
            model_path: 模型文件路径
            vram_budget_mb: VRAM预算(MB)，None表示自动检测

        Returns:
            InferenceConfig: 最优配置
        """
        if vram_budget_mb is None:
            vram_budget_mb = _detect_gpu_vram_mb()

        model_size = _estimate_model_size_mb(model_path)
        per_layer_mb = _estimate_layers_memory_mb(model_path)
        n_layers = _guess_n_layers(model_path) or 32

        if vram_budget_mb <= 0 or per_layer_mb <= 0:
            # CPU-only
            n_gpu = 0
        else:
            # 预留15%空间给KV缓存和开销
            usable = vram_budget_mb * 0.85
            n_gpu = int(usable / per_layer_mb)
            n_gpu = min(n_gpu, n_layers)

        # 根据GPU层数调整批处理大小
        n_batch = 1024 if n_gpu > 0 else 512

        # 上下文大小: 默认4096，显存紧张时降低
        n_ctx = 4096
        if vram_budget_mb > 0 and model_size > 0:
            remaining = vram_budget_mb - (n_gpu * per_layer_mb)
            if remaining < 500:
                n_ctx = 2048

        return InferenceConfig(
            model_path=model_path,
            n_gpu_layers=n_gpu,
            n_batch=n_batch,
            n_ctx=n_ctx,
            use_mmap=True,
            use_mlock=False,
        )

    def __del__(self):
        """析构时确保模型被卸载"""
        self.unload_model()
