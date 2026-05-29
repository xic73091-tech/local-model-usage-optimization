"""
硬件检测模块

检测系统硬件信息（GPU、CPU、内存、存储），
并基于硬件能力推荐最优的本地模型运行配置。
"""

import os
import sys
import platform
import subprocess
import shutil
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# 枚举与数据结构定义
# ============================================================

class GPUVendor(Enum):
    """GPU 厂商枚举"""
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"       # Apple Silicon (M1/M2/M3/M4 系列)
    INTEL = "intel"
    UNKNOWN = "unknown"


class ModelQuant(Enum):
    """模型量化等级"""
    Q2_K = "Q2_K"         # 最小体积，质量最低
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q4_0 = "Q4_0"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"    # 推荐平衡点
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"         # 高质量，体积较大
    F16 = "F16"           # 半精度浮点
    F32 = "F32"           # 全精度浮点（不推荐本地推理）


@dataclass
class GPUInfo:
    """GPU 硬件信息"""
    vendor: GPUVendor = GPUVendor.UNKNOWN
    name: str = "未检测到"
    vram_total_mb: int = 0               # 显存总量 (MB)
    vram_free_mb: int = 0                # 可用显存 (MB)
    compute_capability: str = ""         # CUDA 计算能力 (如 "8.9")
    driver_version: str = ""             # 驱动版本
    cuda_version: str = ""               # CUDA 版本
    rocm_version: str = ""               # ROCm 版本 (AMD)
    metal_support: bool = False          # Metal 支持 (Apple)
    unified_memory: bool = False         # 统一内存架构
    tensor_cores: bool = False           # 是否有 Tensor Core
    gpu_count: int = 0                   # GPU 数量

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024

    @property
    def has_gpu(self) -> bool:
        return self.vendor != GPUVendor.UNKNOWN and self.vram_total_mb > 0

    @property
    def summary(self) -> str:
        if not self.has_gpu:
            return "未检测到独立GPU"
        mem = f"{self.vram_total_gb:.1f}GB"
        return f"{self.name} ({mem})"


@dataclass
class CPUInfo:
    """CPU 硬件信息"""
    brand: str = "未知"
    architecture: str = ""               # x86_64 / arm64
    physical_cores: int = 0
    logical_cores: int = 0
    frequency_mhz: float = 0.0          # 当前频率

    # 指令集支持
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_fma: bool = False
    has_f16c: bool = False               # 半精度转换指令
    has_sse4_2: bool = False
    has_neon: bool = False               # ARM NEON

    @property
    def best_vector_extension(self) -> str:
        """返回当前CPU支持的最优向量指令集"""
        if self.has_avx512:
            return "AVX-512"
        if self.has_avx2:
            return "AVX2"
        if self.has_avx:
            return "AVX"
        if self.has_neon:
            return "NEON"
        if self.has_sse4_2:
            return "SSE4.2"
        return "基础指令集"

    @property
    def summary(self) -> str:
        cores = f"{self.physical_cores}核/{self.logical_cores}线程"
        return f"{self.brand} ({cores}, {self.best_vector_extension})"


@dataclass
class DiskSpeedInfo:
    """磁盘速度信息"""
    sequential_read_mbps: float = 0.0    # 顺序读取速度 (MB/s)
    sequential_write_mbps: float = 0.0   # 顺序写入速度 (MB/s)
    is_nvme: bool = False                # 是否为 NVMe 设备
    pcie_generation: int = 0             # PCIe 代数 (3/4/5)
    interface: str = "未知"              # 接口类型: NVMe/SATA/AHCI

    @property
    def speed_rating(self) -> str:
        """速度评级"""
        if self.sequential_read_mbps >= 5000:
            return "极快 (PCIe 4.0+ NVMe)"
        elif self.sequential_read_mbps >= 3000:
            return "很快 (PCIe 3.0 NVMe)"
        elif self.sequential_read_mbps >= 500:
            return "快 (SATA SSD)"
        elif self.sequential_read_mbps > 0:
            return "慢 (HDD)"
        return "未检测"

    @property
    def summary(self) -> str:
        if self.sequential_read_mbps > 0:
            return f"{self.interface} 读:{self.sequential_read_mbps:.0f}MB/s 写:{self.sequential_write_mbps:.0f}MB/s ({self.speed_rating})"
        return self.interface


@dataclass
class MemoryInfo:
    """内存信息"""
    total_mb: int = 0
    available_mb: int = 0
    swap_total_mb: int = 0
    speed_mhz: int = 0                   # 内存频率 (MHz)
    channels: int = 0                    # 内存通道数
    ddr_type: str = ""                   # DDR4/DDR5/LPDDR5 等

    # 存储信息
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_type: str = "未知"              # SSD / HDD / NVMe
    disk_speed: DiskSpeedInfo = field(default_factory=DiskSpeedInfo)

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024

    @property
    def available_gb(self) -> float:
        return self.available_mb / 1024

    @property
    def summary(self) -> str:
        lines = [
            f"内存: {self.total_gb:.1f}GB (可用 {self.available_gb:.1f}GB)",
            f"磁盘: {self.disk_free_gb:.1f}/{self.disk_total_gb:.1f}GB ({self.disk_type})",
        ]
        if self.disk_speed.sequential_read_mbps > 0:
            lines.append(f"磁盘速度: {self.disk_speed.summary}")
        return " | ".join(lines)


@dataclass
class ModelRecommendation:
    """模型推荐配置"""
    max_model_size_gb: float = 0.0       # 推荐最大模型参数量对应体积
    recommended_quant: ModelQuant = ModelQuant.Q4_K_M
    gpu_offload_layers: int = 0          # 建议 offload 到 GPU 的层数
    context_length: int = 2048           # 推荐上下文长度
    threads: int = 4                     # CPU 线程数
    batch_size: int = 512                # 批处理大小
    use_gpu: bool = False
    use_metal: bool = False              # Apple Silicon
    use_cuda: bool = False
    use_rocm: bool = False
    backend: str = "cpu"                 # 推推理后端: cpu / cuda / rocm / metal / vulkan
    notes: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        lines = [
            f"推荐后端: {self.backend}",
            f"推荐量化: {self.recommended_quant.value}",
            f"推荐最大模型体积: {self.max_model_size_gb:.1f}GB",
            f"上下文长度: {self.context_length}",
            f"CPU线程: {self.threads}",
        ]
        if self.use_gpu:
            lines.append(f"GPU offload 层数: {self.gpu_offload_layers}")
        if self.notes:
            lines.append("备注:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


@dataclass
class HardwareProfile:
    """完整硬件画像"""
    gpu: GPUInfo = field(default_factory=GPUInfo)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    recommendation: ModelRecommendation = field(default_factory=ModelRecommendation)
    os_info: str = ""

    @property
    def summary(self) -> str:
        sep = "=" * 60
        return (
            f"{sep}\n"
            f"系统: {self.os_info}\n"
            f"GPU:  {self.gpu.summary}\n"
            f"CPU:  {self.cpu.summary}\n"
            f"{self.memory.summary}\n"
            f"{sep}\n"
            f"推荐配置:\n{self.recommendation.summary}\n"
            f"{sep}"
        )


# ============================================================
# 硬件检测器
# ============================================================

class HardwareDetector:
    """
    硬件检测器

    跨平台检测 GPU、CPU、内存、存储信息，
    并根据硬件能力给出本地大模型运行的最优配置建议。
    """

    def __init__(self, auto_detect: bool = True):
        """
        初始化硬件检测器。

        Args:
            auto_detect: 是否在初始化时自动执行全部检测
        """
        self.profile = HardwareProfile(
            os_info=f"{platform.system()} {platform.release()} {platform.machine()}"
        )
        if auto_detect:
            self.detect_all()

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------
    def detect_all(self) -> HardwareProfile:
        """执行全部硬件检测并生成推荐配置"""
        self.detect_cpu()
        self.detect_memory()
        self.detect_gpu()
        self._generate_recommendation()
        return self.profile

    # ----------------------------------------------------------
    # CPU 检测
    # ----------------------------------------------------------
    def detect_cpu(self) -> CPUInfo:
        """检测 CPU 信息和指令集特性"""
        cpu = CPUInfo()
        cpu.brand = platform.processor() or "未知"
        cpu.architecture = platform.machine()
        cpu.logical_cores = os.cpu_count() or 1

        try:
            if sys.platform == "win32":
                self._detect_cpu_windows(cpu)
            elif sys.platform == "darwin":
                self._detect_cpu_darwin(cpu)
            else:
                self._detect_cpu_linux(cpu)
        except Exception as e:
            logger.warning(f"CPU 详细信息检测失败: {e}")

        # 物理核心数回退
        if cpu.physical_cores == 0:
            cpu.physical_cores = max(1, cpu.logical_cores // 2)

        self.profile.cpu = cpu
        logger.info(f"CPU 检测完成: {cpu.summary}")
        return cpu

    def _detect_cpu_windows(self, cpu: CPUInfo) -> None:
        """Windows 平台 CPU 检测"""
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get",
                 "Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed",
                 "/format:list"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.startswith("Name=") and line[5:]:
                    cpu.brand = line[5:]
                elif line.startswith("NumberOfCores=") and line[14:]:
                    cpu.physical_cores = int(line[14:])
                elif line.startswith("NumberOfLogicalProcessors=") and line[26:]:
                    cpu.logical_cores = int(line[26:])
                elif line.startswith("MaxClockSpeed=") and line[14:]:
                    cpu.frequency_mhz = float(line[14:])
        except Exception as e:
            logger.debug(f"wmic CPU 检测失败: {e}")

        # 指令集检测 (Windows: 通过 Python 或 CPUID)
        self._detect_cpu_features_fallback(cpu)

    def _detect_cpu_darwin(self, cpu: CPUInfo) -> None:
        """macOS 平台 CPU 检测"""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                cpu.brand = result.stdout.strip()
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu", "hw.logicalcpu", "hw.cpufrequency_max"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().splitlines()
            if len(lines) >= 2:
                cpu.physical_cores = int(lines[0])
                cpu.logical_cores = int(lines[1])
            if len(lines) >= 3 and lines[2].isdigit():
                cpu.frequency_mhz = int(lines[2]) / 1_000_000
        except Exception:
            pass

        # Apple Silicon 使用 NEON
        if cpu.architecture == "arm64":
            cpu.has_neon = True
        else:
            self._detect_cpu_features_fallback(cpu)

    def _detect_cpu_linux(self, cpu: CPUInfo) -> None:
        """Linux 平台 CPU 检测"""
        cpu_info_path = "/proc/cpuinfo"
        if os.path.exists(cpu_info_path):
            try:
                with open(cpu_info_path, "r") as f:
                    content = f.read()
                cores_set = set()
                for line in content.splitlines():
                    if line.startswith("model name") and ":" in line:
                        cpu.brand = line.split(":", 1)[1].strip()
                    elif line.startswith("processor"):
                        cores_set.add(line.split(":", 1)[1].strip())
                    elif line.startswith("flags") or line.startswith("Features"):
                        self._parse_cpu_flags(cpu, line)
                cpu.logical_cores = len(cores_set) if cores_set else os.cpu_count() or 1
            except Exception as e:
                logger.debug(f"/proc/cpuinfo 解析失败: {e}")

        # lscpu 补充
        try:
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if "Core(s) per socket" in line and ":" in line:
                    cores_per = int(line.split(":")[1].strip())
                elif "Socket(s)" in line and ":" in line:
                    sockets = int(line.split(":")[1].strip())
                    cpu.physical_cores = cores_per * sockets if 'cores_per' in dir() else 0
        except Exception:
            pass

        if cpu.physical_cores == 0:
            cpu.physical_cores = max(1, cpu.logical_cores // 2)

    def _detect_cpu_features_fallback(self, cpu: CPUInfo) -> None:
        """回退方案：通过尝试导入 numpy 检测 CPU 特性"""
        try:
            import numpy as np
            config = np.show_config()
            config_str = str(config) if config else ""
            if "AVX512" in config_str:
                cpu.has_avx512 = True
            if "AVX2" in config_str:
                cpu.has_avx2 = True
            if "AVX" in config_str and not cpu.has_avx2:
                cpu.has_avx = True
            if "FMA" in config_str:
                cpu.has_fma = True
            if "F16C" in config_str:
                cpu.has_f16c = True
            if "SSE4" in config_str:
                cpu.has_sse4_2 = True
        except ImportError:
            pass

        # Windows 平台通用回退：使用 ctypes 调用 IsProcessorFeaturePresent
        if sys.platform == "win32" and not any([
            cpu.has_avx, cpu.has_avx2, cpu.has_avx512
        ]):
            self._detect_cpu_features_win32(cpu)

    def _detect_cpu_features_win32(self, cpu: CPUInfo) -> None:
        """Windows 通过 IsProcessorFeaturePresent 检测指令集"""
        try:
            import ctypes
            PF_AVX512F_INSTRUCTIONS_AVAILABLE = 40
            PF_SSE4_2_INSTRUCTIONS_AVAILABLE = 38
            PF_AVX_INSTRUCTIONS_AVAILABLE = 37
            PF_XSAVE_ENABLED = 17

            IsProcessorFeaturePresent = ctypes.windll.kernel32.IsProcessorFeaturePresent

            if IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE):
                cpu.has_avx512 = True
                cpu.has_avx2 = True
                cpu.has_avx = True
            if IsProcessorFeaturePresent(PF_AVX_INSTRUCTIONS_AVAILABLE):
                cpu.has_avx = True
            if IsProcessorFeaturePresent(PF_SSE4_2_INSTRUCTIONS_AVAILABLE):
                cpu.has_sse4_2 = True
        except Exception as e:
            logger.debug(f"Win32 指令集检测失败: {e}")

    @staticmethod
    def _parse_cpu_flags(cpu: CPUInfo, line: str) -> None:
        """从 /proc/cpuinfo 的 flags 行解析指令集"""
        flags = line.lower()
        if "avx512" in flags:
            cpu.has_avx512 = True
        if "avx2" in flags:
            cpu.has_avx2 = True
        if " avx " in flags or flags.startswith("flags") and "avx" in flags:
            cpu.has_avx = True
        if "fma" in flags:
            cpu.has_fma = True
        if "f16c" in flags:
            cpu.has_f16c = True
        if "sse4_2" in flags:
            cpu.has_sse4_2 = True
        if "neon" in flags:
            cpu.has_neon = True

    # ----------------------------------------------------------
    # GPU 检测
    # ----------------------------------------------------------
    def detect_gpu(self) -> GPUInfo:
        """检测 GPU 信息"""
        gpu = GPUInfo()

        # 按优先级依次尝试
        detected = False
        if not detected:
            detected = self._detect_nvidia_gpu(gpu)
        if not detected:
            detected = self._detect_amd_gpu(gpu)
        if not detected:
            detected = self._detect_apple_gpu(gpu)
        if not detected:
            detected = self._detect_intel_gpu(gpu)

        self.profile.gpu = gpu
        logger.info(f"GPU 检测完成: {gpu.summary}")
        return gpu

    def _detect_nvidia_gpu(self, gpu: GPUInfo) -> bool:
        """通过 nvidia-smi 检测 NVIDIA GPU"""
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            # 尝试常见路径
            for path in [
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                "/usr/bin/nvidia-smi",
                "/usr/local/bin/nvidia-smi",
            ]:
                if os.path.exists(path):
                    nvidia_smi = path
                    break

        if not nvidia_smi:
            return False

        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,memory.free,"
                    "driver_version,compute_cap",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                return False

            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if not lines:
                return False

            gpu.gpu_count = len(lines)
            # 取第一块 GPU 的信息
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 5:
                gpu.vendor = GPUVendor.NVIDIA
                gpu.name = parts[0]
                gpu.vram_total_mb = int(float(parts[1]))
                gpu.vram_free_mb = int(float(parts[2]))
                gpu.driver_version = parts[3]
                gpu.compute_capability = parts[4]

                # Tensor Core: compute capability >= 7.0
                try:
                    major = int(gpu.compute_capability.split(".")[0])
                    gpu.tensor_cores = major >= 7
                except (ValueError, IndexError):
                    pass

            # 检测 CUDA 版本
            try:
                cuda_result = subprocess.run(
                    [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )
                # nvidia-smi 输出的 CUDA 版本在 banner 中
                banner = subprocess.run(
                    [nvidia_smi], capture_output=True, text=True, timeout=10
                )
                for bl in banner.stdout.splitlines():
                    if "CUDA Version" in bl:
                        import re
                        match = re.search(r"CUDA Version:\s*([\d.]+)", bl)
                        if match:
                            gpu.cuda_version = match.group(1)
            except Exception:
                pass

            return True

        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning(f"nvidia-smi 检测失败: {e}")
            return False

    def _detect_amd_gpu(self, gpu: GPUInfo) -> bool:
        """检测 AMD GPU (通过 rocm-smi 或系统工具)"""
        # Linux: rocm-smi
        rocm_smi = shutil.which("rocm-smi")
        if rocm_smi:
            try:
                result = subprocess.run(
                    [rocm_smi, "--showmeminfo", "vram", "--showproductname"],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0 and result.stdout.strip():
                    gpu.vendor = GPUVendor.AMD
                    for line in result.stdout.splitlines():
                        line_lower = line.lower()
                        if "card" in line_lower and "series" in line_lower:
                            gpu.name = line.strip()
                        if "total" in line_lower and ("mb" in line_lower or "gb" in line_lower):
                            import re
                            match = re.search(r"([\d.]+)\s*(MB|GB|MiB|GiB)", line, re.IGNORECASE)
                            if match:
                                val = float(match.group(1))
                                unit = match.group(2).upper()
                                if "GB" in unit or "GIB" in unit:
                                    val *= 1024
                                gpu.vram_total_mb = int(val)

                    # 检测 ROCm 版本
                    try:
                        ver_result = subprocess.run(
                            [rocm_smi, "--version"],
                            capture_output=True, text=True, timeout=10
                        )
                        import re
                        match = re.search(r"rocm[_-]smi.*?(\d+\.\d+[\.\d]*)", ver_result.stdout, re.IGNORECASE)
                        if match:
                            gpu.rocm_version = match.group(1)
                    except Exception:
                        pass

                    if gpu.vram_total_mb > 0:
                        return True
            except Exception as e:
                logger.debug(f"rocm-smi 检测失败: {e}")

        # Windows: 通过 WMIC 或 dxdiag 检测 AMD
        if sys.platform == "win32":
            return self._detect_gpu_wmic(gpu, target_vendor="amd")

        return False

    def _detect_apple_gpu(self, gpu: GPUInfo) -> bool:
        """检测 Apple Silicon GPU (Metal)"""
        if sys.platform != "darwin":
            return False

        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=15
            )
            output = result.stdout
            if "Apple" in output:
                gpu.vendor = GPUVendor.APPLE
                gpu.metal_support = True
                gpu.unified_memory = True

                # 提取芯片名称
                for line in output.splitlines():
                    line = line.strip()
                    if "Chipset Model" in line or "Chip" in line:
                        gpu.name = line.split(":")[-1].strip()
                    if "Total Number of Cores" in line:
                        import re
                        match = re.search(r"(\d+)", line)
                        if match:
                            gpu.gpu_count = int(match.group(1))

                # Apple Silicon 统一内存：从 sysctl 获取总内存作为 GPU 可用量
                try:
                    mem_result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True, timeout=10
                    )
                    if mem_result.stdout.strip().isdigit():
                        total_bytes = int(mem_result.stdout.strip())
                        gpu.vram_total_mb = total_bytes // (1024 * 1024)
                        gpu.vram_free_mb = gpu.vram_total_mb  # 统一内存，简化处理
                except Exception:
                    pass

                return True
        except Exception as e:
            logger.debug(f"Apple GPU 检测失败: {e}")

        return False

    def _detect_intel_gpu(self, gpu: GPUInfo) -> bool:
        """检测 Intel 集成显卡 (有限支持)"""
        if sys.platform == "win32":
            return self._detect_gpu_wmic(gpu, target_vendor="intel")
        return False

    def _detect_gpu_wmic(self, gpu: GPUInfo, target_vendor: str = "") -> bool:
        """Windows 通过 WMIC 检测 GPU"""
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get",
                 "Name,AdapterRAM,DriverVersion", "/format:list"],
                capture_output=True, text=True, timeout=15
            )

            current_name = ""
            current_vram = 0
            current_driver = ""

            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.startswith("Name=") and line[5:]:
                    # 处理上一个 GPU
                    if current_name and self._gpu_matches_vendor(current_name, target_vendor):
                        gpu.vendor = self._name_to_vendor(current_name)
                        gpu.name = current_name
                        gpu.vram_total_mb = current_vram
                        gpu.driver_version = current_driver
                        gpu.gpu_count += 1
                        return True
                    current_name = line[5:]
                    current_vram = 0
                    current_driver = ""
                elif line.startswith("AdapterRAM=") and line[11:]:
                    try:
                        current_vram = int(line[11:]) // (1024 * 1024)
                    except ValueError:
                        current_vram = 0
                elif line.startswith("DriverVersion=") and line[14:]:
                    current_driver = line[14:]

            # 最后一个 GPU
            if current_name and self._gpu_matches_vendor(current_name, target_vendor):
                gpu.vendor = self._name_to_vendor(current_name)
                gpu.name = current_name
                gpu.vram_total_mb = current_vram
                gpu.driver_version = current_driver
                gpu.gpu_count += 1
                return current_vram > 0

        except Exception as e:
            logger.debug(f"WMIC GPU 检测失败: {e}")

        return False

    @staticmethod
    def _gpu_matches_vendor(name: str, target: str) -> bool:
        name_lower = name.lower()
        if target == "nvidia":
            return "nvidia" in name_lower or "geforce" in name_lower or "rtx" in name_lower or "gtx" in name_lower
        if target == "amd":
            return "amd" in name_lower or "radeon" in name_lower
        if target == "intel":
            return "intel" in name_lower or "iris" in name_lower or "uhd" in name_lower
        return True

    @staticmethod
    def _name_to_vendor(name: str) -> GPUVendor:
        name_lower = name.lower()
        if "nvidia" in name_lower or "geforce" in name_lower or "rtx" in name_lower or "gtx" in name_lower:
            return GPUVendor.NVIDIA
        if "amd" in name_lower or "radeon" in name_lower:
            return GPUVendor.AMD
        if "apple" in name_lower:
            return GPUVendor.APPLE
        if "intel" in name_lower:
            return GPUVendor.INTEL
        return GPUVendor.UNKNOWN

    # ----------------------------------------------------------
    # 内存与存储检测
    # ----------------------------------------------------------
    def detect_memory(self) -> MemoryInfo:
        """检测内存和存储信息"""
        mem = MemoryInfo()

        try:
            if sys.platform == "win32":
                self._detect_memory_windows(mem)
            elif sys.platform == "darwin":
                self._detect_memory_darwin(mem)
            else:
                self._detect_memory_linux(mem)
        except Exception as e:
            logger.warning(f"内存检测失败: {e}")
            # 回退：通过 Python 获取近似值
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", c_ulonglong),
                        ("ullAvailPhys", c_ulonglong),
                        ("ullTotalPageFile", c_ulonglong),
                        ("ullAvailPageFile", c_ulonglong),
                        ("ullTotalVirtual", c_ulonglong),
                        ("ullAvailVirtual", c_ulonglong),
                        ("ullAvailExtendedVirtual", c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    mem.total_mb = stat.ullTotalPhys // (1024 * 1024)
                    mem.available_mb = stat.ullAvailPhys // (1024 * 1024)
                    mem.swap_total_mb = (stat.ullTotalPageFile - stat.ullTotalPhys) // (1024 * 1024)
            except Exception:
                pass

        # 磁盘检测
        self._detect_disk(mem)

        self.profile.memory = mem
        logger.info(f"内存检测完成: {mem.summary}")
        return mem

    def _detect_memory_windows(self, mem: MemoryInfo) -> None:
        """Windows 内存检测"""
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                mem.total_mb = stat.ullTotalPhys // (1024 * 1024)
                mem.available_mb = stat.ullAvailPhys // (1024 * 1024)
                mem.swap_total_mb = (stat.ullTotalPageFile - stat.ullTotalPhys) // (1024 * 1024)
        except Exception as e:
            logger.debug(f"Windows 内存检测失败: {e}")

        # 检测内存频率和通道
        self._detect_memory_details_windows(mem)

    def _detect_memory_darwin(self, mem: MemoryInfo) -> None:
        """macOS 内存检测"""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip().isdigit():
                mem.total_mb = int(result.stdout.strip()) // (1024 * 1024)
                mem.available_mb = mem.total_mb  # 简化：macOS 内存管理复杂
        except Exception:
            pass

    def _detect_memory_linux(self, mem: MemoryInfo) -> None:
        """Linux 内存检测"""
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem.total_mb = int(line.split()[1]) // 1024
                    elif line.startswith("MemAvailable:"):
                        mem.available_mb = int(line.split()[1]) // 1024
                    elif line.startswith("SwapTotal:"):
                        mem.swap_total_mb = int(line.split()[1]) // 1024
        except Exception:
            pass

        # 检测内存频率和通道
        self._detect_memory_details_linux(mem)

    def _detect_memory_details_windows(self, mem: MemoryInfo) -> None:
        """Windows 检测内存频率、通道数和 DDR 类型"""
        try:
            # 通过 WMIC 获取内存信息
            result = subprocess.run(
                ["wmic", "memorychip", "get",
                 "Speed,MemoryType,DeviceLocator", "/format:list"],
                capture_output=True, text=True, timeout=15
            )
            speeds = []
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.startswith("Speed=") and line[6:]:
                    try:
                        speeds.append(int(line[6:]))
                    except ValueError:
                        pass
                elif line.startswith("MemoryType=") and line[11:]:
                    try:
                        mem_type_id = int(line[11:])
                        # DDR 类型映射
                        ddr_map = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 34: "DDR5"}
                        mem.ddr_type = ddr_map.get(mem_type_id, f"类型{mem_type_id}")
                    except ValueError:
                        pass

            if speeds:
                mem.speed_mhz = min(speeds)  # 取最低频率（双通道以最慢为准）
                mem.channels = len(speeds)

            # 如果 WMIC 没有获取到 DDR 类型，通过 PowerShell 补充
            if not mem.ddr_type:
                ps_result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-CimInstance -ClassName Win32_PhysicalMemory | "
                     "Select-Object -First 1 SMBIOSMemoryType | ConvertTo-Json"],
                    capture_output=True, text=True, timeout=15
                )
                if ps_result.returncode == 0 and ps_result.stdout.strip():
                    import json
                    data = json.loads(ps_result.stdout)
                    smbios_type = data.get("SMBIOSMemoryType", 0)
                    ddr_map = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 34: "DDR5"}
                    mem.ddr_type = ddr_map.get(smbios_type, "")
        except Exception as e:
            logger.debug(f"Windows 内存详细信息检测失败: {e}")

    def _detect_memory_details_linux(self, mem: MemoryInfo) -> None:
        """Linux 检测内存频率、通道数和 DDR 类型"""
        try:
            # 方法1: 通过 dmidecode (需要 root 权限)
            result = subprocess.run(
                ["dmidecode", "-t", "memory"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                speeds = []
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if "Speed:" in line and "MHz" in line:
                        import re
                        match = re.search(r"(\d+)\s*MHz", line)
                        if match:
                            speeds.append(int(match.group(1)))
                    elif "Type:" in line and "DDR" in line:
                        mem.ddr_type = line.split(":")[-1].strip()
                if speeds:
                    mem.speed_mhz = min(speeds)
                    mem.channels = len(speeds) // 2  # 每个通道通常有 1-2 个 DIMM
        except Exception:
            # 方法2: 通过 /proc/meminfo 和 lscpu 推断
            try:
                result = subprocess.run(
                    ["lscpu"], capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.splitlines():
                    if "NUMA" in line and "node" in line:
                        # 粗略估算通道数
                        import re
                        match = re.search(r"(\d+)", line)
                        if match:
                            mem.channels = max(1, int(match.group(1)) // 2)
            except Exception:
                pass

    def _detect_disk(self, mem: MemoryInfo) -> None:
        """检测磁盘空间和类型"""
        try:
            if sys.platform == "win32":
                # Windows: 检测当前驱动器
                drive = os.path.splitdrive(os.getcwd())[0] + "\\"
                usage = shutil.disk_usage(drive)
            else:
                usage = shutil.disk_usage("/")

            mem.disk_total_gb = usage.total / (1024 ** 3)
            mem.disk_free_gb = usage.free / (1024 ** 3)
        except Exception:
            pass

        # 磁盘类型检测
        try:
            if sys.platform == "win32":
                self._detect_disk_type_windows(mem)
            elif sys.platform == "linux":
                self._detect_disk_type_linux(mem)
            elif sys.platform == "darwin":
                mem.disk_type = "NVMe/SSD"  # 现代 Mac 基本都是 SSD
        except Exception:
            pass

    def _detect_disk_type_windows(self, mem: MemoryInfo) -> None:
        """Windows 检测磁盘类型 (SSD/HDD/NVMe)"""
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-PhysicalDisk | Select-Object MediaType,BusType | ConvertTo-Json"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                import json
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    bus_type = item.get("BusType", "")
                    media_type = item.get("MediaType", "Unknown")
                    if bus_type == "NVMe":
                        mem.disk_type = "NVMe"
                        mem.disk_speed.is_nvme = True
                        mem.disk_speed.interface = "NVMe"
                        break
                    elif media_type == "SSD":
                        mem.disk_type = "SSD"
                        mem.disk_speed.interface = "SATA"
                    elif media_type == "HDD":
                        mem.disk_type = "HDD"
                        mem.disk_speed.interface = "SATA"
        except Exception:
            mem.disk_type = "未知"

        # 尝试通过 winsat 获取磁盘速度（需管理员权限，可能失败）
        self._detect_disk_speed_windows(mem)

    def _detect_disk_type_linux(self, mem: MemoryInfo) -> None:
        """Linux 检测磁盘类型 (SSD/HDD/NVMe)"""
        try:
            import glob as glob_mod
            # 检查是否为 NVMe 设备
            nvme_devices = glob_mod.glob("/dev/nvme*")
            if nvme_devices:
                mem.disk_type = "NVMe"
                mem.disk_speed.is_nvme = True
                mem.disk_speed.interface = "NVMe"
                # 尝试通过 lspci 检测 PCIe 代数
                self._detect_nvme_pcie_gen_linux(mem)
            else:
                # 检查 /sys/block/*/queue/rotational
                rotational_files = glob_mod.glob("/sys/block/*/queue/rotational")
                for f in rotational_files:
                    with open(f, "r") as fh:
                        if fh.read().strip() == "0":
                            mem.disk_type = "SSD"
                            mem.disk_speed.interface = "SATA"
                            return
                mem.disk_type = "HDD"
                mem.disk_speed.interface = "SATA"
        except Exception:
            mem.disk_type = "未知"

        # 尝试通过 nvme cli 获取速度信息
        self._detect_nvme_speed_linux(mem)

    def _detect_nvme_pcie_gen_linux(self, mem: MemoryInfo) -> None:
        """Linux 检测 NVMe PCIe 代数"""
        try:
            result = subprocess.run(
                ["lspci", "-v"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if "nvme" in line.lower() or "non-volatile" in line.lower():
                    if "Gen4" in line or "8GT/s" in line:
                        mem.disk_speed.pcie_generation = 4
                    elif "Gen5" in line or "16GT/s" in line:
                        mem.disk_speed.pcie_generation = 5
                    elif "Gen3" in line or "2.5GT/s" in line:
                        mem.disk_speed.pcie_generation = 3
        except Exception:
            pass

    def _detect_nvme_speed_linux(self, mem: MemoryInfo) -> None:
        """Linux 通过 nvme cli 检测 NVMe 速度"""
        nvme_cli = shutil.which("nvme")
        if not nvme_cli:
            return
        try:
            # 获取 NVMe 设备列表
            result = subprocess.run(
                [nvme_cli, "list", "-o", "json"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                return
            import json
            data = json.loads(result.stdout)
            devices = data.get("Devices", [])
            if not devices:
                return
            # 获取第一个 NVMe 设备的速度信息
            device_path = devices[0].get("DevicePath", "")
            if not device_path:
                return
            # 通过 smart-log 获取性能数据
            smart_result = subprocess.run(
                [nvme_cli, "smart-log", device_path, "-o", "json"],
                capture_output=True, text=True, timeout=15
            )
            if smart_result.returncode == 0:
                smart_data = json.loads(smart_result.stdout)
                # NVMe 规范中的性能信息通常在 controller 数据中
                # 估算基于 PCIe 代数的速度
                if mem.disk_speed.pcie_generation >= 4:
                    mem.disk_speed.sequential_read_mbps = 7000
                    mem.disk_speed.sequential_write_mbps = 5000
                elif mem.disk_speed.pcie_generation >= 3:
                    mem.disk_speed.sequential_read_mbps = 3500
                    mem.disk_speed.sequential_write_mbps = 3000
        except Exception:
            pass

    def _detect_disk_speed_windows(self, mem: MemoryInfo) -> None:
        """Windows 检测磁盘速度 (通过 winsat 或 PowerShell)"""
        # 方法1: 尝试通过 PowerShell 获取 NVMe 性能信息
        if mem.disk_speed.is_nvme:
            try:
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-PhysicalDisk | Select-Object DeviceId,FriendlyName,"
                     "MediaType,BusType | ConvertTo-Json"],
                    capture_output=True, text=True, timeout=15
                )
                # 基于 NVMe 类型估算速度
                # PCIe 4.0 NVMe: ~7000MB/s 读, PCIe 3.0: ~3500MB/s
                if result.returncode == 0:
                    # 检查是否有 PCIe 代数信息
                    import json
                    data = json.loads(result.stdout)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        if item.get("BusType") == "NVMe":
                            # 默认按 PCIe 3.0 NVMe 估算
                            mem.disk_speed.sequential_read_mbps = 3500
                            mem.disk_speed.sequential_write_mbps = 3000
                            mem.disk_speed.pcie_generation = 3
                            break
            except Exception:
                pass

        # 方法2: 尝试 winsat (需要管理员权限，可能失败)
        if mem.disk_speed.sequential_read_mbps == 0:
            try:
                # 检查是否以管理员权限运行
                import ctypes
                is_admin = ctypes.windll.shell32.IsUserAnAdmin()
                if is_admin:
                    result = subprocess.run(
                        ["winsat", "disk", "-drive", "c"],
                        capture_output=True, text=True, timeout=60
                    )
                    # 解析 winsat 输出获取速度
                    for line in result.stdout.splitlines():
                        if "Disk Sequential" in line and "Read" in line:
                            import re
                            match = re.search(r"([\d.]+)\s*MB/s", line)
                            if match:
                                mem.disk_speed.sequential_read_mbps = float(match.group(1))
                        elif "Disk Sequential" in line and "Write" in line:
                            import re
                            match = re.search(r"([\d.]+)\s*MB/s", line)
                            if match:
                                mem.disk_speed.sequential_write_mbps = float(match.group(1))
            except Exception:
                pass

    # ----------------------------------------------------------
    # 配置推荐
    # ----------------------------------------------------------
    def _generate_recommendation(self) -> None:
        """根据硬件画像生成最优配置建议"""
        rec = ModelRecommendation()
        gpu = self.profile.gpu
        cpu = self.profile.cpu
        mem = self.profile.memory

        # 1. 确定推理后端
        rec.backend = self._determine_backend(gpu)
        rec.use_gpu = gpu.has_gpu
        rec.use_metal = gpu.vendor == GPUVendor.APPLE and gpu.metal_support
        rec.use_cuda = gpu.vendor == GPUVendor.NVIDIA and gpu.has_gpu
        rec.use_rocm = gpu.vendor == GPUVendor.AMD and gpu.has_gpu

        # 2. 确定可用内存总量（用于模型加载）
        available_for_model_gb = self._estimate_model_memory_budget(gpu, mem)

        # 3. 推荐量化等级
        rec.recommended_quant = self._recommend_quantization(gpu, available_for_model_gb)

        # 4. 推荐最大模型体积
        rec.max_model_size_gb = self._estimate_max_model_size(
            gpu, mem, available_for_model_gb
        )

        # 5. GPU offload 层数
        if rec.use_gpu and rec.use_cuda:
            rec.gpu_offload_layers = self._estimate_gpu_layers(gpu, rec.max_model_size_gb)
        elif rec.use_metal:
            rec.gpu_offload_layers = 999  # Apple Silicon 全部 offload
        else:
            rec.gpu_offload_layers = 0

        # 6. CPU 线程数
        rec.threads = self._recommend_threads(cpu)

        # 7. 上下文长度
        rec.context_length = self._recommend_context_length(gpu, mem, rec.max_model_size_gb)

        # 8. 批处理大小
        rec.batch_size = self._recommend_batch_size(gpu, mem)

        # 9. 备注
        rec.notes = self._generate_notes(gpu, cpu, mem, rec)

        self.profile.recommendation = rec

    def _determine_backend(self, gpu: GPUInfo) -> str:
        """确定最优推理后端"""
        if gpu.vendor == GPUVendor.NVIDIA and gpu.has_gpu:
            return "cuda"
        if gpu.vendor == GPUVendor.APPLE and gpu.metal_support:
            return "metal"
        if gpu.vendor == GPUVendor.AMD and gpu.has_gpu:
            return "rocm"
        return "cpu"

    def _estimate_model_memory_budget(self, gpu: GPUInfo, mem: MemoryInfo) -> float:
        """估算可用于模型的内存预算 (GB)"""
        if gpu.has_gpu and not gpu.unified_memory:
            # 独立 GPU：主要用显存，预留 20% 给系统
            return gpu.vram_total_gb * 0.80
        elif gpu.unified_memory:
            # 统一内存 (Apple Silicon)：内存和 GPU 共享，取较小值
            # 预留 4GB 给系统
            usable = max(0, mem.total_gb - 4.0)
            return usable * 0.70  # 70% 给模型
        else:
            # 纯 CPU 推理：用内存
            return mem.available_gb * 0.60

    @staticmethod
    def _recommend_quantization(gpu: GPUInfo, budget_gb: float) -> ModelQuant:
        """根据可用内存推荐量化等级"""
        if budget_gb >= 32:
            return ModelQuant.Q6_K      # 大显存：高质量
        elif budget_gb >= 16:
            return ModelQuant.Q5_K_M    # 中等显存：平衡
        elif budget_gb >= 8:
            return ModelQuant.Q4_K_M    # 小显存：推荐
        elif budget_gb >= 4:
            return ModelQuant.Q4_K_S    # 极小显存
        else:
            return ModelQuant.Q3_K_M    # 极端受限

    @staticmethod
    def _estimate_max_model_size(gpu: GPUInfo, mem: MemoryInfo, budget_gb: float) -> float:
        """估算推荐最大模型体积 (GB)"""
        if gpu.has_gpu and not gpu.unified_memory:
            # 独立 GPU：纯显存推理的模型大小上限
            return budget_gb
        elif gpu.unified_memory:
            # Apple Silicon：可以用部分内存
            return budget_gb
        else:
            # 纯 CPU：取内存的 60% 作为模型大小上限
            return budget_gb

    @staticmethod
    def _estimate_gpu_layers(gpu: GPUInfo, model_size_gb: float) -> int:
        """估算 GPU offload 层数"""
        if not gpu.has_gpu:
            return 0
        # 粗略估算：按显存与模型比例
        ratio = gpu.vram_total_gb / max(model_size_gb, 0.1)
        # 一个 7B 模型大约 32-40 层
        if ratio >= 1.0:
            return 999  # 全部 offload
        elif ratio >= 0.5:
            return 24
        elif ratio >= 0.25:
            return 12
        else:
            return 4

    @staticmethod
    def _recommend_threads(cpu: CPUInfo) -> int:
        """推荐 CPU 推理线程数"""
        # 通常使用物理核心数，不超过 8
        return min(cpu.physical_cores, 8) if cpu.physical_cores > 0 else 4

    def _recommend_context_length(self, gpu: GPUInfo, mem: MemoryInfo, model_size_gb: float) -> int:
        """推荐上下文长度"""
        if gpu.has_gpu and not gpu.unified_memory:
            vram_left = gpu.vram_total_gb - model_size_gb
            if vram_left >= 6:
                return 8192
            elif vram_left >= 3:
                return 4096
            else:
                return 2048
        else:
            mem_left = mem.available_gb - model_size_gb
            if mem_left >= 16:
                return 8192
            elif mem_left >= 8:
                return 4096
            elif mem_left >= 4:
                return 2048
            else:
                return 1024

    def _recommend_batch_size(self, gpu: GPUInfo, mem: MemoryInfo) -> int:
        """推荐批处理大小"""
        if gpu.has_gpu:
            if gpu.vram_total_gb >= 16:
                return 2048
            elif gpu.vram_total_gb >= 8:
                return 1024
            else:
                return 512
        else:
            if mem.total_gb >= 32:
                return 512
            elif mem.total_gb >= 16:
                return 256
            else:
                return 128

    def _generate_notes(self, gpu: GPUInfo, cpu: CPUInfo, mem: MemoryInfo,
                        rec: ModelRecommendation) -> List[str]:
        """生成硬件相关的备注和优化建议"""
        notes = []

        # GPU 相关建议
        if not gpu.has_gpu:
            notes.append("未检测到独立GPU，将使用纯CPU推理，速度较慢")
            if cpu.has_avx2:
                notes.append("CPU 支持 AVX2，可加速 CPU 推理")
            elif not cpu.has_avx:
                notes.append("CPU 不支持 AVX，推理速度可能非常慢，建议升级硬件")
        else:
            if gpu.vendor == GPUVendor.NVIDIA:
                if gpu.tensor_cores:
                    notes.append(f"GPU 具有 Tensor Core (CC {gpu.compute_capability})，支持高效推理")
                if gpu.vram_total_gb < 6:
                    notes.append("GPU 显存不足 6GB，建议使用较小模型或更低量化")
                if rec.gpu_offload_layers < 999:
                    notes.append("显存不足以完全加载模型，部分层将由CPU处理 (混合推理)")
            elif gpu.vendor == GPUVendor.APPLE:
                notes.append("Apple Silicon 统一内存架构，GPU 可直接访问全部内存")
                notes.append("推荐使用 MLX 或 llama.cpp Metal 后端")

        # 内存相关建议
        if mem.total_gb < 8:
            notes.append("系统内存不足 8GB，运行大模型时可能遇到问题")
        elif mem.total_gb < 16:
            notes.append("系统内存 16GB 以下，建议运行 7B 及以下规模模型")

        # 内存带宽建议
        if mem.speed_mhz > 0 and mem.ddr_type:
            mem_info = f"{mem.ddr_type}-{mem.speed_mhz}"
            if mem.channels > 0:
                mem_info += f" {mem.channels}通道"
            if mem.speed_mhz < 2666 and "DDR4" in mem.ddr_type:
                notes.append(f"内存频率较低 ({mem_info})，CPU 推理速度可能受限")
            elif mem.speed_mhz >= 4800:
                notes.append(f"内存频率较高 ({mem_info})，有利于 CPU 推理性能")

        # 存储相关建议
        disk_speed = mem.disk_speed
        if mem.disk_type == "HDD":
            notes.append("检测到机械硬盘，模型加载速度会很慢，强烈建议升级到 SSD")
        elif disk_speed.is_nvme:
            if disk_speed.sequential_read_mbps >= 5000:
                notes.append(f"NVMe SSD 速度优秀 (读 {disk_speed.sequential_read_mbps:.0f}MB/s)，模型加载快速")
            elif disk_speed.sequential_read_mbps >= 3000:
                notes.append(f"NVMe SSD 速度良好 (读 {disk_speed.sequential_read_mbps:.0f}MB/s)")
            elif disk_speed.pcie_generation > 0:
                notes.append(f"NVMe SSD (PCIe {disk_speed.pcie_generation}.0)，建议检查是否安装在正确的 M.2 插槽")
        elif mem.disk_type == "SSD":
            if disk_speed.sequential_read_mbps > 0:
                notes.append(f"SATA SSD (读 {disk_speed.sequential_read_mbps:.0f}MB/s)，模型加载速度一般")

        # NVMe 特殊建议：模型加载优化
        if disk_speed.is_nvme and disk_speed.sequential_read_mbps >= 3000:
            notes.append("NVMe 高速存储适合运行大型 GGUF 模型 (支持 mmap 快速加载)")

        # CPU 相关建议
        if cpu.logical_cores <= 2:
            notes.append("CPU 核心数过少，多任务运行时性能可能下降")

        return notes

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """将硬件画像导出为字典"""
        from dataclasses import asdict
        return asdict(self.profile)

    def to_json(self, indent: int = 2) -> str:
        """将硬件画像导出为 JSON 字符串"""
        import json
        # 处理 Enum 序列化
        def _default(obj):
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=_default)


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行直接运行：打印硬件检测结果"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    detector = HardwareDetector()
    print(detector.profile.summary)
    print()
    print("JSON 输出:")
    print(detector.to_json())


if __name__ == "__main__":
    main()
