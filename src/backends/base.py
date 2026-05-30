"""推理后端基类定义"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Dict, Iterator, List, Optional


class BackendType(Enum):
    """推理后端类型枚举"""
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    EXLLAMA = "exllama"
    TRANSFORMERS = "transformers"
    VULKAN = "vulkan"          # 跨平台GPU加速 (NVIDIA/AMD/Intel/Apple via MoltenVK)


@dataclass
class InferenceConfig:
    """
    推理配置数据类，包含内存优化参数。

    Attributes:
        model_path: 模型文件路径
        n_gpu_layers: GPU卸载层数，-1表示全部卸载到GPU
        n_batch: 批处理大小，影响prompt处理速度
        n_ctx: 上下文窗口大小
        n_threads: CPU线程数，0表示自动检测
        use_mmap: 使用内存映射加载模型，减少内存占用
        use_mlock: 锁定模型在内存中，防止被交换到磁盘
        rope_freq_base: RoPE频率基数，用于上下文扩展
        rope_freq_scale: RoPE频率缩放因子
        verbose: 是否启用详细日志
        extra_params: 额外的后端特定参数
    """
    model_path: str
    n_gpu_layers: int = -1
    n_batch: int = 512
    n_ctx: int = 4096
    n_threads: int = 0
    use_mmap: bool = True
    use_mlock: bool = False
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0
    verbose: bool = False
    extra_params: Dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """
    生成结果数据类。

    Attributes:
        text: 生成的文本
        tokens_generated: 生成的token数量
        prompt_tokens: 输入prompt的token数量
        tokens_per_second: 生成速度
        finish_reason: 生成结束原因 ("stop", "length", "error")
        total_duration_ms: 总耗时(毫秒)
    """
    text: str = ""
    tokens_generated: int = 0
    prompt_tokens: int = 0
    tokens_per_second: float = 0.0
    finish_reason: str = ""
    total_duration_ms: float = 0.0


@dataclass
class MemoryUsage:
    """
    内存使用报告。

    Attributes:
        vram_used_mb: 当前VRAM使用量(MB)
        vram_total_mb: 总可用VRAM(MB)
        ram_used_mb: 当前系统RAM使用量(MB)
        model_size_mb: 模型文件大小(MB)
        kv_cache_mb: KV缓存占用(MB)
    """
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    ram_used_mb: float = 0.0
    model_size_mb: float = 0.0
    kv_cache_mb: float = 0.0


class InferenceBackend(ABC):
    """推理后端抽象基类"""

    def __init__(self, config: InferenceConfig):
        self._config = config
        self._is_loaded = False

    @property
    def config(self) -> InferenceConfig:
        """获取当前配置"""
        return self._config

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        return self._is_loaded

    @abstractmethod
    def load_model(self) -> None:
        """加载模型到内存

        Raises:
            RuntimeError: 模型加载失败
            FileNotFoundError: 模型文件不存在
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型，释放内存"""
        pass

    @abstractmethod
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
        """同步生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            repeat_penalty: 重复惩罚系数
            stop: 停止词列表
            seed: 随机种子

        Returns:
            GenerationResult: 生成结果

        Raises:
            RuntimeError: 生成失败
            ValueError: 参数无效
        """
        pass

    @abstractmethod
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
        """同步流式生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            repeat_penalty: 重复惩罚系数
            stop: 停止词列表
            seed: 随机种子

        Yields:
            str: 生成的文本片段

        Raises:
            RuntimeError: 生成失败
            ValueError: 参数无效
        """
        pass

    @abstractmethod
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
        """异步流式生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            repeat_penalty: 重复惩罚系数
            stop: 停止词列表
            seed: 随机种子

        Yields:
            str: 生成的文本片段

        Raises:
            RuntimeError: 生成失败
            ValueError: 参数无效
        """
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """获取文本的embedding向量

        Args:
            text: 输入文本

        Returns:
            List[float]: embedding向量

        Raises:
            RuntimeError: 获取失败
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> MemoryUsage:
        """获取当前内存使用情况

        Returns:
            MemoryUsage: 内存使用详情
        """
        pass

    @staticmethod
    @abstractmethod
    def get_optimal_config(
        model_path: str,
        vram_budget_mb: Optional[float] = None
    ) -> InferenceConfig:
        """根据硬件条件计算最优配置

        Args:
            model_path: 模型文件路径
            vram_budget_mb: VRAM预算(MB)，None表示自动检测

        Returns:
            InferenceConfig: 最优配置
        """
        pass

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
        return False
