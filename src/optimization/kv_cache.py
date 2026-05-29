"""
KV Cache 优化模块
实现 KV Cache 的分页管理、量化、压缩和前缀共享等优化策略，
支持动态内存管理和基于注意力分数的智能淘汰。

典型使用流程:
    config = KVCacheConfig(cache_bits=8, max_cache_size_gb=2.0)
    optimizer = KVCacheOptimizer(config)
    size_gb = optimizer.estimate_cache_size(seq_length=4096, num_layers=32, num_heads=32, head_dim=128)
    compressed = optimizer.compress_cache(cache_data, target_size=size_gb * 0.5)
"""

import bisect
import hashlib
import logging
import math
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举与常量
# ---------------------------------------------------------------------------

class EvictionPolicy(Enum):
    """Cache 淘汰策略"""
    LRU = "lru"                         # 最近最少使用
    LFU = "lfu"                         # 最不经常使用
    ATTENTION_SCORE = "attention_score"  # 基于注意力分数


class CachePrecision(Enum):
    """Cache 量化精度"""
    FP16 = 16
    INT8 = 8
    INT4 = 4


# 每个元素字节数 (FP16 = 2 bytes)
_BYTES_PER_ELEMENT: Dict[int, int] = {
    16: 2,
    8: 1,
    4: 0.5,
}


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------

@dataclass
class KVCacheConfig:
    """KV Cache 配置

    Attributes:
        cache_bits: Cache 量化位数，支持 16(FP16) / 8(INT8) / 4(INT4)
        max_cache_size_gb: 最大 Cache 大小 (GB)
        eviction_policy: 淘汰策略 (lru / lfu / attention_score)
        prefix_sharing: 是否启用前缀共享
        compression_ratio: 压缩比例 (0.0 ~ 1.0)，表示压缩后保留的比例
        page_size: PagedAttention 页大小 (tokens 数)
        num_pages: 预分配页数，0 表示自动计算
    """
    cache_bits: int = 16
    max_cache_size_gb: float = 2.0
    eviction_policy: str = "attention_score"
    prefix_sharing: bool = True
    compression_ratio: float = 0.5
    page_size: int = 64
    num_pages: int = 0


@dataclass
class CacheEntry:
    """单个 Cache 条目 (一页)"""
    page_id: str
    seq_id: str
    start_token: int
    end_token: int
    # 实际 KV 数据 (numpy array 或 list)，此处用 Any 保持轻量
    key_data: Any = None
    value_data: Any = None
    # 元信息
    attention_score: float = 0.0
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    is_pinned: bool = False  # 被钉住的页不会被淘汰
    precision_bits: int = 16  # 当前存储精度


@dataclass
class SequenceCache:
    """单个序列的 Cache 管理"""
    seq_id: str
    prefix_hash: Optional[str] = None   # 前缀哈希，用于前缀共享
    pages: List[CacheEntry] = field(default_factory=list)
    total_tokens: int = 0
    is_shared: bool = False             # 是否为共享前缀


@dataclass
class CacheStats:
    """Cache 统计信息"""
    total_pages: int = 0
    used_pages: int = 0
    free_pages: int = 0
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    eviction_count: int = 0
    compression_count: int = 0
    prefix_share_count: int = 0
    quantization_savings_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.cache_hit_count + self.cache_miss_count
        return self.cache_hit_count / total if total > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        """内存利用率"""
        return self.used_memory_mb / self.total_memory_mb if self.total_memory_mb > 0 else 0.0


# ---------------------------------------------------------------------------
# PagedAttention 管理器
# ---------------------------------------------------------------------------

class PagedAttentionManager:
    """PagedAttention 分页管理器

    将 KV Cache 划分为固定大小的页 (page)，减少内存碎片，
    支持非连续物理页映射到连续逻辑地址空间。
    """

    def __init__(self, page_size: int = 64, num_pages: int = 256, precision_bits: int = 16):
        self._page_size = page_size
        self._num_pages = num_pages
        self._precision_bits = precision_bits

        # 页池: page_id -> CacheEntry (None 表示空闲)
        self._page_pool: Dict[str, Optional[CacheEntry]] = {}
        self._free_pages: List[str] = []
        self._used_pages: Dict[str, CacheEntry] = {}

        # 初始化页池
        for i in range(num_pages):
            pid = f"page_{i:04d}"
            self._page_pool[pid] = None
            self._free_pages.append(pid)

        logger.debug(
            "PagedAttention 管理器初始化: page_size=%d, num_pages=%d, precision=%dbit",
            page_size, num_pages, precision_bits,
        )

    @property
    def free_count(self) -> int:
        return len(self._free_pages)

    @property
    def used_count(self) -> int:
        return len(self._used_pages)

    @property
    def page_size(self) -> int:
        return self._page_size

    def allocate_page(self, seq_id: str, start_token: int, end_token: int) -> Optional[CacheEntry]:
        """分配一个空闲页

        :return: 新分配的 CacheEntry，无空闲页则返回 None
        """
        if not self._free_pages:
            return None

        pid = self._free_pages.pop(0)
        entry = CacheEntry(
            page_id=pid,
            seq_id=seq_id,
            start_token=start_token,
            end_token=end_token,
            precision_bits=self._precision_bits,
        )
        self._page_pool[pid] = entry
        self._used_pages[pid] = entry
        return entry

    def free_page(self, page_id: str) -> bool:
        """释放一个页"""
        entry = self._used_pages.pop(page_id, None)
        if entry is None:
            return False
        self._page_pool[page_id] = None
        self._free_pages.append(page_id)
        return True

    def get_page(self, page_id: str) -> Optional[CacheEntry]:
        """获取页内容"""
        return self._used_pages.get(page_id)

    def get_memory_per_page_mb(self) -> float:
        """计算每页内存占用 (MB)

        每页存储 key + value，shape: [num_heads * head_dim * page_size]
        此处只计算单个 head 的开销，外部需乘以实际层数和头数。
        """
        elements = self._page_size
        bytes_per_elem = _BYTES_PER_ELEMENT.get(self._precision_bits, 2)
        # key + value = 2x
        return (elements * bytes_per_elem * 2) / (1024 * 1024)

    def defragment(self) -> int:
        """碎片整理：合并空闲页

        简单实现：对空闲页 ID 排序以提高局部性。
        返回整理后的空闲页数。
        """
        self._free_pages.sort()
        return len(self._free_pages)


# ---------------------------------------------------------------------------
# KV Cache 量化器
# ---------------------------------------------------------------------------

class KVQuantizer:
    """KV Cache 量化工具

    支持 FP16 -> INT8 和 FP16 -> INT4 的 per-channel 对称量化。
    量化参数 (scale / zero_point) 随每页存储。
    """

    @staticmethod
    def quantize_int8(data: List[float]) -> Tuple[List[int], float, float]:
        """FP16 -> INT8 对称量化

        :return: (quantized_data, scale, zero_point)
        """
        if not data:
            return [], 1.0, 0.0

        abs_max = max(abs(v) for v in data)
        if abs_max == 0:
            return [0] * len(data), 1.0, 0.0

        scale = abs_max / 127.0
        quantized = [max(-128, min(127, round(v / scale))) for v in data]
        return quantized, scale, 0.0

    @staticmethod
    def dequantize_int8(quantized: List[int], scale: float, zero_point: float = 0.0) -> List[float]:
        """INT8 -> FP16 反量化"""
        return [q * scale + zero_point for q in quantized]

    @staticmethod
    def quantize_int4(data: List[float]) -> Tuple[List[int], float, float]:
        """FP16 -> INT4 非对称量化

        :return: (quantized_data, scale, zero_point)
        """
        if not data:
            return [], 1.0, 0.0

        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            return [8] * len(data), 1.0, min_val  # 映射到中间值

        scale = (max_val - min_val) / 15.0
        zero_point = min_val
        quantized = [max(0, min(15, round((v - zero_point) / scale))) for v in data]
        return quantized, scale, zero_point

    @staticmethod
    def dequantize_int4(quantized: List[int], scale: float, zero_point: float = 0.0) -> List[float]:
        """INT4 -> FP16 反量化"""
        return [q * scale + zero_point for q in quantized]


# ---------------------------------------------------------------------------
# 前缀共享管理器
# ---------------------------------------------------------------------------

class PrefixShareManager:
    """前缀共享管理器

    检测多个请求中相同的 prompt 前缀，共享对应的 KV Cache，
    减少重复计算和内存占用。
    """

    def __init__(self):
        # prefix_hash -> (prefix_tokens, page_ids, ref_count)
        self._shared_prefixes: Dict[str, Tuple[int, List[str], int]] = {}

    def compute_prefix_hash(self, tokens: List[int], max_prefix_len: int = 512) -> str:
        """计算 token 序列前缀的哈希值

        :param tokens: token ID 序列
        :param max_prefix_len: 最大前缀长度 (token 数)
        :return: 前缀哈希字符串
        """
        prefix = tokens[:max_prefix_len]
        hasher = hashlib.sha256()
        for t in prefix:
            hasher.update(t.to_bytes(4, "little", signed=True))
        return hasher.hexdigest()[:16]

    def find_shared_prefix(self, tokens: List[int], max_prefix_len: int = 512) -> Optional[Tuple[str, List[str]]]:
        """查找已缓存的共享前缀

        :return: (prefix_hash, shared_page_ids) 或 None
        """
        prefix_hash = self.compute_prefix_hash(tokens, max_prefix_len)
        if prefix_hash in self._shared_prefixes:
            _, page_ids, ref_count = self._shared_prefixes[prefix_hash]
            self._shared_prefixes[prefix_hash] = (len(tokens[:max_prefix_len]), page_ids, ref_count + 1)
            logger.debug("前缀命中: hash=%s  ref_count=%d", prefix_hash, ref_count + 1)
            return prefix_hash, page_ids
        return None

    def register_shared_prefix(
        self, prefix_hash: str, prefix_length: int, page_ids: List[str]
    ) -> None:
        """注册共享前缀"""
        if prefix_hash in self._shared_prefixes:
            _, old_pages, ref_count = self._shared_prefixes[prefix_hash]
            self._shared_prefixes[prefix_hash] = (prefix_length, old_pages, ref_count + 1)
        else:
            self._shared_prefixes[prefix_hash] = (prefix_length, page_ids, 1)
        logger.debug("前缀注册: hash=%s  pages=%d", prefix_hash, len(page_ids))

    def release_shared_prefix(self, prefix_hash: str) -> bool:
        """释放共享前缀引用

        :return: True 表示引用归零，可以安全释放页
        """
        if prefix_hash not in self._shared_prefixes:
            return False

        prefix_length, page_ids, ref_count = self._shared_prefixes[prefix_hash]
        if ref_count <= 1:
            del self._shared_prefixes[prefix_hash]
            logger.debug("前缀释放: hash=%s  (ref_count -> 0, 释放)", prefix_hash)
            return True
        else:
            self._shared_prefixes[prefix_hash] = (prefix_length, page_ids, ref_count - 1)
            logger.debug("前缀释放: hash=%s  ref_count=%d", prefix_hash, ref_count - 1)
            return False

    @property
    def shared_count(self) -> int:
        return len(self._shared_prefixes)

    def get_total_shared_pages(self) -> int:
        return sum(len(pages) for _, pages, _ in self._shared_prefixes.values())


# ---------------------------------------------------------------------------
# KV Cache 优化器 (主类)
# ---------------------------------------------------------------------------

class KVCacheOptimizer:
    """KV Cache 优化器

    核心能力:
    1. PagedAttention 分页管理，消除内存碎片
    2. KV Cache 量化 (FP16 -> INT8 / INT4)，降低内存占用
    3. 基于注意力分数的智能压缩
    4. 多请求前缀共享
    5. 动态 Cache 管理：接近上限时自动压缩或淘汰
    6. 内存预估与配置推荐
    """

    def __init__(self, config: Optional[KVCacheConfig] = None):
        self._config = config or KVCacheConfig()

        # 核心组件
        precision_bits = self._config.cache_bits
        num_pages = self._config.num_pages or self._auto_calc_pages()

        self._paged_manager = PagedAttentionManager(
            page_size=self._config.page_size,
            num_pages=num_pages,
            precision_bits=precision_bits,
        )
        self._quantizer = KVQuantizer()
        self._prefix_manager = PrefixShareManager()

        # 序列 Cache 注册表
        self._sequences: Dict[str, SequenceCache] = {}

        # 淘汰策略相关
        self._eviction_policy = EvictionPolicy(self._config.eviction_policy)
        self._lru_order: OrderedDict = OrderedDict()   # page_id -> access_time
        self._lfu_counter: Dict[str, int] = {}          # page_id -> access_count

        # 统计
        self._stats = CacheStats(
            total_pages=num_pages,
            free_pages=num_pages,
        )

        logger.info(
            "KVCacheOptimizer 初始化: bits=%d, max=%.1fGB, policy=%s, prefix_sharing=%s, page_size=%d, num_pages=%d",
            self._config.cache_bits,
            self._config.max_cache_size_gb,
            self._config.eviction_policy,
            self._config.prefix_sharing,
            self._config.page_size,
            num_pages,
        )

    def _auto_calc_pages(self) -> int:
        """根据 max_cache_size_gb 自动计算页数

        假设每页大致开销 = page_size * 2 (K+V) * bytes_per_elem，
        再除以 num_layers * num_heads 的典型值。
        此处给出保守估算，实际使用中由外部传入 num_pages。
        """
        page_bytes = self._config.page_size * 2 * _BYTES_PER_ELEMENT.get(self._config.cache_bits, 2)
        # 以 32 layers * 32 heads 为默认
        estimated_per_page = page_bytes * 32 * 32
        if estimated_per_page == 0:
            return 256
        max_bytes = self._config.max_cache_size_gb * 1024 * 1024 * 1024
        return max(64, int(max_bytes / estimated_per_page))

    # ------------------------------------------------------------------
    # 内存预估
    # ------------------------------------------------------------------

    def estimate_cache_size(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        batch_size: int = 1,
    ) -> float:
        """预估 KV Cache 内存占用 (GB)

        公式: batch_size * seq_length * num_layers * 2 (K+V) * num_heads * head_dim * bytes_per_element

        :param seq_length: 序列长度 (tokens)
        :param num_layers: Transformer 层数
        :param num_heads: 注意力头数
        :param head_dim: 每头维度
        :param batch_size: 批大小
        :return: 预估内存 (GB)
        """
        bytes_per_elem = _BYTES_PER_ELEMENT.get(self._config.cache_bits, 2)
        total_elements = batch_size * seq_length * num_layers * 2 * num_heads * head_dim
        total_bytes = total_elements * bytes_per_elem
        total_gb = total_bytes / (1024 ** 3)

        logger.debug(
            "Cache 预估: seq=%d, layers=%d, heads=%d, dim=%d, batch=%d -> %.3f GB (bits=%d)",
            seq_length, num_layers, num_heads, head_dim, batch_size, total_gb, self._config.cache_bits,
        )
        return total_gb

    def estimate_page_count(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        batch_size: int = 1,
    ) -> int:
        """预估所需页数"""
        tokens_per_page = self._config.page_size
        total_tokens = seq_length * batch_size
        return math.ceil(total_tokens / tokens_per_page)

    # ------------------------------------------------------------------
    # 配置推荐
    # ------------------------------------------------------------------

    def recommend_config(
        self,
        available_memory_gb: float,
        typical_seq_length: int,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        batch_size: int = 1,
    ) -> KVCacheConfig:
        """基于可用内存推荐最优 Cache 配置

        策略:
        1. 先尝试 FP16，若内存不够则降级 INT8 -> INT4
        2. 启用前缀共享 (除非序列很短)
        3. 淘汰策略默认 attention_score

        :param available_memory_gb: 可用内存 (GB)
        :param typical_seq_length: 典型序列长度
        :param num_layers: 层数
        :param num_heads: 头数
        :param head_dim: 每头维度
        :param batch_size: 批大小
        :return: 推荐的 KVCacheConfig
        """
        # 为模型权重预留 50% 内存，KV Cache 最多使用剩余部分
        cache_budget_gb = available_memory_gb * 0.5

        # 尝试不同精度
        for bits in [16, 8, 4]:
            test_config = KVCacheConfig(cache_bits=bits)
            test_optimizer = KVCacheOptimizer.__new__(KVCacheOptimizer)
            test_optimizer._config = test_config
            est = test_optimizer.estimate_cache_size(
                typical_seq_length, num_layers, num_heads, head_dim, batch_size,
            )
            if est <= cache_budget_gb:
                prefix_sharing = typical_seq_length > 128
                compression_ratio = 0.5 if bits == 16 else 0.7

                logger.info(
                    "推荐配置: bits=%d, 预估=%.2fGB, 预算=%.2fGB, prefix=%s",
                    bits, est, cache_budget_gb, prefix_sharing,
                )
                return KVCacheConfig(
                    cache_bits=bits,
                    max_cache_size_gb=cache_budget_gb,
                    eviction_policy="attention_score",
                    prefix_sharing=prefix_sharing,
                    compression_ratio=compression_ratio,
                    page_size=64,
                )

        # 所有精度都不够，返回最激进配置
        logger.warning("内存极度紧张，使用 INT4 + 高压缩配置")
        return KVCacheConfig(
            cache_bits=4,
            max_cache_size_gb=cache_budget_gb,
            eviction_policy="attention_score",
            prefix_sharing=True,
            compression_ratio=0.3,
            page_size=32,
        )

    # ------------------------------------------------------------------
    # Cache 注册与查询
    # ------------------------------------------------------------------

    def register_sequence(
        self,
        seq_id: Optional[str] = None,
        tokens: Optional[List[int]] = None,
    ) -> str:
        """注册一个新序列的 Cache

        :param seq_id: 序列 ID，None 则自动生成
        :param tokens: 该序列的 token 列表 (用于前缀共享检测)
        :return: seq_id
        """
        seq_id = seq_id or f"seq_{uuid.uuid4().hex[:10]}"
        prefix_hash = None

        # 前缀共享检测
        if self._config.prefix_sharing and tokens:
            prefix_hash = self._prefix_manager.compute_prefix_hash(tokens)
            shared = self._prefix_manager.find_shared_prefix(tokens)
            if shared is not None:
                self._stats.prefix_share_count += 1
                logger.debug("序列 %s 命中共享前缀 %s", seq_id, shared[0])
            else:
                # 首次出现此前缀，注册以供后续序列共享
                self._prefix_manager.register_shared_prefix(
                    prefix_hash, len(tokens), []
                )

        seq_cache = SequenceCache(
            seq_id=seq_id,
            prefix_hash=prefix_hash,
        )
        self._sequences[seq_id] = seq_cache
        return seq_id

    def unregister_sequence(self, seq_id: str) -> None:
        """注销序列并释放其所有 Cache 页"""
        seq = self._sequences.pop(seq_id, None)
        if seq is None:
            return

        for page in seq.pages:
            self._release_page(page)

        # 释放前缀引用
        if seq.prefix_hash:
            release_all = self._prefix_manager.release_shared_prefix(seq.prefix_hash)
            if release_all:
                # 前缀引用归零，相关页也释放
                pass  # 页已在上面释放

        logger.debug("序列 %s 已注销，释放 %d 页", seq_id, len(seq.pages))

    def get_sequence_cache(self, seq_id: str) -> Optional[SequenceCache]:
        return self._sequences.get(seq_id)

    # ------------------------------------------------------------------
    # 分配与写入
    # ------------------------------------------------------------------

    def allocate_for_sequence(
        self,
        seq_id: str,
        num_tokens: int,
        key_data: Optional[Any] = None,
        value_data: Optional[Any] = None,
    ) -> List[CacheEntry]:
        """为序列分配 Cache 页并写入 KV 数据

        :param seq_id: 序列 ID
        :param num_tokens: 需要缓存的 token 数
        :param key_data: Key 张量数据 (可选)
        :param value_data: Value 张量数据 (可选)
        :return: 分配成功的页列表
        """
        seq = self._sequences.get(seq_id)
        if seq is None:
            raise ValueError(f"序列 {seq_id} 未注册")

        allocated_pages: List[CacheEntry] = []
        page_size = self._config.page_size
        current_token = seq.total_tokens

        while current_token < seq.total_tokens + num_tokens:
            end_token = min(current_token + page_size, seq.total_tokens + num_tokens)

            # 尝试分配页
            page = self._paged_manager.allocate_page(seq_id, current_token, end_token)
            if page is None:
                # 无空闲页，触发淘汰
                evicted = self._evict_pages(1)
                if evicted == 0:
                    logger.warning("无法分配更多页 (所有页已钉住或已空)")
                    break
                page = self._paged_manager.allocate_page(seq_id, current_token, end_token)
                if page is None:
                    break

            # 量化写入
            if self._config.cache_bits < 16:
                page = self._quantize_page(page, key_data, value_data)

            # 更新淘汰策略数据结构
            self._track_page_access(page)

            seq.pages.append(page)
            allocated_pages.append(page)
            current_token = end_token

        seq.total_tokens += num_tokens
        self._update_stats_after_alloc(len(allocated_pages))

        return allocated_pages

    def _quantize_page(
        self, page: CacheEntry, key_data: Any, value_data: Any
    ) -> CacheEntry:
        """对页内 KV 数据进行量化"""
        if self._config.cache_bits == 8:
            if key_data is not None:
                page.key_data = self._quantizer.quantize_int8(key_data)
            if value_data is not None:
                page.value_data = self._quantizer.quantize_int8(value_data)
            page.precision_bits = 8
        elif self._config.cache_bits == 4:
            if key_data is not None:
                page.key_data = self._quantizer.quantize_int4(key_data)
            if value_data is not None:
                page.value_data = self._quantizer.quantize_int4(value_data)
            page.precision_bits = 4
        self._stats.quantization_savings_mb += self._paged_manager.get_memory_per_page_mb() * (
            1 - _BYTES_PER_ELEMENT.get(self._config.cache_bits, 2) / 2
        )
        return page

    def _track_page_access(self, page: CacheEntry) -> None:
        """更新淘汰策略所需的数据结构"""
        page.access_count += 1
        page.last_access_time = time.time()

        pid = page.page_id
        if self._eviction_policy == EvictionPolicy.LRU:
            self._lru_order[pid] = page.last_access_time
        elif self._eviction_policy == EvictionPolicy.LFU:
            self._lfu_counter[pid] = page.access_count
        # attention_score 策略通过 page.attention_score 字段直接管理

    # ------------------------------------------------------------------
    # 读取与反量化
    # ------------------------------------------------------------------

    def read_page(self, page_id: str) -> Optional[Tuple[Any, Any]]:
        """读取页的 KV 数据 (自动反量化)

        :return: (key_data, value_data) 或 None
        """
        page = self._paged_manager.get_page(page_id)
        if page is None:
            self._stats.cache_miss_count += 1
            return None

        self._stats.cache_hit_count += 1
        self._track_page_access(page)

        key_data = page.key_data
        value_data = page.value_data

        # 反量化
        if page.precision_bits == 8 and key_data is not None:
            if isinstance(key_data, tuple) and len(key_data) == 3:
                key_data = self._quantizer.dequantize_int8(*key_data)
            if isinstance(value_data, tuple) and len(value_data) == 3:
                value_data = self._quantizer.dequantize_int8(*value_data)
        elif page.precision_bits == 4 and key_data is not None:
            if isinstance(key_data, tuple) and len(key_data) == 3:
                key_data = self._quantizer.dequantize_int4(*key_data)
            if isinstance(value_data, tuple) and len(value_data) == 3:
                value_data = self._quantizer.dequantize_int4(*value_data)

        return key_data, value_data

    # ------------------------------------------------------------------
    # 淘汰策略
    # ------------------------------------------------------------------

    def _evict_pages(self, count: int) -> int:
        """淘汰指定数量的页

        :return: 实际淘汰的页数
        """
        evicted = 0
        candidates = self._get_eviction_candidates()

        for page_id in candidates:
            if evicted >= count:
                break
            page = self._paged_manager.get_page(page_id)
            if page is None or page.is_pinned:
                continue

            self._paged_manager.free_page(page_id)
            self._remove_from_tracking(page_id)
            evicted += 1
            self._stats.eviction_count += 1

            logger.debug("淘汰页: %s (policy=%s)", page_id, self._eviction_policy.value)

        return evicted

    def _get_eviction_candidates(self) -> List[str]:
        """根据淘汰策略获取候选页列表 (按淘汰优先级排序)"""
        if self._eviction_policy == EvictionPolicy.LRU:
            # 最久未访问的优先淘汰
            sorted_items = sorted(self._lru_order.items(), key=lambda x: x[1])
            return [pid for pid, _ in sorted_items]

        elif self._eviction_policy == EvictionPolicy.LFU:
            # 访问次数最少的优先淘汰
            sorted_items = sorted(self._lfu_counter.items(), key=lambda x: x[1])
            return [pid for pid, _ in sorted_items]

        elif self._eviction_policy == EvictionPolicy.ATTENTION_SCORE:
            # 注意力分数最低的优先淘汰
            candidates: List[Tuple[str, float]] = []
            for pid, page in self._paged_manager._used_pages.items():
                if not page.is_pinned:
                    candidates.append((pid, page.attention_score))
            candidates.sort(key=lambda x: x[1])
            return [pid for pid, _ in candidates]

        return []

    def _remove_from_tracking(self, page_id: str) -> None:
        """从淘汰策略数据结构中移除"""
        self._lru_order.pop(page_id, None)
        self._lfu_counter.pop(page_id, None)

    # ------------------------------------------------------------------
    # 压缩
    # ------------------------------------------------------------------

    def compress_cache(
        self,
        cache_data: Dict[str, Any],
        target_size: float,
        attention_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """压缩 Cache 到目标大小

        策略:
        1. 若有注意力分数，优先保留高分页
        2. 对保留的页进行更激进的量化 (FP16->INT8->INT4)
        3. 移除低分页

        :param cache_data: 待压缩的 Cache 数据 {page_id: (key, value, score)}
        :param target_size: 目标大小 (GB)
        :param attention_scores: 每页的注意力分数 {page_id: score}
        :return: 压缩后的 Cache 数据
        """
        if not cache_data:
            return {}

        current_size = self._estimate_data_size(cache_data)
        if current_size <= target_size:
            return cache_data

        scores = attention_scores or {}
        # 按注意力分数排序，高分在前
        sorted_pages = sorted(
            cache_data.items(),
            key=lambda x: scores.get(x[0], 0.0),
            reverse=True,
        )

        compressed: Dict[str, Any] = {}
        running_size = 0.0
        bytes_per_elem = _BYTES_PER_ELEMENT.get(self._config.cache_bits, 2)

        for page_id, data in sorted_pages:
            # 估算该页大小
            if isinstance(data, (tuple, list)) and len(data) >= 2:
                page_elements = len(data[0]) + len(data[1]) if isinstance(data[0], list) else 1024
            else:
                page_elements = 1024  # 默认估计
            page_size_gb = (page_elements * bytes_per_elem) / (1024 ** 3)

            if running_size + page_size_gb <= target_size:
                compressed[page_id] = data
                running_size += page_size_gb
            else:
                # 尝试量化后放入
                if self._config.cache_bits == 16:
                    # 降级到 INT8
                    quantized_data = self._quantize_data_int8(data)
                    quant_size_gb = (page_elements * 1) / (1024 ** 3)
                    if running_size + quant_size_gb <= target_size:
                        compressed[page_id] = quantized_data
                        running_size += quant_size_gb
                        self._stats.compression_count += 1
                        continue
                elif self._config.cache_bits == 8:
                    # 降级到 INT4
                    quantized_data = self._quantize_data_int4(data)
                    quant_size_gb = (page_elements * 0.5) / (1024 ** 3)
                    if running_size + quant_size_gb <= target_size:
                        compressed[page_id] = quantized_data
                        running_size += quant_size_gb
                        self._stats.compression_count += 1
                        continue

                # 无法放入，丢弃
                logger.debug("压缩丢弃页 %s (注意力分数=%.4f)", page_id, scores.get(page_id, 0.0))

        self._stats.compression_count += 1
        logger.info(
            "Cache 压缩完成: %d -> %d 页 (%.2fGB -> %.2fGB)",
            len(cache_data), len(compressed), current_size, running_size,
        )
        return compressed

    def _quantize_data_int8(self, data: Any) -> Any:
        """将数据量化到 INT8"""
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            k, v = data[0], data[1]
            if isinstance(k, list):
                k_q = self._quantizer.quantize_int8(k)
                v_q = self._quantizer.quantize_int8(v) if isinstance(v, list) else v
                return (k_q, v_q) + tuple(data[2:])
        return data

    def _quantize_data_int4(self, data: Any) -> Any:
        """将数据量化到 INT4"""
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            k, v = data[0], data[1]
            if isinstance(k, list):
                k_q = self._quantizer.quantize_int4(k)
                v_q = self._quantizer.quantize_int4(v) if isinstance(v, list) else v
                return (k_q, v_q) + tuple(data[2:])
        return data

    def _estimate_data_size(self, cache_data: Dict[str, Any]) -> float:
        """估算 Cache 数据大小 (GB)"""
        total_elements = 0
        for data in cache_data.values():
            if isinstance(data, (tuple, list)):
                for item in data[:2]:
                    if isinstance(item, list):
                        total_elements += len(item)
                    elif isinstance(item, tuple) and len(item) >= 1 and isinstance(item[0], list):
                        total_elements += len(item[0])
            else:
                total_elements += 1024  # 默认估计
        bytes_per_elem = _BYTES_PER_ELEMENT.get(self._config.cache_bits, 2)
        return (total_elements * bytes_per_elem) / (1024 ** 3)

    # ------------------------------------------------------------------
    # 注意力分数驱动的动态管理
    # ------------------------------------------------------------------

    def update_attention_scores(self, scores: Dict[str, float]) -> None:
        """更新页的注意力分数

        :param scores: {page_id: attention_score}
        """
        for pid, score in scores.items():
            page = self._paged_manager.get_page(pid)
            if page is not None:
                page.attention_score = score

    def auto_manage(self) -> Dict[str, Any]:
        """自动管理 Cache: 检测内存压力并执行压缩或淘汰

        :return: 管理操作摘要
        """
        actions: List[str] = []
        memory_util = self._stats.memory_utilization

        if memory_util > 0.95:
            # 极度紧张: 淘汰低分页
            evict_count = max(1, int(self._paged_manager.used_count * 0.1))
            evicted = self._evict_pages(evict_count)
            actions.append(f"evicted_{evicted}_pages")
            logger.warning("内存压力 %.1f%%，淘汰 %d 页", memory_util * 100, evicted)

        elif memory_util > 0.85:
            # 中度紧张: 量化降级 + 轻度淘汰
            evict_count = max(1, int(self._paged_manager.used_count * 0.05))
            evicted = self._evict_pages(evict_count)
            actions.append(f"evicted_{evicted}_pages")
            # 对剩余页尝试更激进量化
            self._try_aggressive_quantization()
            actions.append("quantization_downgrade")
            logger.info("内存压力 %.1f%%，淘汰 %d 页 + 量化降级", memory_util * 100, evicted)

        elif memory_util > 0.70:
            # 轻度压力: 碎片整理
            free_count = self._paged_manager.defragment()
            actions.append(f"defragmented (free={free_count})")

        return {
            "memory_utilization": round(memory_util, 4),
            "actions": actions,
            "stats": self._stats,
        }

    def _try_aggressive_quantization(self) -> int:
        """尝试对当前 FP16 的页降级到更低位宽"""
        downgraded = 0
        if self._config.cache_bits >= 16:
            # FP16 -> INT8
            for pid, page in self._paged_manager._used_pages.items():
                if page.precision_bits == 16 and not page.is_pinned:
                    page.precision_bits = 8
                    downgraded += 1
        elif self._config.cache_bits >= 8:
            # INT8 -> INT4
            for pid, page in self._paged_manager._used_pages.items():
                if page.precision_bits == 8 and not page.is_pinned:
                    page.precision_bits = 4
                    downgraded += 1
        return downgraded

    def _release_page(self, page: CacheEntry) -> None:
        """释放单个页"""
        self._paged_manager.free_page(page.page_id)
        self._remove_from_tracking(page.page_id)

    def _update_stats_after_alloc(self, allocated: int) -> None:
        """分配后更新统计"""
        self._stats.used_pages = self._paged_manager.used_count
        self._stats.free_pages = self._paged_manager.free_count
        mem_per_page = self._paged_manager.get_memory_per_page_mb()
        self._stats.used_memory_mb = self._stats.used_pages * mem_per_page
        self._stats.total_memory_mb = self._stats.total_pages * mem_per_page

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_stats(self) -> CacheStats:
        """获取 Cache 统计信息"""
        self._stats.used_pages = self._paged_manager.used_count
        self._stats.free_pages = self._paged_manager.free_count
        return self._stats

    def get_config(self) -> KVCacheConfig:
        """获取当前配置"""
        return self._config

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存摘要"""
        stats = self.get_stats()
        return {
            "config_bits": self._config.cache_bits,
            "max_cache_size_gb": self._config.max_cache_size_gb,
            "total_pages": stats.total_pages,
            "used_pages": stats.used_pages,
            "free_pages": stats.free_pages,
            "used_memory_mb": round(stats.used_memory_mb, 2),
            "total_memory_mb": round(stats.total_memory_mb, 2),
            "memory_utilization": round(stats.memory_utilization, 4),
            "hit_rate": round(stats.hit_rate, 4),
            "eviction_count": stats.eviction_count,
            "compression_count": stats.compression_count,
            "prefix_shares": stats.prefix_share_count,
            "quantization_savings_mb": round(stats.quantization_savings_mb, 2),
        }

    def get_sequence_list(self) -> List[Dict[str, Any]]:
        """获取所有序列摘要"""
        result = []
        for seq_id, seq in self._sequences.items():
            result.append({
                "seq_id": seq_id,
                "total_tokens": seq.total_tokens,
                "page_count": len(seq.pages),
                "is_shared": seq.is_shared,
                "prefix_hash": seq.prefix_hash,
            })
        return result
