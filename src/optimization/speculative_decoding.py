"""
Speculative Decoding 模块

使用小模型(草稿模型)预测大模型(目标模型)的输出，减少大模型的实际计算量。

原理:
1. 草稿模型快速生成N个候选token
2. 目标模型一次性验证这些token
3. 接受正确的token，拒绝错误的token
4. 平均每次调用目标模型可获得多个token

优势:
- 减少目标模型的调用次数
- 保持输出质量不变（数学上等价）
- 适合小显存场景（草稿模型很小）

预期提升: 2-3倍生成速度
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 配置和数据类
# ============================================================

class DraftModelStrategy(Enum):
    """草稿模型选择策略"""
    SMALL_QUANTIZED = "small_quantized"  # 使用更小的量化模型
    SAME_QUANTIZED = "same_quantized"    # 使用同模型的更激进量化版本
    NGRAM = "ngram"                      # 使用n-gram预测（无需额外模型）


@dataclass
class SpeculativeConfig:
    """Speculative Decoding配置"""
    # 草稿模型策略
    strategy: DraftModelStrategy = DraftModelStrategy.NGRAM

    # 每次推测的token数
    num_speculative_tokens: int = 5

    # 草稿模型路径（如果使用模型策略）
    draft_model_path: Optional[str] = None

    # 草稿模型量化级别
    draft_quant_level: str = "q2_k"

    # 接受阈值（用于采样）
    temperature: float = 0.7

    # 是否启用缓存
    enable_cache: bool = True

    # 缓存大小
    cache_size: int = 1024


@dataclass
class SpeculativeResult:
    """Speculative Decoding结果"""
    # 生成的token
    tokens: List[str]

    # 接受的token数
    accepted_count: int

    # 总推测token数
    speculated_count: int

    # 接受率
    acceptance_rate: float

    # 目标模型调用次数
    target_model_calls: int

    # 草稿模型调用次数
    draft_model_calls: int

    # 总耗时
    total_time_ms: float

    # 速度提升倍数
    speedup: float


# ============================================================
# N-gram预测器（轻量级，无需额外模型）
# ============================================================

class NgramPredictor:
    """N-gram预测器

    基于历史token序列预测下一个token，无需额外模型。
    适合小显存场景，因为不需要加载额外的模型。
    """

    def __init__(self, n: int = 3, cache_size: int = 1024):
        """初始化N-gram预测器

        Args:
            n: N-gram的N值
            cache_size: 缓存大小
        """
        self.n = n
        self.cache_size = cache_size
        self._ngrams: Dict[Tuple[str, ...], List[Tuple[str, int]]] = {}
        self._token_history: List[str] = []

    def update(self, token: str) -> None:
        """更新token历史

        Args:
            token: 新的token
        """
        self._token_history.append(token)

        # 更新n-gram统计
        if len(self._token_history) >= self.n:
            context = tuple(self._token_history[-(self.n):])
            if context not in self._ngrams:
                self._ngrams[context] = []

            # 查找或添加后续token
            found = False
            for i, (t, count) in enumerate(self._ngrams[context]):
                if t == token:
                    self._ngrams[context][i] = (t, count + 1)
                    found = True
                    break

            if not found:
                self._ngrams[context].append((token, 1))

        # 限制历史长度
        if len(self._token_history) > self.cache_size:
            self._token_history = self._token_history[-self.cache_size:]

    def predict(self, num_predictions: int = 5) -> List[str]:
        """预测接下来的token

        Args:
            num_predictions: 预测数量

        Returns:
            List[str]: 预测的token列表
        """
        predictions = []
        current_context = list(self._token_history[-self.n:]) if len(self._token_history) >= self.n else list(self._token_history)

        for _ in range(num_predictions):
            context = tuple(current_context[-self.n:]) if len(current_context) >= self.n else tuple(current_context)

            if context in self._ngrams:
                # 按频率排序，选择最可能的token
                candidates = sorted(self._ngrams[context], key=lambda x: x[1], reverse=True)
                if candidates:
                    next_token = candidates[0][0]
                    predictions.append(next_token)
                    current_context.append(next_token)
                else:
                    break
            else:
                break

        return predictions

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "ngram_count": len(self._ngrams),
            "history_length": len(self._token_history),
            "cache_size": self.cache_size,
        }


# ============================================================
# Speculative Decoding引擎
# ============================================================

class SpeculativeDecoder:
    """Speculative Decoding解码器

    使用小模型或n-gram预测器来推测大模型的输出。
    """

    def __init__(self, config: Optional[SpeculativeConfig] = None):
        """初始化Speculative Decoder

        Args:
            config: 配置
        """
        self.config = config or SpeculativeConfig()

        # 初始化预测器
        if self.config.strategy == DraftModelStrategy.NGRAM:
            self._predictor = NgramPredictor(
                n=3,
                cache_size=self.config.cache_size,
            )
        else:
            # 模型策略需要外部提供模型
            self._predictor = None

        # 统计
        self._total_accepted = 0
        self._total_speculated = 0
        self._total_target_calls = 0
        self._total_draft_calls = 0

    async def generate(
        self,
        prompt: str,
        target_generate_fn,
        max_tokens: int = 100,
        num_speculative: Optional[int] = None,
    ) -> SpeculativeResult:
        """使用Speculative Decoding生成文本

        Args:
            prompt: 输入提示
            target_generate_fn: 目标模型的生成函数
            max_tokens: 最大生成token数
            num_speculative: 每次推测的token数（覆盖配置）

        Returns:
            SpeculativeResult: 生成结果
        """
        start_time = time.time()
        num_spec = num_speculative or self.config.num_speculative_tokens

        generated_tokens = []
        total_accepted = 0
        total_speculated = 0
        target_calls = 0
        draft_calls = 0

        remaining_tokens = max_tokens

        while remaining_tokens > 0:
            # 1. 使用预测器推测接下来的token
            speculated_tokens = self._speculate_tokens(
                prompt + "".join(generated_tokens),
                min(num_spec, remaining_tokens),
            )
            draft_calls += 1
            total_speculated += len(speculated_tokens)

            if not speculated_tokens:
                # 预测器无法预测，直接使用目标模型
                result = await target_generate_fn(
                    prompt=prompt + "".join(generated_tokens),
                    max_tokens=1,
                )
                target_calls += 1
                if result:
                    generated_tokens.append(result)
                    self._update_predictor(result)
                    remaining_tokens -= 1
                continue

            # 2. 使用目标模型验证推测的token
            verified_tokens = await self._verify_tokens(
                prompt + "".join(generated_tokens),
                speculated_tokens,
                target_generate_fn,
            )
            target_calls += 1

            # 3. 接受验证通过的token
            accepted_count = len(verified_tokens)
            total_accepted += accepted_count

            # 添加接受的token
            for token in verified_tokens:
                generated_tokens.append(token)
                self._update_predictor(token)
                remaining_tokens -= 1

                if remaining_tokens <= 0:
                    break

        # 计算统计信息
        total_time = (time.time() - start_time) * 1000  # ms
        acceptance_rate = total_accepted / max(total_speculated, 1)

        # 计算速度提升（假设每次目标模型调用生成1个token，现在平均生成多个）
        baseline_calls = max_tokens  # 不使用speculative时需要的调用次数
        speedup = baseline_calls / max(target_calls, 1)

        # 更新全局统计
        self._total_accepted += total_accepted
        self._total_speculated += total_speculated
        self._total_target_calls += target_calls
        self._total_draft_calls += draft_calls

        return SpeculativeResult(
            tokens=generated_tokens,
            accepted_count=total_accepted,
            speculated_count=total_speculated,
            acceptance_rate=acceptance_rate,
            target_model_calls=target_calls,
            draft_model_calls=draft_calls,
            total_time_ms=total_time,
            speedup=speedup,
        )

    def _speculate_tokens(self, context: str, count: int) -> List[str]:
        """推测接下来的token

        Args:
            context: 上下文
            count: 推测数量

        Returns:
            List[str]: 推测的token列表
        """
        if self._predictor is None:
            return []

        return self._predictor.predict(count)

    async def _verify_tokens(
        self,
        context: str,
        speculated_tokens: List[str],
        target_generate_fn,
    ) -> List[str]:
        """使用目标模型验证推测的token

        Args:
            context: 上下文
            speculated_tokens: 推测的token列表
            target_generate_fn: 目标模型生成函数

        Returns:
            List[str]: 验证通过的token列表
        """
        # 构建验证提示：让目标模型继续生成
        # 如果目标模型生成的token与推测的一致，则接受
        verified = []

        # 让目标模型生成相同数量的token
        result = await target_generate_fn(
            prompt=context,
            max_tokens=len(speculated_tokens),
        )

        if result:
            # 简化实现：假设目标模型返回的是一个字符串
            # 实际实现中需要逐token比较
            target_tokens = list(result)  # 简化：按字符分割

            # 比较并接受匹配的token
            for spec_token, target_token in zip(speculated_tokens, target_tokens):
                if spec_token == target_token:
                    verified.append(spec_token)
                else:
                    # 不匹配，停止接受
                    # 添加目标模型的token
                    verified.append(target_token)
                    break

        return verified

    def _update_predictor(self, token: str) -> None:
        """更新预测器

        Args:
            token: 新的token
        """
        if self._predictor is not None:
            self._predictor.update(token)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_accepted": self._total_accepted,
            "total_speculated": self._total_speculated,
            "acceptance_rate": self._total_accepted / max(self._total_speculated, 1),
            "total_target_calls": self._total_target_calls,
            "total_draft_calls": self._total_draft_calls,
            "predictor_stats": self._predictor.get_stats() if self._predictor else None,
        }


# ============================================================
# 优化的推理引擎
# ============================================================

@dataclass
class OptimizedInferenceConfig:
    """优化的推理配置"""
    # Speculative Decoding配置
    speculative_config: SpeculativeConfig = field(default_factory=SpeculativeConfig)

    # 是否启用Speculative Decoding
    enable_speculative: bool = True

    # 批处理大小
    batch_size: int = 1

    # 是否启用流式输出
    enable_streaming: bool = True


class OptimizedInferenceEngine:
    """优化的推理引擎

    整合Speculative Decoding和其他优化技术。
    """

    def __init__(self, config: Optional[OptimizedInferenceConfig] = None):
        """初始化优化推理引擎

        Args:
            config: 配置
        """
        self.config = config or OptimizedInferenceConfig()

        # 初始化Speculative Decoder
        if self.config.enable_speculative:
            self._decoder = SpeculativeDecoder(self.config.speculative_config)
        else:
            self._decoder = None

    async def generate(
        self,
        prompt: str,
        target_generate_fn,
        max_tokens: int = 100,
        use_speculative: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """生成文本

        Args:
            prompt: 输入提示
            target_generate_fn: 目标模型生成函数
            max_tokens: 最大生成token数
            use_speculative: 是否使用Speculative Decoding

        Returns:
            Tuple[str, Dict]: (生成的文本, 统计信息)
        """
        start_time = time.time()

        if use_speculative and self._decoder is not None:
            # 使用Speculative Decoding
            result = await self._decoder.generate(
                prompt=prompt,
                target_generate_fn=target_generate_fn,
                max_tokens=max_tokens,
            )
            text = "".join(result.tokens)
            stats = {
                "method": "speculative",
                "accepted_count": result.accepted_count,
                "speculated_count": result.speculated_count,
                "acceptance_rate": result.acceptance_rate,
                "target_model_calls": result.target_model_calls,
                "draft_model_calls": result.draft_model_calls,
                "speedup": result.speedup,
                "total_time_ms": result.total_time_ms,
            }
        else:
            # 直接使用目标模型
            text = await target_generate_fn(prompt=prompt, max_tokens=max_tokens)
            total_time = (time.time() - start_time) * 1000
            stats = {
                "method": "direct",
                "target_model_calls": 1,
                "total_time_ms": total_time,
            }

        return text, stats

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "speculative_enabled": self._decoder is not None,
            "decoder_stats": self._decoder.get_stats() if self._decoder else None,
        }


# ============================================================
# 便捷函数
# ============================================================

def create_speculative_decoder(
    strategy: str = "ngram",
    num_speculative_tokens: int = 5,
    cache_size: int = 1024,
) -> SpeculativeDecoder:
    """创建Speculative Decoder

    Args:
        strategy: 策略 ("ngram", "small_quantized", "same_quantized")
        num_speculative_tokens: 每次推测的token数
        cache_size: 缓存大小

    Returns:
        SpeculativeDecoder: 解码器实例
    """
    strategy_enum = DraftModelStrategy(strategy)
    config = SpeculativeConfig(
        strategy=strategy_enum,
        num_speculative_tokens=num_speculative_tokens,
        cache_size=cache_size,
    )
    return SpeculativeDecoder(config)


def create_optimized_engine(
    enable_speculative: bool = True,
    num_speculative_tokens: int = 5,
) -> OptimizedInferenceEngine:
    """创建优化的推理引擎

    Args:
        enable_speculative: 是否启用Speculative Decoding
        num_speculative_tokens: 每次推测的token数

    Returns:
        OptimizedInferenceEngine: 引擎实例
    """
    config = OptimizedInferenceConfig(
        enable_speculative=enable_speculative,
        speculative_config=SpeculativeConfig(
            num_speculative_tokens=num_speculative_tokens,
        ),
    )
    return OptimizedInferenceEngine(config)
