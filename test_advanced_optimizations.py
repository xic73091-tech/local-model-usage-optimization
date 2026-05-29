"""
高级优化测试脚本

测试新实现的优化功能：
1. Speculative Decoding
2. 混合精度量化
3. 运行时自适应优化
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.speculative_decoding import (
    SpeculativeDecoder,
    SpeculativeConfig,
    DraftModelStrategy,
    NgramPredictor,
    create_speculative_decoder,
    create_optimized_engine,
)
from src.optimization.mixed_precision import (
    MixedPrecisionQuantizer,
    MixedPrecisionConfig,
    QuantLevel,
    LayerType,
    get_mixed_precision_config,
    compare_quantization_strategies,
)
from src.optimization.adaptive_optimizer import (
    AdaptiveOptimizer,
    AdaptiveConfig,
    OptimizationGoal,
    PerformanceMonitor,
    create_adaptive_optimizer,
)


def test_ngram_predictor():
    """测试N-gram预测器"""
    print("\n" + "=" * 60)
    print("N-gram预测器测试")
    print("=" * 60)

    predictor = NgramPredictor(n=3, cache_size=100)

    # 模拟token序列
    tokens = ["The", "cat", "sat", "on", "the", "mat", "The", "cat", "sat", "on"]

    # 更新预测器
    for token in tokens:
        predictor.update(token)

    # 测试预测
    predictions = predictor.predict(3)
    print(f"  输入序列: {tokens}")
    print(f"  预测结果: {predictions}")

    stats = predictor.get_stats()
    print(f"  统计: {stats}")


def test_speculative_decoder():
    """测试Speculative Decoder"""
    print("\n" + "=" * 60)
    print("Speculative Decoder测试")
    print("=" * 60)

    # 创建解码器
    decoder = create_speculative_decoder(
        strategy="ngram",
        num_speculative_tokens=5,
        cache_size=100,
    )

    # 模拟目标模型生成函数
    async def mock_generate(prompt: str, max_tokens: int = 1) -> str:
        await asyncio.sleep(0.01)  # 模拟延迟
        # 简单的模拟：返回固定的token
        return "token" * max_tokens

    # 测试生成
    async def run_test():
        result = await decoder.generate(
            prompt="Hello world",
            target_generate_fn=mock_generate,
            max_tokens=10,
        )

        print(f"  生成的token数: {len(result.tokens)}")
        print(f"  接受的token数: {result.accepted_count}")
        print(f"  推测的token数: {result.speculated_count}")
        print(f"  接受率: {result.acceptance_rate:.2%}")
        print(f"  目标模型调用: {result.target_model_calls}")
        print(f"  草稿模型调用: {result.draft_model_calls}")
        print(f"  速度提升: {result.speedup:.2f}x")
        print(f"  总耗时: {result.total_time_ms:.1f} ms")

    asyncio.run(run_test())

    stats = decoder.get_stats()
    print(f"  统计: {stats}")


def test_mixed_precision():
    """测试混合精度量化"""
    print("\n" + "=" * 60)
    print("混合精度量化测试")
    print("=" * 60)

    # 测试不同模型大小
    model_sizes = [7, 13, 30, 70]

    for size in model_sizes:
        print(f"\n  模型大小: {size}B")

        # 获取混合精度配置
        result = get_mixed_precision_config(
            model_size_b=size,
            vram_gb=8.0,
            target_quality=0.7,
        )

        print(f"    平均位数: {result.total_bits:.1f}")
        print(f"    质量分数: {result.total_quality:.2f}")
        print(f"    速度因子: {result.total_speed_factor:.2f}")
        print(f"    显存占用: {result.total_size_gb:.2f} GB")

        # 显示各层配置
        for lc in result.layer_configs:
            print(f"    {lc.layer_type.value}: {lc.quant_level.value} ({lc.bits} bits)")


def test_mixed_precision_comparison():
    """测试混合精度量化对比"""
    print("\n" + "=" * 60)
    print("混合精度量化对比测试")
    print("=" * 60)

    model_size = 13

    # 对比不同策略
    comparison = compare_quantization_strategies(model_size, 8.0)

    print(f"\n  模型大小: {model_size}B")
    print(f"\n  混合精度:")
    mp = comparison["mixed_precision"]
    print(f"    平均位数: {mp['avg_bits']:.1f}")
    print(f"    质量: {mp['quality']:.2f}")
    print(f"    速度因子: {mp['speed_factor']:.2f}")
    print(f"    大小: {mp['size_gb']:.2f} GB")

    print(f"\n  Q4_K_M:")
    q4 = comparison["q4_k_m"]
    print(f"    平均位数: {q4['avg_bits']:.1f}")
    print(f"    质量: {q4['quality']:.2f}")
    print(f"    速度因子: {q4['speed_factor']:.2f}")
    print(f"    大小: {q4['size_gb']:.2f} GB")

    print(f"\n  Q2_K:")
    q2 = comparison["q2_k"]
    print(f"    平均位数: {q2['avg_bits']:.1f}")
    print(f"    质量: {q2['quality']:.2f}")
    print(f"    速度因子: {q2['speed_factor']:.2f}")
    print(f"    大小: {q2['size_gb']:.2f} GB")

    print(f"\n  推荐: {comparison['recommendation']}")


def test_adaptive_optimizer():
    """测试自适应优化器"""
    print("\n" + "=" * 60)
    print("自适应优化器测试")
    print("=" * 60)

    # 创建优化器
    optimizer = create_adaptive_optimizer(
        goal="balanced",
        min_tps=5.0,
        max_latency_ms=500.0,
    )

    # 模拟性能数据
    test_data = [
        (10.0, 100.0, 4.0),   # 正常
        (8.0, 150.0, 4.5),    # 稍慢
        (4.0, 300.0, 5.0),    # 太慢，应该触发调整
        (3.0, 400.0, 5.5),    # 更慢
        (12.0, 80.0, 3.5),    # 恢复正常
    ]

    async def run_test():
        for tps, latency, vram in test_data:
            adjustment = await optimizer.record_performance(
                tokens_per_second=tps,
                latency_ms=latency,
                vram_usage_gb=vram,
            )

            print(f"\n  TPS: {tps}, 延迟: {latency}ms, 显存: {vram}GB")
            if adjustment:
                print(f"    触发调整: {adjustment}")
            else:
                print(f"    无需调整")

    asyncio.run(run_test())

    # 获取统计信息
    stats = optimizer.get_stats()
    print(f"\n  统计信息:")
    print(f"    当前量化: {stats['current_state']['quant_level']}")
    print(f"    当前上下文: {stats['current_state']['context_length']}")
    print(f"    调整次数: {stats['adjustment_count']}")

    # 获取建议
    recommendation = optimizer.get_recommendation()
    print(f"\n  优化建议:")
    if recommendation.get("recommendations"):
        for rec in recommendation["recommendations"]:
            print(f"    - {rec['suggestion']}")
    else:
        print(f"    当前配置最优")


def test_optimized_engine():
    """测试优化推理引擎"""
    print("\n" + "=" * 60)
    print("优化推理引擎测试")
    print("=" * 60)

    # 创建引擎
    engine = create_optimized_engine(
        enable_speculative=True,
        num_speculative_tokens=5,
    )

    # 模拟目标模型
    async def mock_target_generate(prompt: str, max_tokens: int = 1) -> str:
        await asyncio.sleep(0.01)
        return "token" * max_tokens

    # 测试生成
    async def run_test():
        text, stats = await engine.generate(
            prompt="Hello world",
            target_generate_fn=mock_target_generate,
            max_tokens=20,
            use_speculative=True,
        )

        print(f"  生成的文本长度: {len(text)}")
        print(f"  方法: {stats['method']}")
        if "speedup" in stats:
            print(f"  速度提升: {stats['speedup']:.2f}x")
        print(f"  总耗时: {stats['total_time_ms']:.1f} ms")

    asyncio.run(run_test())


def main():
    """运行所有测试"""
    print("=" * 60)
    print("高级优化功能测试")
    print("=" * 60)

    test_ngram_predictor()
    test_speculative_decoder()
    test_mixed_precision()
    test_mixed_precision_comparison()
    test_adaptive_optimizer()
    test_optimized_engine()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
