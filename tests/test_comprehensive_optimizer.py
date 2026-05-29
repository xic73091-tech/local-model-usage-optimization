"""
全方位优化器测试
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.comprehensive_optimizer import (
    ComprehensiveConfig,
    ComprehensiveOptimizer,
    OptimizationLevel,
    SparseAttentionOptimizer,
    ChunkedPrefillOptimizer,
    ActivationCheckpointOptimizer,
    TokenMergingOptimizer,
    ContinuousBatcher,
    create_comprehensive_optimizer,
    quick_optimization_analysis,
)


def test_sparse_attention():
    """测试稀疏注意力"""
    print("=== Sparse Attention ===")

    optimizer = SparseAttentionOptimizer(sparse_ratio=0.5)

    # 模拟选择重要token
    selected = optimizer.select_important_tokens(None, seq_len=1024)
    print(f"Selected tokens: {len(selected)}/1024")

    savings = optimizer.get_savings()
    print(f"Savings: {savings}")

    print("[OK]\n")


def test_chunked_prefill():
    """测试分块预填充"""
    print("=== Chunked Prefill ===")

    optimizer = ChunkedPrefillOptimizer(chunk_size=512)

    # 分块
    chunks = optimizer.split_prompt(prompt_length=2048)
    print(f"Chunks: {len(chunks)}")

    # 内存节省
    savings = optimizer.estimate_memory_savings(2048)
    print(f"Memory savings: {savings:.1%}")

    stats = optimizer.get_stats()
    print(f"Stats: {stats}")

    print("[OK]\n")


def test_activation_checkpoint():
    """测试激活检查点"""
    print("=== Activation Checkpoint ===")

    optimizer = ActivationCheckpointOptimizer(checkpoint_ratio=0.5)

    # 选择检查点层
    selected = optimizer.select_checkpoint_layers(num_layers=32)
    print(f"Checkpoint layers: {len(selected)}")

    # 内存节省
    savings = optimizer.estimate_memory_savings(32)
    print(f"Memory savings: {savings:.1%}")

    # 计算开销
    overhead = optimizer.estimate_compute_overhead()
    print(f"Compute overhead: {overhead:.1%}")

    print("[OK]\n")


def test_token_merging():
    """测试Token合并"""
    print("=== Token Merging ===")

    optimizer = TokenMergingOptimizer(merge_ratio=0.25)

    # 合并token
    _, new_len = optimizer.merge_tokens(None, seq_len=1024)
    print(f"New sequence length: {new_len}")

    # 加速比
    speedup = optimizer.estimate_speedup()
    print(f"Speedup: {speedup:.2f}x")

    print("[OK]\n")


def test_continuous_batcher():
    """测试连续批处理器"""
    print("=== Continuous Batcher ===")

    batcher = ContinuousBatcher(max_batch_size=8, max_wait_ms=100.0)

    # 添加请求
    for i in range(20):
        batcher.add_request({"id": i, "prompt": f"test {i}"})

    # 获取批次
    batch1 = batcher.get_next_batch()
    batch2 = batcher.get_next_batch()
    print(f"Batch 1: {len(batch1)} requests")
    print(f"Batch 2: {len(batch2)} requests")

    # 吞吐量提升
    improvement = batcher.estimate_throughput_improvement()
    print(f"Throughput improvement: {improvement:.1f}x")

    stats = batcher.get_stats()
    print(f"Stats: {stats}")

    print("[OK]\n")


def test_comprehensive_optimizer():
    """测试全方位优化器"""
    print("=== Comprehensive Optimizer ===")

    # 测试不同优化级别
    for level in ["conservative", "balanced", "aggressive"]:
        optimizer = create_comprehensive_optimizer(level=level, vram_gb=6.0)

        # 分析优化潜力
        estimate = optimizer.analyze_optimization_potential(
            model_size_b=7.0,
            vram_gb=6.0,
            seq_len=2048,
            num_layers=32,
        )

        print(f"\n{level.upper()}:")
        print(f"  Memory saved: {estimate.memory_saved_gb:.2f} GB")
        print(f"  Speedup: {estimate.speedup_ratio:.2f}x")
        print(f"  Quality impact: {estimate.quality_impact:.3f}")
        print(f"  Stability risk: {estimate.stability_risk:.3f}")

    print("\n[OK]\n")


def test_optimization_plan():
    """测试优化计划"""
    print("=== Optimization Plan ===")

    optimizer = create_comprehensive_optimizer(level="balanced", vram_gb=6.0)

    # 获取优化计划
    plan = optimizer.get_optimization_plan(
        model_size_b=7.0,
        vram_gb=6.0,
    )

    print(f"Model: {plan['model_size_b']}B")
    print(f"VRAM: {plan['vram_gb']}GB")
    print(f"Level: {plan['optimization_level']}")
    print(f"Enabled optimizations: {len(plan['enabled_optimizations'])}")

    for opt in plan["enabled_optimizations"]:
        print(f"  - {opt['name']}: {opt['config']}")

    print(f"\nRecommendations:")
    for rec in plan["recommendations"]:
        print(f"  - {rec}")

    print("\n[OK]\n")


def test_quick_analysis():
    """测试快速分析"""
    print("=== Quick Analysis ===")

    # 测试不同显存配置
    for vram in [4, 6, 8, 12]:
        result = quick_optimization_analysis(
            model_size_b=7.0,
            vram_gb=float(vram),
            level="balanced",
        )

        estimate = result["estimate"]
        print(f"\n{vram}GB VRAM:")
        print(f"  Memory saved: {estimate['memory_saved_gb']} GB")
        print(f"  Speedup: {estimate['speedup_ratio']}x")
        print(f"  Quality impact: {estimate['quality_impact']}")

    print("\n[OK]\n")


def test_optimization_levels():
    """测试优化级别"""
    print("=== Optimization Levels ===")

    config = ComprehensiveConfig()

    for level in OptimizationLevel:
        config.apply_level(level)
        print(f"\n{level.value}:")
        print(f"  Sparse attention: {config.attention.sparse_attention}")
        print(f"  Sparse ratio: {config.attention.sparse_ratio}")
        print(f"  Sliding window: {config.attention.sliding_window}")
        print(f"  Token merging: {config.prefill.token_merging}")
        print(f"  Merge ratio: {config.prefill.merge_ratio}")
        print(f"  Activation checkpoint: {config.activation.activation_checkpointing}")
        print(f"  Checkpoint ratio: {config.activation.checkpoint_ratio}")
        print(f"  Torch compile: {config.compilation.torch_compile}")

    print("\n[OK]\n")


async def main():
    """主测试函数"""
    print("=== Comprehensive Optimizer Tests ===\n")

    test_sparse_attention()
    test_chunked_prefill()
    test_activation_checkpoint()
    test_token_merging()
    test_continuous_batcher()
    test_comprehensive_optimizer()
    test_optimization_plan()
    test_quick_analysis()
    test_optimization_levels()

    print("=== All Tests Passed ===")


if __name__ == "__main__":
    asyncio.run(main())
