"""
增强优化测试脚本

测试小显存运行大模型的优化效果
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.enhanced_optimizer import (
    EnhancedOptimizer,
    PCIeDetector,
    KVCacheCalculator,
    OOMProtector,
    FlashAttentionOptimizer,
    quick_optimize,
    get_optimization_report,
)


def test_pcie_detection():
    """测试PCIe检测"""
    print("\n" + "=" * 60)
    print("PCIe带宽检测测试")
    print("=" * 60)

    pcie = PCIeDetector.detect()
    print(f"  PCIe代际: {pcie.generation.value}.0")
    print(f"  通道数: x{pcie.lanes}")
    print(f"  带宽: {pcie.bandwidth_gb_s:.1f} GB/s")
    print(f"  开销系数: {pcie.overhead_factor:.1%}")
    return pcie


def test_gqa_detection():
    """测试GQA架构检测"""
    print("\n" + "=" * 60)
    print("GQA架构检测测试")
    print("=" * 60)

    test_cases = [
        (7, "llama-7b"),
        (13, "llama-13b"),
        (70, "llama-70b"),
        (7, "mistral-7b"),
        (7, None),  # 基于参数量估算
        (70, None),
    ]

    for size, name in test_cases:
        arch = KVCacheCalculator.detect_architecture(size, name)
        savings = KVCacheCalculator.calculate_memory_savings(arch, 4096)
        print(f"\n  模型: {size}B ({name or '自动检测'})")
        print(f"    层数: {arch.num_layers}")
        print(f"    注意力头: {arch.num_attention_heads}")
        print(f"    KV头数: {arch.num_kv_heads}")
        print(f"    GQA: {'是' if arch.use_gqa else '否'}")
        if arch.use_gqa:
            print(f"    KV Cache节省: {savings['savings_ratio']:.1%}")


def test_kv_cache_optimization():
    """测试KV Cache优化"""
    print("\n" + "=" * 60)
    print("KV Cache优化测试")
    print("=" * 60)

    test_configs = [
        (7, 4096, 16),
        (7, 4096, 8),
        (7, 4096, 4),
        (13, 8192, 16),
        (13, 8192, 8),
        (70, 4096, 16),
    ]

    for model_size, context, kv_bits in test_configs:
        arch = KVCacheCalculator.detect_architecture(model_size)
        kv_size = KVCacheCalculator.calculate_kv_cache_size(
            arch, context, kv_quant_bits=kv_bits
        )
        print(f"\n  {model_size}B, 上下文{context}, KV量化{kv_bits}bit:")
        print(f"    KV Cache大小: {kv_size:.3f} GB")
        print(f"    GQA节省: {arch.use_gqa}")


def test_oom_protection():
    """测试OOM防护"""
    print("\n" + "=" * 60)
    print("OOM防护测试")
    print("=" * 60)

    protector = OOMProtector(6.0, safety_margin_gb=0.5)

    # 模拟显存使用增加
    test_usages = [3.0, 4.0, 4.5, 5.0, 5.5, 5.8]
    for usage in test_usages:
        status = protector.update_usage(usage)
        print(f"\n  显存使用: {usage:.1f} GB / {protector.usable_vram_gb:.1f} GB")
        print(f"    级别: {status.level.name}")
        print(f"    使用率: {status.usage_ratio:.1%}")
        print(f"    建议: {status.recommended_action}")

    # 测试自动降级
    print("\n  自动降级测试:")
    new_ctx, new_quant, new_gpu, msg = protector.auto_degrade(
        current_context=4096,
        current_quant="q4_k_m",
        current_gpu_layers=20,
    )
    print(f"    降级结果: {msg}")


def test_flash_attention():
    """测试Flash Attention优化"""
    print("\n" + "=" * 60)
    print("Flash Attention优化测试")
    print("=" * 60)

    test_configs = [
        (32, 32, 128, 4096),   # 7B
        (40, 40, 128, 8192),   # 13B
        (80, 64, 128, 4096),   # 70B
    ]

    for layers, heads, head_dim, context in test_configs:
        savings = FlashAttentionOptimizer.calculate_memory_savings(
            context, layers, heads, head_dim
        )
        print(f"\n  {layers}层, {heads}头, 上下文{context}:")
        print(f"    标准注意力: {savings['standard_attention_gb']:.2f} GB")
        print(f"    Flash Attention: {savings['flash_attention_gb']:.2f} GB")
        print(f"    节省: {savings['savings_gb']:.2f} GB ({savings['savings_ratio']:.1%})")


def test_enhanced_optimizer():
    """测试增强优化器"""
    print("\n" + "=" * 60)
    print("增强优化器完整测试")
    print("=" * 60)

    # 测试不同显存配置
    test_configs = [
        (4, 7, "7B模型, 4GB显存"),
        (6, 7, "7B模型, 6GB显存"),
        (6, 13, "13B模型, 6GB显存"),
        (8, 13, "13B模型, 8GB显存"),
        (8, 30, "30B模型, 8GB显存"),
        (12, 30, "30B模型, 12GB显存"),
        (24, 70, "70B模型, 24GB显存"),
    ]

    for vram, model_size, desc in test_configs:
        print(f"\n  【{desc}】")
        result = quick_optimize(vram, model_size)

        print(f"    量化: {result.quant_level} ({result.quant_bits} bits)")
        print(f"    GPU层数: {result.gpu_layers}/{result.total_layers}")
        print(f"    上下文: {result.context_length} (滑动窗口: {result.sliding_window})")
        print(f"    KV Cache: {result.kv_cache_gb:.3f} GB")
        print(f"    总显存: {result.total_vram_gb:.2f} GB")
        print(f"    预估速度: {result.speed_estimate.estimated_tps:.1f} tokens/s")
        print(f"    适合实时: {'是' if result.speed_estimate.suitable_for_realtime else '否'}")

        if result.warnings:
            for w in result.warnings:
                print(f"    警告: {w}")


def test_optimization_report():
    """测试优化报告生成"""
    print("\n" + "=" * 60)
    print("优化报告生成测试")
    print("=" * 60)

    # 6GB显存运行13B模型的报告
    report = get_optimization_report(6, 13, "llama-13b")
    print(report)


def main():
    """运行所有测试"""
    print("=" * 60)
    print("小显存运行大模型优化测试")
    print("=" * 60)

    test_pcie_detection()
    test_gqa_detection()
    test_kv_cache_optimization()
    test_oom_protection()
    test_flash_attention()
    test_enhanced_optimizer()
    test_optimization_report()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
