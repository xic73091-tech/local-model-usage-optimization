"""
显存优化测试脚本

测试6GB显存运行大模型的各种配置。
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optimization.ultra_quantizer import UltraQuantizer, UltraQuantLevel
from src.optimization.vram_optimizer import VRAMOptimizer, OptimizationTarget


def test_ultra_quantizer():
    """测试极低精度量化器"""
    print("=" * 80)
    print("测试极低精度量化器")
    print("=" * 80)

    quantizer = UltraQuantizer()

    # 测试不同模型大小
    for model_size in [7.0, 13.0, 30.0, 70.0]:
        print(f"\n{'=' * 60}")
        print(f"模型: {model_size}B, 显存: 6GB")
        print(f"{'=' * 60}")

        recommendation = quantizer.recommend_for_vram(
            model_size_b=model_size,
            available_vram_gb=6.0,
            context_length=2048,
        )

        print(f"推荐量化: {recommendation.recommended_level.value}")
        print(f"  有效位数: {recommendation.profile.bits_per_param} bits")
        print(f"  每B参数显存: {recommendation.profile.vram_per_billion_gb} GB")
        print(f"  质量评分: {recommendation.profile.quality_score:.2f}")

        vram = recommendation.vram_estimate
        print(f"\n显存估算:")
        print(f"  模型权重: {vram.model_weight_gb:.2f} GB")
        print(f"  KV Cache: {vram.kv_cache_gb:.3f} GB")
        print(f"  总计: {vram.total_gb:.2f} GB")
        fits_str = "YES" if vram.fits_in_vram else "NO"
        print(f"  是否可运行: {fits_str}")


def test_vram_optimizer():
    """测试显存优化器"""
    print("\n" + "=" * 80)
    print("测试显存优化器")
    print("=" * 80)

    optimizer = VRAMOptimizer(vram_gb=6.0)

    # 测试不同模型和优化目标
    for model_size in [7.0, 13.0, 30.0, 70.0]:
        print(f"\n{'=' * 60}")
        print(f"模型: {model_size}B")
        print(f"{'=' * 60}")

        # 对比不同优化目标
        comparisons = optimizer.compare_targets(model_size)

        print(f"\n{'目标':<15} {'量化':<12} {'GPU层':<10} {'显存(GB)':<10} {'速度(t/s)':<12} {'可运行'}")
        print(f"{'-' * 60}")

        for comp in comparisons:
            fits = "YES" if comp["fits_in_vram"] else "NO"
            print(
                f"{comp['target']:<15} "
                f"{comp['quantization']:<12} "
                f"{comp['gpu_layers']}/{comp['total_layers']:<8} "
                f"{comp['vram_gb']:<10.2f} "
                f"{comp['speed_tps']:<12.1f} "
                f"{fits}"
            )


def test_specific_scenario():
    """测试特定场景"""
    print("\n" + "=" * 80)
    print("特定场景测试: 6GB显存运行13B模型")
    print("=" * 80)

    optimizer = VRAMOptimizer(vram_gb=6.0)

    # 场景1: 平衡模式
    result = optimizer.optimize(13.0, OptimizationTarget.BALANCED)
    print(f"\n场景1: BALANCED")
    print(f"  量化: {result.quantization.value}")
    print(f"  GPU层数: {result.layer_allocation.gpu_layers}/{result.layer_allocation.total_layers}")
    print(f"  上下文: {result.context_length}")
    print(f"  KV量化: {result.kv_quant_bits}bit")
    print(f"  预估显存: {result.estimated_vram_gb:.2f} GB")
    print(f"  预估速度: {result.estimated_speed_tps:.1f} tokens/s")
    print(f"  质量评分: {result.quality_score:.3f}")

    # 场景2: 最小显存模式
    result = optimizer.optimize(13.0, OptimizationTarget.MINIMAL_VRAM)
    print(f"\n场景2: MINIMAL_VRAM")
    print(f"  量化: {result.quantization.value}")
    print(f"  GPU层数: {result.layer_allocation.gpu_layers}/{result.layer_allocation.total_layers}")
    print(f"  上下文: {result.context_length}")
    print(f"  KV量化: {result.kv_quant_bits}bit")
    print(f"  预估显存: {result.estimated_vram_gb:.2f} GB")
    print(f"  预估速度: {result.estimated_speed_tps:.1f} tokens/s")
    print(f"  质量评分: {result.quality_score:.3f}")

    # 场景3: 最高质量模式
    result = optimizer.optimize(13.0, OptimizationTarget.MAX_QUALITY)
    print(f"\n场景3: MAX_QUALITY")
    print(f"  量化: {result.quantization.value}")
    print(f"  GPU层数: {result.layer_allocation.gpu_layers}/{result.layer_allocation.total_layers}")
    print(f"  上下文: {result.context_length}")
    print(f"  KV量化: {result.kv_quant_bits}bit")
    print(f"  预估显存: {result.estimated_vram_gb:.2f} GB")
    print(f"  预估速度: {result.estimated_speed_tps:.1f} tokens/s")
    print(f"  质量评分: {result.quality_score:.3f}")


def print_optimization_summary():
    """打印优化总结"""
    print("\n" + "=" * 80)
    print("6GB显存运行大模型 - 优化总结")
    print("=" * 80)

    optimizer = VRAMOptimizer(vram_gb=6.0)

    print(f"\n{'模型':<10} {'量化':<12} {'GPU层':<10} {'显存(GB)':<10} {'速度(t/s)':<12} {'质量':<8} {'可用性'}")
    print(f"{'-' * 80}")

    for model_size in [7.0, 13.0, 30.0, 70.0]:
        result = optimizer.optimize(model_size, OptimizationTarget.BALANCED)

        # 判断可用性
        if result.estimated_vram_gb <= 6.0 and result.estimated_speed_tps >= 10:
            usability = "PERFECT"
        elif result.estimated_vram_gb <= 6.0 and result.estimated_speed_tps >= 5:
            usability = "GOOD"
        elif result.estimated_vram_gb <= 6.0:
            usability = "OK"
        else:
            usability = "SLOW"

        print(
            f"{model_size}B{'':<7} "
            f"{result.quantization.value:<12} "
            f"{result.layer_allocation.gpu_layers}/{result.layer_allocation.total_layers:<8} "
            f"{result.estimated_vram_gb:<10.2f} "
            f"{result.estimated_speed_tps:<12.1f} "
            f"{result.quality_score:<8.3f} "
            f"{usability}"
        )

    print(f"\n{'=' * 80}")
    print("核心优化技术:")
    print("  1. 极低精度量化 (Q2_K/IQ2_XXS) - 压缩70-80%显存")
    print("  2. GPU-CPU混合卸载 - 关键层放GPU，其他放CPU")
    print("  3. KV Cache INT4量化 - 节省75% KV显存")
    print("  4. 滑动窗口注意力 - 减少KV Cache大小")
    print("  5. 动态层加载 - 按需加载减少GPU占用")
    print(f"{'=' * 80}")


def main():
    """主函数"""
    print("开始显存优化测试...\n")

    # 测试极低精度量化器
    test_ultra_quantizer()

    # 测试显存优化器
    test_vram_optimizer()

    # 测试特定场景
    test_specific_scenario()

    # 打印优化总结
    print_optimization_summary()

    print("\n测试完成！")


if __name__ == "__main__":
    main()
