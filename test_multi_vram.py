"""
多显存优化测试脚本

测试所有显存配置(4GB/6GB/8GB/12GB/16GB/24GB)的优化方案。
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optimization.multi_vram_optimizer import (
    MultiVRAMOptimizer,
    VRAM_CONFIGS,
    MODEL_SIZES,
)


def test_multi_vram_optimizer():
    """测试多显存优化器"""
    print("=" * 100)
    print("多显存优化器测试")
    print("=" * 100)

    optimizer = MultiVRAMOptimizer()

    # 打印对比表
    optimizer.print_comparison_table()

    # 打印推荐配置
    optimizer.print_recommendations()


def test_vram_specific():
    """测试特定显存配置"""
    print("\n" + "=" * 80)
    print("特定显存配置测试")
    print("=" * 80)

    optimizer = MultiVRAMOptimizer()

    # 测试每个显存配置
    for vram_gb in [4, 6, 8, 12, 16, 24]:
        print(f"\n{'=' * 60}")
        print(f"{vram_gb}GB 显存配置")
        print(f"{'=' * 60}")

        # 对比不同模型
        comparisons = optimizer.compare_models_for_vram(vram_gb)

        print(f"\n{'模型':<10} {'量化':<12} {'GPU层':<10} {'显存(GB)':<10} {'速度(t/s)':<12} {'质量':<8} {'可运行'}")
        print(f"{'-' * 70}")

        for comp in comparisons:
            fits = "YES" if comp["fits"] else "NO"
            print(
                f"{comp['model_size_b']}B{'':<7} "
                f"{comp['quantization']:<12} "
                f"{comp['gpu_layers']}/{comp['total_layers']:<8} "
                f"{comp['vram_used_gb']:<10.2f} "
                f"{comp['speed_tps']:<12.1f} "
                f"{comp['quality']:<8.3f} "
                f"{fits}"
            )


def test_model_specific():
    """测试特定模型配置"""
    print("\n" + "=" * 80)
    print("特定模型配置测试")
    print("=" * 80)

    optimizer = MultiVRAMOptimizer()

    # 测试13B模型在不同显存下的配置
    print("\n13B模型在不同显存下的配置:")
    print(f"{'显存':<10} {'量化':<12} {'GPU层':<10} {'显存(GB)':<10} {'速度(t/s)':<12} {'质量':<8} {'可运行'}")
    print(f"{'-' * 70}")

    comparisons = optimizer.compare_vram_for_model(13.0)
    for comp in comparisons:
        fits = "YES" if comp["fits"] else "NO"
        print(
            f"{comp['vram_gb']}GB{'':<7} "
            f"{comp['quantization']:<12} "
            f"{comp['gpu_layers']}/{comp['total_layers']:<8} "
            f"{comp['vram_used_gb']:<10.2f} "
            f"{comp['speed_tps']:<12.1f} "
            f"{comp['quality']:<8.3f} "
            f"{fits}"
        )


def test_best_model_finder():
    """测试最佳模型查找"""
    print("\n" + "=" * 80)
    print("最佳模型查找测试")
    print("=" * 80)

    optimizer = MultiVRAMOptimizer()

    for vram_gb in [4, 6, 8, 12, 16, 24]:
        print(f"\n{vram_gb}GB 显存可用模型:")

        best_models = optimizer.find_best_model_for_vram(
            vram_gb,
            min_speed=5.0,
            min_quality=0.4,
        )

        if best_models:
            for model in best_models[:3]:  # 显示前3个
                print(f"  - {model['model_size_b']}B {model['quantization']} "
                      f"{model['speed_tps']}t/s quality={model['quality']:.3f}")
        else:
            print("  无满足条件的模型")


def export_configs():
    """导出配置"""
    print("\n" + "=" * 80)
    print("导出配置")
    print("=" * 80)

    optimizer = MultiVRAMOptimizer()

    # 导出JSON配置
    optimizer.export_configs("vram_configs.json")
    print("已导出: vram_configs.json")

    # 导出Markdown表格
    optimizer.export_markdown_table("VRAM_CONFIGS.md")
    print("已导出: VRAM_CONFIGS.md")


def main():
    """主函数"""
    print("开始多显存优化测试...\n")

    # 测试多显存优化器
    test_multi_vram_optimizer()

    # 测试特定显存配置
    test_vram_specific()

    # 测试特定模型配置
    test_model_specific()

    # 测试最佳模型查找
    test_best_model_finder()

    # 导出配置
    export_configs()

    print("\n测试完成！")


if __name__ == "__main__":
    main()
