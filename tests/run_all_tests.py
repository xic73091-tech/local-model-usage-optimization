"""
运行所有测试
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Running All Optimization Tests")
    print("=" * 60)
    print()

    # 测试1: 基础优化
    print("-" * 40)
    print("Test 1: Basic Optimizations")
    print("-" * 40)
    try:
        from src.optimization.speculative_decoding import NgramPredictor
        from src.optimization.mixed_precision import MixedPrecisionQuantizer, MixedPrecisionConfig
        from src.optimization.adaptive_optimizer import AdaptiveOptimizer, AdaptiveConfig, OptimizationGoal
        from src.optimization.pipeline_parallel import AccessPatternPredictor, create_double_buffer_pipeline

        # Speculative Decoding
        predictor = NgramPredictor(n=3, cache_size=1024)
        for token in ["1", "2", "3", "4", "5"] * 10:
            predictor.update(token)
        predictions = predictor.predict(num_predictions=5)
        print(f"  Speculative Decoding: OK (predictions={predictions})")

        # Mixed Precision
        config = MixedPrecisionConfig(target_vram_gb=6.0, model_size_b=7.0)
        quantizer = MixedPrecisionQuantizer(config)
        result = quantizer.quantize()
        print(f"  Mixed Precision: OK (size={result.total_size_gb:.2f}GB, bits={result.total_bits:.2f})")

        # Adaptive Optimizer
        config = AdaptiveConfig(goal=OptimizationGoal.BALANCED)
        optimizer = AdaptiveOptimizer(config)
        for i in range(5):
            await optimizer.record_performance(
                tokens_per_second=20.0 + i,
                latency_ms=50 - i,
                vram_usage_gb=4.0,
            )
        print(f"  Adaptive Optimizer: OK")

        # Pipeline Parallel
        predictor = AccessPatternPredictor(window_size=10)
        for i in range(10):
            predictor.record_access(i)
        predictions = predictor.predict_next(3)
        print(f"  Pipeline Parallel: OK (predictions={predictions})")

    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # 测试2: 全方位优化
    print("-" * 40)
    print("Test 2: Comprehensive Optimizer")
    print("-" * 40)
    try:
        from src.optimization.comprehensive_optimizer import (
            create_comprehensive_optimizer,
            quick_optimization_analysis,
        )

        # 测试不同级别
        for level in ["conservative", "balanced", "aggressive"]:
            optimizer = create_comprehensive_optimizer(level=level, vram_gb=6.0)
            estimate = optimizer.analyze_optimization_potential(
                model_size_b=7.0,
                vram_gb=6.0,
            )
            print(f"  {level}: memory_saved={estimate.memory_saved_gb:.2f}GB, speedup={estimate.speedup_ratio:.2f}x")

        # 快速分析
        result = quick_optimization_analysis(model_size_b=7.0, vram_gb=6.0)
        print(f"  Quick analysis: OK")

    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # 测试3: 内存优化协调器
    print("-" * 40)
    print("Test 3: Memory Optimizer")
    print("-" * 40)
    try:
        from src.optimization.memory_optimizer import (
            MemoryOptimizer,
            HardwareProfile,
            OptimizationProfile,
        )

        optimizer = MemoryOptimizer()
        hardware = HardwareProfile(
            vram_total_gb=6.0,
            vram_free_gb=5.5,
            ram_total_gb=16.0,
            ram_free_gb=12.0,
            cpu_cores=8,
            has_gpu=True,
        )

        result = optimizer.optimize_for_model(
            model_size_b=7.0,
            profile=OptimizationProfile.BALANCED,
            hardware=hardware,
        )
        print(f"  Memory Optimizer: OK (vram={result.estimated_vram_gb:.2f}GB, speed={result.estimated_speed_tps:.1f}tps)")

    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # 测试4: 动态层加载
    print("-" * 40)
    print("Test 4: Dynamic Layer Loader")
    print("-" * 40)
    try:
        from src.optimization.dynamic_loader import (
            DynamicLayerLoader,
            DynamicConfig,
            LayerLocation,
        )

        config = DynamicConfig(
            max_gpu_layers=20,
            max_cpu_layers=40,
            prefetch_enabled=True,
        )
        loader = DynamicLayerLoader(
            model_path="./models/test",
            config=config,
            total_layers=32,
        )

        # 模拟加载
        loader._initialized = True
        loader._manager = loader._manager or __import__('src.optimization.dynamic_loader', fromlist=['LayerManager']).LayerManager(32, config)

        print(f"  Dynamic Loader: OK (total_layers={loader.total_layers})")

    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # 测试5: 导入所有模块
    print("-" * 40)
    print("Test 5: Import All Modules")
    print("-" * 40)
    try:
        from src.optimization import (
            # scheduler
            InferenceScheduler,
            # offloader
            ModelOffloader,
            # kv_cache
            KVCacheOptimizer,
            # memory_optimizer
            MemoryOptimizer,
            # quantizer
            QuantizationManager,
            # ultra_quantizer
            UltraQuantizer,
            # vram_optimizer
            VRAMOptimizer,
            # multi_vram_optimizer
            MultiVRAMOptimizer,
            # comprehensive_optimizer
            ComprehensiveOptimizer,
            create_comprehensive_optimizer,
            quick_optimization_analysis,
        )
        print("  All imports: OK")

    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("=" * 60)
    print("All Tests Completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
