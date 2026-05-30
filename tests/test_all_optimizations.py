"""
所有优化模块综合测试
"""

import asyncio
import sys
from pathlib import Path

import pytest

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.speculative_decoding import (
    NgramPredictor,
    SpeculativeDecoder,
    DraftModelStrategy,
    create_speculative_decoder,
)
from src.optimization.mixed_precision import (
    QuantLevel,
    LayerType,
    MixedPrecisionQuantizer,
    MixedPrecisionConfig,
)
from src.optimization.adaptive_optimizer import (
    AdaptiveOptimizer,
    AdaptiveConfig,
    OptimizationGoal,
    PerformanceMetrics,
)
from src.optimization.pipeline_parallel import (
    AccessPatternPredictor,
    PipelineParallelManager,
    DoubleBufferPipeline,
    create_pipeline_manager,
    create_double_buffer_pipeline,
)


@pytest.mark.asyncio
async def test_speculative_decoding():
    """测试Speculative Decoding"""
    print("=== Speculative Decoding ===")

    predictor = NgramPredictor(n=3, cache_size=1024)

    # 更新token历史
    tokens = ["1", "2", "3", "4", "5"] * 10
    for token in tokens:
        predictor.update(token)

    # 预测
    predictions = predictor.predict(num_predictions=5)
    print(f"Predictions: {predictions}")
    print("[OK]\n")


@pytest.mark.asyncio
async def test_mixed_precision():
    """测试混合精度量化"""
    print("=== Mixed Precision ===")

    config = MixedPrecisionConfig(
        target_vram_gb=6.0,
        model_size_b=7.0,
    )
    quantizer = MixedPrecisionQuantizer(config)

    # 执行量化
    result = quantizer.quantize()

    print(f"Total size: {result.total_size_gb:.2f} GB")
    print(f"Avg bits: {result.total_bits:.2f}")
    print(f"Quality: {result.total_quality:.2f}")
    print(f"Speed factor: {result.total_speed_factor:.2f}")

    # 显示各层配置
    for info in result.layer_configs:
        print(f"{info.layer_type.value}: {info.quant_level.value}")

    print("[OK]\n")


@pytest.mark.asyncio
async def test_adaptive_optimizer():
    """测试自适应优化器"""
    print("=== Adaptive Optimizer ===")

    config = AdaptiveConfig(
        goal=OptimizationGoal.BALANCED,
    )
    optimizer = AdaptiveOptimizer(config)

    # 模拟性能数据
    for i in range(10):
        await optimizer.record_performance(
            tokens_per_second=20.0 + i,
            latency_ms=50 - i,
            vram_usage_gb=4.0 + i * 0.1,
            ram_usage_gb=8.0,
            gpu_utilization=0.8,
        )

    recommendation = optimizer.get_recommendation()
    print(f"Recommendation: {recommendation}")
    print("[OK]\n")


@pytest.mark.asyncio
async def test_pipeline_parallel():
    """测试流水线并行"""
    print("=== Pipeline Parallel ===")

    predictor = AccessPatternPredictor(window_size=10)

    # 记录访问
    for i in range(10):
        predictor.record_access(i)

    predictions = predictor.predict_next(3)
    print(f"Predictions: {predictions}")

    # 测试流水线
    pipeline = create_double_buffer_pipeline(buffer_size_mb=256)

    def mock_load(layer_idx: int) -> dict:
        return {"layer": layer_idx}

    def mock_compute(data: dict) -> str:
        return f"result_{data['layer']}"

    for i in range(3):
        result = await pipeline.process_layer(i, mock_load, mock_compute)
        print(f"Layer {i}: {result}")

    print("[OK]\n")


async def main():
    """主测试"""
    print("=== All Optimizations Test ===\n")

    await test_speculative_decoding()
    await test_mixed_precision()
    await test_adaptive_optimizer()
    await test_pipeline_parallel()

    print("=== All Tests Passed ===")


if __name__ == "__main__":
    asyncio.run(main())
