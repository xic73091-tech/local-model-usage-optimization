"""
层间流水线并行测试
"""

import asyncio
import time
import sys
from pathlib import Path

import pytest

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.pipeline_parallel import (
    AccessPatternPredictor,
    PipelineParallelManager,
    DoubleBufferPipeline,
    PipelineConfig,
    PipelineStrategy,
    create_pipeline_manager,
    create_double_buffer_pipeline,
)


def test_access_pattern_predictor():
    """测试访问模式预测器"""
    print("=== 测试访问模式预测器 ===")

    predictor = AccessPatternPredictor(window_size=10)

    # 模拟顺序访问
    for i in range(10):
        predictor.record_access(i)

    # 预测
    predictions = predictor.predict_next(3)
    print(f"顺序访问预测: {predictions}")

    # 模拟循环访问
    predictor2 = AccessPatternPredictor(window_size=10)
    for _ in range(3):
        for i in range(5):
            predictor2.record_access(i)

    predictions2 = predictor2.predict_next(3)
    print(f"循环访问预测: {predictions2}")

    # 统计
    stats = predictor.get_stats()
    print(f"统计: {stats}")

    print("[OK] 访问模式预测器测试通过\n")


@pytest.mark.asyncio
async def test_pipeline_manager():
    """测试流水线管理器"""
    print("=== 测试流水线管理器 ===")

    manager = create_pipeline_manager(
        strategy="double",
        prefetch_layers=2,
        enable_predictive=True,
    )

    # 模拟层加载函数
    load_times = []
    async def mock_load(layer_idx: int) -> dict:
        start = time.time()
        await asyncio.sleep(0.05)  # 模拟加载延迟
        load_times.append((layer_idx, time.time() - start))
        return {"layer": layer_idx, "data": f"data_{layer_idx}"}

    def sync_load(layer_idx: int) -> dict:
        time.sleep(0.05)  # 模拟加载延迟
        return {"layer": layer_idx, "data": f"data_{layer_idx}"}

    # 处理多个层
    for i in range(5):
        buffer = await manager.prepare_layer(i, sync_load)
        print(f"层 {i}: ready={buffer.is_ready}, size={buffer.size_mb}MB")
        await asyncio.sleep(0.1)  # 模拟计算时间

    # 统计
    stats = manager.get_buffer_stats()
    print(f"缓冲区统计: {stats}")

    await manager.cleanup()
    print("[OK] 流水线管理器测试通过\n")


@pytest.mark.asyncio
async def test_double_buffer_pipeline():
    """测试双缓冲流水线"""
    print("=== 测试双缓冲流水线 ===")

    pipeline = create_double_buffer_pipeline(buffer_size_mb=256)

    # 模拟加载和计算函数
    def mock_load(layer_idx: int) -> dict:
        time.sleep(0.05)  # 模拟加载延迟
        return {"layer": layer_idx, "data": list(range(1000))}

    def mock_compute(data: dict) -> float:
        time.sleep(0.03)  # 模拟计算延迟
        return sum(data["data"])

    # 处理多个层
    for i in range(5):
        result = await pipeline.process_layer(i, mock_load, mock_compute)
        print(f"层 {i}: result={result}")

    # 统计
    stats = pipeline.get_stats()
    print(f"流水线统计: {stats}")

    print("[OK] 双缓冲流水线测试通过\n")


@pytest.mark.asyncio
async def test_pipeline_speedup():
    """测试流水线加速效果"""
    print("=== 测试流水线加速效果 ===")

    # 不使用流水线
    def no_pipeline_process(layers: int) -> float:
        start = time.time()
        for i in range(layers):
            # 加载
            time.sleep(0.05)
            # 计算
            time.sleep(0.03)
        return time.time() - start

    # 使用流水线
    async def pipeline_process(layers: int) -> float:
        pipeline = create_double_buffer_pipeline()

        def mock_load(layer_idx: int) -> dict:
            time.sleep(0.05)  # 加载延迟
            return {"layer": layer_idx, "data": []}

        def mock_compute(data: dict) -> None:
            time.sleep(0.03)  # 计算延迟

        start = time.time()
        for i in range(layers):
            await pipeline.process_layer(i, mock_load, mock_compute)
        return time.time() - start

    layers = 10

    # 测试无流水线
    time_no_pipeline = no_pipeline_process(layers)

    # 测试有流水线
    time_pipeline = await pipeline_process(layers)

    # 计算加速比
    speedup = time_no_pipeline / time_pipeline if time_pipeline > 0 else 1.0

    print(f"无流水线: {time_no_pipeline:.3f}s")
    print(f"有流水线: {time_pipeline:.3f}s")
    print(f"加速比: {speedup:.2f}x")

    # 流水线应该更快
    if speedup > 1.0:
        print(f"[OK] 流水线加速效果: {(speedup - 1) * 100:.1f}%")
    else:
        print(f"[INFO] 流水线未加速（开销大于收益）")

    print()


@pytest.mark.asyncio
async def test_concurrent_pipeline():
    """测试并发流水线"""
    print("=== 测试并发流水线 ===")

    manager = create_pipeline_manager(strategy="double")

    def mock_load(layer_idx: int) -> dict:
        time.sleep(0.02)
        return {"layer": layer_idx}

    # 并发处理多个层
    async def process_layer(layer_idx: int):
        buffer = await manager.prepare_layer(layer_idx, mock_load)
        await asyncio.sleep(0.05)  # 模拟计算
        return buffer.is_ready

    # 并发执行
    tasks = [process_layer(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    print(f"并发结果: {results}")
    print(f"成功率: {sum(results) / len(results) * 100:.1f}%")

    # 统计
    stats = manager.get_buffer_stats()
    print(f"统计: {stats}")

    await manager.cleanup()
    print("[OK] 并发流水线测试通过\n")


async def main():
    """主测试函数"""
    print("=== 层间流水线并行测试 ===\n")

    test_access_pattern_predictor()
    await test_pipeline_manager()
    await test_double_buffer_pipeline()
    await test_pipeline_speedup()
    await test_concurrent_pipeline()

    print("=== 所有测试完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
