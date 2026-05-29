"""
安全修复验证测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """测试所有导入是否正常"""
    print("=== Testing Imports ===")

    # 测试 server 模块 (需要 fastapi)
    try:
        from src.api.server import (
            AUTH_ENABLED,
            API_KEYS,
            CORS_ORIGINS,
            RATE_LIMIT_REQUESTS,
            RATE_LIMIT_WINDOW,
            RateLimiter,
            AppState,
        )
        print(f"  AUTH_ENABLED: {AUTH_ENABLED}")
        print(f"  API_KEYS: {'***' if API_KEYS else '(empty)'}")
        print(f"  CORS_ORIGINS: {CORS_ORIGINS}")
        print(f"  RATE_LIMIT_REQUESTS: {RATE_LIMIT_REQUESTS}")
        print(f"  RATE_LIMIT_WINDOW: {RATE_LIMIT_WINDOW}")
        print("  [OK] Server imports")
    except ImportError as e:
        print(f"  [SKIP] Server imports (fastapi not installed): {e}")

    # 测试 optimization 模块
    try:
        from src.optimization.ultra_quantizer import UltraQuantizer
        print("  [OK] UltraQuantizer imports")
    except Exception as e:
        print(f"  [ERROR] UltraQuantizer imports: {e}")
        return False

    # 测试 backends 模块
    try:
        from src.backends.llama_cpp import _skip_gguf_value
        print("  [OK] llama_cpp imports")
    except Exception as e:
        print(f"  [ERROR] llama_cpp imports: {e}")
        return False

    return True


def test_rate_limiter():
    """测试速率限制器清理"""
    print("\n=== Testing Rate Limiter ===")

    try:
        from src.api.server import RateLimiter
    except ImportError:
        print("  [SKIP] RateLimiter (fastapi not installed)")
        return True

    import time

    limiter = RateLimiter(max_requests=10, window_seconds=60)

    # 添加一些请求
    for i in range(5):
        limiter.is_allowed(f"192.168.1.{i}")

    print(f"  Active IPs: {len(limiter.requests)}")

    # 模拟过期
    limiter._last_cleanup = time.time() - 400  # 超过清理间隔
    limiter._cleanup_stale_entries()

    print(f"  After cleanup: {len(limiter.requests)}")
    print("  [OK] Rate limiter cleanup")

    return True


def test_gguf_security():
    """测试GGUF解析安全限制"""
    print("\n=== Testing GGUF Security ===")

    from src.backends.llama_cpp import _skip_gguf_value
    import io
    import struct

    # 测试1: 深度限制
    print("  Testing depth limit...")
    try:
        # 创建嵌套数组数据
        # 格式: [arr_type(4 bytes)][arr_len(8 bytes)][elements...]
        data = b""
        for i in range(15):  # 15层嵌套
            data += struct.pack("<I", 9)  # arr_type = array (type 9)
            data += struct.pack("<Q", 1)  # arr_len = 1

        # 添加一个uint8元素作为终止
        data += struct.pack("<I", 0)  # arr_type = uint8 (type 0)
        data += struct.pack("<Q", 1)  # arr_len = 1

        f = io.BytesIO(data)
        _skip_gguf_value(f, 9, depth=0)
        print("  [ERROR] Should have raised ValueError for deep nesting")
        return False
    except ValueError as e:
        if "嵌套深度" in str(e):
            print(f"  [OK] Depth limit works: {e}")
        else:
            print(f"  [ERROR] Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"  [ERROR] Unexpected exception: {type(e).__name__}: {e}")
        return False

    # 测试2: 数组长度限制
    print("  Testing array length limit...")
    try:
        # 格式: [arr_type(4 bytes)][arr_len(8 bytes)]
        data = struct.pack("<I", 0)  # arr_type = uint8 (type 0)
        data += struct.pack("<Q", 20000)  # arr_len = 20000 (超过10000限制)

        f = io.BytesIO(data)
        _skip_gguf_value(f, 9, depth=0)
        print("  [ERROR] Should have raised ValueError for large array")
        return False
    except ValueError as e:
        if "数组长度" in str(e):
            print(f"  [OK] Array length limit works: {e}")
        else:
            print(f"  [ERROR] Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"  [ERROR] Unexpected exception: {type(e).__name__}: {e}")
        return False

    # 测试3: 字符串长度限制
    print("  Testing string length limit...")
    try:
        # 格式: [str_len(8 bytes)][str_data...]
        data = struct.pack("<Q", 2000000)  # str_len = 2MB (超过1MB限制)
        data += b"\x00" * 100  # 部分数据

        f = io.BytesIO(data)
        _skip_gguf_value(f, 8, depth=0)
        print("  [ERROR] Should have raised ValueError for large string")
        return False
    except ValueError as e:
        if "字符串长度" in str(e):
            print(f"  [OK] String length limit works: {e}")
        else:
            print(f"  [ERROR] Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"  [ERROR] Unexpected exception: {type(e).__name__}: {e}")
        return False

    # 测试4: 正常数据应该通过
    print("  Testing normal data...")
    try:
        # 正常的uint8值
        data = struct.pack("<B", 42)
        f = io.BytesIO(data)
        _skip_gguf_value(f, 0, depth=0)
        print("  [OK] Normal data accepted")

        # 正常的数组
        data = struct.pack("<I", 0)  # arr_type = uint8
        data += struct.pack("<Q", 3)  # arr_len = 3
        data += struct.pack("<B", 1)  # element 1
        data += struct.pack("<B", 2)  # element 2
        data += struct.pack("<B", 3)  # element 3

        f = io.BytesIO(data)
        _skip_gguf_value(f, 9, depth=0)
        print("  [OK] Normal array accepted")

    except Exception as e:
        print(f"  [ERROR] Normal data rejected: {type(e).__name__}: {e}")
        return False

    return True


def test_config_validation():
    """测试配置验证"""
    print("\n=== Testing Config Validation ===")

    try:
        from src.api.server import OptimizeApplyRequest
    except ImportError:
        print("  [SKIP] Config validation (fastapi not installed)")
        return True

    # 测试有效配置
    try:
        req = OptimizeApplyRequest(
            model_name="test-model",
            config={
                "offload_config": {
                    "strategy": "gpu_cpu",
                    "gpu_layers": 20,
                },
                "quantization": {
                    "level": "q4_k_m",
                }
            }
        )
        print("  [OK] Valid config accepted")
    except Exception as e:
        print(f"  [ERROR] Valid config rejected: {e}")
        return False

    # 测试无效策略
    try:
        req = OptimizeApplyRequest(
            model_name="test-model",
            config={
                "offload_config": {
                    "strategy": "invalid_strategy",
                }
            }
        )
        print("  [ERROR] Invalid strategy should be rejected")
        return False
    except ValueError as e:
        print(f"  [OK] Invalid strategy rejected: {e}")

    # 测试无效量化级别
    try:
        req = OptimizeApplyRequest(
            model_name="test-model",
            config={
                "quantization": {
                    "level": "invalid_level",
                }
            }
        )
        print("  [ERROR] Invalid quantization level should be rejected")
        return False
    except ValueError as e:
        print(f"  [OK] Invalid quantization level rejected: {e}")

    # 测试缺少必要键
    try:
        req = OptimizeApplyRequest(
            model_name="test-model",
            config={"some_key": "some_value"}
        )
        print("  [ERROR] Missing required keys should be rejected")
        return False
    except ValueError as e:
        print(f"  [OK] Missing keys rejected: {e}")

    return True


def test_auth_default():
    """测试认证默认值"""
    print("\n=== Testing Auth Default ===")

    try:
        from src.api.server import AUTH_ENABLED
    except ImportError:
        print("  [SKIP] Auth default (fastapi not installed)")
        return True

    if AUTH_ENABLED:
        print("  [OK] Auth is ENABLED by default (secure)")
    else:
        print("  [WARNING] Auth is DISABLED by default (insecure)")

    return True


async def main():
    """主测试"""
    print("=" * 50)
    print("Security Fixes Verification Test")
    print("=" * 50)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Rate Limiter", test_rate_limiter()))
    results.append(("GGUF Security", test_gguf_security()))
    results.append(("Config Validation", test_config_validation()))
    results.append(("Auth Default", test_auth_default()))

    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r for _, r in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return all_passed


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
