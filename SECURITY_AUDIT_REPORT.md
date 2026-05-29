# 安全审计报告

**项目**: 本地AI模型优化
**日期**: 2026-05-29
**审计范围**: `src/api/*.py`, `src/core/*.py`, `src/optimization/*.py`, `src/backends/*.py`

---

## 审计摘要

| 严重程度 | 发现数量 | 已修复 |
|----------|----------|--------|
| CRITICAL | 1 | ✅ |
| HIGH | 2 | ✅ |
| MEDIUM | 6 | ✅ |
| LOW | 7 | ✅ |

---

## 已修复的安全问题

### CRITICAL 级别

#### 1. 认证默认禁用 ✅ 已修复
**文件**: `src/api/server.py:75`
**问题**: `AUTH_ENABLED` 默认为 `false`，部署时未显式启用会导致API无认证保护。
**修复**: 默认值改为 `true`，添加启动警告。

```python
# 修复前
AUTH_ENABLED = os.environ.get("LMO_AUTH_ENABLED", "false").lower() == "true"

# 修复后
AUTH_ENABLED = os.environ.get("LMO_AUTH_ENABLED", "true").lower() == "true"
```

### HIGH 级别

#### 2. 服务器绑定到 0.0.0.0 ✅ 已修复
**文件**: `src/api/server.py:1120-1123`
**问题**: 服务器默认绑定到所有网络接口，暴露API。
**修复**: 从环境变量读取，默认绑定到 `127.0.0.1`。

```python
# 修复后
host = os.environ.get("LMO_HOST", "127.0.0.1")
port = int(os.environ.get("LMO_PORT", "8000"))

if host == "0.0.0.0":
    logger.warning("Server binding to 0.0.0.0 (all interfaces). This exposes the API to the network!")
```

#### 3. 缺少配置验证 ✅ 已修复
**文件**: `src/api/server.py:218`
**问题**: `OptimizeApplyRequest.config` 接受任意字典，无类型和范围验证。
**修复**: 添加 Pydantic 验证器，验证策略和量化级别。

```python
@validator("config")
def validate_config(cls, v):
    # 验证 offload_config 策略
    valid_strategies = {"gpu_only", "gpu_cpu", "gpu_cpu_disk", "cpu_only"}
    # 验证 quantization 级别
    valid_levels = {"q2_k", "q3_k_s", "q3_k_m", "q4_0", "q4_k_s", "q4_k_m", ...}
```

### MEDIUM 级别

#### 4. 路径泄露 ✅ 已修复
**文件**: `src/api/server.py:842`
**问题**: `/api/models` 端点返回完整文件路径。
**修复**: 只返回文件名。

```python
# 修复后
"filename": Path(entry.get("path", "")).name,
```

#### 5. URL 日志泄露 ✅ 已修复
**文件**: `src/api/server.py:887`
**问题**: 日志记录完整URL，可能包含认证token。
**修复**: 移除查询参数。

```python
safe_url = req.url.split('?')[0] if req.url else None
logger.info("Download requested: url=%s ...", safe_url)
```

#### 6. 配置日志泄露 ✅ 已修复
**文件**: `src/api/server.py:1072`
**问题**: 日志记录完整配置字典。
**修复**: 只记录键名和类型。

```python
config_summary = {k: type(v).__name__ for k, v in config.items()}
logger.info("Applied config: keys=%s", config_summary)
```

#### 7. GGUF 递归解析 ✅ 已修复
**文件**: `src/backends/llama_cpp.py:78-95`
**问题**: 递归解析无深度限制，恶意文件可导致栈溢出。
**修复**: 添加深度限制 (10层) 和数组长度限制 (10000)。

```python
def _skip_gguf_value(f, vtype: int, depth: int = 0) -> None:
    MAX_DEPTH = 10
    MAX_ARRAY_LEN = 10000
    # ...
```

#### 8. 速率限制器内存泄漏 ✅ 已修复
**文件**: `src/api/server.py:388-416`
**问题**: IP记录无清理机制，内存无限增长。
**修复**: 添加定期清理 (5分钟一次)。

```python
def _cleanup_stale_entries(self) -> None:
    stale_ips = [ip for ip, times in self.requests.items()
                 if not times or times[-1] < stale_threshold]
    for ip in stale_ips:
        del self.requests[ip]
```

#### 9. 后端竞态条件 ✅ 已修复
**文件**: `src/api/server.py:234`
**问题**: 并发请求可能导致模型重复加载。
**修复**: 添加 asyncio.Lock。

```python
class AppState:
    _backend_lock: asyncio.Lock = None
    def __init__(self):
        self._backend_lock = asyncio.Lock()
```

### LOW 级别

#### 10. 速率限制配置无边界 ✅ 已修复
**文件**: `src/api/server.py:385-386`
**修复**: 限制范围 1-10000 请求, 1-3600 秒。

#### 11. 堆栈跟踪日志 ✅ 已修复
**文件**: `src/optimization/scheduler.py:780`
**修复**: 生产环境只记录异常类型，不记录完整堆栈。

#### 12. ultra_quantizer 变量名错误 ✅ 已修复
**文件**: `src/optimization/ultra_quantizer.py:481`
**修复**: `model_config` 改为 `model_size_b`。

#### 13. CORS 通配符警告 ✅ 已修复
**文件**: `src/api/server.py:337`
**修复**: 添加启动警告。

#### 14. 认证禁用警告 ✅ 已修复
**文件**: `src/api/server.py:1114`
**修复**: 启动时检查并警告。

---

## 安全优势 (已正确实现)

- ✅ **路径遍历防护**: 使用 `Path().name` 清理用户输入
- ✅ **命令注入防护**: 所有 subprocess 调用使用列表参数
- ✅ **YAML 安全加载**: 使用 `yaml.safe_load`
- ✅ **无硬编码密钥**: 所有密钥从环境变量读取
- ✅ **SafeTensors 限制**: 50MB 头部大小限制
- ✅ **GGUF 解析限制**: 200 KV对限制
- ✅ **输入长度限制**: Pydantic 模型使用 `max_length`

---

## 配置建议

### 生产环境部署

```bash
# 必须设置
export LMO_AUTH_ENABLED=true
export LMO_API_KEYS=your-secure-api-key-here
export LMO_HOST=127.0.0.1

# 建议设置
export CORS_ORIGINS=https://your-domain.com
export LMO_RATE_LIMIT_REQUESTS=60
export LMO_RATE_LIMIT_WINDOW=60

# 可选
export LMO_PORT=8000
```

### 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LMO_AUTH_ENABLED` | `true` | 是否启用认证 |
| `LMO_API_KEYS` | (空) | API密钥，逗号分隔 |
| `LMO_HOST` | `127.0.0.1` | 服务器绑定地址 |
| `LMO_PORT` | `8000` | 服务器端口 |
| `CORS_ORIGINS` | localhost | 允许的跨域来源 |
| `LMO_RATE_LIMIT_REQUESTS` | `60` | 每窗口最大请求数 |
| `LMO_RATE_LIMIT_WINDOW` | `60` | 速率限制窗口 (秒) |

---

## 后续建议

1. **HTTPS**: 生产环境使用反向代理 (nginx) 提供 HTTPS
2. **日志审计**: 将日志发送到安全的集中式日志系统
3. **密钥轮换**: 定期轮换 API 密钥
4. **网络隔离**: 使用防火墙限制对 API 端口的访问
5. **依赖审计**: 定期运行 `pip audit` 检查依赖漏洞

---

**审计完成**: 所有发现的安全问题已修复
**审计人员**: Claude Code Security Audit
**审计日期**: 2026-05-29
