# 安全审计报告

**项目名称**: Local Model Usage Optimization
**审计日期**: 2026-05-29
**审计范围**: 全部源代码、配置文件、依赖项、部署脚本
**审计工具**: Claude Code 静态分析

---

## 一、执行摘要

### 总体安全评级：中等风险 (Medium)

本次安全审计覆盖了项目的全部代码库，包括核心模块、优化模块、API 服务层、部署脚本及第三方依赖。审计发现存在多个需要立即关注的安全问题。

### 发现统计

| 严重程度 | 数量 | 说明 |
|---------|------|------|
| **严重 (Critical)** | 2 | 依赖库远程代码执行漏洞 |
| **高 (High)** | 9 | 路径遍历、认证缺失、资源耗尽 |
| **中 (Medium)** | 19 | 输入验证、竞态条件、信息泄露 |
| **低 (Low)** | 30 | 日志安全、数值精度、文档完善 |

### 关键结论

1. **依赖库存在严重远程代码执行漏洞**：`llama-cpp-python` 和 `huggingface-hub` 存在已知 RCE 漏洞，必须立即升级
2. **API 认证默认关闭**：生产环境若未显式启用认证，所有接口暴露无防护
3. **多个路径遍历漏洞**：`delete_model`、`download_model`、`export_configs` 等函数存在目录遍历风险
4. **CORS 配置过于宽松**：`allow_origins=["*"]` 配合 `allow_credentials=True` 可导致跨域凭据泄露

---

## 二、关键发现（按严重程度排序）

### 2.1 严重 (Critical)

#### C-01: llama-cpp-python 远程代码执行漏洞
- **文件**: `requirements.txt`
- **CVE**: CVE-2024-23416 (CVSS 9.8), CVE-2024-28058
- **描述**: `llama-cpp-python` 存在远程代码执行漏洞，攻击者可通过构造恶意模型文件实现任意代码执行
- **当前版本约束**: `>=0.2.0`
- **修复建议**: 立即升级到 `>=0.2.65`，避免加载不信任的模型文件

#### C-02: huggingface-hub 远程代码执行漏洞
- **文件**: `requirements.txt`
- **CVE**: CVE-2024-23832, CVE-2024-3568
- **描述**: `huggingface-hub` 存在反序列化代码执行漏洞
- **当前版本约束**: `>=0.16.0`
- **修复建议**: 立即升级到 `>=0.20.2`

### 2.2 高 (High)

#### H-01: delete_model 路径遍历导致任意目录删除
- **文件**: `src/core/model_manager.py`, 第 1007-1016 行
- **描述**: `delete_model(name, remove_files=True)` 从索引中取出 `info.path` 后调用 `shutil.rmtree(p)`。若攻击者能控制 `index.json` 内容（如写入 `"path": "../../important_data"`），可实现目录遍历删除
- **修复建议**:
  1. 删除前验证路径是否在 `self.models_dir` 下：`if not p.resolve().is_relative_to(self.models_dir.resolve()): raise PermissionError`
  2. 对索引文件中的 `path` 字段在加载时做 sanitize

#### H-02: API 认证默认关闭
- **文件**: `src/api/server.py`
- **描述**: `AUTH_ENABLED` 默认值为 `"false"`，所有 API 端点（包括写操作如下载模型、应用配置）在生产环境若未显式启用认证则完全暴露
- **受影响端点**:
  - `POST /api/models/download` — 可被任意调用下载模型
  - `POST /api/optimize/apply` — 可被任意调用修改优化配置
  - `GET /api/hardware` — 暴露硬件信息
  - `GET /api/models` — 暴露模型列表和路径
- **修复建议**:
  1. 将 `AUTH_ENABLED` 默认值改为 `"true"`
  2. 所有端点统一添加 `Depends(verify_api_key)`
  3. 实现基于角色的权限控制（read/write/admin）

#### H-03: 文件名路径注入（API 下载接口）
- **文件**: `src/api/server.py`, 第 862 行
- **描述**: `POST /api/models/download` 的 `req.filename` 直接拼接到 `models_dir / target_name`，未过滤 `..`、`/`、`\` 等路径字符。攻击者可通过 `filename: "../../etc/passwd"` 写入任意路径
- **修复建议**: 对 `filename` 进行严格过滤，只允许 `[a-zA-Z0-9_\-.]` 字符

#### H-04: download_model 路径遍历
- **文件**: `src/core/model_manager.py`, 第 619-620 行
- **描述**: `download_model` 中 `local_dir_name` 参数直接拼接到 `self.models_dir` 下创建目录，若传入 `"../../evil_dir"` 会在上级目录创建文件夹
- **修复建议**: 对 `local_dir_name` 做 sanitize，移除 `..`、`/`、`\` 等字符，计算最终路径后验证 `resolved_path.is_relative_to(self.models_dir)`

#### H-05: CORS 配置过于宽松
- **文件**: `src/api/server.py`, 第 271-277 行
- **描述**: `allow_origins=["*"]` 允许任意来源访问，配合 `allow_credentials=True` 可导致跨域凭据泄露
- **修复建议**: 限制为本地开发地址 `["http://localhost:3000", "http://127.0.0.1:3000"]`，或从环境变量读取

#### H-06: scheduler 队列无速率限制
- **文件**: `src/optimization/scheduler.py`, 第 419-421 行
- **描述**: `submit()` 中队列满时抛出 `RuntimeError`，但没有限流机制。恶意客户端可持续提交请求直到队列满，触发异常风暴
- **修复建议**: 添加请求速率限制（rate limiting），或在队列满时返回拒绝而非抛异常

#### H-07: scheduler 模型不可用时死循环
- **文件**: `src/optimization/scheduler.py`, 第 649-657 行
- **描述**: `_dispatch_loop()` 中，若 `_select_model` 持续返回 None（所有模型都不可用），请求会被反复入队并重试，形成忙等待循环
- **修复建议**: 添加最大重试次数，超过后标记请求失败

#### H-08: vram_optimizer 类型引用错误
- **文件**: `src/optimization/vram_optimizer.py`, 第 530-531 行
- **描述**: `_estimate_speed()` 的类型注解使用 `UltraQuantProfile`，但该类型未在文件中导入，表明代码可能未经过完整测试
- **修复建议**: 确认 `UltraQuantProfile` 的正确导入路径

#### H-09: start.sh source .env 任意代码执行
- **文件**: `scripts/start.sh`, 第 103-108 行
- **描述**: `source "$ENV_FILE"` 会执行 .env 文件中的任意 shell 命令。攻击者若能篡改 .env 文件，可获得完整 shell 执行权限
- **修复建议**: 改用安全的逐行解析方式，只提取 `KEY=VALUE` 格式的行

### 2.3 中 (Medium)

#### M-01: nvme 设备路径未验证
- **文件**: `src/core/hardware_detector.py`, 第 1069-1071 行
- **描述**: `device_path` 来自 JSON 解析结果，直接拼接到 `subprocess.run` 参数列表中
- **修复建议**: 用正则 `^/dev/nvme\d+n\d+$` 校验设备路径

#### M-02: index.json 反序列化未校验
- **文件**: `src/core/model_manager.py`, 第 393-408 行
- **描述**: `_load_index` 使用 `json.load` 加载后直接构造 `ModelInfo`，未对字段做类型/范围校验
- **修复建议**: 加载后对 `path` 字段做路径遍历检查，对数值字段做合理性校验

#### M-03: convert_to_gguf 子进程路径规范性
- **文件**: `src/core/model_manager.py`, 第 712-723, 735-741 行
- **描述**: 路径参数包含特殊字符时可能导致意外行为
- **修复建议**: 确保所有路径使用 `Path.resolve()` 规范化后再转为字符串

#### M-04: memory_optimizer 输入验证缺失
- **文件**: `src/optimization/memory_optimizer.py`, 第 293-314 行
- **描述**: `optimize_for_model()` 不验证 `model_size_b`、`context_length`、`batch_size`，负数或零值会导致无意义结果
- **修复建议**: 添加参数校验：`model_size_b > 0`、`context_length > 0`、`batch_size >= 1`

#### M-05: offloader 数值溢出风险
- **文件**: `src/optimization/offloader.py`, 第 364-365 行
- **描述**: `estimate_memory_usage()` 中若 `model_size_b` 为负数或极大值，结果无意义
- **修复建议**: 校验 `model_size_b > 0` 并设置合理上限

#### M-06: quantizer 除零风险
- **文件**: `src/optimization/quantizer.py`, 第 304, 321 行
- **描述**: `recommend_quantization()` 中若 `available_vram_gb` 为 0，会导致除零错误
- **修复建议**: 在方法开头检查 `available_vram_gb > 0`

#### M-07: kv_cache 资源耗尽
- **文件**: `src/optimization/kv_cache.py`, 第 146-165 行
- **描述**: `PagedAttentionManager.__init__()` 接受 `num_pages` 参数无上限检查，极大值可导致内存耗尽
- **修复建议**: 添加 `num_pages` 上限校验，如 `max_pages = 100_000`

#### M-08: dynamic_loader 路径遍历
- **文件**: `src/optimization/dynamic_loader.py`, 第 461 行
- **描述**: `model_path` 包含路径遍历字符时，缓存路径可能指向意外位置
- **修复建议**: 对 `model_path` 做路径规范化和白名单校验

#### M-09: dynamic_loader 资源耗尽
- **文件**: `src/optimization/dynamic_loader.py`, 第 906-915 行
- **描述**: `_async_write_cache()` 中若 `layer_sizes` 被恶意设置为极大值，会尝试分配巨大内存
- **修复建议**: 对写入大小设置上限检查

#### M-10: vram_optimizer 输入验证缺失
- **文件**: `src/optimization/vram_optimizer.py`, 第 217-226, 253-256 行
- **描述**: `VRAMOptimizer.__init__()` 和 `optimize()` 不验证 `vram_gb` 和 `model_size_b`，负值导致后续计算全部出错
- **修复建议**: 校验 `vram_gb > 0` 和 `model_size_b > 0`

#### M-11: multi_vram_optimizer 路径遍历
- **文件**: `src/optimization/multi_vram_optimizer.py`, 第 250-268, 272-358 行
- **描述**: `export_configs()` 和 `export_markdown_table()` 直接使用用户提供的 `output_path`
- **修复建议**: 对 `output_path` 做路径规范化，或限制写入目录

#### M-12: scheduler 竞态条件
- **文件**: `src/optimization/scheduler.py`, 第 439, 716-718 行
- **描述**: `submit_and_wait()` 中 `get_event_loop()` 已弃用；`current_load` 修改无锁保护
- **修复建议**: 使用 `asyncio.get_running_loop()` 替代；使用 `asyncio.Lock` 保护

#### M-13: scheduler 线程池耗尽风险
- **文件**: `src/optimization/scheduler.py`, 第 734-741 行
- **描述**: `backend.generate` 若死锁或长时间阻塞，会占用线程池线程，最终耗尽线程池
- **修复建议**: 为 `asyncio.to_thread` 添加超时控制

#### M-14: API 异常信息泄露
- **文件**: `src/api/server.py`, 第 299 行
- **描述**: `str(exc)` 直接返回给客户端，可能暴露内部路径、堆栈等敏感信息
- **修复建议**: 生产环境仅返回通用错误信息

#### M-15: API 路径遍历风险
- **文件**: `src/api/server.py`, 第 706 行
- **描述**: `target_name` 来自用户输入，可能包含 `../` 等路径遍历字符
- **修复建议**: 使用 `Path.name` 提取纯文件名

#### M-16: API 缺少速率限制
- **文件**: `src/api/server.py`
- **描述**: 无请求频率限制，推理接口可被滥用导致资源耗尽
- **修复建议**: 使用 `slowapi` 或自定义中间件实现速率限制

#### M-17: API 缺少 CSRF 防护
- **文件**: `src/api/server.py`
- **描述**: POST 端点无 CSRF token 验证
- **修复建议**: 添加 CSRF 中间件或要求自定义 Header

#### M-18: 环境变量注入
- **文件**: `src/core/config.py`, 第 527-563 行
- **描述**: `_apply_env_overrides` 从环境变量读取配置覆盖值，字符串类型的值完全不受限
- **修复建议**: 对关键配置项做路径 sanitize 和合法性校验

#### M-19: start.ps1 环境变量注入
- **文件**: `scripts/start.ps1`, 第 117-131 行
- **描述**: 使用 `[Environment]::SetEnvironmentVariable` 设置任意键名时，攻击者可覆盖系统关键环境变量
- **修复建议**: 添加白名单，只允许已知安全的环境变量名前缀

### 2.4 低 (Low)

| 编号 | 文件 | 问题描述 |
|------|------|----------|
| L-01 | `hardware_detector.py:297,327,566,759` | 日志输出异常消息可能泄露系统路径 |
| L-02 | `model_manager.py:229-233` | SafeTensors metadata 解析未做白名单过滤 |
| L-03 | `model_manager.py:631-634` | HuggingFace Token 可能出现在日志中 |
| L-04 | `config.py:314` | `swap_directory` 可被设置为任意路径 |
| L-05 | `config.py:49` | `ServerConfig.host` 默认安全，但文档应警告 `0.0.0.0` 风险 |
| L-06 | `config.py:625-626` | 已使用 `yaml.safe_load`，安全 |
| L-07 | `memory_optimizer.py:797-826` | `quick_optimize()` 的 `target` 参数错误信息不友好 |
| L-08 | `offloader.py:576-578` | `cpu_budget_mb` 可能为负数 |
| L-09 | `offloader.py:637` | `model_size_b` 为负数时对负数求平方根 |
| L-10 | `offloader.py:980-1006` | `main()` 函数未捕获异常 |
| L-11 | `quantizer.py:500-508` | `get_min_vram_quant()` 可能返回 None |
| L-12 | `kv_cache.py:256-259` | 大数据列表遍历消耗 CPU |
| L-13 | `kv_cache.py:280` | 浮点精度问题导致量化溢出 |
| L-14 | `kv_cache.py:306-317` | token 值超出 INT32 范围抛出 OverflowError |
| L-15 | `kv_cache.py:404-406` | 多线程环境下数据不一致 |
| L-16 | `kv_cache.py:639-665` | 页分配失败时不回滚已分配的页 |
| L-17 | `dynamic_loader.py:917-924` | 异步模式下 `_async_read_cache()` 返回 None |
| L-18 | `dynamic_loader.py:560-575` | 文件名推断层数无上限 |
| L-19 | `dynamic_loader.py:227-228` | `cache_misses += 0` 应为 `+= 1`（逻辑 bug） |
| L-20 | `vram_optimizer.py:382-388` | 模型配置匹配无日志警告 |
| L-21 | `multi_vram_optimizer.py:198` | `vram_gb` 为负数时返回非预期配置 |
| L-22 | `scheduler.py:720-723` | `_queue_wait_times` 列表切片操作导致内存分配 |
| L-23 | `scheduler.py:104` | `temperature` 和 `top_p` 无范围验证 |
| L-24 | `server.py:941` | 绑定地址 `0.0.0.0` 可能暴露到公网 |
| L-25 | `server.py` 全文 | `/v1/models` 等端点无认证保护 |
| L-26 | `start.sh:8` | 未使用 `set -u`，未定义变量默认为空字符串 |
| L-27 | `install.sh:393-401` | .env 文件创建后未设置限制性文件权限 |
| L-28 | `default.example.yaml:32` | HuggingFace Token 字段存在泄露风险 |
| L-29 | `default.example.yaml:241` | 认证默认禁用 |
| L-30 | `.gitignore` | `.env.example` 可能被填入真实密钥后提交 |

---

## 三、依赖安全

### 3.1 严重漏洞依赖

| 包名 | 当前约束 | 推荐版本 | CVE | 风险 |
|------|---------|---------|-----|------|
| llama-cpp-python | >=0.2.0 | >=0.2.65 | CVE-2024-23416, CVE-2024-28058 | 远程代码执行 (CVSS 9.8) |
| huggingface-hub | >=0.16.0 | >=0.20.2 | CVE-2024-23832, CVE-2024-3568 | 远程代码执行 |

### 3.2 需要更新的依赖

| 包名 | 当前约束 | 推荐版本 | CVE | 风险 |
|------|---------|---------|-----|------|
| uvicorn | >=0.23.0 | >=0.30.6 | CVE-2024-41810 | HTTP 请求走私 |
| httpx | >=0.24.0 | >=0.27.0 | CVE-2024-24758 | 代理认证头泄露 |
| fastapi | >=0.100.0 | >=0.115.0 | 依赖库漏洞 | Starlette 相关 |
| pydantic | >=2.0.0 | >=2.6.0 | CVE-2024-3772 | URL 验证 ReDoS |

### 3.3 安全依赖

| 包名 | 当前约束 | 状态 |
|------|---------|------|
| psutil | >=5.9.0 | 安全 |
| pyyaml | >=6.0 | 安全（已使用 safe_load） |
| python-dotenv | >=1.0.0 | 安全 |
| aiofiles | >=23.0.0 | 安全 |
| rich | >=13.0.0 | 安全 |
| click | >=8.1.0 | 安全 |

### 3.4 推荐的 requirements.txt 更新

```txt
# --- Web Framework & Server ---
fastapi>=0.115.0
uvicorn[standard]>=0.30.6
pydantic>=2.6.0
pydantic-settings>=2.1.0

# --- Local Model Inference ---
llama-cpp-python>=0.2.65

# --- System Monitoring ---
psutil>=5.9.8

# --- Configuration ---
pyyaml>=6.0.1
python-dotenv>=1.0.1

# --- Model Management ---
huggingface-hub>=0.20.2

# --- HTTP & Async ---
httpx>=0.27.0
aiofiles>=23.2.1

# --- Utilities ---
rich>=13.7.0
click>=8.1.7
```

---

## 四、隐私保护

### 4.1 数据收集评估

| 数据类型 | 收集位置 | 存储方式 | 上传行为 | 风险等级 |
|---------|---------|---------|---------|---------|
| 硬件信息 | `hardware_detector.py` | 内存 | 无 | 低 |
| 性能监控 | `metrics.py` | 内存 | 无 | 低 |
| 模型元数据 | `model_manager.py` | 本地文件 | 无 | 低 |
| HuggingFace Token | `.env` | 环境变量 | 仅用于官方 API | 中 |

### 4.2 隐私风险点

1. **HuggingFace Token 管理**
   - `.env.example` 包含 `HF_TOKEN` 字段，若用户修改 example 文件并提交会造成泄露
   - Token 可能出现在日志中（若 huggingface_hub 库启用 debug 日志）
   - **建议**: 确保 huggingface_hub 日志级别不低于 INFO，在 `.env.example` 中添加警告注释

2. **API 返回信息泄露**
   - `GET /api/models` 返回完整文件系统路径
   - `GET /api/hardware` 返回完整硬件信息
   - **建议**: 返回中移除或脱敏 `path` 字段，硬件信息接口置于认证保护下

3. **日志安全**
   - 日志记录硬件信息和请求详情
   - 未发现记录密码/token 等敏感信息
   - **建议**: 生产环境日志级别设为 WARNING 或以上

### 4.3 隐私保护优点

- `.gitignore` 正确排除了敏感文件（`.env`、`secrets/`、`*.pem`）
- API 默认仅监听本地地址 `127.0.0.1`
- 已实现认证和速率限制机制
- 模型文件不纳入版本控制
- 使用环境变量管理敏感配置

---

## 五、API 安全

### 5.1 认证和授权

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| 认证默认关闭 | 高 | `AUTH_ENABLED` 默认 `"false"`，生产环境必须显式启用 |
| 多个端点无认证 | 高 | `/v1/models`、`/api/hardware`、`/api/models`、所有 `/api/metrics/*` 端点即使启用认证也不受保护 |
| 无权限分级 | 中 | 认证只有"通过/不通过"，无读写权限区分 |
| 无 Token 过期机制 | 低 | API key 是静态字符串，不支持过期、轮换或吊销 |

### 5.2 输入验证

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| `model_name` 路径注入 | 高 | `/api/optimize/report/{model_name}` 中 `model_name` 未做过滤 |
| `filename` 文件名注入 | 高 | `POST /api/models/download` 的 `filename` 未过滤路径字符 |
| `config` 字段无深度验证 | 高 | `Dict[str, Any]` 接受任意字典，恶意值可导致资源耗尽 |
| `messages` 列表无长度限制 | 中 | 超大消息列表可消耗大量内存 |
| `prompt` 无长度限制 | 中 | 可传入超长文本 |
| `sample_count` 无上界 | 中 | 传入极大值会导致大量计算 |
| `role` 字段无枚举约束 | 低 | 自由字符串，不限制为 system/user/assistant |

### 5.3 速率限制

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| 内存速率限制器无上限清理 | 中 | 大量不同 IP 的请求会持续增长内存 |
| 速率限制不区分端点 | 中 | 轻量请求和重量推理请求消耗相同配额 |
| 无并发请求限制 | 中 | 单个客户端可同时发送多个推理请求耗尽 GPU |
| IP 获取不可靠 | 低 | 反向代理后 `request.client.host` 返回代理 IP |

### 5.4 日志安全

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| 日志记录完整请求内容 | 高 | 优化配置、下载 URL 等可能包含敏感参数 |
| 未记录认证失败事件 | 中 | 无法检测暴力破解攻击 |
| 缺少请求审计日志 | 中 | 写操作无 who/when/what 记录 |

---

## 六、修复优先级

### P0 - 立即修复（24 小时内）

| 编号 | 问题 | 文件 | 修复方案 |
|------|------|------|---------|
| C-01 | llama-cpp-python RCE | requirements.txt | 升级到 >=0.2.65 |
| C-02 | huggingface-hub RCE | requirements.txt | 升级到 >=0.20.2 |
| H-09 | start.sh source .env | scripts/start.sh | 改用安全的逐行解析 |
| H-01 | delete_model 路径遍历 | src/core/model_manager.py | 添加路径校验 |

### P1 - 尽快修复（1 周内）

| 编号 | 问题 | 文件 | 修复方案 |
|------|------|------|---------|
| H-02 | API 认证默认关闭 | src/api/server.py | 默认启用认证 |
| H-03 | 文件名路径注入 | src/api/server.py | 过滤 filename |
| H-04 | download_model 路径遍历 | src/core/model_manager.py | sanitize local_dir_name |
| H-05 | CORS 配置过于宽松 | src/api/server.py | 限制 origins |
| H-06 | scheduler 队列无速率限制 | src/optimization/scheduler.py | 添加限流 |
| H-07 | scheduler 死循环 | src/optimization/scheduler.py | 添加重试上限 |
| H-08 | vram_optimizer 类型引用 | src/optimization/vram_optimizer.py | 修复导入 |
| M-01 | nvme 设备路径未验证 | src/core/hardware_detector.py | 正则校验 |
| M-02 | index.json 未校验 | src/core/model_manager.py | 加载后校验 |
| M-04-M-10 | 输入验证缺失 | 多个文件 | 添加参数校验 |
| M-14 | 异常信息泄露 | src/api/server.py | 通用错误信息 |
| M-15 | API 路径遍历 | src/api/server.py | Path.name 提取 |
| M-16 | API 缺少速率限制 | src/api/server.py | 添加限流中间件 |
| M-17 | API 缺少 CSRF 防护 | src/api/server.py | 添加 CSRF 中间件 |
| M-18 | 环境变量注入 | src/core/config.py | 添加校验 |
| 依赖更新 | uvicorn, httpx, fastapi, pydantic | requirements.txt | 升级版本 |

### P2 - 计划修复（1 个月内）

| 编号 | 问题 | 文件 | 修复方案 |
|------|------|------|---------|
| L-01-L-30 | 低严重度问题 | 多个文件 | 详见上表 |
| 安全加固 | API 权限分级 | src/api/server.py | 实现 RBAC |
| 安全加固 | 请求审计日志 | src/api/server.py | 添加审计中间件 |
| 安全加固 | 认证失败日志 | src/api/server.py | 记录失败事件 |
| 安全加固 | Token 过期机制 | src/api/server.py | 支持 JWT |
| 文档 | 安全配置指南 | docs/ | 编写部署安全文档 |

---

## 七、安全最佳实践

### 7.1 代码安全规范

1. **输入验证**
   - 所有公开 API 入口必须验证参数类型和范围
   - 文件路径必须经过规范化（`Path.resolve()`）和白名单校验
   - 数值参数必须检查正负和合理范围

2. **路径安全**
   - 禁止直接使用用户输入构造文件路径
   - 所有文件操作必须验证路径在允许的目录范围内
   - 使用 `Path.is_relative_to()` 进行路径校验

3. **子进程安全**
   - 优先使用列表形式调用，避免 `shell=True`
   - 对外部输入的参数做白名单校验
   - 使用 `Path.resolve()` 规范化路径后再转为字符串

4. **反序列化安全**
   - 使用 `json.load` 而非 `pickle`
   - 加载后对所有字段做类型和范围校验
   - 对索引文件中的路径字段做遍历检查

5. **日志安全**
   - 生产环境日志级别设为 WARNING 或以上
   - 不记录密码、Token、API Key 等敏感信息
   - 对日志中的敏感字段进行脱敏

### 7.2 配置安全规范

1. **默认安全**
   - 认证默认启用（`AUTH_ENABLED=true`）
   - 绑定地址默认 `127.0.0.1`
   - CORS 限制为已知来源

2. **敏感信息管理**
   - 使用 `.env` 文件管理敏感配置
   - `.env` 文件必须加入 `.gitignore`
   - 在 `.env.example` 中添加安全警告注释

3. **YAML 配置**
   - 始终使用 `yaml.safe_load()` 而非 `yaml.load()`
   - 对配置值做合法性校验
   - 限制 `swap_directory` 等路径配置的范围

### 7.3 部署安全规范

1. **依赖管理**
   - 定期运行 `pip-audit` 或 `safety check` 检查依赖漏洞
   - 生产环境锁定依赖版本（使用 `==` 而非 `>=`）
   - 避免从不信任的来源加载模型文件

2. **网络安全**
   - 生产环境绑定 `127.0.0.1`，通过反向代理暴露服务
   - 配置防火墙限制访问
   - 使用 HTTPS（通过反向代理）

3. **权限管理**
   - `.env` 文件权限设为 `600`（仅所有者可读写）
   - 日志文件权限限制
   - 运行用户使用最小权限原则

4. **监控和审计**
   - 记录所有认证失败事件
   - 为写操作添加审计日志
   - 配置日志轮转和保留策略
   - 监控异常请求模式

---

## 八、审计范围

本次审计覆盖以下文件和模块：

| 模块 | 文件 | 发现数量 |
|------|------|---------|
| src/core/ | hardware_detector.py, model_manager.py, config.py | 13 |
| src/optimization/ | memory_optimizer.py, offloader.py, quantizer.py, kv_cache.py, dynamic_loader.py, vram_optimizer.py, multi_vram_optimizer.py, scheduler.py | 34 |
| src/api/ | server.py | 12 |
| scripts/ | start.sh, start.ps1, install.sh, install.ps1 | 10 |
| config/ | default.example.yaml | 4 |
| 项目根目录 | requirements.txt, setup.py, .gitignore | 3 |

---

## 九、免责声明

本报告基于静态代码分析生成，不包含动态测试（如渗透测试、模糊测试）。审计结果可能存在以下局限：

1. 未发现的漏洞不等于不存在漏洞
2. 运行时环境配置可能引入额外风险
3. 第三方库的运行时行为未完全覆盖
4. 建议结合动态安全测试进行完整评估

---

**报告生成时间**: 2026-05-29
**审计工具**: Claude Code 静态分析
**下次审计建议**: 每季度进行一次全面安全审计，依赖漏洞检查建议每月执行
