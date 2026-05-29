"""
HTTP API 模块

提供 RESTful API 接口，用于推理请求、模型管理和系统监控。

兼容 OpenAI API:
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models

管理接口:
    GET  /api/hardware
    GET  /api/models
    POST /api/models/download
    GET  /api/metrics/current
    GET  /api/metrics/history
    GET  /api/metrics/bottleneck
    GET  /api/metrics/suggestions

优化接口:
    POST /api/optimize
    GET  /api/optimize/report/{model_name}
    POST /api/optimize/apply
"""

from .server import app

__all__ = [
    "app",
]
