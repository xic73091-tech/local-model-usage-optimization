"""
FastAPI server implementing OpenAI-compatible, management, and optimization APIs.

Endpoints:
    OpenAI-compatible:
        POST /v1/chat/completions
        POST /v1/completions
        GET  /v1/models

    Management:
        GET  /api/hardware
        GET  /api/models
        POST /api/models/download
        GET  /api/metrics/current
        GET  /api/metrics/history
        GET  /api/metrics/bottleneck
        GET  /api/metrics/suggestions

    Optimization:
        POST /api/optimize
        GET  /api/optimize/report/{model_name}
        POST /api/optimize/apply
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator

from src.core.hardware_detector import HardwareDetector
from src.core.model_manager import ModelManager
from src.monitor.analyzer import PerformanceAnalyzer
from src.monitor.metrics import MetricsCollector
from src.optimization.scheduler import (
    InferenceRequest,
    InferenceScheduler,
    ModelProfile,
    Priority,
    TaskType,
)
from src.optimization.memory_optimizer import (
    MemoryOptimizer,
    HardwareProfile as OptHardwareProfile,
    OptimizationProfile,
)
from src.backends.base import InferenceBackend, InferenceConfig
from src.backends.llama_cpp import LlamaCppBackend

logger = logging.getLogger(__name__)


# ================================================================
# Security: API Key Authentication
# ================================================================

# API keys from environment variable (comma-separated)
# Set LMO_API_KEYS=key1,key2,key3 to enable authentication
# If not set or empty, authentication is disabled (local development mode)
API_KEYS: set = set(
    k.strip() for k in os.environ.get("LMO_API_KEYS", "").split(",") if k.strip()
)
# Security: Default to auth ENABLED for production safety
# Set LMO_AUTH_ENABLED=false explicitly to disable in trusted environments
AUTH_ENABLED = os.environ.get("LMO_AUTH_ENABLED", "true").lower() == "true"

security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """
    Verify API key from Authorization header.

    If AUTH_ENABLED is false, authentication is skipped (development mode).
    If AUTH_ENABLED is true, a valid Bearer token is required.

    Returns the API key if valid, None if auth is disabled.
    """
    if not AUTH_ENABLED:
        return None  # Auth disabled - allow all requests

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing Authorization header",
                    "type": "authentication_error",
                    "code": "missing_api_key",
                }
            },
        )

    if credentials.credentials not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
        )

    return credentials.credentials


# ================================================================
# Pydantic models -- OpenAI-compatible request/response schemas
# ================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., max_length=50)
    content: str = Field(..., max_length=100000)
    name: Optional[str] = Field(default=None, max_length=100)


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default", max_length=200)
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    user: Optional[str] = Field(default=None, max_length=100)


class CompletionRequest(BaseModel):
    model: str = Field(default="default", max_length=200)
    prompt: str = Field(default="", max_length=100000)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    suffix: Optional[str] = Field(default=None, max_length=1000)
    user: Optional[str] = Field(default=None, max_length=100)


class ModelDownloadRequest(BaseModel):
    url: Optional[str] = Field(default=None, max_length=2000)
    model_id: Optional[str] = Field(default=None, max_length=200)
    filename: Optional[str] = Field(default=None, max_length=200)


# ================================================================
# Pydantic models -- Management response schemas
# ================================================================

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"
    path: Optional[str] = None
    loaded: bool = False
    memory_mb: float = 0.0


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ================================================================
# Pydantic models -- Optimization request/response schemas
# ================================================================

class OptimizeRequest(BaseModel):
    """Request body for POST /api/optimize."""
    model_name: str = Field(..., max_length=200, description="Model name or identifier")
    model_size_b: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=1000,
        description="Model parameter count in billions (inferred from registry if omitted)",
    )
    target: str = Field(
        default="balanced",
        max_length=50,
        description="Optimization target: minimal_vram / balanced / max_speed / quality",
    )
    context_length: int = Field(default=4096, ge=1, le=1048576)
    batch_size: int = Field(default=1, ge=1, le=1024)
    vram_gb: Optional[float] = Field(
        default=None,
        ge=0,
        le=256,
        description="Override available VRAM in GB (auto-detected if omitted)",
    )
    ram_gb: Optional[float] = Field(
        default=None,
        ge=0,
        le=1024,
        description="Override available RAM in GB (auto-detected if omitted)",
    )


class OffloadConfigModel(BaseModel):
    """Validation model for offload configuration."""
    strategy: str = Field(default="gpu_only", max_length=50)
    gpu_layers: int = Field(default=-1, ge=-1, le=1000)
    cpu_threads: int = Field(default=4, ge=1, le=128)
    context_length: int = Field(default=4096, ge=1, le=1048576)
    batch_size: int = Field(default=1, ge=1, le=1024)


class QuantizationConfigModel(BaseModel):
    """Validation model for quantization configuration."""
    level: str = Field(default="q4_k_m", max_length=50)
    vram_per_b: float = Field(default=0.0, ge=0, le=100)
    quality_score: float = Field(default=0.0, ge=0, le=1)
    speed_score: float = Field(default=0.0, ge=0, le=1)
    bits: int = Field(default=4, ge=1, le=16)


class OptimizeApplyRequest(BaseModel):
    """Request body for POST /api/optimize/apply."""
    model_name: str = Field(..., max_length=200, description="Model name to apply config to")
    config: Dict[str, Any] = Field(..., max_length=10000, description="Optimization config dict from /api/optimize")

    @validator("config")
    def validate_config(cls, v):
        """Validate config structure and values."""
        # Check for required keys
        if "offload_config" not in v and "quantization" not in v:
            raise ValueError("Config must contain 'offload_config' or 'quantization'")

        # Validate offload_config if present
        if "offload_config" in v:
            offload = v["offload_config"]
            if not isinstance(offload, dict):
                raise ValueError("'offload_config' must be a dict")
            # Validate strategy
            valid_strategies = {"gpu_only", "gpu_cpu", "gpu_cpu_disk", "cpu_only"}
            strategy = offload.get("strategy", "gpu_only")
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")
            # Validate numeric bounds
            if "gpu_layers" in offload and not isinstance(offload["gpu_layers"], int):
                raise ValueError("'gpu_layers' must be an integer")

        # Validate quantization if present
        if "quantization" in v:
            quant = v["quantization"]
            if not isinstance(quant, dict):
                raise ValueError("'quantization' must be a dict")
            # Validate level
            valid_levels = {
                "q2_k", "q3_k_s", "q3_k_m", "q4_0", "q4_k_s", "q4_k_m",
                "q5_k_s", "q5_k_m", "q6_k", "q8_0", "fp16"
            }
            level = quant.get("level", "q4_k_m")
            if level not in valid_levels:
                raise ValueError(f"Invalid quantization level: {level}")

        return v


# ================================================================
# Global state -- initialised in lifespan
# ================================================================

class AppState:
    """Holds all shared services."""

    hardware_detector: Optional[HardwareDetector] = None
    metrics_collector: Optional[MetricsCollector] = None
    analyzer: Optional[PerformanceAnalyzer] = None
    scheduler: Optional[InferenceScheduler] = None
    memory_optimizer: Optional[MemoryOptimizer] = None
    model_manager: Optional[ModelManager] = None
    backends: Dict[str, InferenceBackend] = {}
    model_registry: Dict[str, Dict[str, Any]] = {}
    models_dir: Path = Path("models")
    active_configs: Dict[str, Dict[str, Any]] = {}  # model_name -> applied config
    # Lock for thread-safe backend operations
    _backend_lock: asyncio.Lock = None

    def __init__(self):
        self._backend_lock = asyncio.Lock()


app_state = AppState()


# ================================================================
# Lifespan -- startup / shutdown
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    logger.info("Starting server...")

    # Hardware detection (sync, fast)
    app_state.hardware_detector = HardwareDetector(auto_detect=True)
    hw = app_state.hardware_detector.profile
    logger.info("Hardware: GPU=%s  CPU=%s  Memory=%.1fGB",
                hw.gpu.summary, hw.cpu.brand, hw.memory.total_gb)

    # Metrics collector
    app_state.metrics_collector = MetricsCollector(
        history_size=300, collect_interval=1.0
    )
    await app_state.metrics_collector.start()
    logger.info("Metrics collector started")

    # Performance analyzer
    app_state.analyzer = PerformanceAnalyzer(app_state.metrics_collector)

    # Inference scheduler
    app_state.scheduler = InferenceScheduler(
        max_queue_size=1000,
        max_global_concurrency=4,
        model_select_strategy="balanced",
    )
    await app_state.scheduler.start()
    logger.info("Inference scheduler started")

    # Memory optimizer
    app_state.memory_optimizer = MemoryOptimizer()
    logger.info("Memory optimizer initialized")

    # Ensure models directory exists
    app_state.models_dir.mkdir(parents=True, exist_ok=True)

    # Model manager
    app_state.model_manager = ModelManager(models_dir=app_state.models_dir)
    app_state.model_manager.scan_models(deep=False)

    # Auto-discover models in models directory
    _discover_models()

    yield  # ---- server is running ----

    # Shutdown
    logger.info("Shutting down server...")
    if app_state.scheduler:
        await app_state.scheduler.stop()
    if app_state.metrics_collector:
        await app_state.metrics_collector.stop()
    for backend in app_state.backends.values():
        try:
            backend.unload_model()
        except Exception as e:
            logger.warning("Error unloading model: %s", e)
    app_state.backends.clear()
    logger.info("Server stopped")


def _discover_models():
    """Scan models directory and register discovered GGUF models."""
    models_dir = app_state.models_dir
    if not models_dir.exists():
        return
    for gguf_file in models_dir.glob("*.gguf"):
        model_id = gguf_file.stem
        if model_id not in app_state.model_registry:
            app_state.model_registry[model_id] = {
                "id": model_id,
                "path": str(gguf_file),
                "loaded": False,
            }
            logger.info("Discovered model: %s", model_id)


# ================================================================
# FastAPI application
# ================================================================

app = FastAPI(
    title="Local Model Optimization Server",
    description="OpenAI-compatible API with hardware monitoring and optimization",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS -- restrict origins in production
# For local development, set CORS_ORIGINS=["*"] in environment
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
# Security warning for wildcard CORS
if "*" in CORS_ORIGINS:
    logger.warning("CORS allows all origins ('*'). This is INSECURE for production!")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ================================================================
# Error handling middleware
# ================================================================

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Catch unhandled exceptions and return structured JSON errors."""
    request_id = str(uuid.uuid4())
    try:
        response = await call_next(request)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        # Log full error details internally
        logger.exception("Unhandled exception on %s %s [request_id=%s]",
                        request.method, request.url.path, request_id)
        # Return generic error to client (no sensitive info)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                    "code": "internal_error",
                    "request_id": request_id,  # For support reference
                }
            },
        )


# ================================================================
# Rate limiting middleware
# ================================================================

from collections import defaultdict

# Rate limit configuration (from environment or defaults)
# Security: Clamp to reasonable ranges to prevent misconfiguration
RATE_LIMIT_REQUESTS = max(1, min(10000, int(os.environ.get("LMO_RATE_LIMIT_REQUESTS", "60"))))
RATE_LIMIT_WINDOW = max(1, min(3600, int(os.environ.get("LMO_RATE_LIMIT_WINDOW", "60"))))  # seconds

class RateLimiter:
    """Simple in-memory rate limiter per IP address."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes

    def _cleanup_stale_entries(self) -> None:
        """Remove entries for IPs that haven't made requests recently."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        stale_threshold = now - (self.window_seconds * 2)  # Double window for safety

        stale_ips = [
            ip for ip, times in self.requests.items()
            if not times or times[-1] < stale_threshold
        ]
        for ip in stale_ips:
            del self.requests[ip]

        if stale_ips:
            logger.debug("Rate limiter cleanup: removed %d stale IP entries", len(stale_ips))

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds

        # Periodic cleanup of stale entries
        self._cleanup_stale_entries()

        # Clean old entries for this IP
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            return False

        self.requests[client_ip].append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - self.window_seconds
        recent = [t for t in self.requests[client_ip] if t > window_start]
        return max(0, self.max_requests - len(recent))


rate_limiter = RateLimiter(
    max_requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW,
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce rate limits per client IP."""
    # Skip rate limiting for health checks
    if request.url.path == "/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        remaining = rate_limiter.get_remaining(client_ip)
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "too_many_requests",
                }
            },
            headers={
                "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(RATE_LIMIT_WINDOW),
            },
        )

    response = await call_next(request)

    # Add rate limit headers to response
    remaining = rate_limiter.get_remaining(client_ip)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    return response


# ================================================================
# Helper: get or load backend for a model
# ================================================================

def _get_backend(model_id: str) -> InferenceBackend:
    """Return a loaded backend for *model_id*, loading on demand."""
    if model_id in app_state.backends and app_state.backends[model_id].is_loaded:
        return app_state.backends[model_id]

    entry = app_state.model_registry.get(model_id)
    if entry is None:
        # Try default
        if "default" in app_state.backends:
            return app_state.backends["default"]
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    config = InferenceConfig(
        model_path=entry["path"],
        n_gpu_layers=-1,
        n_ctx=4096,
        use_mmap=True,
    )

    # Apply active optimization config if present
    active = app_state.active_configs.get(model_id)
    if active:
        offload = active.get("offload_config", {})
        if "gpu_layers" in offload:
            config.n_gpu_layers = offload["gpu_layers"]
        if "cpu_threads" in offload and offload["cpu_threads"] > 0:
            config.n_threads = offload["cpu_threads"]
        if "context_length" in offload:
            config.n_ctx = offload["context_length"]
        if "batch_size" in offload:
            config.n_batch = offload["batch_size"]
        config.use_mmap = offload.get("use_mmap", True)

    backend = LlamaCppBackend(config)
    backend.load_model()
    app_state.backends[model_id] = backend
    entry["loaded"] = True

    # Register with scheduler
    if app_state.scheduler:
        profile = ModelProfile(
            model_id=model_id,
            model_path=entry["path"],
            supported_tasks=set(TaskType),
            memory_mb=backend.get_memory_usage().model_size_mb,
            is_loaded=True,
        )
        app_state.scheduler.register_model(profile, backend)

    return backend


def _format_sse(data: Dict[str, Any]) -> str:
    """Format a dict as a Server-Sent Event."""
    import json
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_opt_hardware_profile() -> OptHardwareProfile:
    """Build an optimization HardwareProfile from the detected hardware."""
    hw = app_state.hardware_detector.profile
    return OptHardwareProfile.from_detector(hw)


def _resolve_model_size_b(model_name: str, explicit: Optional[float] = None) -> float:
    """Resolve model parameter count in billions.

    Priority: explicit argument > model_manager metadata > model_registry > default 7.0
    """
    if explicit is not None:
        return explicit

    # Try model_manager
    if app_state.model_manager:
        info = app_state.model_manager.get_model(model_name)
        if info and info.parameter_count and info.parameter_count > 0:
            return info.parameter_count / 1e9

    # Try registry (look for size hints in path)
    entry = app_state.model_registry.get(model_name)
    if entry:
        path = entry.get("path", "")
        import re
        match = re.search(r"(\d+\.?\d*)[Bb]", Path(path).stem)
        if match:
            return float(match.group(1))

    # Default fallback
    logger.warning("Cannot determine model size for '%s', defaulting to 7.0B", model_name)
    return 7.0


def _serialize_enums(obj: Any) -> Any:
    """Recursively convert Enum values to their string values."""
    if isinstance(obj, dict):
        return {k: _serialize_enums(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_enums(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


# ================================================================
# OpenAI-compatible API
# ================================================================

@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """OpenAI-compatible chat completions endpoint."""
    # Build prompt from messages
    prompt_parts: List[str] = []
    for msg in req.messages:
        prefix = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
            msg.role, msg.role.capitalize()
        )
        prompt_parts.append(f"{prefix}: {msg.content}")
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    if req.stream:
        return StreamingResponse(
            _chat_stream(req.model, prompt, req),
            media_type="text/event-stream",
        )

    # Non-streaming
    backend = _get_backend(req.model)
    result = await asyncio.to_thread(
        backend.generate,
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop,
        seed=req.seed,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.tokens_generated,
            "total_tokens": result.prompt_tokens + result.tokens_generated,
        },
    }


async def _chat_stream(model_id: str, prompt: str, req: ChatCompletionRequest):
    """SSE generator for streaming chat completions."""
    backend = _get_backend(model_id)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        async for chunk in backend.generate_async(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
            seed=req.seed,
        ):
            event = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield _format_sse(event)
    except Exception as e:
        logger.error("Stream error: %s", e)
        event = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "error"}
            ],
        }
        yield _format_sse(event)

    # Final chunk
    event = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}
        ],
    }
    yield _format_sse(event)
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(
    req: CompletionRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """OpenAI-compatible text completions endpoint."""
    if req.stream:
        return StreamingResponse(
            _completion_stream(req.model, req.prompt, req),
            media_type="text/event-stream",
        )

    backend = _get_backend(req.model)
    result = await asyncio.to_thread(
        backend.generate,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop,
        seed=req.seed,
    )

    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "text": result.text,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.tokens_generated,
            "total_tokens": result.prompt_tokens + result.tokens_generated,
        },
    }


async def _completion_stream(model_id: str, prompt: str, req: CompletionRequest):
    """SSE generator for streaming text completions."""
    backend = _get_backend(model_id)
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    try:
        async for chunk in backend.generate_async(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
            seed=req.seed,
        ):
            event = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": chunk,
                        "finish_reason": None,
                    }
                ],
            }
            yield _format_sse(event)
    except Exception as e:
        logger.error("Stream error: %s", e)

    event = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {"index": 0, "text": "", "finish_reason": "stop"}
        ],
    }
    yield _format_sse(event)
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models_v1():
    """OpenAI-compatible model listing."""
    models: List[Dict[str, Any]] = []
    for model_id, entry in app_state.model_registry.items():
        models.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        })
    # Always include a default entry
    if not models:
        models.append({
            "id": "default",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        })
    return {"object": "list", "data": models}


# ================================================================
# Management API -- Hardware
# ================================================================

@app.get("/api/hardware")
async def get_hardware():
    """Return detected hardware profile."""
    if app_state.hardware_detector is None:
        raise HTTPException(status_code=503, detail="Hardware detection not available")

    profile = app_state.hardware_detector.profile
    data = asdict(profile)
    data = _serialize_enums(data)
    return data


# ================================================================
# Management API -- Models
# ================================================================

@app.get("/api/models")
async def list_models():
    """List all known models with their status."""
    result = []
    for model_id, entry in app_state.model_registry.items():
        backend = app_state.backends.get(model_id)
        memory_usage = None
        if backend and backend.is_loaded:
            try:
                mu = backend.get_memory_usage()
                memory_usage = mu.model_size_mb
            except Exception:
                pass

        # Enrich with model_manager metadata if available
        extra = {}
        if app_state.model_manager:
            info = app_state.model_manager.get_model(model_id)
            if info:
                extra = {
                    "parameter_count_b": round(info.parameter_count_b, 2) if info.parameter_count_b else None,
                    "quantization": info.quantization,
                    "architecture": info.architecture,
                    "size_gb": round(info.size_gb, 2),
                    "format": info.format.value,
                }

        result.append({
            "id": model_id,
            # Security: Return only filename, not full path
            "filename": Path(entry.get("path", "")).name,
            "loaded": entry.get("loaded", False),
            "memory_mb": memory_usage,
            "active_config": app_state.active_configs.get(model_id),
            **extra,
        })
    return {"models": result, "count": len(result)}


@app.post("/api/models/download")
async def download_model(
    req: ModelDownloadRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Trigger model download.

    In a full implementation this would download from HuggingFace or a URL.
    This stub validates input and returns a task placeholder.
    """
    if not req.url and not req.model_id:
        raise HTTPException(
            status_code=400,
            detail="Either 'url' or 'model_id' must be provided",
        )

    target_name = req.filename or req.model_id or "downloaded_model"
    if not target_name.endswith(".gguf"):
        target_name += ".gguf"

    # Security: sanitize filename to prevent path traversal
    target_name = Path(target_name).name
    if not target_name or target_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    target_path = app_state.models_dir / target_name
    if target_path.exists():
        return {
            "status": "exists",
            "model_id": target_path.stem,
            "path": str(target_path),
            "message": "Model file already exists",
        }

    # Placeholder -- real implementation would start an async download task
    task_id = uuid.uuid4().hex[:12]
    # Security: Redact URL to avoid logging tokens/secrets in query params
    safe_url = req.url.split('?')[0] if req.url else None
    logger.info("Download requested: url=%s model_id=%s -> %s", safe_url, req.model_id, target_path.name)

    return {
        "status": "queued",
        "task_id": task_id,
        "model_id": target_path.stem,
        "target_path": str(target_path),
        "message": "Download task queued (not yet implemented)",
    }


# ================================================================
# Management API -- Metrics
# ================================================================

@app.get("/api/metrics/current")
async def get_current_metrics():
    """Return the latest performance metrics snapshot."""
    if app_state.metrics_collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector not available")
    metrics = app_state.metrics_collector.get_current()
    return metrics.to_dict()


@app.get("/api/metrics/history")
async def get_metrics_history(count: Optional[int] = None):
    """Return historical metrics, optionally limited to *count* entries."""
    if app_state.metrics_collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector not available")
    history = app_state.metrics_collector.get_history(count)
    return {
        "count": len(history),
        "metrics": [m.to_dict() for m in history],
    }


@app.get("/api/metrics/bottleneck")
async def get_bottleneck(sample_count: int = 60):
    """Run bottleneck analysis and return detected bottlenecks."""
    if app_state.analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not available")
    result = app_state.analyzer.analyze(sample_count)
    return {
        "bottlenecks": result.to_dict()["bottlenecks"],
        "summary": result.summary,
    }


@app.get("/api/metrics/suggestions")
async def get_suggestions(sample_count: int = 60):
    """Run full analysis and return optimization suggestions."""
    if app_state.analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not available")
    result = app_state.analyzer.analyze(sample_count)
    return {
        "suggestions": result.to_dict()["suggestions"],
        "summary": result.summary,
    }


# ================================================================
# Optimization API
# ================================================================

@app.post("/api/optimize")
async def optimize_model(
    req: OptimizeRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get optimization configuration for a model.

    Input: model name, target optimization profile, optional hardware overrides.
    Output: recommended configuration and estimated performance.
    """
    if app_state.memory_optimizer is None:
        raise HTTPException(status_code=503, detail="Memory optimizer not available")

    # Validate target
    valid_targets = {p.value for p in OptimizationProfile}
    if req.target not in valid_targets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target '{req.target}'. Must be one of: {sorted(valid_targets)}",
        )

    profile = OptimizationProfile(req.target)
    model_size_b = _resolve_model_size_b(req.model_name, req.model_size_b)

    # Build hardware profile, apply overrides
    hw_profile = _build_opt_hardware_profile()
    if req.vram_gb is not None:
        hw_profile.vram_free_gb = req.vram_gb
        hw_profile.vram_total_gb = max(req.vram_gb, hw_profile.vram_total_gb)
    if req.ram_gb is not None:
        hw_profile.ram_free_gb = req.ram_gb
        hw_profile.ram_total_gb = max(req.ram_gb, hw_profile.ram_total_gb)

    try:
        result = app_state.memory_optimizer.optimize_for_model(
            model_size_b=model_size_b,
            profile=profile,
            hardware=hw_profile,
            context_length=req.context_length,
            batch_size=req.batch_size,
        )
    except Exception as e:
        logger.exception("Optimization failed for model '%s'", req.model_name)
        raise HTTPException(
            status_code=500,
            detail="Optimization failed. Check server logs for details.",
        )

    response = {
        "model_name": req.model_name,
        "model_size_b": model_size_b,
        "target": req.target,
        "config": result.to_dict(),
        "estimated_performance": {
            "tokens_per_second": result.estimated_speed_tps,
            "vram_gb": result.estimated_vram_gb,
            "ram_gb": result.estimated_ram_gb,
            "quality_score": result.quality_score,
            "gpu_utilization": result.gpu_utilization,
        },
        "notes": result.notes,
    }
    return response


@app.get("/api/optimize/report/{model_name}")
async def get_optimization_report(model_name: str):
    """Generate a detailed optimization report for a model.

    Compares all optimization profiles and strategies.
    """
    if app_state.memory_optimizer is None:
        raise HTTPException(status_code=503, detail="Memory optimizer not available")

    model_size_b = _resolve_model_size_b(model_name)
    hw_profile = _build_opt_hardware_profile()

    try:
        report = app_state.memory_optimizer.get_optimization_report(
            model_size_b=model_size_b,
            hardware=hw_profile,
        )
    except Exception as e:
        logger.exception("Report generation failed for model '%s'", model_name)
        raise HTTPException(
            status_code=500,
            detail="Report generation failed. Check server logs for details.",
        )

    report["model_name"] = model_name
    return report


@app.post("/api/optimize/apply")
async def apply_optimization(
    req: OptimizeApplyRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Apply an optimization configuration to a model.

    Stores the config so that subsequent model loading uses optimized parameters.
    If the model is currently loaded, it will be unloaded and reloaded with new config.
    """
    model_name = req.model_name
    config = req.config

    if model_name not in app_state.model_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in registry",
        )

    # Validate config has expected structure
    if "offload_config" not in config and "quantization" not in config:
        raise HTTPException(
            status_code=400,
            detail="Config must contain at least 'offload_config' or 'quantization' keys",
        )

    # Store the active config
    app_state.active_configs[model_name] = config
    # Security: Log summary instead of full config to avoid leaking sensitive data
    config_summary = {k: type(v).__name__ for k, v in config.items()}
    logger.info("Applied optimization config for model '%s': keys=%s", model_name, config_summary)

    # If model is currently loaded, reload it
    reloaded = False
    if model_name in app_state.backends and app_state.backends[model_name].is_loaded:
        try:
            app_state.backends[model_name].unload_model()
            del app_state.backends[model_name]
            app_state.model_registry[model_name]["loaded"] = False
            # Next request will trigger reload with new config
            reloaded = True
            logger.info("Unloaded model '%s' for config reload", model_name)
        except Exception as e:
            logger.warning("Error unloading model '%s' for reload: %s", model_name, e)

    return {
        "status": "applied",
        "model_name": model_name,
        "config": config,
        "reloaded": reloaded,
        "message": (
            f"Optimization config applied for '{model_name}'."
            + (" Model unloaded and will reload with new config on next request." if reloaded
               else " Config will take effect on next model load.")
        ),
    }


# ================================================================
# Health check
# ================================================================

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": int(time.time())}


# ================================================================
# Entry point
# ================================================================

def main():
    """Start the server with uvicorn."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Security: Read host/port from environment, default to localhost
    host = os.environ.get("LMO_HOST", "127.0.0.1")
    port = int(os.environ.get("LMO_PORT", "8000"))

    # Security warning for non-localhost binding
    if host == "0.0.0.0":
        logger.warning("Server binding to 0.0.0.0 (all interfaces). This exposes the API to the network!")

    # Security warning if auth is disabled
    if not AUTH_ENABLED:
        logger.warning("Authentication is DISABLED. Set LMO_AUTH_ENABLED=true for production.")

    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
