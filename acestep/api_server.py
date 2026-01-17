"""FastAPI server for ACE-Step V1.5.

Endpoints:
- POST /v1/music/generate  Create an async music generation job (queued)
    - Supports application/json and multipart/form-data (with file upload)
- GET  /v1/jobs/{job_id}   Poll job status/result (+ queue position/eta when queued)

NOTE:
- In-memory queue and job store -> run uvicorn with workers=1.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
import tempfile
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

try:
    from dotenv import load_dotenv
except ImportError:  # Optional dependency
    load_dotenv = None  # type: ignore

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.constants import (
    DEFAULT_DIT_INSTRUCTION,
    DEFAULT_LM_INSTRUCTION,
)
from acestep.inference import (
    GenerationParams,
    GenerationConfig,
    generate_music,
    create_sample,
    format_sample,
)
from acestep.gradio_ui.events.results_handlers import _build_generation_info


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class GenerateMusicRequest(BaseModel):
    caption: str = Field(default="", description="Text caption describing the music")
    lyrics: str = Field(default="", description="Lyric text")

    # New API semantics:
    # - thinking=True: use 5Hz LM to generate audio codes (lm-dit behavior)
    # - thinking=False: do not use LM to generate codes (dit behavior)
    # Regardless of thinking, if some metas are missing, server may use LM to fill them.
    thinking: bool = False
    # Sample-mode requests auto-generate caption/lyrics/metas via LM (no user prompt).
    sample_mode: bool = False
    # Description for sample mode: auto-generate caption/lyrics from description query
    sample_query: str = Field(default="", description="Query/description for sample mode (use create_sample)")
    # Whether to use format_sample() to enhance input caption/lyrics
    use_format: bool = Field(default=False, description="Use format_sample() to enhance input (default: False)")
    # Model name for multi-model support (select which DiT model to use)
    model: Optional[str] = Field(default=None, description="Model name to use (e.g., 'acestep-v15-turbo')")

    bpm: Optional[int] = None
    # Accept common client keys via manual parsing (see _build_req_from_mapping).
    key_scale: str = ""
    time_signature: str = ""
    vocal_language: str = "en"
    inference_steps: int = 8
    guidance_scale: float = 7.0
    use_random_seed: bool = True
    seed: int = -1

    reference_audio_path: Optional[str] = None
    src_audio_path: Optional[str] = None
    audio_duration: Optional[float] = None
    batch_size: Optional[int] = None

    audio_code_string: str = ""

    repainting_start: float = 0.0
    repainting_end: Optional[float] = None

    instruction: str = DEFAULT_DIT_INSTRUCTION
    audio_cover_strength: float = 1.0
    task_type: str = "text2music"

    use_adg: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    infer_method: str = "ode"  # "ode" or "sde" - diffusion inference method
    shift: float = Field(
        default=3.0,
        description="Timestep shift factor (range 1.0~5.0, default 3.0). Only effective for base models, not turbo models."
    )
    timesteps: Optional[str] = Field(
        default=None,
        description="Custom timesteps (comma-separated, e.g., '0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0'). Overrides inference_steps and shift."
    )

    audio_format: str = "mp3"
    use_tiled_decode: bool = True

    # 5Hz LM (server-side): used for metadata completion and (when thinking=True) codes generation.
    lm_model_path: Optional[str] = None  # e.g. "acestep-5Hz-lm-0.6B"
    lm_backend: Literal["vllm", "pt"] = "vllm"

    constrained_decoding: bool = True
    constrained_decoding_debug: bool = False
    use_cot_caption: bool = True
    use_cot_language: bool = True
    is_format_caption: bool = False

    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.0
    lm_top_k: Optional[int] = None
    lm_top_p: Optional[float] = 0.9
    lm_repetition_penalty: float = 1.0
    lm_negative_prompt: str = "NO USER INPUT"

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True


_LM_DEFAULT_TEMPERATURE = 0.85
_LM_DEFAULT_CFG_SCALE = 2.0
_LM_DEFAULT_TOP_P = 0.9
_DEFAULT_DIT_INSTRUCTION = DEFAULT_DIT_INSTRUCTION
_DEFAULT_LM_INSTRUCTION = DEFAULT_LM_INSTRUCTION


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    queue_position: int = 0  # 1-based best-effort position when queued


class JobResult(BaseModel):
    first_audio_path: Optional[str] = None
    second_audio_path: Optional[str] = None
    audio_paths: list[str] = Field(default_factory=list)

    generation_info: str = ""
    status_message: str = ""
    seed_value: str = ""

    metas: Dict[str, Any] = Field(default_factory=dict)
    bpm: Optional[int] = None
    duration: Optional[float] = None
    genres: Optional[str] = None
    keyscale: Optional[str] = None
    timesignature: Optional[str] = None
    
    # Model information
    lm_model: Optional[str] = None
    dit_model: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # queue observability
    queue_position: int = 0
    eta_seconds: Optional[float] = None
    avg_job_seconds: Optional[float] = None

    result: Optional[JobResult] = None
    error: Optional[str] = None


@dataclass
class _JobRecord:
    job_id: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class _JobStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: Dict[str, _JobRecord] = {}

    def create(self) -> _JobRecord:
        job_id = str(uuid4())
        rec = _JobRecord(job_id=job_id, status="queued", created_at=time.time())
        with self._lock:
            self._jobs[job_id] = rec
        return rec

    def get(self, job_id: str) -> Optional[_JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            rec = self._jobs[job_id]
            rec.status = "running"
            rec.started_at = time.time()

    def mark_succeeded(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            rec = self._jobs[job_id]
            rec.status = "succeeded"
            rec.finished_at = time.time()
            rec.result = result
            rec.error = None

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            rec = self._jobs[job_id]
            rec.status = "failed"
            rec.finished_at = time.time()
            rec.result = None
            rec.error = error


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_project_root() -> str:
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current_file))


def _get_model_name(config_path: str) -> str:
    """
    Extract model name from config_path.
    
    Args:
        config_path: Path like "acestep-v15-turbo" or "/path/to/acestep-v15-turbo"
        
    Returns:
        Model name (last directory name from config_path)
    """
    if not config_path:
        return ""
    normalized = config_path.rstrip("/\\")
    return os.path.basename(normalized)


def _load_project_env() -> None:
    if load_dotenv is None:
        return
    try:
        project_root = _get_project_root()
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
    except Exception:
        # Optional best-effort: continue even if .env loading fails.
        pass


_load_project_env()


def _to_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s == "":
        return default
    try:
        return int(s)
    except Exception:
        return default


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, float):
        return v
    s = str(v).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def _to_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s == "":
        return default
    return s in {"1", "true", "yes", "y", "on"}


async def _save_upload_to_temp(upload: StarletteUploadFile, *, prefix: str) -> str:
    suffix = Path(upload.filename or "").suffix
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=suffix)
    os.close(fd)
    try:
        with open(path, "wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        raise
    finally:
        try:
            await upload.close()
        except Exception:
            pass
    return path


def create_app() -> FastAPI:
    store = _JobStore()

    QUEUE_MAXSIZE = int(os.getenv("ACESTEP_QUEUE_MAXSIZE", "200"))
    WORKER_COUNT = int(os.getenv("ACESTEP_QUEUE_WORKERS", "1"))  # 单 GPU 建议 1

    INITIAL_AVG_JOB_SECONDS = float(os.getenv("ACESTEP_AVG_JOB_SECONDS", "5.0"))
    AVG_WINDOW = int(os.getenv("ACESTEP_AVG_WINDOW", "50"))

    def _path_to_audio_url(path: str) -> str:
        """将本地文件路径转换为可下载的相对 URL"""
        if not path:
            return path
        if path.startswith("http://") or path.startswith("https://"):
            return path
        encoded_path = urllib.parse.quote(path, safe="")
        return f"/v1/audio?path={encoded_path}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Clear proxy env that may affect downstream libs
        for proxy_var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            os.environ.pop(proxy_var, None)

        # Ensure compilation/temp caches do not fill up small default /tmp.
        # Triton/Inductor (and the system compiler) can create large temporary files.
        project_root = _get_project_root()
        cache_root = os.path.join(project_root, ".cache", "acestep")
        tmp_root = (os.getenv("ACESTEP_TMPDIR") or os.path.join(cache_root, "tmp")).strip()
        triton_cache_root = (os.getenv("TRITON_CACHE_DIR") or os.path.join(cache_root, "triton")).strip()
        inductor_cache_root = (os.getenv("TORCHINDUCTOR_CACHE_DIR") or os.path.join(cache_root, "torchinductor")).strip()

        for p in [cache_root, tmp_root, triton_cache_root, inductor_cache_root]:
            try:
                os.makedirs(p, exist_ok=True)
            except Exception:
                # Best-effort: do not block startup if directory creation fails.
                pass

        # Respect explicit user overrides; if ACESTEP_TMPDIR is set, it should win.
        if os.getenv("ACESTEP_TMPDIR"):
            os.environ["TMPDIR"] = tmp_root
            os.environ["TEMP"] = tmp_root
            os.environ["TMP"] = tmp_root
        else:
            os.environ.setdefault("TMPDIR", tmp_root)
            os.environ.setdefault("TEMP", tmp_root)
            os.environ.setdefault("TMP", tmp_root)

        os.environ.setdefault("TRITON_CACHE_DIR", triton_cache_root)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", inductor_cache_root)

        handler = AceStepHandler()
        llm_handler = LLMHandler()
        init_lock = asyncio.Lock()
        app.state._initialized = False
        app.state._init_error = None
        app.state._init_lock = init_lock

        app.state.llm_handler = llm_handler
        app.state._llm_initialized = False
        app.state._llm_init_error = None
        app.state._llm_init_lock = Lock()

        # Multi-model support: secondary DiT handlers
        handler2 = None
        handler3 = None
        config_path2 = os.getenv("ACESTEP_CONFIG_PATH2", "").strip()
        config_path3 = os.getenv("ACESTEP_CONFIG_PATH3", "").strip()
        
        if config_path2:
            handler2 = AceStepHandler()
        if config_path3:
            handler3 = AceStepHandler()
        
        app.state.handler2 = handler2
        app.state.handler3 = handler3
        app.state._initialized2 = False
        app.state._initialized3 = False
        app.state._config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo-rl")
        app.state._config_path2 = config_path2
        app.state._config_path3 = config_path3

        max_workers = int(os.getenv("ACESTEP_API_WORKERS", "1"))
        executor = ThreadPoolExecutor(max_workers=max_workers)

        # Queue & observability
        app.state.job_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)  # (job_id, req)
        app.state.pending_ids = deque()  # queued job_ids
        app.state.pending_lock = asyncio.Lock()

        # temp files per job (from multipart uploads)
        app.state.job_temp_files = {}  # job_id -> list[path]
        app.state.job_temp_files_lock = asyncio.Lock()

        # stats
        app.state.stats_lock = asyncio.Lock()
        app.state.recent_durations = deque(maxlen=AVG_WINDOW)
        app.state.avg_job_seconds = INITIAL_AVG_JOB_SECONDS

        app.state.handler = handler
        app.state.executor = executor
        app.state.job_store = store
        app.state._python_executable = sys.executable
        
        # Temporary directory for saving generated audio files
        app.state.temp_audio_dir = os.path.join(tmp_root, "api_audio")
        os.makedirs(app.state.temp_audio_dir, exist_ok=True)

        async def _ensure_initialized() -> None:
            h: AceStepHandler = app.state.handler

            if getattr(app.state, "_initialized", False):
                return
            if getattr(app.state, "_init_error", None):
                raise RuntimeError(app.state._init_error)

            async with app.state._init_lock:
                if getattr(app.state, "_initialized", False):
                    return
                if getattr(app.state, "_init_error", None):
                    raise RuntimeError(app.state._init_error)

                project_root = _get_project_root()
                config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo-rl")
                device = os.getenv("ACESTEP_DEVICE", "auto")

                use_flash_attention = _env_bool("ACESTEP_USE_FLASH_ATTENTION", True)
                offload_to_cpu = _env_bool("ACESTEP_OFFLOAD_TO_CPU", False)
                offload_dit_to_cpu = _env_bool("ACESTEP_OFFLOAD_DIT_TO_CPU", False)

                # Initialize primary model
                status_msg, ok = h.initialize_service(
                    project_root=project_root,
                    config_path=config_path,
                    device=device,
                    use_flash_attention=use_flash_attention,
                    compile_model=False,
                    offload_to_cpu=offload_to_cpu,
                    offload_dit_to_cpu=offload_dit_to_cpu,
                )
                if not ok:
                    app.state._init_error = status_msg
                    raise RuntimeError(status_msg)
                app.state._initialized = True
                
                # Initialize secondary model if configured
                if app.state.handler2 and app.state._config_path2:
                    try:
                        status_msg2, ok2 = app.state.handler2.initialize_service(
                            project_root=project_root,
                            config_path=app.state._config_path2,
                            device=device,
                            use_flash_attention=use_flash_attention,
                            compile_model=False,
                            offload_to_cpu=offload_to_cpu,
                            offload_dit_to_cpu=offload_dit_to_cpu,
                        )
                        app.state._initialized2 = ok2
                        if ok2:
                            print(f"[API Server] Secondary model loaded: {_get_model_name(app.state._config_path2)}")
                        else:
                            print(f"[API Server] Warning: Secondary model failed to load: {status_msg2}")
                    except Exception as e:
                        print(f"[API Server] Warning: Failed to initialize secondary model: {e}")
                        app.state._initialized2 = False
                
                # Initialize third model if configured
                if app.state.handler3 and app.state._config_path3:
                    try:
                        status_msg3, ok3 = app.state.handler3.initialize_service(
                            project_root=project_root,
                            config_path=app.state._config_path3,
                            device=device,
                            use_flash_attention=use_flash_attention,
                            compile_model=False,
                            offload_to_cpu=offload_to_cpu,
                            offload_dit_to_cpu=offload_dit_to_cpu,
                        )
                        app.state._initialized3 = ok3
                        if ok3:
                            print(f"[API Server] Third model loaded: {_get_model_name(app.state._config_path3)}")
                        else:
                            print(f"[API Server] Warning: Third model failed to load: {status_msg3}")
                    except Exception as e:
                        print(f"[API Server] Warning: Failed to initialize third model: {e}")
                        app.state._initialized3 = False

        async def _cleanup_job_temp_files(job_id: str) -> None:
            async with app.state.job_temp_files_lock:
                paths = app.state.job_temp_files.pop(job_id, [])
            for p in paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

        async def _run_one_job(job_id: str, req: GenerateMusicRequest) -> None:
            job_store: _JobStore = app.state.job_store
            llm: LLMHandler = app.state.llm_handler
            executor: ThreadPoolExecutor = app.state.executor

            await _ensure_initialized()
            job_store.mark_running(job_id)
            
            # Select DiT handler based on user's model choice
            # Default: use primary handler
            selected_handler: AceStepHandler = app.state.handler
            selected_model_name = _get_model_name(app.state._config_path)
            
            if req.model:
                model_matched = False
                
                # Check if it matches the second model
                if app.state.handler2 and getattr(app.state, "_initialized2", False):
                    model2_name = _get_model_name(app.state._config_path2)
                    if req.model == model2_name:
                        selected_handler = app.state.handler2
                        selected_model_name = model2_name
                        model_matched = True
                        print(f"[API Server] Job {job_id}: Using second model: {model2_name}")
                
                # Check if it matches the third model
                if not model_matched and app.state.handler3 and getattr(app.state, "_initialized3", False):
                    model3_name = _get_model_name(app.state._config_path3)
                    if req.model == model3_name:
                        selected_handler = app.state.handler3
                        selected_model_name = model3_name
                        model_matched = True
                        print(f"[API Server] Job {job_id}: Using third model: {model3_name}")
                
                if not model_matched:
                    available_models = [_get_model_name(app.state._config_path)]
                    if app.state.handler2 and getattr(app.state, "_initialized2", False):
                        available_models.append(_get_model_name(app.state._config_path2))
                    if app.state.handler3 and getattr(app.state, "_initialized3", False):
                        available_models.append(_get_model_name(app.state._config_path3))
                    print(f"[API Server] Job {job_id}: Model '{req.model}' not found in {available_models}, using primary: {selected_model_name}")
            
            # Use selected handler for generation
            h: AceStepHandler = selected_handler

            def _blocking_generate() -> Dict[str, Any]:
                """Generate music using unified inference logic from acestep.inference"""
                
                def _ensure_llm_ready() -> None:
                    """Ensure LLM handler is initialized when needed"""
                    with app.state._llm_init_lock:
                        initialized = getattr(app.state, "_llm_initialized", False)
                        had_error = getattr(app.state, "_llm_init_error", None)
                        if initialized or had_error is not None:
                            return

                        project_root = _get_project_root()
                        checkpoint_dir = os.path.join(project_root, "checkpoints")
                        lm_model_path = (req.lm_model_path or os.getenv("ACESTEP_LM_MODEL_PATH") or "acestep-5Hz-lm-0.6B-v3").strip()
                        backend = (req.lm_backend or os.getenv("ACESTEP_LM_BACKEND") or "vllm").strip().lower()
                        if backend not in {"vllm", "pt"}:
                            backend = "vllm"

                        lm_device = os.getenv("ACESTEP_LM_DEVICE", os.getenv("ACESTEP_DEVICE", "auto"))
                        lm_offload = _env_bool("ACESTEP_LM_OFFLOAD_TO_CPU", False)

                        status, ok = llm.initialize(
                            checkpoint_dir=checkpoint_dir,
                            lm_model_path=lm_model_path,
                            backend=backend,
                            device=lm_device,
                            offload_to_cpu=lm_offload,
                            dtype=h.dtype,
                        )
                        if not ok:
                            app.state._llm_init_error = status
                        else:
                            app.state._llm_initialized = True

                def _normalize_metas(meta: Dict[str, Any]) -> Dict[str, Any]:
                    """Ensure a stable `metas` dict (keys always present)."""
                    meta = meta or {}
                    out: Dict[str, Any] = dict(meta)

                    # Normalize key aliases
                    if "keyscale" not in out and "key_scale" in out:
                        out["keyscale"] = out.get("key_scale")
                    if "timesignature" not in out and "time_signature" in out:
                        out["timesignature"] = out.get("time_signature")

                    # Ensure required keys exist
                    for k in ["bpm", "duration", "genres", "keyscale", "timesignature"]:
                        if out.get(k) in (None, ""):
                            out[k] = "N/A"
                    return out

                # Normalize LM sampling parameters
                lm_top_k = req.lm_top_k if req.lm_top_k and req.lm_top_k > 0 else 0
                lm_top_p = req.lm_top_p if req.lm_top_p and req.lm_top_p < 1.0 else 0.9

                # Determine if LLM is needed
                thinking = bool(req.thinking)
                sample_mode = bool(req.sample_mode)
                has_sample_query = bool(req.sample_query and req.sample_query.strip())
                use_format = bool(req.use_format)
                use_cot_caption = bool(req.use_cot_caption)
                use_cot_language = bool(req.use_cot_language)
                
                # LLM is needed for:
                # - thinking mode (LM generates audio codes)
                # - sample_mode (LM generates random caption/lyrics/metas)
                # - sample_query/description (LM generates from description)
                # - use_format (LM enhances caption/lyrics)
                # - use_cot_caption or use_cot_language (LM enhances metadata)
                need_llm = thinking or sample_mode or has_sample_query or use_format or use_cot_caption or use_cot_language
                
                print(f"[api_server] Request params: req.thinking={req.thinking}, req.sample_mode={req.sample_mode}, req.use_cot_caption={req.use_cot_caption}, req.use_cot_language={req.use_cot_language}, req.use_format={req.use_format}")
                print(f"[api_server] Determined: thinking={thinking}, sample_mode={sample_mode}, use_cot_caption={use_cot_caption}, use_cot_language={use_cot_language}, use_format={use_format}, need_llm={need_llm}")
                
                # Ensure LLM is ready if needed
                if need_llm:
                    _ensure_llm_ready()
                    if getattr(app.state, "_llm_init_error", None):
                        raise RuntimeError(f"5Hz LM init failed: {app.state._llm_init_error}")

                # Handle sample mode or description: generate caption/lyrics/metas via LM
                caption = req.caption
                lyrics = req.lyrics
                bpm = req.bpm
                key_scale = req.key_scale
                time_signature = req.time_signature
                audio_duration = req.audio_duration
                
                if sample_mode or has_sample_query:
                    if has_sample_query:
                        # Use create_sample() with description query
                        print(f"[api_server] Description mode: generating sample from query: {req.sample_query[:100]}")
                        sample_result = create_sample(
                            llm_handler=llm,
                            query=req.sample_query,
                            instrumental=False,  # Could be extracted from description
                            vocal_language=req.vocal_language if req.vocal_language != "en" else None,
                            temperature=req.lm_temperature,
                            top_k=lm_top_k if lm_top_k > 0 else None,
                            top_p=lm_top_p if lm_top_p < 1.0 else None,
                            use_constrained_decoding=req.constrained_decoding,
                        )
                        
                        if not sample_result.success:
                            raise RuntimeError(f"create_sample failed: {sample_result.error or sample_result.status_message}")
                        
                        # Use generated sample data
                        caption = sample_result.caption
                        lyrics = sample_result.lyrics
                        bpm = sample_result.bpm
                        key_scale = sample_result.keyscale
                        time_signature = sample_result.timesignature
                        audio_duration = sample_result.duration
                        
                        print(f"[api_server] Sample from description generated: caption_len={len(caption)}, lyrics_len={len(lyrics)}, bpm={bpm}")
                    else:
                        # Original sample_mode behavior: random generation
                        print("[api_server] Sample mode: generating random caption/lyrics via LM")
                        sample_metadata, sample_status = llm.understand_audio_from_codes(
                            audio_codes="NO USER INPUT",
                            temperature=req.lm_temperature,
                            top_k=lm_top_k if lm_top_k > 0 else None,
                            top_p=lm_top_p if lm_top_p < 1.0 else None,
                            repetition_penalty=req.lm_repetition_penalty,
                            use_constrained_decoding=req.constrained_decoding,
                            constrained_decoding_debug=req.constrained_decoding_debug,
                        )

                        if not sample_metadata or str(sample_status).startswith("❌"):
                            raise RuntimeError(f"Sample generation failed: {sample_status}")

                        # Use generated values with fallback defaults
                        caption = sample_metadata.get("caption", "")
                        lyrics = sample_metadata.get("lyrics", "")
                        bpm = _to_int(sample_metadata.get("bpm"), None) or _to_int(os.getenv("ACESTEP_SAMPLE_DEFAULT_BPM", "120"), 120)
                        key_scale = sample_metadata.get("keyscale", "") or os.getenv("ACESTEP_SAMPLE_DEFAULT_KEY", "C Major")
                        time_signature = sample_metadata.get("timesignature", "") or os.getenv("ACESTEP_SAMPLE_DEFAULT_TIMESIGNATURE", "4/4")
                        audio_duration = _to_float(sample_metadata.get("duration"), None) or _to_float(os.getenv("ACESTEP_SAMPLE_DEFAULT_DURATION_SECONDS", "120"), 120.0)
                        
                        print(f"[api_server] Sample generated: caption_len={len(caption)}, lyrics_len={len(lyrics)}, bpm={bpm}, duration={audio_duration}")
                
                # Apply format_sample() if use_format is True and caption/lyrics are provided
                if req.use_format and (caption or lyrics):
                    print(f"[api_server] Applying format_sample to enhance input...")
                    _ensure_llm_ready()
                    if getattr(app.state, "_llm_init_error", None):
                        raise RuntimeError(f"5Hz LM init failed (needed for format): {app.state._llm_init_error}")
                    
                    format_result = format_sample(
                        llm_handler=llm,
                        caption=caption,
                        lyrics=lyrics,
                        temperature=req.lm_temperature,
                        top_k=lm_top_k if lm_top_k > 0 else None,
                        top_p=lm_top_p if lm_top_p < 1.0 else None,
                        use_constrained_decoding=req.constrained_decoding,
                    )
                    
                    if format_result.success:
                        caption = format_result.caption
                        lyrics = format_result.lyrics
                        print(f"[api_server] Format applied: new caption_len={len(caption)}, lyrics_len={len(lyrics)}")
                    else:
                        print(f"[api_server] Warning: format_sample failed: {format_result.error}, using original input")
                
                print(f"[api_server] Before GenerationParams: thinking={thinking}, sample_mode={sample_mode}")
                # Parse timesteps string to list of floats if provided
                parsed_timesteps = None
                if req.timesteps and req.timesteps.strip():
                    try:
                        parsed_timesteps = [float(t.strip()) for t in req.timesteps.split(",") if t.strip()]
                    except ValueError:
                        print(f"[api_server] Warning: Failed to parse timesteps '{req.timesteps}', using default")
                        parsed_timesteps = None

                print(f"[api_server] Caption/Lyrics to use: caption_len={len(caption)}, lyrics_len={len(lyrics)}")

                # Parse timesteps if provided
                parsed_timesteps = None
                if req.timesteps:
                    try:
                        parsed_timesteps = [float(t.strip()) for t in req.timesteps.split(",") if t.strip()]
                        print(f"[api_server] Using custom timesteps: {parsed_timesteps}")
                    except Exception as e:
                        print(f"[api_server] Warning: Failed to parse timesteps '{req.timesteps}': {e}")
                        parsed_timesteps = None
                
                # Determine actual inference steps (timesteps override inference_steps)
                actual_inference_steps = len(parsed_timesteps) if parsed_timesteps else req.inference_steps

                # Build GenerationParams using unified interface
                # Note: thinking controls LM code generation, sample_mode only affects CoT metas
                params = GenerationParams(
                    task_type=req.task_type,
                    instruction=req.instruction,
                    reference_audio=req.reference_audio_path,
                    src_audio=req.src_audio_path,
                    audio_codes=req.audio_code_string,
                    caption=caption,
                    lyrics=lyrics,
                    instrumental=False,
                    vocal_language=req.vocal_language,
                    bpm=bpm,
                    keyscale=key_scale,
                    timesignature=time_signature,
                    duration=audio_duration if audio_duration else -1.0,
                    inference_steps=req.inference_steps,
                    seed=req.seed,
                    guidance_scale=req.guidance_scale,
                    use_adg=req.use_adg,
                    cfg_interval_start=req.cfg_interval_start,
                    cfg_interval_end=req.cfg_interval_end,
                    shift=req.shift,
                    infer_method=req.infer_method,
                    timesteps=parsed_timesteps,
                    repainting_start=req.repainting_start,
                    repainting_end=req.repainting_end if req.repainting_end else -1,
                    audio_cover_strength=req.audio_cover_strength,
                    # LM parameters
                    thinking=thinking,  # Use LM for code generation when thinking=True
                    lm_temperature=req.lm_temperature,
                    lm_cfg_scale=req.lm_cfg_scale,
                    lm_top_k=lm_top_k,
                    lm_top_p=lm_top_p,
                    lm_negative_prompt=req.lm_negative_prompt,
                    use_cot_metas=not sample_mode,  # Sample mode already generated metas, don't regenerate
                    use_cot_caption=req.use_cot_caption,
                    use_cot_language=req.use_cot_language,
                    use_constrained_decoding=req.constrained_decoding,
                )

                # Build GenerationConfig - default to 2 audios like gradio_ui
                batch_size = req.batch_size if req.batch_size is not None else 2
                config = GenerationConfig(
                    batch_size=batch_size,
                    use_random_seed=req.use_random_seed,
                    seeds=None,  # Let unified logic handle seed generation
                    audio_format=req.audio_format,
                    constrained_decoding_debug=req.constrained_decoding_debug,
                )

                # Check LLM initialization status
                llm_is_initialized = getattr(app.state, "_llm_initialized", False)
                llm_to_pass = llm if llm_is_initialized else None
                
                print(f"[api_server] Generating music with unified interface:")
                print(f"  - thinking={params.thinking}")
                print(f"  - use_cot_caption={params.use_cot_caption}")
                print(f"  - use_cot_language={params.use_cot_language}")
                print(f"  - use_cot_metas={params.use_cot_metas}")
                print(f"  - batch_size={batch_size}")
                print(f"  - llm_initialized={llm_is_initialized}")
                print(f"  - llm_handler={'Available' if llm_to_pass else 'None'}")
                if llm_to_pass:
                    print(f"  - LLM will be used for: CoT caption={params.use_cot_caption}, CoT language={params.use_cot_language}, CoT metas={params.use_cot_metas}, thinking={params.thinking}")
                else:
                    print(f"  - WARNING: LLM features requested but LLM not initialized!")

                # Generate music using unified interface
                result = generate_music(
                    dit_handler=h,
                    llm_handler=llm_to_pass,
                    params=params,
                    config=config,
                    save_dir=app.state.temp_audio_dir,
                    progress=None,
                )
                
                print(f"[api_server] Generation completed. Success={result.success}, Audios={len(result.audios)}")
                print(f"[api_server] Time costs keys: {list(result.extra_outputs.get('time_costs', {}).keys())}")

                if not result.success:
                    raise RuntimeError(f"Music generation failed: {result.error or result.status_message}")

                # Extract results
                audio_paths = [audio["path"] for audio in result.audios if audio.get("path")]
                first_audio = audio_paths[0] if len(audio_paths) > 0 else None
                second_audio = audio_paths[1] if len(audio_paths) > 1 else None

                # Get metadata from LM or CoT results
                lm_metadata = result.extra_outputs.get("lm_metadata", {})
                metas_out = _normalize_metas(lm_metadata)
                
                # Update metas with actual values used
                if params.cot_bpm:
                    metas_out["bpm"] = params.cot_bpm
                elif bpm:
                    metas_out["bpm"] = bpm
                    
                if params.cot_duration:
                    metas_out["duration"] = params.cot_duration
                elif audio_duration:
                    metas_out["duration"] = audio_duration
                    
                if params.cot_keyscale:
                    metas_out["keyscale"] = params.cot_keyscale
                elif key_scale:
                    metas_out["keyscale"] = key_scale
                    
                if params.cot_timesignature:
                    metas_out["timesignature"] = params.cot_timesignature
                elif time_signature:
                    metas_out["timesignature"] = time_signature

                # Ensure caption and lyrics are in metas
                if caption:
                    metas_out["caption"] = caption
                if lyrics:
                    metas_out["lyrics"] = lyrics

                # Extract seed values for response (comma-separated for multiple audios)
                seed_values = []
                for audio in result.audios:
                    audio_params = audio.get("params", {})
                    seed = audio_params.get("seed")
                    if seed is not None:
                        seed_values.append(str(seed))
                seed_value = ",".join(seed_values) if seed_values else ""

                # Build generation_info using the helper function (like gradio_ui)
                time_costs = result.extra_outputs.get("time_costs", {})
                generation_info = _build_generation_info(
                    lm_metadata=lm_metadata,
                    time_costs=time_costs,
                    seed_value=seed_value,
                    inference_steps=req.inference_steps,
                    num_audios=len(result.audios),
                )

                def _none_if_na_str(v: Any) -> Optional[str]:
                    if v is None:
                        return None
                    s = str(v).strip()
                    if s in {"", "N/A"}:
                        return None
                    return s

                # Get model information
                lm_model_name = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B-v3")
                # Use selected_model_name (set at the beginning of _run_one_job)
                dit_model_name = selected_model_name
                
                return {
                    "first_audio_path": _path_to_audio_url(first_audio) if first_audio else None,
                    "second_audio_path": _path_to_audio_url(second_audio) if second_audio else None,
                    "audio_paths": [_path_to_audio_url(p) for p in audio_paths],
                    "generation_info": generation_info,
                    "status_message": result.status_message,
                    "seed_value": seed_value,
                    "metas": metas_out,
                    "bpm": metas_out.get("bpm") if isinstance(metas_out.get("bpm"), int) else None,
                    "duration": metas_out.get("duration") if isinstance(metas_out.get("duration"), (int, float)) else None,
                    "genres": _none_if_na_str(metas_out.get("genres")),
                    "keyscale": _none_if_na_str(metas_out.get("keyscale")),
                    "timesignature": _none_if_na_str(metas_out.get("timesignature")),
                    "lm_model": lm_model_name,
                    "dit_model": dit_model_name,
                }

            t0 = time.time()
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(executor, _blocking_generate)
                job_store.mark_succeeded(job_id, result)
            except Exception:
                job_store.mark_failed(job_id, traceback.format_exc())
            finally:
                dt = max(0.0, time.time() - t0)
                async with app.state.stats_lock:
                    app.state.recent_durations.append(dt)
                    if app.state.recent_durations:
                        app.state.avg_job_seconds = sum(app.state.recent_durations) / len(app.state.recent_durations)

        async def _queue_worker(worker_idx: int) -> None:
            while True:
                job_id, req = await app.state.job_queue.get()
                try:
                    async with app.state.pending_lock:
                        try:
                            app.state.pending_ids.remove(job_id)
                        except ValueError:
                            pass

                    await _run_one_job(job_id, req)
                finally:
                    await _cleanup_job_temp_files(job_id)
                    app.state.job_queue.task_done()

        worker_count = max(1, WORKER_COUNT)
        workers = [asyncio.create_task(_queue_worker(i)) for i in range(worker_count)]
        app.state.worker_tasks = workers

        try:
            yield
        finally:
            for t in workers:
                t.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="ACE-Step API", version="1.0", lifespan=lifespan)

    async def _queue_position(job_id: str) -> int:
        async with app.state.pending_lock:
            try:
                return list(app.state.pending_ids).index(job_id) + 1
            except ValueError:
                return 0

    async def _eta_seconds_for_position(pos: int) -> Optional[float]:
        if pos <= 0:
            return None
        async with app.state.stats_lock:
            avg = float(getattr(app.state, "avg_job_seconds", INITIAL_AVG_JOB_SECONDS))
        return pos * avg

    @app.post("/v1/music/generate", response_model=CreateJobResponse)
    async def create_music_generate_job(request: Request) -> CreateJobResponse:
        content_type = (request.headers.get("content-type") or "").lower()
        temp_files: list[str] = []

        def _build_req_from_mapping(mapping: Any, *, reference_audio_path: Optional[str], src_audio_path: Optional[str]) -> GenerateMusicRequest:
            get = getattr(mapping, "get", None)
            if not callable(get):
                raise HTTPException(status_code=400, detail="Invalid request payload")

            def _get_any(*keys: str, default: Any = None) -> Any:
                # 1) Top-level keys
                for k in keys:
                    v = get(k, None)
                    if v is not None:
                        return v

                # 2) Nested metas/metadata/user_metadata (dict or JSON string)
                nested = (
                    get("metas", None)
                    or get("meta", None)
                    or get("metadata", None)
                    or get("user_metadata", None)
                    or get("userMetadata", None)
                )

                if isinstance(nested, str):
                    s = nested.strip()
                    if s.startswith("{") and s.endswith("}"):
                        try:
                            nested = json.loads(s)
                        except Exception:
                            nested = None

                if isinstance(nested, dict):
                    g2 = nested.get
                    for k in keys:
                        v = g2(k, None)
                        if v is not None:
                            return v

                return default

            normalized_audio_duration = _to_float(_get_any("audio_duration", "duration", "audioDuration"), None)
            normalized_bpm = _to_int(_get_any("bpm"), None)
            normalized_keyscale = str(_get_any("key_scale", "keyscale", "keyScale", default="") or "")
            normalized_timesig = str(_get_any("time_signature", "timesignature", "timeSignature", default="") or "")

            # Accept it as an alias to avoid clients needing to special-case server.
            if normalized_audio_duration is None:
                normalized_audio_duration = _to_float(_get_any("target_duration", "targetDuration"), None)

            return GenerateMusicRequest(
                caption=str(get("caption", "") or ""),
                lyrics=str(get("lyrics", "") or ""),
                thinking=_to_bool(get("thinking"), False),
                sample_mode=_to_bool(_get_any("sample_mode", "sampleMode"), False),
                sample_query=str(_get_any("sample_query", "sampleQuery", "description", "desc", default="") or ""),
                use_format=_to_bool(_get_any("use_format", "useFormat", "format"), False),
                model=str(_get_any("model", "dit_model", "ditModel", default="") or "").strip() or None,
                bpm=normalized_bpm,
                key_scale=normalized_keyscale,
                time_signature=normalized_timesig,
                vocal_language=str(_get_any("vocal_language", "vocalLanguage", default="en") or "en"),
                inference_steps=_to_int(_get_any("inference_steps", "inferenceSteps"), 8) or 8,
                guidance_scale=_to_float(_get_any("guidance_scale", "guidanceScale"), 7.0) or 7.0,
                use_random_seed=_to_bool(_get_any("use_random_seed", "useRandomSeed"), True),
                seed=_to_int(get("seed"), -1) or -1,
                reference_audio_path=reference_audio_path,
                src_audio_path=src_audio_path,
                audio_duration=normalized_audio_duration,
                batch_size=_to_int(get("batch_size"), None),
                audio_code_string=str(_get_any("audio_code_string", "audioCodeString", default="") or ""),
                repainting_start=_to_float(get("repainting_start"), 0.0) or 0.0,
                repainting_end=_to_float(get("repainting_end"), None),
                instruction=str(get("instruction", _DEFAULT_DIT_INSTRUCTION) or ""),
                audio_cover_strength=_to_float(_get_any("audio_cover_strength", "audioCoverStrength"), 1.0) or 1.0,
                task_type=str(_get_any("task_type", "taskType", default="text2music") or "text2music"),
                use_adg=_to_bool(get("use_adg"), False),
                cfg_interval_start=_to_float(get("cfg_interval_start"), 0.0) or 0.0,
                cfg_interval_end=_to_float(get("cfg_interval_end"), 1.0) or 1.0,
                infer_method=str(_get_any("infer_method", "inferMethod", default="ode") or "ode"),
                shift=_to_float(_get_any("shift"), 3.0) or 3.0,
                audio_format=str(get("audio_format", "mp3") or "mp3"),
                use_tiled_decode=_to_bool(_get_any("use_tiled_decode", "useTiledDecode"), True),
                lm_model_path=str(get("lm_model_path") or "").strip() or None,
                lm_backend=str(get("lm_backend", "vllm") or "vllm"),
                lm_temperature=_to_float(get("lm_temperature"), _LM_DEFAULT_TEMPERATURE) or _LM_DEFAULT_TEMPERATURE,
                lm_cfg_scale=_to_float(get("lm_cfg_scale"), _LM_DEFAULT_CFG_SCALE) or _LM_DEFAULT_CFG_SCALE,
                lm_top_k=_to_int(get("lm_top_k"), None),
                lm_top_p=_to_float(get("lm_top_p"), _LM_DEFAULT_TOP_P),
                lm_repetition_penalty=_to_float(get("lm_repetition_penalty"), 1.0) or 1.0,
                lm_negative_prompt=str(get("lm_negative_prompt", "NO USER INPUT") or "NO USER INPUT"),
                constrained_decoding=_to_bool(_get_any("constrained_decoding", "constrainedDecoding", "constrained"), True),
                constrained_decoding_debug=_to_bool(_get_any("constrained_decoding_debug", "constrainedDecodingDebug"), False),
                use_cot_caption=_to_bool(_get_any("use_cot_caption", "cot_caption", "cot-caption"), True),
                use_cot_language=_to_bool(_get_any("use_cot_language", "cot_language", "cot-language"), True),
                is_format_caption=_to_bool(_get_any("is_format_caption", "isFormatCaption"), False),
            )

        def _first_value(v: Any) -> Any:
            if isinstance(v, list) and v:
                return v[0]
            return v

        if content_type.startswith("application/json"):
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="JSON payload must be an object")
            req = _build_req_from_mapping(body, reference_audio_path=None, src_audio_path=None)

        elif content_type.endswith("+json"):
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="JSON payload must be an object")
            req = _build_req_from_mapping(body, reference_audio_path=None, src_audio_path=None)

        elif content_type.startswith("multipart/form-data"):
            form = await request.form()

            ref_up = form.get("reference_audio")
            src_up = form.get("src_audio")

            reference_audio_path = None
            src_audio_path = None

            if isinstance(ref_up, StarletteUploadFile):
                reference_audio_path = await _save_upload_to_temp(ref_up, prefix="reference_audio")
                temp_files.append(reference_audio_path)
            else:
                reference_audio_path = str(form.get("reference_audio_path") or "").strip() or None

            if isinstance(src_up, StarletteUploadFile):
                src_audio_path = await _save_upload_to_temp(src_up, prefix="src_audio")
                temp_files.append(src_audio_path)
            else:
                src_audio_path = str(form.get("src_audio_path") or "").strip() or None

            req = _build_req_from_mapping(form, reference_audio_path=reference_audio_path, src_audio_path=src_audio_path)

        elif content_type.startswith("application/x-www-form-urlencoded"):
            form = await request.form()
            reference_audio_path = str(form.get("reference_audio_path") or "").strip() or None
            src_audio_path = str(form.get("src_audio_path") or "").strip() or None
            req = _build_req_from_mapping(form, reference_audio_path=reference_audio_path, src_audio_path=src_audio_path)

        else:
            raw = await request.body()
            raw_stripped = raw.lstrip()
            # Best-effort: accept missing/incorrect Content-Type if payload is valid JSON.
            if raw_stripped.startswith(b"{") or raw_stripped.startswith(b"["):
                try:
                    body = json.loads(raw.decode("utf-8"))
                    if isinstance(body, dict):
                        req = _build_req_from_mapping(body, reference_audio_path=None, src_audio_path=None)
                    else:
                        raise HTTPException(status_code=400, detail="JSON payload must be an object")
                except HTTPException:
                    raise
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid JSON body (hint: set 'Content-Type: application/json')",
                    )
            # Best-effort: parse key=value bodies even if Content-Type is missing.
            elif raw_stripped and b"=" in raw:
                parsed = urllib.parse.parse_qs(raw.decode("utf-8"), keep_blank_values=True)
                flat = {k: _first_value(v) for k, v in parsed.items()}
                reference_audio_path = str(flat.get("reference_audio_path") or "").strip() or None
                src_audio_path = str(flat.get("src_audio_path") or "").strip() or None
                req = _build_req_from_mapping(flat, reference_audio_path=reference_audio_path, src_audio_path=src_audio_path)
            else:
                raise HTTPException(
                    status_code=415,
                    detail=(
                        f"Unsupported Content-Type: {content_type or '(missing)'}; "
                        "use application/json, application/x-www-form-urlencoded, or multipart/form-data"
                    ),
                )

        rec = store.create()

        q: asyncio.Queue = app.state.job_queue
        if q.full():
            for p in temp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass
            raise HTTPException(status_code=429, detail="Server busy: queue is full")

        if temp_files:
            async with app.state.job_temp_files_lock:
                app.state.job_temp_files[rec.job_id] = temp_files

        async with app.state.pending_lock:
            app.state.pending_ids.append(rec.job_id)
            position = len(app.state.pending_ids)

        await q.put((rec.job_id, req))
        return CreateJobResponse(job_id=rec.job_id, status="queued", queue_position=position)

    @app.post("/v1/music/random", response_model=CreateJobResponse)
    async def create_random_sample_job(request: Request) -> CreateJobResponse:
        """Create a sample-mode job that auto-generates caption/lyrics via LM."""

        thinking_value: Any = None
        content_type = (request.headers.get("content-type") or "").lower()
        body_dict: Dict[str, Any] = {}

        if "json" in content_type:
            try:
                payload = await request.json()
                if isinstance(payload, dict):
                    body_dict = payload
            except Exception:
                body_dict = {}

        if not body_dict and request.query_params:
            body_dict = dict(request.query_params)

        thinking_value = body_dict.get("thinking")
        if thinking_value is None:
            thinking_value = body_dict.get("Thinking")

        thinking_flag = _to_bool(thinking_value, True)

        req = GenerateMusicRequest(
            caption="",
            lyrics="",
            thinking=thinking_flag,
            sample_mode=True,
        )

        rec = store.create()
        q: asyncio.Queue = app.state.job_queue
        if q.full():
            raise HTTPException(status_code=429, detail="Server busy: queue is full")

        async with app.state.pending_lock:
            app.state.pending_ids.append(rec.job_id)
            position = len(app.state.pending_ids)

        await q.put((rec.job_id, req))
        return CreateJobResponse(job_id=rec.job_id, status="queued", queue_position=position)

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str) -> JobResponse:
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Job not found")

        pos = 0
        eta = None
        async with app.state.stats_lock:
            avg = float(getattr(app.state, "avg_job_seconds", INITIAL_AVG_JOB_SECONDS))

        if rec.status == "queued":
            pos = await _queue_position(job_id)
            eta = await _eta_seconds_for_position(pos)

        return JobResponse(
            job_id=rec.job_id,
            status=rec.status,
            created_at=rec.created_at,
            started_at=rec.started_at,
            finished_at=rec.finished_at,
            queue_position=pos,
            eta_seconds=eta,
            avg_job_seconds=avg,
            result=JobResult(**rec.result) if rec.result else None,
            error=rec.error,
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint for service status."""
        return {
            "status": "ok",
            "service": "ACE-Step API",
            "version": "1.0",
        }

    @app.get("/v1/audio")
    async def get_audio(path: str):
        """Serve audio file by path."""
        from fastapi.responses import FileResponse

        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        media_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        media_type = media_types.get(ext, "audio/mpeg")

        return FileResponse(path, media_type=media_type)

    return app


app = create_app()


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="ACE-Step API server")
    parser.add_argument(
        "--host",
        default=os.getenv("ACESTEP_API_HOST", "127.0.0.1"),
        help="Bind host (default from ACESTEP_API_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ACESTEP_API_PORT", "8001")),
        help="Bind port (default from ACESTEP_API_PORT or 8001)",
    )
    args = parser.parse_args()

    # IMPORTANT: in-memory queue/store -> workers MUST be 1
    uvicorn.run(
        "acestep.api_server:app",
        host=str(args.host),
        port=int(args.port),
        reload=False,
        workers=1,
    )

if __name__ == "__main__":
    main()
