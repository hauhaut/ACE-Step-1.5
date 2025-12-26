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
import re
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

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile

from .handler import AceStepHandler
from .llm_inference import LLMHandler


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class GenerateMusicRequest(BaseModel):
    caption: str = Field(default="", description="Text caption describing the music")
    lyrics: str = Field(default="", description="Lyric text")

    # New API semantics:
    # - thinking=True: use 5Hz LM to generate audio codes (lm-dit behavior)
    # - thinking=False: do not use LM to generate codes (dit behavior)
    # Regardless of thinking, if some metas are missing, server may use LM to fill them.
    thinking: bool = False

    bpm: Optional[int] = None
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

    instruction: str = "Fill the audio semantic mask based on the given conditions:"
    audio_cover_strength: float = 1.0
    task_type: str = "text2music"

    use_adg: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0

    audio_format: str = "mp3"
    use_tiled_decode: bool = True

    # 5Hz LM (server-side): used for metadata completion and (when thinking=True) codes generation.
    lm_model_path: Optional[str] = None  # e.g. "acestep-5Hz-lm-0.6B"
    lm_backend: Literal["vllm", "pt"] = "vllm"

    # Align defaults with `acestep/gradio_ui.py` and `feishu_bot/config.py`
    # to improve lyric adherence in lm-dit mode.
    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.0
    lm_top_k: Optional[int] = None
    lm_top_p: Optional[float] = 0.9
    lm_repetition_penalty: float = 1.0
    lm_negative_prompt: str = "NO USER INPUT"


_LM_DEFAULT_TEMPERATURE = 0.85
_LM_DEFAULT_CFG_SCALE = 2.0
_LM_DEFAULT_TOP_P = 0.9
_DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
_DEFAULT_LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"


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

    # 5Hz LM metadata (present when server invoked LM)
    # Keep a raw-ish dict for clients that expect a `metas` object.
    metas: Dict[str, Any] = Field(default_factory=dict)
    bpm: Optional[int] = None
    duration: Optional[float] = None
    genres: Optional[str] = None
    keyscale: Optional[str] = None
    timesignature: Optional[str] = None


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


def _to_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s == "":
        return default
    return int(s)


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, float):
        return v
    s = str(v).strip()
    if s == "":
        return default
    return float(s)


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
                config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
                device = os.getenv("ACESTEP_DEVICE", "auto")

                use_flash_attention = _env_bool("ACESTEP_USE_FLASH_ATTENTION", True)
                offload_to_cpu = _env_bool("ACESTEP_OFFLOAD_TO_CPU", False)
                offload_dit_to_cpu = _env_bool("ACESTEP_OFFLOAD_DIT_TO_CPU", False)

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
            h: AceStepHandler = app.state.handler
            llm: LLMHandler = app.state.llm_handler
            executor: ThreadPoolExecutor = app.state.executor

            await _ensure_initialized()
            job_store.mark_running(job_id)

            def _blocking_generate() -> Dict[str, Any]:
                def _normalize_optional_int(v: Any) -> Optional[int]:
                    if v is None:
                        return None
                    try:
                        iv = int(v)
                    except Exception:
                        return None
                    return None if iv == 0 else iv

                def _normalize_optional_float(v: Any) -> Optional[float]:
                    if v is None:
                        return None
                    try:
                        fv = float(v)
                    except Exception:
                        return None
                    # gradio treats 1.0 as disabled for top_p
                    return None if fv >= 1.0 else fv

                def _maybe_fill_from_metadata(current: GenerateMusicRequest, meta: Dict[str, Any]) -> tuple[Optional[int], str, str, Optional[float]]:
                    # Fill only when user did not provide values
                    bpm_val = current.bpm
                    if bpm_val is None:
                        try:
                            m = meta.get("bpm")
                            if m not in (None, "", "N/A"):
                                bpm_val = int(float(m))
                        except Exception:
                            bpm_val = current.bpm

                    key_scale_val = current.key_scale
                    if not key_scale_val:
                        m = meta.get("keyscale", meta.get("key_scale", ""))
                        if m not in (None, "", "N/A"):
                            key_scale_val = str(m)

                    time_sig_val = current.time_signature
                    if not time_sig_val:
                        m = meta.get("timesignature", meta.get("time_signature", ""))
                        if m not in (None, "", "N/A"):
                            time_sig_val = str(m)

                    dur_val = current.audio_duration
                    if dur_val is None:
                        m = meta.get("duration", meta.get("audio_duration"))
                        try:
                            if m not in (None, "", "N/A"):
                                dur_val = float(m)
                                if dur_val <= 0:
                                    dur_val = None
                                # Avoid truncating lyrical songs when LM predicts a very short duration.
                                # (Users can still force a short duration by explicitly setting `audio_duration`.)
                                if dur_val is not None and (current.lyrics or "").strip():
                                    min_dur = float(os.getenv("ACESTEP_LM_MIN_DURATION_SECONDS", "30"))
                                    if dur_val < min_dur:
                                        dur_val = None
                        except Exception:
                            dur_val = current.audio_duration

                    return bpm_val, key_scale_val, time_sig_val, dur_val

                def _estimate_duration_from_lyrics(lyrics: str) -> Optional[float]:
                    lyrics = (lyrics or "").strip()
                    if not lyrics:
                        return None

                    # Best-effort heuristic: singing rate ~ 2.2 words/sec for English-like lyrics.
                    # For languages without spaces, fall back to non-space char count.
                    words = re.findall(r"[A-Za-z0-9']+", lyrics)
                    if len(words) >= 8:
                        words_per_sec = float(os.getenv("ACESTEP_LYRICS_WORDS_PER_SEC", "2.2"))
                        est = len(words) / max(0.5, words_per_sec)
                    else:
                        non_space = len(re.sub(r"\s+", "", lyrics))
                        chars_per_sec = float(os.getenv("ACESTEP_LYRICS_CHARS_PER_SEC", "12"))
                        est = non_space / max(4.0, chars_per_sec)

                    min_dur = float(os.getenv("ACESTEP_LYRICS_MIN_DURATION_SECONDS", "45"))
                    max_dur = float(os.getenv("ACESTEP_LYRICS_MAX_DURATION_SECONDS", "180"))
                    return float(min(max(est, min_dur), max_dur))

                def _extract_lm_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
                    def _none_if_na(v: Any) -> Any:
                        if v is None:
                            return None
                        if isinstance(v, str) and v.strip() in {"", "N/A"}:
                            return None
                        return v

                    out: Dict[str, Any] = {}

                    bpm_raw = _none_if_na(meta.get("bpm"))
                    try:
                        out["bpm"] = int(float(bpm_raw)) if bpm_raw is not None else None
                    except Exception:
                        out["bpm"] = None

                    dur_raw = _none_if_na(meta.get("duration"))
                    try:
                        out["duration"] = float(dur_raw) if dur_raw is not None else None
                    except Exception:
                        out["duration"] = None

                    genres_raw = _none_if_na(meta.get("genres"))
                    out["genres"] = str(genres_raw) if genres_raw is not None else None

                    keyscale_raw = _none_if_na(meta.get("keyscale", meta.get("key_scale")))
                    out["keyscale"] = str(keyscale_raw) if keyscale_raw is not None else None

                    ts_raw = _none_if_na(meta.get("timesignature", meta.get("time_signature")))
                    out["timesignature"] = str(ts_raw) if ts_raw is not None else None

                    return out

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

                # Optional: generate 5Hz LM codes server-side
                audio_code_string = req.audio_code_string
                bpm_val = req.bpm
                key_scale_val = req.key_scale
                time_sig_val = req.time_signature
                audio_duration_val = req.audio_duration

                thinking = bool(getattr(req, "thinking", False))

                # If LM-generated code hints are used, a too-strong cover strength can suppress lyric/vocal conditioning.
                # We keep backward compatibility: only auto-adjust when user didn't override (still at default 1.0).
                audio_cover_strength_val = float(req.audio_cover_strength)

                lm_fields: Dict[str, Any] = {}

                # Determine effective batch size (used for per-sample LM code diversity)
                effective_batch_size = req.batch_size
                if effective_batch_size is None:
                    try:
                        effective_batch_size = int(getattr(h, "batch_size", 1))
                    except Exception:
                        effective_batch_size = 1
                effective_batch_size = max(1, int(effective_batch_size))

                has_codes = bool(audio_code_string and str(audio_code_string).strip())
                need_lm_codes = bool(thinking) and (not has_codes)
                need_lm_metas = (
                    (bpm_val is None)
                    or (not (key_scale_val or "").strip())
                    or (not (time_sig_val or "").strip())
                    or (audio_duration_val is None)
                )

                if need_lm_metas or need_lm_codes:
                    # Lazy init 5Hz LM once
                    with app.state._llm_init_lock:
                        if getattr(app.state, "_llm_initialized", False) is False and getattr(app.state, "_llm_init_error", None) is None:
                            project_root = _get_project_root()
                            checkpoint_dir = os.path.join(project_root, "checkpoints")
                            lm_model_path = (req.lm_model_path or os.getenv("ACESTEP_LM_MODEL_PATH") or "acestep-5Hz-lm-0.6B").strip()
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

                    if getattr(app.state, "_llm_init_error", None):
                        # If codes generation is required, fail hard.
                        if need_lm_codes:
                            raise RuntimeError(f"5Hz LM init failed: {app.state._llm_init_error}")
                        # Otherwise, skip LM best-effort (fallback to default/meta-less behavior)
                    else:
                        lm_infer = "llm_dit" if need_lm_codes else "dit"

                        def _lm_call() -> tuple[Dict[str, Any], str, str]:
                            return llm.generate_with_stop_condition(
                                caption=req.caption,
                                lyrics=req.lyrics,
                                infer_type=lm_infer,
                                temperature=float(req.lm_temperature),
                                cfg_scale=max(1.0, float(req.lm_cfg_scale)),
                                negative_prompt=str(req.lm_negative_prompt or "NO USER INPUT"),
                                top_k=_normalize_optional_int(req.lm_top_k),
                                top_p=_normalize_optional_float(req.lm_top_p),
                                repetition_penalty=float(req.lm_repetition_penalty),
                            )

                        meta, codes, status = _lm_call()

                        if need_lm_codes:
                            if not codes:
                                raise RuntimeError(f"5Hz LM generation failed: {status}")

                            # LM once per job; rely on DiT seeds for batch diversity.
                            # For convenience, replicate the same codes across the batch.
                            if effective_batch_size > 1:
                                audio_code_string = [codes] * effective_batch_size
                            else:
                                audio_code_string = codes

                        # Always expose LM metas when we invoked LM (even if user already set some fields).
                        lm_fields = {
                            "metas": _normalize_metas(meta),
                            **_extract_lm_fields(meta),
                        }

                        # Fill only missing fields (user-provided values win)
                        bpm_val, key_scale_val, time_sig_val, audio_duration_val = _maybe_fill_from_metadata(req, meta)

                        # If user provided lyrics but LM didn't provide a usable duration, estimate a longer duration.
                        if audio_duration_val is None and (req.audio_duration is None):
                            est = _estimate_duration_from_lyrics(req.lyrics)
                            if est is not None:
                                audio_duration_val = est

                        # Optional: auto-tune LM cover strength (opt-in) to avoid suppressing lyric/vocal conditioning.
                        if thinking and audio_cover_strength_val >= 0.999 and (req.lyrics or "").strip():
                            tuned = os.getenv("ACESTEP_LM_COVER_STRENGTH")
                            if tuned is not None and tuned.strip() != "":
                                audio_cover_strength_val = float(tuned)

                # Align behavior:
                # - thinking=False: metas only (ignore audio codes), keep text2music.
                # - thinking=True: metas + audio codes, run in cover mode with LM instruction.
                instruction_val = req.instruction
                task_type_val = (req.task_type or "").strip() or "text2music"

                if not thinking:
                    audio_code_string = ""
                    if task_type_val == "cover":
                        task_type_val = "text2music"
                    if (instruction_val or "").strip() in {"", _DEFAULT_LM_INSTRUCTION}:
                        instruction_val = _DEFAULT_DIT_INSTRUCTION

                if thinking:
                    task_type_val = "cover"
                    if (instruction_val or "").strip() in {"", _DEFAULT_DIT_INSTRUCTION}:
                        instruction_val = _DEFAULT_LM_INSTRUCTION

                    if not (audio_code_string and str(audio_code_string).strip()):
                        # thinking=True requires codes generation.
                        raise RuntimeError("thinking=true requires non-empty audio codes (LM generation failed).")

                first, second, paths, gen_info, status_msg, seed_value, *_ = h.generate_music(
                    captions=req.caption,
                    lyrics=req.lyrics,
                    bpm=bpm_val,
                    key_scale=key_scale_val,
                    time_signature=time_sig_val,
                    vocal_language=req.vocal_language,
                    inference_steps=req.inference_steps,
                    guidance_scale=req.guidance_scale,
                    use_random_seed=req.use_random_seed,
                    seed=("-1" if (req.use_random_seed and int(req.seed) < 0) else str(req.seed)),
                    reference_audio=req.reference_audio_path,
                    audio_duration=audio_duration_val,
                    batch_size=req.batch_size,
                    src_audio=req.src_audio_path,
                    audio_code_string=audio_code_string,
                    repainting_start=req.repainting_start,
                    repainting_end=req.repainting_end,
                    instruction=instruction_val,
                    audio_cover_strength=audio_cover_strength_val,
                    task_type=task_type_val,
                    use_adg=req.use_adg,
                    cfg_interval_start=req.cfg_interval_start,
                    cfg_interval_end=req.cfg_interval_end,
                    audio_format=req.audio_format,
                    use_tiled_decode=req.use_tiled_decode,
                    progress=None,
                )
                return {
                    "first_audio_path": first,
                    "second_audio_path": second,
                    "audio_paths": paths,
                    "generation_info": gen_info,
                    "status_message": status_msg,
                    "seed_value": seed_value,
                    **lm_fields,
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

            return GenerateMusicRequest(
                caption=str(get("caption", "") or ""),
                lyrics=str(get("lyrics", "") or ""),
                thinking=_to_bool(get("thinking"), False),
                bpm=_to_int(get("bpm"), None),
                key_scale=str(get("key_scale", "") or ""),
                time_signature=str(get("time_signature", "") or ""),
                vocal_language=str(get("vocal_language", "en") or "en"),
                inference_steps=_to_int(get("inference_steps"), 8) or 8,
                guidance_scale=_to_float(get("guidance_scale"), 7.0) or 7.0,
                use_random_seed=_to_bool(get("use_random_seed"), True),
                seed=_to_int(get("seed"), -1) or -1,
                reference_audio_path=reference_audio_path,
                src_audio_path=src_audio_path,
                audio_duration=_to_float(get("audio_duration"), None),
                batch_size=_to_int(get("batch_size"), None),
                audio_code_string=str(get("audio_code_string", "") or ""),
                repainting_start=_to_float(get("repainting_start"), 0.0) or 0.0,
                repainting_end=_to_float(get("repainting_end"), None),
                instruction=str(get("instruction", _DEFAULT_DIT_INSTRUCTION) or ""),
                audio_cover_strength=_to_float(get("audio_cover_strength"), 1.0) or 1.0,
                task_type=str(get("task_type", "text2music") or "text2music"),
                use_adg=_to_bool(get("use_adg"), False),
                cfg_interval_start=_to_float(get("cfg_interval_start"), 0.0) or 0.0,
                cfg_interval_end=_to_float(get("cfg_interval_end"), 1.0) or 1.0,
                audio_format=str(get("audio_format", "mp3") or "mp3"),
                use_tiled_decode=_to_bool(get("use_tiled_decode"), True),
                lm_model_path=str(get("lm_model_path") or "").strip() or None,
                lm_backend=str(get("lm_backend", "vllm") or "vllm"),
                lm_temperature=_to_float(get("lm_temperature"), _LM_DEFAULT_TEMPERATURE) or _LM_DEFAULT_TEMPERATURE,
                lm_cfg_scale=_to_float(get("lm_cfg_scale"), _LM_DEFAULT_CFG_SCALE) or _LM_DEFAULT_CFG_SCALE,
                lm_top_k=_to_int(get("lm_top_k"), None),
                lm_top_p=_to_float(get("lm_top_p"), _LM_DEFAULT_TOP_P),
                lm_repetition_penalty=_to_float(get("lm_repetition_penalty"), 1.0) or 1.0,
                lm_negative_prompt=str(get("lm_negative_prompt", "NO USER INPUT") or "NO USER INPUT"),
            )

        def _first_value(v: Any) -> Any:
            if isinstance(v, list) and v:
                return v[0]
            return v

        if content_type.startswith("application/json"):
            body = await request.json()
            req = GenerateMusicRequest(**body)

        elif content_type.endswith("+json"):
            body = await request.json()
            req = GenerateMusicRequest(**body)

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
                        req = GenerateMusicRequest(**body)
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

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.getenv("ACESTEP_API_HOST", "127.0.0.1")
    port = int(os.getenv("ACESTEP_API_PORT", "8001"))

    # IMPORTANT: in-memory queue/store -> workers MUST be 1
    uvicorn.run("acestep.api_server:app", host=host, port=port, reload=False, workers=1)


if __name__ == "__main__":
    main()
