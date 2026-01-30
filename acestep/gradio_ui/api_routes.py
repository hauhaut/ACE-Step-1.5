"""
Gradio API Routes Module
Add API endpoints compatible with api_server.py to Gradio application
"""
import json
import os
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Depends, Header
from fastapi.responses import FileResponse

# API Key storage (set via setup_api_routes)
_api_key: Optional[str] = None


def set_api_key(key: Optional[str]):
    """Set the API key for authentication"""
    global _api_key
    _api_key = key


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key from Authorization header"""
    if _api_key is None:
        return  # No auth required

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Support "Bearer <key>" format
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization

    if token != _api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# Use diskcache to store results
try:
    import diskcache
    _cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "api_results")
    os.makedirs(_cache_dir, exist_ok=True)
    _result_cache = diskcache.Cache(_cache_dir)
    DISKCACHE_AVAILABLE = True
except ImportError:
    _result_cache = {}
    DISKCACHE_AVAILABLE = False

RESULT_EXPIRE_SECONDS = 7 * 24 * 60 * 60  # 7 days expiration
RESULT_KEY_PREFIX = "ace_step_v1.5_"


def store_result(task_id: str, result: dict, status: str = "succeeded"):
    """Store result to diskcache"""
    data = {
        "result": result,
        "created_at": time.time(),
        "status": status
    }
    key = f"{RESULT_KEY_PREFIX}{task_id}"
    if DISKCACHE_AVAILABLE:
        _result_cache.set(key, data, expire=RESULT_EXPIRE_SECONDS)
    else:
        _result_cache[key] = data


def get_result(task_id: str) -> Optional[dict]:
    """Get result from diskcache"""
    key = f"{RESULT_KEY_PREFIX}{task_id}"
    if DISKCACHE_AVAILABLE:
        return _result_cache.get(key)
    else:
        return _result_cache.get(key)


router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "ACE-Step Gradio API",
        "version": "1.0",
    }


@router.get("/v1/models")
async def list_models(request: Request, _: None = Depends(verify_api_key)):
    """List available DiT models"""
    dit_handler = request.app.state.dit_handler

    models = []
    if dit_handler and dit_handler.model is not None:
        # Get current loaded model name
        config_path = getattr(dit_handler, 'config_path', '') or ''
        model_name = os.path.basename(config_path.rstrip("/\\")) if config_path else "unknown"
        models.append({
            "name": model_name,
            "is_default": True,
        })

    return {
        "models": models,
        "default_model": models[0]["name"] if models else None,
    }


@router.get("/v1/audio")
async def get_audio(path: str, _: None = Depends(verify_api_key)):
    """Download audio file"""
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


@router.post("/create_random_sample")
async def create_random_sample(request: Request, _: None = Depends(verify_api_key)):
    """Generate random music parameters via LLM"""
    llm_handler = request.app.state.llm_handler

    if not llm_handler or not llm_handler.llm_initialized:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    body = await request.json()
    query = body.get("query", "NO USER INPUT") or "NO USER INPUT"
    temperature = float(body.get("temperature", 0.85))

    from acestep.inference import create_sample

    try:
        result = create_sample(
            llm_handler=llm_handler,
            query=query,
            temperature=temperature,
            use_constrained_decoding=True,
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.status_message)

        return {
            "caption": result.caption,
            "lyrics": result.lyrics,
            "bpm": result.bpm,
            "key_scale": result.keyscale,
            "time_signature": result.timesignature,
            "duration": result.duration,
            "vocal_language": result.language or "unknown",
            "instrumental": result.instrumental,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query_result")
async def query_result(request: Request, _: None = Depends(verify_api_key)):
    """Batch query task results"""
    body = await request.json()
    task_ids = body.get("task_id_list", [])

    if isinstance(task_ids, str):
        try:
            task_ids = json.loads(task_ids)
        except Exception:
            task_ids = []

    results = []
    for task_id in task_ids:
        data = get_result(task_id)
        if data and data.get("status") == "succeeded":
            results.append({
                "task_id": task_id,
                "status": 1,
                "result": json.dumps(data["result"], ensure_ascii=False)
            })
        else:
            results.append({
                "task_id": task_id,
                "status": 0,
                "result": "[]"
            })

    return results


@router.post("/format_lyrics")
async def format_lyrics(request: Request, _: None = Depends(verify_api_key)):
    """Format lyrics via LLM"""
    llm_handler = request.app.state.llm_handler

    if not llm_handler or not llm_handler.llm_initialized:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    body = await request.json()
    caption = body.get("prompt", "") or ""
    lyrics = body.get("lyrics", "") or ""
    temperature = float(body.get("temperature", 0.85))

    from acestep.inference import format_sample

    try:
        result = format_sample(
            llm_handler=llm_handler,
            caption=caption,
            lyrics=lyrics,
            temperature=temperature,
            use_constrained_decoding=True,
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.status_message)

        return {
            "caption": result.caption or caption,
            "lyrics": result.lyrics or lyrics,
            "bpm": result.bpm,
            "key_scale": result.keyscale,
            "time_signature": result.timesignature,
            "duration": result.duration,
            "vocal_language": result.language or "unknown",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/release_task")
async def release_task(request: Request, _: None = Depends(verify_api_key)):
    """Create music generation task"""
    dit_handler = request.app.state.dit_handler
    llm_handler = request.app.state.llm_handler

    if not dit_handler or dit_handler.model is None:
        raise HTTPException(status_code=500, detail="DiT model not initialized")

    body = await request.json()
    task_id = str(uuid4())

    from acestep.inference import generate_music, GenerationParams, GenerationConfig

    try:
        # Build generation params
        params = GenerationParams(
            task_type=body.get("task_type", "text2music"),
            caption=body.get("prompt", ""),
            lyrics=body.get("lyrics", ""),
            bpm=body.get("bpm"),
            keyscale=body.get("key_scale", ""),
            timesignature=body.get("time_signature", ""),
            duration=body.get("audio_duration", -1),
            vocal_language=body.get("vocal_language", "en"),
            inference_steps=body.get("inference_steps", 8),
            guidance_scale=body.get("guidance_scale", 7.0),
            seed=body.get("seed", -1),
            thinking=body.get("thinking", False),
        )

        config = GenerationConfig(
            batch_size=body.get("batch_size", 2),
            use_random_seed=body.get("use_random_seed", True),
            audio_format=body.get("audio_format", "mp3"),
        )

        # Get temp directory
        import tempfile
        save_dir = tempfile.gettempdir()

        # Call generation function
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler if llm_handler and llm_handler.llm_initialized else None,
            params=params,
            config=config,
            save_dir=save_dir,
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or result.status_message)

        # Extract audio paths
        audio_paths = [a["path"] for a in result.audios if a.get("path")]

        # Build result data
        result_data = [{
            "file": p,
            "status": 1,
            "create_time": int(time.time()),
        } for p in audio_paths]

        # Store result
        store_result(task_id, result_data)

        return {"task_id": task_id, "status": "succeeded"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def setup_api_routes(demo, dit_handler, llm_handler, api_key: Optional[str] = None):
    """
    Mount API routes to Gradio application

    Args:
        demo: Gradio Blocks instance
        dit_handler: DiT handler
        llm_handler: LLM handler
        api_key: Optional API key for authentication
    """
    set_api_key(api_key)
    app = demo.app
    app.state.dit_handler = dit_handler
    app.state.llm_handler = llm_handler
    app.include_router(router)

