"""OpenRouter-compatible API server for ACE-Step V1.5.

Provides OpenAI Chat Completions API format for text-to-music generation.

Endpoints:
- GET  /api/v1/models       List available models with pricing
- POST /v1/chat/completions Generate music from text prompt
- GET  /health              Health check

Usage:
    python -m openrouter.openrouter_api_server --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import (
    GenerationParams,
    GenerationConfig,
    generate_music,
)

# =============================================================================
# Constants
# =============================================================================

MODEL_ID = "acestep/music-v1.5"
MODEL_NAME = "ACE-Step Music Generator V1.5"
MODEL_CREATED = 1706688000  # Unix timestamp

# Pricing (USD per token/unit) - adjust as needed
PRICING_PROMPT = "0.000005"
PRICING_COMPLETION = "0.02"
PRICING_REQUEST = "0"

# =============================================================================
# API Key Authentication
# =============================================================================

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


# =============================================================================
# Request/Response Models (OpenAI Compatible)
# =============================================================================

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[ChatMessage] = Field(default_factory=list)
    modalities: List[str] = Field(default=["audio"])
    temperature: float = 0.85
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    # ACE-Step specific parameters (optional)
    lyrics: str = ""
    duration: Optional[float] = None
    bpm: Optional[int] = None
    vocal_language: str = "en"
    instrumental: bool = False


class AudioOutput(BaseModel):
    id: str = ""
    data: str = ""  # Base64 encoded audio
    transcript: str = ""  # Optional transcript/lyrics


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    audio: Optional[AudioOutput] = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = MODEL_ID
    choices: List[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class ModelInfo(BaseModel):
    id: str
    name: str
    created: int
    description: str
    input_modalities: List[str]
    output_modalities: List[str]
    context_length: int
    pricing: Dict[str, str]
    supported_sampling_parameters: List[str]


class ModelsResponse(BaseModel):
    data: List[ModelInfo]


# =============================================================================
# Helper Functions
# =============================================================================

def _get_project_root() -> str:
    """Get the project root directory."""
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current_file))


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_prompt_and_lyrics(messages: List[ChatMessage]) -> tuple[str, str]:
    """
    Extract prompt (caption) and lyrics from messages.
    
    Simple parsing:
    - Last user message content is used as the prompt
    - If content contains [lyrics]...[/lyrics], extract lyrics separately
    """
    prompt = ""
    lyrics = ""
    
    # Get the last user message
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            content = msg.content.strip()
            
            # Check for [lyrics] tag
            lyrics_start = content.lower().find("[lyrics]")
            lyrics_end = content.lower().find("[/lyrics]")
            
            if lyrics_start != -1 and lyrics_end != -1:
                lyrics = content[lyrics_start + 8:lyrics_end].strip()
                prompt = (content[:lyrics_start] + content[lyrics_end + 9:]).strip()
            else:
                prompt = content
            break
    
    return prompt, lyrics


def _read_audio_as_base64(file_path: str) -> str:
    """Read audio file and return Base64 encoded string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # API Key from environment
    api_key = os.getenv("OPENROUTER_API_KEY", None)
    set_api_key(api_key)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan: initialize and cleanup resources."""
        
        # Setup cache directories
        project_root = _get_project_root()
        cache_root = os.path.join(project_root, ".cache", "openrouter")
        tmp_root = os.path.join(cache_root, "tmp")
        
        for p in [cache_root, tmp_root]:
            os.makedirs(p, exist_ok=True)
        
        # Initialize handlers
        handler = AceStepHandler()
        llm_handler = LLMHandler()
        
        app.state.handler = handler
        app.state.llm_handler = llm_handler
        app.state._initialized = False
        app.state._init_error = None
        app.state._init_lock = asyncio.Lock()
        app.state._llm_initialized = False
        app.state.temp_audio_dir = tmp_root
        
        # Thread pool for blocking operations
        executor = ThreadPoolExecutor(max_workers=1)
        app.state.executor = executor
        
        async def _ensure_initialized() -> None:
            """Lazy initialization of models."""
            if app.state._initialized:
                return
            if app.state._init_error:
                raise RuntimeError(app.state._init_error)
            
            async with app.state._init_lock:
                if app.state._initialized:
                    return
                if app.state._init_error:
                    raise RuntimeError(app.state._init_error)
                
                # Initialize DiT model
                config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
                device = os.getenv("ACESTEP_DEVICE", "auto")
                use_flash_attention = _env_bool("ACESTEP_USE_FLASH_ATTENTION", True)
                offload_to_cpu = _env_bool("ACESTEP_OFFLOAD_TO_CPU", False)
                
                status_msg, ok = handler.initialize_service(
                    project_root=project_root,
                    config_path=config_path,
                    device=device,
                    use_flash_attention=use_flash_attention,
                    compile_model=False,
                    offload_to_cpu=offload_to_cpu,
                )
                
                if not ok:
                    app.state._init_error = status_msg
                    raise RuntimeError(status_msg)
                
                app.state._initialized = True
                
                # Initialize LLM (optional, for thinking mode)
                try:
                    checkpoint_dir = os.path.join(project_root, "checkpoints")
                    lm_model_path = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
                    backend = os.getenv("ACESTEP_LM_BACKEND", "vllm")
                    
                    lm_status, lm_ok = llm_handler.initialize(
                        checkpoint_dir=checkpoint_dir,
                        lm_model_path=lm_model_path,
                        backend=backend,
                        device=device,
                    )
                    app.state._llm_initialized = lm_ok
                except Exception:
                    app.state._llm_initialized = False
        
        app.state._ensure_initialized = _ensure_initialized
        
        try:
            yield
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    
    app = FastAPI(
        title="ACE-Step OpenRouter API",
        version="1.0",
        description="OpenRouter-compatible API for text-to-music generation",
        lifespan=lifespan,
    )
    
    # -------------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------------
    
    @app.get("/api/v1/models", response_model=ModelsResponse)
    async def list_models(_: None = Depends(verify_api_key)) -> ModelsResponse:
        """List available models with capabilities and pricing."""
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=MODEL_ID,
                    name=MODEL_NAME,
                    created=MODEL_CREATED,
                    description="High-performance text-to-music generation model. Supports multiple styles, lyrics input, and various audio durations.",
                    input_modalities=["text"],
                    output_modalities=["audio"],
                    context_length=4096,
                    pricing={
                        "prompt": PRICING_PROMPT,
                        "completion": PRICING_COMPLETION,
                        "request": PRICING_REQUEST,
                    },
                    supported_sampling_parameters=["temperature", "top_p"],
                )
            ]
        )
    
    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
        _: None = Depends(verify_api_key),
    ) -> ChatCompletionResponse:
        """
        Generate music from text prompt (OpenAI Chat Completions format).
        
        The last user message is used as the music description (caption).
        Optionally include lyrics using [lyrics]...[/lyrics] tags.
        """
        await app.state._ensure_initialized()
        
        # Extract prompt and lyrics from messages
        prompt, lyrics_from_msg = _extract_prompt_and_lyrics(request.messages)
        lyrics = request.lyrics or lyrics_from_msg
        
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided in messages")
        
        # Determine if instrumental
        instrumental = request.instrumental or not lyrics
        
        def _blocking_generate() -> Dict[str, Any]:
            """Run music generation in thread pool."""
            h: AceStepHandler = app.state.handler
            llm = app.state.llm_handler if app.state._llm_initialized else None
            
            # Build generation parameters
            params = GenerationParams(
                task_type="text2music",
                caption=prompt,
                lyrics=lyrics,
                instrumental=instrumental,
                vocal_language=request.vocal_language,
                bpm=request.bpm,
                duration=request.duration if request.duration else -1.0,
                inference_steps=8,
                guidance_scale=7.0,
                lm_temperature=request.temperature,
                lm_top_p=request.top_p,
                thinking=False,  # Simple mode, no LLM thinking
                use_cot_caption=False,
                use_cot_language=False,
            )
            
            config = GenerationConfig(
                batch_size=1,  # Single audio output
                use_random_seed=True,
                audio_format="mp3",
            )
            
            result = generate_music(
                dit_handler=h,
                llm_handler=llm,
                params=params,
                config=config,
                save_dir=app.state.temp_audio_dir,
            )
            
            if not result.success:
                raise RuntimeError(result.error or "Music generation failed")
            
            # Get first audio path
            audio_path = None
            if result.audios and result.audios[0].get("path"):
                audio_path = result.audios[0]["path"]
            
            if not audio_path or not os.path.exists(audio_path):
                raise RuntimeError("No audio file generated")
            
            # Read and encode audio
            audio_base64 = _read_audio_as_base64(audio_path)
            
            return {
                "audio_data": audio_base64,
                "audio_path": audio_path,
                "lyrics": lyrics,
            }
        
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(app.state.executor, _blocking_generate)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        
        # Build response
        audio_id = f"audio_{int(time.time())}_{os.urandom(4).hex()}"
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{os.urandom(8).hex()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(
                        role="assistant",
                        audio=AudioOutput(
                            id=audio_id,
                            data=result["audio_data"],
                            transcript=result.get("lyrics", ""),
                        ),
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=100,  # Placeholder
                total_tokens=len(prompt.split()) + 100,
            ),
        )
        
        return response
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "service": "ACE-Step OpenRouter API",
            "version": "1.0",
        }
    
    return app


# Create app instance
app = create_app()


def main() -> None:
    """Run the server."""
    import uvicorn
    
    parser = argparse.ArgumentParser(description="ACE-Step OpenRouter API server")
    parser.add_argument(
        "--host",
        default=os.getenv("OPENROUTER_HOST", "127.0.0.1"),
        help="Bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("OPENROUTER_PORT", "8002")),
        help="Bind port (default: 8002)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="API key for authentication",
    )
    args = parser.parse_args()
    
    if args.api_key:
        os.environ["OPENROUTER_API_KEY"] = args.api_key
    
    uvicorn.run(
        "openrouter.openrouter_api_server:app",
        host=str(args.host),
        port=int(args.port),
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
