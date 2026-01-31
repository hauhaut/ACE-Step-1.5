"""
OpenRouter-Compatible API Server for ACE-Step Music Generation (Minimal)

Endpoints:
    - GET  /api/v1/models       - List available models
    - POST /v1/chat/completions - Generate music

Usage:
    uvicorn openrouter_server:app --host 0.0.0.0 --port 8000
"""
import os
import asyncio
import base64
import io
import tempfile
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from acestep.inference import GenerationConfig, GenerationParams, generate_music
from contextlib import asynccontextmanager

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler

# =============================================================================
# Config
# =============================================================================

MODEL_ID = "acestep/music-gen-v1"
MODEL_NAME = "ACE-Step Music Generator"
PRICE_PER_REQUEST = "0.05"  # USD
MAX_CONCURRENT = 4
TIMEOUT = 300

# Model handlers (init at startup)
dit_handler = None
llm_handler = None
semaphore: Optional[asyncio.Semaphore] = None


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    # OpenRouter 多模态参数
    modalities: Optional[List[str]] = None
    # Music params
    lyrics: Optional[str] = None
    bpm: Optional[int] = Field(None, ge=30, le=300)
    duration: Optional[float] = Field(None, ge=10, le=600)
    instrumental: Optional[bool] = False
    seed: Optional[int] = None


class AudioOutput(BaseModel):
    id: str
    data: str  # Base64
    expires_at: int


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    audio: Optional[AudioOutput] = None


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    name: str
    created: int
    description: str
    input_modalities: List[str]
    output_modalities: List[str]
    context_length: int
    pricing: Dict[str, str]
    supported_sampling_parameters: Optional[List[str]] = None


class ModelsResponse(BaseModel):
    data: List[ModelInfo]




def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global semaphore, dit_handler, llm_handler
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # 创建 handler 实例
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()
    
    # 初始化 DiT 模型
    project_root = _get_project_root()
    config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
    device = os.getenv("ACESTEP_DEVICE", "auto")
    
    status_msg, ok = dit_handler.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device=device,
        use_flash_attention=True,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"DiT init failed: {status_msg}")
    
    # 初始化 LLM 模型
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    lm_model_path = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
    lm_backend = os.getenv("ACESTEP_LM_BACKEND", "vllm")
    lm_device = os.getenv("ACESTEP_LM_DEVICE", device)
    
    status_msg, ok = llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model_path,
        backend=lm_backend,
        device=lm_device,
        offload_to_cpu=False,
        dtype=dit_handler.dtype,
    )
    if not ok:
        raise RuntimeError(f"LLM init failed: {status_msg}")
    
    logger.info("Models initialized")
    
    yield  # 应用运行中
    
    # Shutdown 清理（可选）
    logger.info("Shutting down")



# =============================================================================
# App
# =============================================================================

app = FastAPI(title="ACE-Step Music API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# =============================================================================
# Helpers
# =============================================================================

def tensor_to_base64(tensor: torch.Tensor, sr: int = 48000) -> str:
    """将音频张量转换为 base64 编码的字符串
    
    注意：MP3 格式需要使用临时文件，因为 torchaudio 的 ffmpeg 后端不支持 BytesIO
    """
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    # 使用临时文件保存 MP3，然后读取并编码
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_file:
        torchaudio.save(tmp_file.name, tensor, sr, format="mp3")
        with open(tmp_file.name, "rb") as f:
            return base64.b64encode(f.read()).decode()


async def run_gen(params, config, save_dir):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: generate_music(dit_handler, llm_handler, params, config, save_dir)
    )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/api/v1/models", response_model=ModelsResponse)
@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(data=[ModelInfo(
        id=MODEL_ID,
        name=MODEL_NAME,
        created=1704067200,
        description="Text-to-music generation",
        input_modalities=["text"],
        output_modalities=["audio"],
        context_length=4096,
        pricing={"prompt": "0.000005", "completion": "0.02", "request": PRICE_PER_REQUEST},
        supported_sampling_parameters=["temperature", "top_p"],
    )])


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    now = int(time.time())
    
    if req.model != MODEL_ID:
        raise HTTPException(400, f"Model not found: {req.model}")
    
    # 检查 modalities 参数，确保包含 "audio" 或为空
    if req.modalities is not None and "audio" not in req.modalities:
        raise HTTPException(400, "This model only supports 'audio' modality")
    
    # 尝试非阻塞方式获取信号量
    acquired = False
    try:
        # 使用 acquire_nowait() 避免竞争条件
        acquired = semaphore.acquire_nowait()
        if not acquired:
            raise HTTPException(429, "Server at capacity", headers={"Retry-After": "30"})
        
        # Get prompt
        user_msgs = [m for m in req.messages if m.role == "user"]
        if not user_msgs:
            raise HTTPException(400, "No user message")
        caption = user_msgs[-1].content
        
        # 修复 duration 默认值处理，避免 0 被当作 False
        duration = -1.0
        if req.duration is not None:
            duration = req.duration
        
        # Build params
        params = GenerationParams(
            task_type="text2music",
            caption=caption,
            lyrics=req.lyrics or ("[Instrumental]" if req.instrumental else ""),
            instrumental=req.instrumental or False,
            bpm=req.bpm,
            duration=duration,
            seed=req.seed if req.seed is not None else -1,
        )
        config = GenerationConfig(batch_size=1, audio_format="mp3")
        
        # Generate
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = await asyncio.wait_for(run_gen(params, config, tmp), timeout=TIMEOUT)
            except asyncio.TimeoutError:
                raise HTTPException(504, "Timeout")
            
            if not result.success or not result.audios:
                raise HTTPException(500, result.error or "Failed")
            
            audio = result.audios[0]
            if audio.get("tensor") is None:
                raise HTTPException(500, "No audio")
            
            b64 = tensor_to_base64(audio["tensor"], audio.get("sample_rate", 48000))
            
            return ChatResponse(
                id=req_id,
                created=now,
                model=req.model,
                choices=[Choice(
                    index=0,
                    message=ResponseMessage(
                        content=f"Generated: {caption[:50]}",
                        audio=AudioOutput(id=audio.get("key", "audio_0"), data=b64, expires_at=now + 3600),
                    ),
                    finish_reason="stop",
                )],
                usage=Usage(),
            )
    finally:
        # 确保在函数退出时释放信号量
        if acquired:
            semaphore.release()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": dit_handler is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
