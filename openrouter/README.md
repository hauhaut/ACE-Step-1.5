# ACE-Step OpenRouter API Server

将 ACE-Step 音乐生成模型包装成 OpenRouter 兼容的 API。

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn torch torchaudio loguru pydantic
```

### 2. 配置模型

编辑 `openrouter_server.py` 中的 `startup()` 函数：

```python
@app.on_event("startup")
async def startup():
    global semaphore, dit_handler, llm_handler
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # 初始化你的模型
    dit_handler = YourDiTHandler(...)
    llm_handler = YourLLMHandler(...)
```

### 3. 启动

```bash
uvicorn openrouter_server:app --host 0.0.0.0 --port 8000
```

## API 端点

### GET /api/v1/models

返回模型信息，用于 OpenRouter 发现。

```json
{
  "data": [{
    "id": "acestep/music-gen-v1",
    "name": "ACE-Step Music Generator",
    "pricing": {"prompt": "0", "completion": "0", "request": "0.05"}
  }]
}
```

### POST /v1/chat/completions

生成音乐。

**请求：**
```json
{
  "model": "acestep/music-gen-v1",
  "messages": [{"role": "user", "content": "Upbeat electronic dance music"}],
  "lyrics": "[Verse]\nHello world...",
  "bpm": 128,
  "duration": 60,
  "instrumental": false,
  "seed": 42
}
```

**响应：**
```json
{
  "id": "chatcmpl-xxx",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Generated: Upbeat electronic...",
      "audio": {
        "id": "audio_uuid",
        "data": "<base64 mp3>",
        "expires_at": 1234567890
      }
    },
    "finish_reason": "stop"
  }]
}
```

## 配置

在文件顶部修改：

```python
MODEL_ID = "acestep/music-gen-v1"  # 模型 ID
PRICE_PER_REQUEST = "0.05"         # 每请求价格 (USD)
MAX_CONCURRENT = 4                  # 最大并发
TIMEOUT = 300                       # 超时秒数
```

## 测试

```bash
python test_client.py
```

## OpenRouter 集成

1. 部署服务器到公网
2. 访问 https://openrouter.ai/docs/guides/guides/for-providers 提交申请
3. 加入 Discord: https://discord.gg/openrouter
