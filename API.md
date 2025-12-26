# ACE-Step API Client Documentation

This service provides an HTTP-based asynchronous music generation API.

**Basic Workflow**:
1. Call `POST /v1/music/generate` to submit a task and obtain a `job_id`.
2. Call `GET /v1/jobs/{job_id}` to poll the task status until `status` is `succeeded` or `failed`.

---

## 1. Task Status Description

Task status (`status`) includes the following types:

- `queued`: Task has entered the queue and is waiting to be executed. You can check `queue_position` and `eta_seconds` at this time.
- `running`: Generation is in progress.
- `succeeded`: Generation succeeded, results are in the `result` field.
- `failed`: Generation failed, error information is in the `error` field.

---

## 2. Create Generation Task

### 2.1 API Definition

- **URL**: `/v1/music/generate`
- **Method**: `POST`
- **Content-Type**: `application/json` or `multipart/form-data`

### 2.2 Request Parameters

#### Method A: JSON Request (application/json)

Suitable for passing only text parameters, or referencing audio file paths that already exist on the server.

**Basic Parameters**:

| Parameter Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `caption` | string | `""` | Music description prompt |
| `lyrics` | string | `""` | Lyrics content |
| `thinking` | bool | `false` | Whether to use 5Hz LM to generate audio codes (lm-dit behavior). |
| `vocal_language` | string | `"en"` | Lyrics language (en, zh, ja, etc.) |
| `audio_format` | string | `"mp3"` | Output format (mp3, wav, flac) |

**thinking Semantics (Important)**:

- `thinking=false`:
  - The server will **NOT** use 5Hz LM to generate `audio_code_string`.
  - DiT runs in **text2music** mode and **ignores** any provided `audio_code_string`.
- `thinking=true`:
  - The server will use 5Hz LM to generate `audio_code_string` (lm-dit behavior).
  - DiT runs in **cover** mode and uses `audio_code_string`.

**Metadata Auto-Completion (Always On)**:

Regardless of `thinking`, if any of the following fields are missing, the server may call 5Hz LM to **fill only the missing fields** based on `caption`/`lyrics`:

- `bpm`
- `key_scale`
- `time_signature`
- `audio_duration`

User-provided values always win; LM only fills the fields that are empty/missing.

**Music Attribute Parameters**:

| Parameter Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bpm` | int | null | Specify tempo (BPM) |
| `key_scale` | string | `""` | Key/scale (e.g., "C Major") |
| `time_signature` | string | `""` | Time signature (e.g., "4/4") |
| `audio_duration` | float | null | Generation duration (seconds) |

**Audio Codes (Optional)**:

| Parameter Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `audio_code_string` | string or string[] | `""` | Audio semantic tokens (5Hz) for `llm_dit`. If provided as an array, it should match `batch_size` (or the server batch size). |

**Generation Control Parameters**:

| Parameter Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `inference_steps` | int | `8` | Number of inference steps |
| `guidance_scale` | float | `7.0` | Prompt guidance coefficient |
| `use_random_seed` | bool | `true` | Whether to use random seed |
| `seed` | int | `-1` | Specify seed (when use_random_seed=false) |
| `batch_size` | int | null | Batch generation count |

**5Hz LM Parameters (Optional, server-side)**:

These parameters control 5Hz LM sampling, used for metadata auto-completion and (when `thinking=true`) codes generation.

| Parameter Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lm_model_path` | string | null | 5Hz LM checkpoint dir name (e.g. `acestep-5Hz-lm-0.6B`) |
| `lm_backend` | string | `"vllm"` | `vllm` or `pt` |
| `lm_temperature` | float | `0.85` | Sampling temperature |
| `lm_cfg_scale` | float | `2.0` | CFG scale (>1 enables CFG) |
| `lm_negative_prompt` | string | `"NO USER INPUT"` | Negative prompt used by CFG |
| `lm_top_k` | int | null | Top-k (0/null disables) |
| `lm_top_p` | float | `0.9` | Top-p (>=1 will be treated as disabled) |
| `lm_repetition_penalty` | float | `1.0` | Repetition penalty |

**Edit/Reference Audio Parameters** (requires absolute path on server):

| Parameter Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `reference_audio_path` | string | null | Reference audio path (Style Transfer) |
| `src_audio_path` | string | null | Source audio path (Repainting/Cover) |
| `task_type` | string | `"text2music"` | Task type (text2music, cover, repaint) |
| `instruction` | string | `"Fill..."` | Edit instruction |
| `repainting_start` | float | `0.0` | Repainting start time |
| `repainting_end` | float | null | Repainting end time |
| `audio_cover_strength` | float | `1.0` | Cover strength |

#### Method B: File Upload (multipart/form-data)

Use this when you need to upload local audio files as reference or source audio.

In addition to supporting all the above fields as Form Fields, the following file fields are also supported:

- `reference_audio`: (File) Upload reference audio file
- `src_audio`: (File) Upload source audio file

> **Note**: After uploading files, the corresponding `_path` parameters will be automatically ignored, and the system will use the temporary file path after upload.

### 2.3 Response Example

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "queue_position": 1
}
```

### 2.4 Usage Examples (cURL)

**JSON Method**:

```bash
curl -X POST http://localhost:8001/v1/music/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "caption": "upbeat pop song",
    "lyrics": "Hello world",
    "inference_steps": 16
  }'
```

**JSON Method (thinking=true: generate codes + fill missing metas)**:

```bash
curl -X POST http://localhost:8001/v1/music/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "caption": "upbeat pop song",
    "lyrics": "Hello world",
    "thinking": true,
    "lm_temperature": 0.85,
    "lm_cfg_scale": 2.0,
    "lm_top_k": null,
    "lm_top_p": 0.9,
    "lm_repetition_penalty": 1.0
  }'
```

**JSON Method (thinking=false: do NOT generate codes, but fill missing metas)**:

Example: user specifies `bpm` but omits `audio_duration`. The server may call LM to infer `duration` from `caption`/`lyrics` and use it only if the user did not set it.

```bash
curl -X POST http://localhost:8001/v1/music/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "caption": "slow emotional ballad",
    "lyrics": "...",
    "thinking": false,
    "bpm": 72
  }'
```

When the server invokes the 5Hz LM (to fill metas and/or generate codes), the job `result` may include the following optional fields:

- `bpm`
- `duration`
- `genres`
- `keyscale`
- `timesignature`
- `metas` (raw-ish metadata dict)

> Note: If you use `curl -d` but **forget** to add `-H 'Content-Type: application/json'`, curl will default to sending `application/x-www-form-urlencoded`, and older server versions will return 415.

**Form Method (no file upload, application/x-www-form-urlencoded)**:

```bash
curl -X POST http://localhost:8001/v1/music/generate \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'caption=upbeat pop song' \
  --data-urlencode 'lyrics=Hello world' \
  --data-urlencode 'inference_steps=16'
```

**File Upload Method**:

```bash
curl -X POST http://localhost:8001/v1/music/generate \
  -F "caption=remix this song" \
  -F "src_audio=@/path/to/local/song.mp3" \
  -F "task_type=repaint"
```

---

## 3. Query Task Results

### 3.1 API Definition

- **URL**: `/v1/jobs/{job_id}`
- **Method**: `GET`

### 3.2 Response Parameters

The response contains basic task information, queue status, and final results.

**Main Fields**:

- `status`: Current status
- `queue_position`: Current queue position (0 means running or completed)
- `eta_seconds`: Estimated remaining wait time (seconds)
- `result`: Result object when successful
  - `audio_paths`: List of generated audio file URLs/paths
  - `first_audio_path`: Preferred audio path
  - `generation_info`: Generation parameter details
  - `status_message`: Brief result description
- `error`: Error information when failed

### 3.3 Response Examples

**Queued**:

```json
{
  "job_id": "...",
  "status": "queued",
  "created_at": 1700000000.0,
  "queue_position": 5,
  "eta_seconds": 25.0,
  "result": null,
  "error": null
}
```

**Execution Successful**:

```json
{
  "job_id": "...",
  "status": "succeeded",
  "created_at": 1700000000.0,
  "finished_at": 1700000010.0,
  "queue_position": 0,
  "result": {
    "first_audio_path": "/tmp/generated_1.mp3",
    "second_audio_path": "/tmp/generated_2.mp3",
    "audio_paths": ["/tmp/generated_1.mp3", "/tmp/generated_2.mp3"],
    "generation_info": "Steps: 8, Scale: 7.0 ...",
    "status_message": "✅ Generation completed successfully!",
    "seed_value": "12345"
  },
  "error": null
}
```
