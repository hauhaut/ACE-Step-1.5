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
| `vocal_language` | string | `"en"` | Lyrics language (en, zh, ja, etc.) |
| `audio_format` | string | `"mp3"` | Output format (mp3, wav, flac) |

**Music Attribute Parameters**:

| Parameter Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bpm` | int | null | Specify tempo (BPM) |
| `key_scale` | string | `""` | Key/scale (e.g., "C Major") |
| `time_signature` | string | `""` | Time signature (e.g., "4/4") |
| `audio_duration` | float | null | Generation duration (seconds) |

**Generation Control Parameters**:

| Parameter Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `inference_steps` | int | `8` | Number of inference steps |
| `guidance_scale` | float | `7.0` | Prompt guidance coefficient |
| `use_random_seed` | bool | `true` | Whether to use random seed |
| `seed` | int | `-1` | Specify seed (when use_random_seed=false) |
| `batch_size` | int | null | Batch generation count |

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
