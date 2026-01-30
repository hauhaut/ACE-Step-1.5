---
name: acestep
description: Use ACE-Step API to generate music, edit songs, and remix music. Supports text-to-music, lyrics generation, audio continuation, and audio repainting. Use this skill when users mention generating music, creating songs, music production, remix, or audio continuation.
allowed-tools: Read, Write, Bash
---

# ACE-Step Music Generation Skill

Use ACE-Step V1.5 API for music generation and editing.

## Output Files

After generation, the script automatically saves results to the `acestep_output` folder in the project root (same level as `.claude`):

```
project_root/
├── .claude/
│   └── skills/acestep/...
├── acestep_output/          # Output directory
│   ├── <job_id>.json         # Complete task result (JSON)
│   ├── <job_id>_1.mp3        # First audio file
│   ├── <job_id>_2.mp3        # Second audio file (if batch_size > 1)
│   └── ...
└── ...
```

## Configuration

The script uses `scripts/config.json` to manage default settings.

### Configuration Priority Rules

**Important**: Configuration follows this priority (high to low):

1. **Command line arguments** > **config.json defaults**
2. User-specified parameters **temporarily override** defaults but **do not modify** config.json
3. Only `config --set` command **permanently modifies** config.json

**Example**:
```bash
# config.json has thinking=true

# Use default config (thinking=true)
./scripts/acestep.sh generate "Pop music"

# Temporary override (thinking=false for this run, config.json unchanged)
./scripts/acestep.sh generate "Pop music" --no-thinking

# Permanently modify default config
./scripts/acestep.sh config --set generation.thinking false
```

### API Connection Flow

1. **Load config**: Read `scripts/config.json` (use built-in defaults if not exists)
2. **Health check**: Request `/health` endpoint to verify service availability
3. **Connection failure**: Prompt user for correct API address and save to config.json

### Default Config File (`scripts/config.json`)

```json
{
  "api_url": "http://127.0.0.1:8001",
  "api_key": "",
  "generation": {
    "thinking": true,
    "use_format": true,
    "use_cot_caption": true,
    "use_cot_language": true,
    "batch_size": 1,
    "audio_format": "mp3",
    "vocal_language": "en"
  }
}
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `api_url` | `http://127.0.0.1:8001` | API server address |
| `api_key` | `""` | API authentication key (optional) |
| `generation.thinking` | `true` | Enable 5Hz LM model (high quality mode) |
| `generation.use_format` | `true` | Use LM to enhance caption/lyrics |
| `generation.use_cot_caption` | `true` | Use CoT to enhance caption |
| `generation.use_cot_language` | `true` | Use CoT to enhance language detection |
| `generation.audio_format` | `mp3` | Output format |
| `generation.vocal_language` | `en` | Vocal language |

## Script Usage

### Config Management (Permanently modify config.json)

```bash
# View all config
./scripts/acestep.sh config

# List all config options and current values
./scripts/acestep.sh config --list

# Get single config value
./scripts/acestep.sh config --get generation.thinking

# Permanently modify config value (writes to config.json)
./scripts/acestep.sh config --set generation.thinking false
./scripts/acestep.sh config --set api_url http://192.168.1.100:8001

# Reset to default config
./scripts/acestep.sh config --reset
```

### Generate Music (Command line args temporarily override, don't modify config.json)

Supports two generation modes:

**Caption Mode** - Directly specify music style description
```bash
./scripts/acestep.sh generate "Pop music with guitar"
./scripts/acestep.sh generate -c "Lyrical pop" -l "[Verse] Lyrics content"
```

**Simple Mode** - Use simple description, LM auto-generates caption and lyrics
```bash
./scripts/acestep.sh generate -d "A cheerful song about spring"
./scripts/acestep.sh generate -d "A love song for February"
```

**Other Options**
```bash
# Temporarily disable thinking mode (this run only, config file unchanged)
./scripts/acestep.sh generate "EDM" --no-thinking

# Temporarily disable format mode
./scripts/acestep.sh generate "Classical piano" --no-format

# Temporarily specify other parameters
./scripts/acestep.sh generate "Jazz" --steps 16 --guidance 8.0

# Random generation
./scripts/acestep.sh random

# Query task status (completed tasks auto-download audio)
./scripts/acestep.sh status <job_id>

# List available models
./scripts/acestep.sh models

# Check API health
./scripts/acestep.sh health
```

### Shell Script (Linux/macOS/Git Bash, requires curl + jq)

```bash
# Config management
./scripts/acestep.sh config --list
./scripts/acestep.sh config --set generation.thinking false

# Caption mode (auto-save results and download audio on completion)
./scripts/acestep.sh generate "Pop music with guitar"
./scripts/acestep.sh generate -c "Lyrical pop" -l "[Verse] Lyrics content"

# Simple mode (LM auto-generates caption/lyrics)
./scripts/acestep.sh generate -d "A cheerful song about spring"

# Random generation
./scripts/acestep.sh random

# Other commands
./scripts/acestep.sh status <job_id>
./scripts/acestep.sh models
./scripts/acestep.sh health
```

## Script Dependencies

| Script | Dependencies | Platform |
|------|------|------|
| `acestep.sh` | curl, jq | Linux/macOS/Git Bash |

Install jq:
- Ubuntu/Debian: `apt install jq`
- macOS: `brew install jq`
- Windows (choco): `choco install jq`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/release_task` | POST | Create music generation task |
| `/query_result` | POST | Batch query task results |
| `/v1/models` | GET | List available models |
| `/v1/audio?path={path}` | GET | Download generated audio file |

## Main Parameters

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "" | Music description text (Caption mode) |
| `sample_query` | string | "" | Simple description, LM auto-generates caption/lyrics (Simple mode) |
| `lyrics` | string | "" | Lyrics content |
| `thinking` | bool | false | Enable 5Hz LM model for audio code generation (high quality) |
| `sample_mode` | bool | false | Random sampling mode (LM auto-generates) |
| `use_format` | bool | false | Use LM to enhance caption/lyrics |
| `model` | string | - | Specify DiT model name |

### Music Attributes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bpm` | int | - | Tempo (beats per minute) |
| `key_scale` | string | "" | Key (e.g. "C Major") |
| `time_signature` | string | "" | Time signature (e.g. "4/4") |
| `vocal_language` | string | "en" | Vocal language |
| `audio_duration` | float | - | Audio duration (seconds) |

### LM Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_cot_caption` | bool | true | Use CoT to enhance caption |
| `use_cot_language` | bool | true | Use CoT to enhance language detection |

### Task Types

| Value | Description |
|-------|-------------|
| `text2music` | Text to music (default) |
| `continuation` | Audio continuation |
| `repainting` | Audio repainting |

## Task Status

| Status | Description |
|--------|-------------|
| `queued` | Waiting in queue |
| `running` | Generating |
| `succeeded` | Generation successful |
| `failed` | Generation failed |

## Response Examples

### Create Task Response (`/release_task`)

```json
{
  "task_id": "abc123-def456",
  "status": "queued",
  "queue_position": 1
}
```

### Query Result Request (`/query_result`)

```json
{
  "task_id_list": ["abc123-def456", "xyz789"]
}
```

### Query Result Response

```json
[
  {
    "task_id": "abc123-def456",
    "status": 1,
    "result": "[{\"file\":\"/v1/audio?path=...\",\"status\":1,\"metas\":{\"bpm\":120,\"duration\":60,\"keyscale\":\"C Major\"}}]"
  }
]
```

Status codes: `0` = processing, `1` = success, `2` = failed

## Notes

1. **Config priority**: Command line args > config.json defaults. User-specified params take effect temporarily without modifying config file
2. **Modify default config**: Only `config --set` command permanently modifies config.json
3. **Thinking mode**: When enabled, uses 5Hz LM to generate audio code, higher quality but slower
4. **Async tasks**: All generation tasks are async, poll results via `POST /query_result`
5. **Auto download**: After completion, auto-saves JSON results and downloads audio files to `acestep_output/` directory
6. **Status codes**: Status in query results is integer: 0=processing, 1=success, 2=failed

## References
- Shell script: [scripts/acestep.sh](scripts/acestep.sh) (Linux/macOS/Git Bash)
- Default config: [scripts/config.json](scripts/config.json)
- Output directory: `acestep_output/` in project root
