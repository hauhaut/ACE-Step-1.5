# ACE-Step Inference API Documentation

This document provides comprehensive documentation for the ACE-Step inference API, including parameter specifications for all supported task types.

## Table of Contents

- [Quick Start](#quick-start)
- [API Overview](#api-overview)
- [GenerationParams Parameters](#generationparams-parameters)
- [GenerationConfig Parameters](#generationconfig-parameters)
- [Task Types](#task-types)
- [Complete Examples](#complete-examples)
- [Best Practices](#best-practices)

---

## Quick Start

### Basic Usage

```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# Initialize handlers
dit_handler = AceStepHandler()
llm_handler = LLMHandler()

# Initialize services
dit_handler.initialize_service(
    project_root="/path/to/project",
    config_path="acestep-v15-turbo-rl",
    device="cuda"
)

llm_handler.initialize(
    checkpoint_dir="/path/to/checkpoints",
    lm_model_path="acestep-5Hz-lm-0.6B-v3",
    backend="vllm",
    device="cuda"
)

# Configure generation parameters
params = GenerationParams(
    caption="upbeat electronic dance music with heavy bass",
    bpm=128,
    duration=30,
)

# Configure generation settings
config = GenerationConfig(
    batch_size=2,
    audio_format="flac",
)

# Generate music
result = generate_music(dit_handler, llm_handler, params, config, save_dir="/path/to/output")

# Access results
if result.success:
    for audio in result.audios:
        print(f"Generated: {audio['path']}")
        print(f"Key: {audio['key']}")
        print(f"Seed: {audio['params']['seed']}")
else:
    print(f"Error: {result.error}")
```

---

## API Overview

### Main Function

```python
def generate_music(
    dit_handler,
    llm_handler,
    params: GenerationParams,
    config: GenerationConfig,
    save_dir: Optional[str] = None,
    progress=None,
) -> GenerationResult
```

### Configuration Objects

The API uses two configuration dataclasses:

**GenerationParams** - Contains all music generation parameters:

```python
@dataclass
class GenerationParams:
    # Task & Instruction
    task_type: str = "text2music"
    instruction: str = "Fill the audio semantic mask based on the given conditions:"
    
    # Audio Uploads
    reference_audio: Optional[str] = None
    src_audio: Optional[str] = None
    
    # LM Codes Hints
    audio_codes: str = ""
    
    # Text Inputs
    caption: str = ""
    lyrics: str = ""
    instrumental: bool = False
    
    # Metadata
    vocal_language: str = "unknown"
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    duration: float = -1.0
    
    # Advanced Settings
    inference_steps: int = 8
    seed: int = -1
    guidance_scale: float = 7.0
    use_adg: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    
    repainting_start: float = 0.0
    repainting_end: float = -1
    audio_cover_strength: float = 1.0
    
    # 5Hz Language Model Parameters
    thinking: bool = True
    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.0
    lm_top_k: int = 0
    lm_top_p: float = 0.9
    lm_negative_prompt: str = "NO USER INPUT"
    use_cot_metas: bool = True
    use_cot_caption: bool = True
    use_cot_lyrics: bool = False
    use_cot_language: bool = True
    use_constrained_decoding: bool = True
    
    # CoT Generated Values (auto-filled by LM)
    cot_bpm: Optional[int] = None
    cot_keyscale: str = ""
    cot_timesignature: str = ""
    cot_duration: Optional[float] = None
    cot_vocal_language: str = "unknown"
    cot_caption: str = ""
    cot_lyrics: str = ""
```

**GenerationConfig** - Contains batch and output configuration:

```python
@dataclass
class GenerationConfig:
    batch_size: int = 2
    allow_lm_batch: bool = False
    use_random_seed: bool = True
    seeds: Optional[List[int]] = None
    lm_batch_chunk_size: int = 8
    constrained_decoding_debug: bool = False
    audio_format: str = "flac"
```

### Result Object

```python
@dataclass
class GenerationResult:
    # Audio Outputs
    audios: List[Dict[str, Any]]  # List of audio dictionaries
    
    # Generation Information
    status_message: str           # Status message from generation
    extra_outputs: Dict[str, Any] # Extra outputs (latents, masks, lm_metadata, time_costs)
    
    # Success Status
    success: bool                 # Whether generation succeeded
    error: Optional[str]          # Error message if failed
```

**Audio Dictionary Structure:**

Each item in `audios` list contains:

```python
{
    "path": str,           # File path to saved audio
    "tensor": Tensor,      # Audio tensor [channels, samples], CPU, float32
    "key": str,            # Unique audio key (UUID based on params)
    "sample_rate": int,    # Sample rate (default: 48000)
    "params": Dict,        # Generation params for this audio (includes seed, audio_codes, etc.)
}
```

---

## GenerationParams Parameters

### Text Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `caption` | `str` | `""` | Text description of the desired music. Can be a simple prompt like "relaxing piano music" or detailed description with genre, mood, instruments, etc. Max 512 characters. |
| `lyrics` | `str` | `""` | Lyrics text for vocal music. Use `"[Instrumental]"` for instrumental tracks. Supports multiple languages. Max 4096 characters. |
| `instrumental` | `bool` | `False` | If True, generate instrumental music regardless of lyrics. |

### Music Metadata

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bpm` | `Optional[int]` | `None` | Beats per minute (30-300). `None` enables auto-detection via LM. |
| `keyscale` | `str` | `""` | Musical key (e.g., "C Major", "Am", "F# minor"). Empty string enables auto-detection. |
| `timesignature` | `str` | `""` | Time signature (2 for '2/4', 3 for '3/4', 4 for '4/4', 6 for '6/8'). Empty string enables auto-detection. |
| `vocal_language` | `str` | `"unknown"` | Language code for vocals (ISO 639-1). Supported: `"en"`, `"zh"`, `"ja"`, `"es"`, `"fr"`, etc. Use `"unknown"` for auto-detection. |
| `duration` | `float` | `-1.0` | Target audio length in seconds (10-600). If <= 0 or None, model chooses automatically based on lyrics length. |

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inference_steps` | `int` | `8` | Number of denoising steps. Turbo model: 1-8 (recommended 8). Base model: 1-100 (recommended 32-64). Higher = better quality but slower. |
| `guidance_scale` | `float` | `7.0` | Classifier-free guidance scale (1.0-15.0). Higher values increase adherence to text prompt. Only supported for non-turbo model. Typical range: 5.0-9.0. |
| `seed` | `int` | `-1` | Random seed for reproducibility. Use `-1` for random seed, or any positive integer for fixed seed. |

### Advanced DiT Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_adg` | `bool` | `False` | Use Adaptive Dual Guidance (base model only). Improves quality at the cost of speed. |
| `cfg_interval_start` | `float` | `0.0` | CFG application start ratio (0.0-1.0). Controls when to start applying classifier-free guidance. |
| `cfg_interval_end` | `float` | `1.0` | CFG application end ratio (0.0-1.0). Controls when to stop applying classifier-free guidance. |

### Task-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | `str` | `"text2music"` | Generation task type. See [Task Types](#task-types) section for details. |
| `instruction` | `str` | `"Fill the audio semantic mask based on the given conditions:"` | Task-specific instruction prompt. |
| `reference_audio` | `Optional[str]` | `None` | Path to reference audio file for style transfer or continuation tasks. |
| `src_audio` | `Optional[str]` | `None` | Path to source audio file for audio-to-audio tasks (cover, repaint, etc.). |
| `audio_codes` | `str` | `""` | Pre-extracted 5Hz audio semantic codes as a string. Advanced use only. |
| `repainting_start` | `float` | `0.0` | Repainting start time in seconds (for repaint/lego tasks). |
| `repainting_end` | `float` | `-1` | Repainting end time in seconds. Use `-1` for end of audio. |
| `audio_cover_strength` | `float` | `1.0` | Strength of audio cover/codes influence (0.0-1.0). Set smaller (0.2) for style transfer tasks. |

### 5Hz Language Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thinking` | `bool` | `True` | Enable 5Hz Language Model "Chain-of-Thought" reasoning for semantic/music metadata and codes. |
| `lm_temperature` | `float` | `0.85` | LM sampling temperature (0.0-2.0). Higher = more creative/diverse, lower = more conservative. |
| `lm_cfg_scale` | `float` | `2.0` | LM classifier-free guidance scale. Higher = stronger adherence to prompt. |
| `lm_top_k` | `int` | `0` | LM top-k sampling. `0` disables top-k filtering. Typical values: 40-100. |
| `lm_top_p` | `float` | `0.9` | LM nucleus sampling (0.0-1.0). `1.0` disables nucleus sampling. Typical values: 0.9-0.95. |
| `lm_negative_prompt` | `str` | `"NO USER INPUT"` | Negative prompt for LM guidance. Helps avoid unwanted characteristics. |
| `use_cot_metas` | `bool` | `True` | Generate metadata using LM CoT reasoning (BPM, key, duration, etc.). |
| `use_cot_caption` | `bool` | `True` | Refine user caption using LM CoT reasoning. |
| `use_cot_language` | `bool` | `True` | Detect vocal language using LM CoT reasoning. |
| `use_cot_lyrics` | `bool` | `False` | (Reserved for future use) Generate/refine lyrics using LM CoT. |
| `use_constrained_decoding` | `bool` | `True` | Enable constrained decoding for structured LM output. |

### CoT Generated Values

These fields are automatically populated by the LM when CoT reasoning is enabled:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cot_bpm` | `Optional[int]` | `None` | LM-generated BPM value. |
| `cot_keyscale` | `str` | `""` | LM-generated key/scale. |
| `cot_timesignature` | `str` | `""` | LM-generated time signature. |
| `cot_duration` | `Optional[float]` | `None` | LM-generated duration. |
| `cot_vocal_language` | `str` | `"unknown"` | LM-detected vocal language. |
| `cot_caption` | `str` | `""` | LM-refined caption. |
| `cot_lyrics` | `str` | `""` | LM-generated/refined lyrics. |

---

## GenerationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `2` | Number of samples to generate in parallel (1-8). Higher values require more GPU memory. |
| `allow_lm_batch` | `bool` | `False` | Allow batch processing in LM. Faster when `batch_size >= 2` and `thinking=True`. |
| `use_random_seed` | `bool` | `True` | Whether to use random seed. `True` for different results each time, `False` for reproducible results. |
| `seeds` | `Optional[List[int]]` | `None` | List of seeds for batch generation. If provided, will be padded with random seeds if fewer than batch_size. Can also be single int. |
| `lm_batch_chunk_size` | `int` | `8` | Maximum batch size per LM inference chunk (GPU memory constraint). |
| `constrained_decoding_debug` | `bool` | `False` | Enable debug logging for constrained decoding. |
| `audio_format` | `str` | `"flac"` | Output audio format. Options: `"mp3"`, `"wav"`, `"flac"`. Default is FLAC for fast saving. |

---

## Task Types

ACE-Step supports 6 different generation task types, each optimized for specific use cases.

### 1. Text2Music (Default)

**Purpose**: Generate music from text descriptions and optional metadata.

**Key Parameters**:
```python
params = GenerationParams(
    task_type="text2music",
    caption="energetic rock music with electric guitar",
    lyrics="[Instrumental]",  # or actual lyrics
    bpm=140,
    duration=30,
)
```

**Required**:
- `caption` or `lyrics` (at least one)

**Optional but Recommended**:
- `bpm`: Controls tempo
- `keyscale`: Controls musical key
- `timesignature`: Controls rhythm structure
- `duration`: Controls length
- `vocal_language`: Controls vocal characteristics

**Use Cases**:
- Generate music from text descriptions
- Create backing tracks from prompts
- Generate songs with lyrics

---

### 2. Cover

**Purpose**: Transform existing audio while maintaining structure but changing style/timbre.

**Key Parameters**:
```python
params = GenerationParams(
    task_type="cover",
    src_audio="original_song.mp3",
    caption="jazz piano version",
    audio_cover_strength=0.8,  # 0.0-1.0
)
```

**Required**:
- `src_audio`: Path to source audio file
- `caption`: Description of desired style/transformation

**Optional**:
- `audio_cover_strength`: Controls influence of original audio
  - `1.0`: Strong adherence to original structure
  - `0.5`: Balanced transformation
  - `0.1`: Loose interpretation
- `lyrics`: New lyrics (if changing vocals)

**Use Cases**:
- Create covers in different styles
- Change instrumentation while keeping melody
- Genre transformation

---

### 3. Repaint

**Purpose**: Regenerate a specific time segment of audio while keeping the rest unchanged.

**Key Parameters**:
```python
params = GenerationParams(
    task_type="repaint",
    src_audio="original.mp3",
    repainting_start=10.0,  # seconds
    repainting_end=20.0,    # seconds
    caption="smooth transition with piano solo",
)
```

**Required**:
- `src_audio`: Path to source audio file
- `repainting_start`: Start time in seconds
- `repainting_end`: End time in seconds (use `-1` for end of file)
- `caption`: Description of desired content for repainted section

**Use Cases**:
- Fix specific sections of generated music
- Add variations to parts of a song
- Create smooth transitions
- Replace problematic segments

---

### 4. Lego (Base Model Only)

**Purpose**: Generate a specific instrument track in context of existing audio.

**Key Parameters**:
```python
params = GenerationParams(
    task_type="lego",
    src_audio="backing_track.mp3",
    instruction="Generate the guitar track based on the audio context:",
    caption="lead guitar melody with bluesy feel",
    repainting_start=0.0,
    repainting_end=-1,
)
```

**Required**:
- `src_audio`: Path to source/backing audio
- `instruction`: Must specify the track type (e.g., "Generate the {TRACK_NAME} track...")
- `caption`: Description of desired track characteristics

**Available Tracks**:
- `"vocals"`, `"backing_vocals"`, `"drums"`, `"bass"`, `"guitar"`, `"keyboard"`, 
- `"percussion"`, `"strings"`, `"synth"`, `"fx"`, `"brass"`, `"woodwinds"`

**Use Cases**:
- Add specific instrument tracks
- Layer additional instruments over backing tracks
- Create multi-track compositions iteratively

---

### 5. Extract (Base Model Only)

**Purpose**: Extract/isolate a specific instrument track from mixed audio.

**Key Parameters**:
```python
params = GenerationParams(
    task_type="extract",
    src_audio="full_mix.mp3",
    instruction="Extract the vocals track from the audio:",
)
```

**Required**:
- `src_audio`: Path to mixed audio file
- `instruction`: Must specify track to extract

**Available Tracks**: Same as Lego task

**Use Cases**:
- Stem separation
- Isolate specific instruments
- Create remixes
- Analyze individual tracks

---

### 6. Complete (Base Model Only)

**Purpose**: Complete/extend partial tracks with specified instruments.

**Key Parameters**:
```python
params = GenerationParams(
    task_type="complete",
    src_audio="incomplete_track.mp3",
    instruction="Complete the input track with drums, bass, guitar:",
    caption="rock style completion",
)
```

**Required**:
- `src_audio`: Path to incomplete/partial track
- `instruction`: Must specify which tracks to add
- `caption`: Description of desired style

**Use Cases**:
- Arrange incomplete compositions
- Add backing tracks
- Auto-complete musical ideas

---

## Complete Examples

### Example 1: Simple Text-to-Music Generation

```python
from acestep.inference import GenerationParams, GenerationConfig, generate_music

params = GenerationParams(
    task_type="text2music",
    caption="calm ambient music with soft piano and strings",
    duration=60,
    bpm=80,
    keyscale="C Major",
)

config = GenerationConfig(
    batch_size=2,  # Generate 2 variations
    audio_format="flac",
)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")

if result.success:
    for i, audio in enumerate(result.audios, 1):
        print(f"Variation {i}: {audio['path']}")
```

### Example 2: Song Generation with Lyrics

```python
params = GenerationParams(
    task_type="text2music",
    caption="pop ballad with emotional vocals",
    lyrics="""Verse 1:
Walking down the street today
Thinking of the words you used to say
Everything feels different now
But I'll find my way somehow

Chorus:
I'm moving on, I'm staying strong
This is where I belong
""",
    vocal_language="en",
    bpm=72,
    duration=45,
)

config = GenerationConfig(batch_size=1)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")
```

### Example 3: Style Cover with LM Reasoning

```python
params = GenerationParams(
    task_type="cover",
    src_audio="original_pop_song.mp3",
    caption="orchestral symphonic arrangement",
    audio_cover_strength=0.7,
    thinking=True,  # Enable LM for metadata
    use_cot_metas=True,
)

config = GenerationConfig(batch_size=1)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")

# Access LM-generated metadata
if result.extra_outputs.get("lm_metadata"):
    lm_meta = result.extra_outputs["lm_metadata"]
    print(f"LM detected BPM: {lm_meta.get('bpm')}")
    print(f"LM detected Key: {lm_meta.get('keyscale')}")
```

### Example 4: Repaint Section of Audio

```python
params = GenerationParams(
    task_type="repaint",
    src_audio="generated_track.mp3",
    repainting_start=15.0,  # Start at 15 seconds
    repainting_end=25.0,    # End at 25 seconds
    caption="dramatic orchestral buildup",
    inference_steps=32,  # Higher quality for base model
)

config = GenerationConfig(batch_size=1)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")
```

### Example 5: Batch Generation with Specific Seeds

```python
params = GenerationParams(
    task_type="text2music",
    caption="epic cinematic trailer music",
)

config = GenerationConfig(
    batch_size=4,           # Generate 4 variations
    seeds=[42, 123, 456],   # Specify 3 seeds, 4th will be random
    use_random_seed=False,  # Use provided seeds
    lm_batch_chunk_size=2,  # Process 2 at a time (GPU memory)
)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")

if result.success:
    print(f"Generated {len(result.audios)} variations")
    for audio in result.audios:
        print(f"  Seed {audio['params']['seed']}: {audio['path']}")
```

### Example 6: High-Quality Generation (Base Model)

```python
params = GenerationParams(
    task_type="text2music",
    caption="intricate jazz fusion with complex harmonies",
    inference_steps=64,     # High quality
    guidance_scale=8.0,
    use_adg=True,           # Adaptive Dual Guidance
    cfg_interval_start=0.0,
    cfg_interval_end=1.0,
    seed=42,                # Reproducible results
)

config = GenerationConfig(
    batch_size=1,
    use_random_seed=False,
    audio_format="wav",     # Lossless format
)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")
```

### Example 7: Extract Vocals from Mix

```python
params = GenerationParams(
    task_type="extract",
    src_audio="full_song_mix.mp3",
    instruction="Extract the vocals track from the audio:",
)

config = GenerationConfig(batch_size=1)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")

if result.success:
    print(f"Extracted vocals: {result.audios[0]['path']}")
```

### Example 8: Add Guitar Track (Lego)

```python
params = GenerationParams(
    task_type="lego",
    src_audio="drums_and_bass.mp3",
    instruction="Generate the guitar track based on the audio context:",
    caption="funky rhythm guitar with wah-wah effect",
    repainting_start=0.0,
    repainting_end=-1,  # Full duration
)

config = GenerationConfig(batch_size=1)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")
```

### Example 9: Instrumental Generation

```python
params = GenerationParams(
    task_type="text2music",
    caption="upbeat electronic dance music",
    instrumental=True,  # Force instrumental output
    duration=120,
    bpm=128,
)

config = GenerationConfig(batch_size=2)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")
```

---

## Best Practices

### 1. Caption Writing

**Good Captions**:
```python
# Specific and descriptive
caption="upbeat electronic dance music with heavy bass and synthesizer leads"

# Include mood and genre
caption="melancholic indie folk with acoustic guitar and soft vocals"

# Specify instruments
caption="jazz trio with piano, upright bass, and brush drums"
```

**Avoid**:
```python
# Too vague
caption="good music"

# Contradictory
caption="fast slow music"  # Conflicting tempos
```

### 2. Parameter Tuning

**For Best Quality**:
- Use base model with `inference_steps=64` or higher
- Enable `use_adg=True`
- Set `guidance_scale=7.0-9.0`
- Use lossless audio format (`audio_format="wav"`)

**For Speed**:
- Use turbo model with `inference_steps=8`
- Disable ADG (`use_adg=False`)
- Lower `guidance_scale=5.0-7.0`
- Use compressed format (`audio_format="mp3"`) or default FLAC

**For Consistency**:
- Set `use_random_seed=False` in config
- Use fixed `seeds` list or single `seed` in params
- Keep `lm_temperature` lower (0.7-0.85)

**For Diversity**:
- Set `use_random_seed=True` in config
- Increase `lm_temperature` (0.9-1.1)
- Use `batch_size > 1` for variations

### 3. Duration Guidelines

- **Instrumental**: 30-180 seconds works well
- **With Lyrics**: Auto-detection recommended (set `duration=-1` or leave default)
- **Short clips**: 10-20 seconds minimum
- **Long form**: Up to 600 seconds (10 minutes) maximum

### 4. LM Usage

**When to Enable LM (`thinking=True`)**:
- Need automatic metadata detection
- Want caption refinement
- Generating from minimal input
- Need diverse outputs

**When to Disable LM (`thinking=False`)**:
- Have precise metadata already
- Need faster generation
- Want full control over parameters

### 5. Batch Processing

```python
# Efficient batch generation
config = GenerationConfig(
    batch_size=8,           # Max supported
    allow_lm_batch=True,    # Enable for speed (when thinking=True)
    lm_batch_chunk_size=4,  # Adjust based on GPU memory
)
```

### 6. Error Handling

```python
result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")

if not result.success:
    print(f"Generation failed: {result.error}")
    print(f"Status: {result.status_message}")
else:
    # Process successful result
    for audio in result.audios:
        path = audio['path']
        key = audio['key']
        seed = audio['params']['seed']
        # ... process audio files
```

### 7. Memory Management

For large batch sizes or long durations:
- Monitor GPU memory usage
- Reduce `batch_size` if OOM errors occur
- Reduce `lm_batch_chunk_size` for LM operations
- Consider using `offload_to_cpu=True` during initialization

### 8. Accessing Time Costs

```python
result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")

if result.success:
    time_costs = result.extra_outputs.get("time_costs", {})
    print(f"LM Phase 1 Time: {time_costs.get('lm_phase1_time', 0):.2f}s")
    print(f"LM Phase 2 Time: {time_costs.get('lm_phase2_time', 0):.2f}s")
    print(f"DiT Total Time: {time_costs.get('dit_total_time_cost', 0):.2f}s")
    print(f"Pipeline Total: {time_costs.get('pipeline_total_time', 0):.2f}s")
```

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory errors
- **Solution**: Reduce `batch_size`, `inference_steps`, or enable CPU offloading

**Issue**: Poor quality results
- **Solution**: Increase `inference_steps`, adjust `guidance_scale`, use base model

**Issue**: Results don't match prompt
- **Solution**: Make caption more specific, increase `guidance_scale`, enable LM refinement (`thinking=True`)

**Issue**: Slow generation
- **Solution**: Use turbo model, reduce `inference_steps`, disable ADG

**Issue**: LM not generating codes
- **Solution**: Verify `llm_handler` is initialized, check `thinking=True` and `use_cot_metas=True`

**Issue**: Seeds not being respected
- **Solution**: Set `use_random_seed=False` in config and provide `seeds` list or `seed` in params

---

## API Reference Summary

### GenerationParams Fields

See [GenerationParams Parameters](#generationparams-parameters) for complete documentation.

### GenerationConfig Fields

See [GenerationConfig Parameters](#generationconfig-parameters) for complete documentation.

### GenerationResult Fields

```python
@dataclass
class GenerationResult:
    # Audio Outputs
    audios: List[Dict[str, Any]]
    # Each audio dict contains:
    #   - "path": str (file path)
    #   - "tensor": Tensor (audio data)
    #   - "key": str (unique identifier)
    #   - "sample_rate": int (48000)
    #   - "params": Dict (generation params with seed, audio_codes, etc.)
    
    # Generation Information
    status_message: str
    extra_outputs: Dict[str, Any]
    # extra_outputs contains:
    #   - "lm_metadata": Dict (LM-generated metadata)
    #   - "time_costs": Dict (timing information)
    #   - "latents": Tensor (intermediate latents, if available)
    #   - "masks": Tensor (attention masks, if available)
    
    # Success Status
    success: bool
    error: Optional[str]
```

---

## Version History

- **v1.5.1**: Current version with refactored inference API
  - Split `GenerationConfig` into `GenerationParams` and `GenerationConfig`
  - Renamed parameters for consistency (`key_scale` → `keyscale`, `time_signature` → `timesignature`, `audio_duration` → `duration`, `use_llm_thinking` → `thinking`, `audio_code_string` → `audio_codes`)
  - Added `instrumental` parameter
  - Added `use_constrained_decoding` parameter
  - Added CoT auto-filled fields (`cot_*`)
  - Changed default `audio_format` to "flac"
  - Changed default `batch_size` to 2
  - Changed default `thinking` to True
  - Simplified `GenerationResult` structure with unified `audios` list
  - Added unified `time_costs` in `extra_outputs`

- **v1.5**: Previous version
  - Introduced `GenerationConfig` and `GenerationResult` dataclasses
  - Simplified parameter passing
  - Added comprehensive documentation

---

For more information, see:
- Main README: [`README.md`](README.md)
- REST API Documentation: [`API.md`](API.md)
- Project repository: [ACE-Step-1.5](https://github.com/yourusername/ACE-Step-1.5)
