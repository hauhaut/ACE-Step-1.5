# ACE-Step Inference API Documentation

This document provides comprehensive documentation for the ACE-Step inference API, including parameter specifications for all supported task types.

## Table of Contents

- [Quick Start](#quick-start)
- [API Overview](#api-overview)
- [Configuration Parameters](#configuration-parameters)
- [Task Types](#task-types)
- [Complete Examples](#complete-examples)
- [Best Practices](#best-practices)

---

## Quick Start

### Basic Usage

```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationConfig, generate_music

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

# Configure generation
config = GenerationConfig(
    caption="upbeat electronic dance music with heavy bass",
    bpm=128,
    audio_duration=30,
    batch_size=1,
)

# Generate music
result = generate_music(dit_handler, llm_handler, config)

# Access results
if result.success:
    for audio_path in result.audio_paths:
        print(f"Generated: {audio_path}")
else:
    print(f"Error: {result.error}")
```

---

## API Overview

### Main Function

```python
def generate_music(
    dit_handler: AceStepHandler,
    llm_handler: LLMHandler,
    config: GenerationConfig,
) -> GenerationResult
```

### Configuration Object

The `GenerationConfig` dataclass consolidates all generation parameters:

```python
@dataclass
class GenerationConfig:
    # Required parameters with sensible defaults
    caption: str = ""
    lyrics: str = ""
    # ... (see full parameter list below)
```

### Result Object

```python
@dataclass
class GenerationResult:
    audio_paths: List[str]          # Paths to generated audio files
    generation_info: str            # Markdown-formatted info
    status_message: str             # Status message
    seed_value: str                 # Seed used
    lm_metadata: Optional[Dict]     # LM-generated metadata
    success: bool                   # Success flag
    error: Optional[str]            # Error message if failed
    # ... (see full fields below)
```

---

## Configuration Parameters

### Text Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `caption` | `str` | `""` | Text description of the desired music. Can be a simple prompt like "relaxing piano music" or detailed description with genre, mood, instruments, etc. |
| `lyrics` | `str` | `""` | Lyrics text for vocal music. Use `"[Instrumental]"` for instrumental tracks. Supports multiple languages. |

### Music Metadata

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bpm` | `Optional[int]` | `None` | Beats per minute (30-300). `None` enables auto-detection via LM. |
| `key_scale` | `str` | `""` | Musical key (e.g., "C Major", "Am", "F# minor"). Empty string enables auto-detection. |
| `time_signature` | `str` | `""` | Time signature (e.g., "4/4", "3/4", "6/8"). Empty string enables auto-detection. |
| `vocal_language` | `str` | `"unknown"` | Language code for vocals (ISO 639-1). Supported: `"en"`, `"zh"`, `"ja"`, `"es"`, `"fr"`, etc. Use `"unknown"` for auto-detection. |
| `audio_duration` | `Optional[float]` | `None` | Duration in seconds (10-600). `None` enables auto-detection based on lyrics length. |

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inference_steps` | `int` | `8` | Number of denoising steps. Turbo model: 1-8 (recommended 8). Base model: 1-100 (recommended 32-64). Higher = better quality but slower. |
| `guidance_scale` | `float` | `7.0` | Classifier-free guidance scale (1.0-15.0). Higher values increase adherence to text prompt. Typical range: 5.0-9.0. |
| `use_random_seed` | `bool` | `True` | Whether to use random seed. `True` for different results each time, `False` for reproducible results. |
| `seed` | `int` | `-1` | Random seed for reproducibility. Use `-1` for random seed, or any positive integer for fixed seed. |
| `batch_size` | `int` | `1` | Number of samples to generate in parallel (1-8). Higher values require more GPU memory. |

### Advanced DiT Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_adg` | `bool` | `False` | Use Adaptive Dual Guidance (base model only). Improves quality at the cost of speed. |
| `cfg_interval_start` | `float` | `0.0` | CFG application start ratio (0.0-1.0). Controls when to start applying classifier-free guidance. |
| `cfg_interval_end` | `float` | `1.0` | CFG application end ratio (0.0-1.0). Controls when to stop applying classifier-free guidance. |
| `audio_format` | `str` | `"mp3"` | Output audio format. Options: `"mp3"`, `"wav"`, `"flac"`. |

### Task-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | `str` | `"text2music"` | Generation task type. See [Task Types](#task-types) section for details. |
| `reference_audio` | `Optional[str]` | `None` | Path to reference audio file for style transfer or continuation tasks. |
| `src_audio` | `Optional[str]` | `None` | Path to source audio file for audio-to-audio tasks (cover, repaint, etc.). |
| `audio_code_string` | `Union[str, List[str]]` | `""` | Pre-extracted 5Hz audio codes. Can be single string or list for batch mode. Advanced use only. |
| `repainting_start` | `float` | `0.0` | Repainting start time in seconds (for repaint/lego tasks). |
| `repainting_end` | `float` | `-1` | Repainting end time in seconds. Use `-1` for end of audio. |
| `audio_cover_strength` | `float` | `1.0` | Strength of audio cover/codes influence (0.0-1.0). Higher = stronger influence from source audio. |
| `instruction` | `str` | `""` | Task-specific instruction prompt. Auto-generated if empty. |

### 5Hz Language Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_llm_thinking` | `bool` | `False` | Enable LM-based Chain-of-Thought reasoning. When enabled, LM generates metadata and/or audio codes. |
| `lm_temperature` | `float` | `0.85` | LM sampling temperature (0.0-2.0). Higher = more creative/diverse, lower = more conservative. |
| `lm_cfg_scale` | `float` | `2.0` | LM classifier-free guidance scale (1.0-5.0). Higher = stronger adherence to prompt. |
| `lm_top_k` | `int` | `0` | LM top-k sampling. `0` disables top-k filtering. Typical values: 40-100. |
| `lm_top_p` | `float` | `0.9` | LM nucleus sampling (0.0-1.0). `1.0` disables nucleus sampling. Typical values: 0.9-0.95. |
| `lm_negative_prompt` | `str` | `"NO USER INPUT"` | Negative prompt for LM guidance. Helps avoid unwanted characteristics. |
| `use_cot_metas` | `bool` | `True` | Generate metadata using LM CoT reasoning (BPM, key, duration, etc.). |
| `use_cot_caption` | `bool` | `True` | Refine user caption using LM CoT reasoning. |
| `use_cot_language` | `bool` | `True` | Detect vocal language using LM CoT reasoning. |
| `is_format_caption` | `bool` | `False` | Whether caption is already formatted/refined (skip LM refinement). |
| `constrained_decoding_debug` | `bool` | `False` | Enable debug logging for constrained decoding. |

### Batch LM Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_lm_batch` | `bool` | `False` | Allow batch LM code generation. Faster when `batch_size >= 2` and `use_llm_thinking=True`. |
| `lm_batch_chunk_size` | `int` | `4` | Maximum batch size per LM inference chunk (GPU memory constraint). |

---

## Task Types

ACE-Step supports 6 different generation task types, each optimized for specific use cases.

### 1. Text2Music (Default)

**Purpose**: Generate music from text descriptions and optional metadata.

**Key Parameters**:
```python
config = GenerationConfig(
    task_type="text2music",
    caption="energetic rock music with electric guitar",
    lyrics="[Instrumental]",  # or actual lyrics
    bpm=140,
    audio_duration=30,
)
```

**Required**:
- `caption` or `lyrics` (at least one)

**Optional but Recommended**:
- `bpm`: Controls tempo
- `key_scale`: Controls musical key
- `time_signature`: Controls rhythm structure
- `audio_duration`: Controls length
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
config = GenerationConfig(
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
config = GenerationConfig(
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
config = GenerationConfig(
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
config = GenerationConfig(
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
config = GenerationConfig(
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
from acestep.inference import GenerationConfig, generate_music

config = GenerationConfig(
    task_type="text2music",
    caption="calm ambient music with soft piano and strings",
    audio_duration=60,
    bpm=80,
    key_scale="C Major",
    batch_size=2,  # Generate 2 variations
)

result = generate_music(dit_handler, llm_handler, config)

if result.success:
    for i, path in enumerate(result.audio_paths, 1):
        print(f"Variation {i}: {path}")
```

### Example 2: Song Generation with Lyrics

```python
config = GenerationConfig(
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
    audio_duration=45,
)

result = generate_music(dit_handler, llm_handler, config)
```

### Example 3: Style Cover with LM Reasoning

```python
config = GenerationConfig(
    task_type="cover",
    src_audio="original_pop_song.mp3",
    caption="orchestral symphonic arrangement",
    audio_cover_strength=0.7,
    use_llm_thinking=True,  # Enable LM for metadata
    use_cot_metas=True,
)

result = generate_music(dit_handler, llm_handler, config)

# Access LM-generated metadata
if result.lm_metadata:
    print(f"LM detected BPM: {result.lm_metadata.get('bpm')}")
    print(f"LM detected Key: {result.lm_metadata.get('keyscale')}")
```

### Example 4: Repaint Section of Audio

```python
config = GenerationConfig(
    task_type="repaint",
    src_audio="generated_track.mp3",
    repainting_start=15.0,  # Start at 15 seconds
    repainting_end=25.0,    # End at 25 seconds
    caption="dramatic orchestral buildup",
    inference_steps=32,  # Higher quality for base model
)

result = generate_music(dit_handler, llm_handler, config)
```

### Example 5: Batch Generation with LM

```python
config = GenerationConfig(
    task_type="text2music",
    caption="epic cinematic trailer music",
    batch_size=4,  # Generate 4 variations
    use_llm_thinking=True,
    use_cot_metas=True,
    allow_lm_batch=True,  # Faster batch processing
    lm_batch_chunk_size=2,  # Process 2 at a time (GPU memory)
)

result = generate_music(dit_handler, llm_handler, config)

if result.success:
    print(f"Generated {len(result.audio_paths)} variations")
```

### Example 6: High-Quality Generation (Base Model)

```python
config = GenerationConfig(
    task_type="text2music",
    caption="intricate jazz fusion with complex harmonies",
    inference_steps=64,  # High quality
    guidance_scale=8.0,
    use_adg=True,  # Adaptive Dual Guidance
    cfg_interval_start=0.0,
    cfg_interval_end=1.0,
    audio_format="wav",  # Lossless format
    use_random_seed=False,
    seed=42,  # Reproducible results
)

result = generate_music(dit_handler, llm_handler, config)
```

### Example 7: Extract Vocals from Mix

```python
config = GenerationConfig(
    task_type="extract",
    src_audio="full_song_mix.mp3",
    instruction="Extract the vocals track from the audio:",
)

result = generate_music(dit_handler, llm_handler, config)

if result.success:
    print(f"Extracted vocals: {result.audio_paths[0]}")
```

### Example 8: Add Guitar Track (Lego)

```python
config = GenerationConfig(
    task_type="lego",
    src_audio="drums_and_bass.mp3",
    instruction="Generate the guitar track based on the audio context:",
    caption="funky rhythm guitar with wah-wah effect",
    repainting_start=0.0,
    repainting_end=-1,  # Full duration
)

result = generate_music(dit_handler, llm_handler, config)
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
- Use compressed format (`audio_format="mp3"`)

**For Consistency**:
- Set `use_random_seed=False`
- Use fixed `seed` value
- Keep `lm_temperature` lower (0.7-0.85)

**For Diversity**:
- Set `use_random_seed=True`
- Increase `lm_temperature` (0.9-1.1)
- Use `batch_size > 1` for variations

### 3. Duration Guidelines

- **Instrumental**: 30-180 seconds works well
- **With Lyrics**: Auto-detection recommended (set `audio_duration=None`)
- **Short clips**: 10-20 seconds minimum
- **Long form**: Up to 600 seconds (10 minutes) maximum

### 4. LM Usage

**When to Enable LM (`use_llm_thinking=True`)**:
- Need automatic metadata detection
- Want caption refinement
- Generating from minimal input
- Need diverse outputs

**When to Disable LM**:
- Have precise metadata already
- Need faster generation
- Want full control over parameters

### 5. Batch Processing

```python
# Efficient batch generation
config = GenerationConfig(
    batch_size=8,  # Max supported
    use_llm_thinking=True,
    allow_lm_batch=True,  # Enable for speed
    lm_batch_chunk_size=4,  # Adjust based on GPU memory
)
```

### 6. Error Handling

```python
result = generate_music(dit_handler, llm_handler, config)

if not result.success:
    print(f"Generation failed: {result.error}")
    # Check logs for details
else:
    # Process successful result
    for path in result.audio_paths:
        # ... process audio files
        pass
```

### 7. Memory Management

For large batch sizes or long durations:
- Monitor GPU memory usage
- Reduce `batch_size` if OOM errors occur
- Reduce `lm_batch_chunk_size` for LM operations
- Consider using `offload_to_cpu=True` during initialization

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory errors
- **Solution**: Reduce `batch_size`, `inference_steps`, or enable CPU offloading

**Issue**: Poor quality results
- **Solution**: Increase `inference_steps`, adjust `guidance_scale`, use base model

**Issue**: Results don't match prompt
- **Solution**: Make caption more specific, increase `guidance_scale`, enable LM refinement

**Issue**: Slow generation
- **Solution**: Use turbo model, reduce `inference_steps`, disable ADG

**Issue**: LM not generating codes
- **Solution**: Verify `llm_handler` is initialized, check `use_llm_thinking=True` and `use_cot_metas=True`

---

## API Reference Summary

### GenerationConfig Fields

See [Configuration Parameters](#configuration-parameters) for complete documentation.

### GenerationResult Fields

```python
@dataclass
class GenerationResult:
    # Audio outputs
    audio_paths: List[str]              # List of generated audio file paths
    first_audio: Optional[str]          # First audio (backward compatibility)
    second_audio: Optional[str]         # Second audio (backward compatibility)
    
    # Generation metadata
    generation_info: str                # Markdown-formatted generation info
    status_message: str                 # Status message
    seed_value: str                     # Seed value used
    
    # LM outputs
    lm_metadata: Optional[Dict[str, Any]]  # LM-generated metadata
    
    # Alignment scores (if available)
    align_score_1: Optional[float]
    align_text_1: Optional[str]
    align_plot_1: Optional[Any]
    align_score_2: Optional[float]
    align_text_2: Optional[str]
    align_plot_2: Optional[Any]
    
    # Status
    success: bool                       # Whether generation succeeded
    error: Optional[str]                # Error message if failed
```

---

## Version History

- **v1.5**: Current version with refactored inference API
  - Introduced `GenerationConfig` and `GenerationResult` dataclasses
  - Simplified parameter passing
  - Added comprehensive documentation
  - Maintained backward compatibility with Gradio UI

---

For more information, see:
- Main README: [`README.md`](README.md)
- REST API Documentation: [`API.md`](API.md)
- Project repository: [ACE-Step-1.5](https://github.com/yourusername/ACE-Step-1.5)
