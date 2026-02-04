# ACE-Step-1.5 Memory Bank

*Session handoff context for continuation agents*

---

## Project Overview

**ACE-Step-1.5** is a text-to-speech (TTS) system with two-stage architecture:

1. **LLM Stage**: Qwen 4B variant with 65,535 custom audio code tokens (49152-114687)
   - Input: Text prompts (with optional speaker context)
   - Output: Sequences of audio codes representing phonemes/prosody

2. **DiT Stage**: Diffusion transformer (Differential in Time)
   - Input: Audio codes from LLM
   - Output: Raw audio waveform
   - Architecture: Transformer with rotary position embeddings, attention layers

**Key constraint**: Cannot swap LLM to vanilla models without retraining - custom vocabulary is fundamental to architecture.

---

## Session Summary

### Security Hardening
- Fixed timing attacks in audio token range checks (constant-time comparisons)
- Added path traversal protection in file handlers (strict directory validation)
- DoS prevention: response size limits, queue depth limits, connection limits
- Error handling hardening across API server

### Code Quality
- Null safety: Added guards for None returns in `get_current_user()` across multiple endpoints
- Type annotations cleanup
- Removed unused imports

### Model Investigation
- Investigated replacing Qwen 4B with Qwen3-30B-A3B for Romanian
- **FINDING**: Incompatible due to custom audio code tokens (65,535 special tokens)
- Vanilla Qwen3 has standard vocabulary (~152K tokens, no audio codes)
- Conclusion: LoRA fine-tuning on existing model is correct approach

### LoRA Training Infrastructure
- Created complete CLI training system for Romanian pronunciation
- Targets DiT attention layers (where pronunciation knowledge lives)
- Ready to train once user provides Romanian audio dataset

---

## Key Technical Findings

### Audio Code Tokens (Critical)

```python
# From acestep/constrained_logits_processor.py
AUDIO_CODE_RANGES = [
    (49152, 65536),   # 16,384 tokens
    (65536, 81920),   # 16,384 tokens
    (81920, 98304),   # 16,384 tokens
    (98304, 114688),  # 16,384 tokens
]
# Total: 65,536 custom audio tokens
```

**Implications**:
- These tokens are phoneme/prosody representations learned during original training
- LLM outputs sequences of these codes, DiT converts codes → audio
- Swapping LLM requires either:
  1. Retraining from scratch with new LLM + audio code vocabulary
  2. LoRA fine-tuning existing model (preserves learned audio mappings)

### Why Vanilla Qwen3-30B-A3B Won't Work

```
Qwen 4B (ACE-Step variant):
  - Vocabulary: ~152K + 65,536 audio codes = ~217K tokens
  - Audio codes: 49152-114687
  - Tokenizer: Modified for audio tokens

Qwen3-30B-A3B (vanilla):
  - Vocabulary: ~152K standard tokens
  - No audio code concepts
  - Would output gibberish (no learned phoneme mappings)
```

### LoRA Strategy

Target DiT attention layers where pronunciation knowledge lives:

```python
# From acestep/training/finetune_lora.py
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "dit.*.attn.qkv",           # Attention projections
        "dit.*.attn.proj",
        "dit.*.mlp.fc1",            # MLP layers
        "dit.*.mlp.fc2",
    ],
    lora_dropout=0.05,
    bias="none",
)
```

**Rationale**: DiT converts audio codes → waveform. Attention layers learn phoneme-to-sound mappings. LoRA adapts these for Romanian phonetics.

---

## Files Created/Modified

### Training Infrastructure (New)

```
acestep/training/
├── __init__.py                 # Package marker
├── finetune_lora.py           # Core LoRA training logic
├── train_cli.py               # CLI entry point
├── data/
│   ├── __init__.py
│   └── audio_dataset.py       # AudioDataset class (waveform + transcripts)
└── utils/
    ├── __init__.py
    └── checkpoint.py          # Save/load LoRA adapters
```

### Modified Files

```
acestep/constrained_logits_processor.py   # Timing attack fixes (constant-time comparisons)
acestep/api_server.py                     # DoS limits, null safety, error handling
acestep/api_server_result.py              # Null safety in user lookups
pyproject.toml                            # Added peft, datasets dependencies
```

### Security Commits

```
774e5ba - fix: security hardening and error handling improvements
  - Timing attack fixes
  - Path traversal protection
  - DoS prevention (size/queue/connection limits)
  - Null safety across endpoints
  - Type annotation cleanup
```

---

## Romanian LoRA Training

### Current State

**Ready to train** - infrastructure complete, awaiting dataset.

### Dataset Requirements

User needs to provide Romanian audio files:

```
dataset/
├── audio/
│   ├── sample001.wav
│   ├── sample002.wav
│   └── ...
└── transcripts.csv      # Columns: filename, transcript
```

**transcripts.csv format**:
```csv
filename,transcript
sample001.wav,Bună ziua, cum te numești?
sample002.wav,Aceasta este o propoziție în română.
```

### Training Command

```bash
python -m acestep.training.train_cli \
  --audio_dir dataset/audio \
  --transcript_file dataset/transcripts.csv \
  --output_dir lora_checkpoints/romanian_v1 \
  --epochs 10 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --lora_r 16 \
  --lora_alpha 32
```

### Training Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--epochs` | 10 | Increase if pronunciation still poor |
| `--batch_size` | 4 | Reduce if GPU OOM |
| `--learning_rate` | 1e-4 | Standard for LoRA |
| `--lora_r` | 16 | LoRA rank (quality vs size tradeoff) |
| `--lora_alpha` | 32 | LoRA scaling (typically 2x rank) |

### Loading Trained LoRA

```python
from acestep.training.utils.checkpoint import load_lora_model

model = load_lora_model(
    base_model_path="checkpoints/acestep-v15-turbo",
    lora_checkpoint="lora_checkpoints/romanian_v1/final"
)
```

---

## Pending/Future Work

### Immediate Next Steps

1. **User provides Romanian dataset**
   - Audio files (16kHz WAV recommended)
   - Transcripts CSV
   - Minimum 100 samples, ideally 1000+

2. **Run initial training**
   - Start with default params
   - Monitor loss curves
   - Test pronunciation quality

3. **Iterate if needed**
   - Increase epochs if underfitting
   - Adjust LoRA rank if capacity issues
   - Consider data augmentation

### If Pronunciation Still Poor After LoRA

**Lyric Encoder Fine-Tuning** (more invasive):

The LLM stage has a "lyric encoder" component that might need Romanian-specific tuning:

```python
# From checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py
class AcestepModel:
    def forward(self, input_ids, attention_mask, lyric_hidden_states=None, ...):
        # Lyric encoder processes text → phoneme features
        # If LoRA doesn't fix pronunciation, this is next target
```

**Approach**:
- Freeze DiT, LoRA on LLM's lyric encoder layers
- Requires Romanian phoneme-aligned dataset
- More complex than DiT LoRA

---

## Important Code Locations

### Training Infrastructure

```
acestep/training/finetune_lora.py       # Core training loop
acestep/training/train_cli.py           # CLI entry point
acestep/training/data/audio_dataset.py  # Dataset loader
acestep/training/utils/checkpoint.py    # Save/load utilities
```

### Model Architecture

```
checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py
  - class AcestepDiT              # DiT stage (LoRA target)
  - class AcestepModel            # Full two-stage model
  - Attention layers: dit.*.attn.*
  - MLP layers: dit.*.mlp.*
```

### Audio Code Handling

```
acestep/constrained_logits_processor.py
  - AUDIO_CODE_RANGES             # 65,536 special tokens
  - ConstrainedLogitsProcessor    # Forces LLM to output valid codes
```

### API Server

```
acestep/api_server.py              # Main FastAPI server
acestep/api_server_result.py       # Result types, user lookups
```

---

## Commands Reference

### Training

```bash
# Basic training
python -m acestep.training.train_cli \
  --audio_dir path/to/audio \
  --transcript_file path/to/transcripts.csv \
  --output_dir lora_checkpoints/romanian

# With custom params
python -m acestep.training.train_cli \
  --audio_dir dataset/audio \
  --transcript_file dataset/transcripts.csv \
  --output_dir lora_checkpoints/romanian_v2 \
  --epochs 20 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_r 32 \
  --lora_alpha 64
```

### Inference (After Training)

```python
from acestep.training.utils.checkpoint import load_lora_model

# Load base model + LoRA
model = load_lora_model(
    base_model_path="checkpoints/acestep-v15-turbo",
    lora_checkpoint="lora_checkpoints/romanian/final"
)

# Generate audio
audio = model.generate(
    text="Bună ziua, cum te numești?",
    # ... other generation params
)
```

### API Server

```bash
# Start server
python start-api.bat

# Or directly
python -m acestep.api_server
```

---

## Git State

### Current Branch

```
main (pushed to hauhaut/ACE-Step-1.5 fork)
```

### Recent Commits

```
774e5ba - fix: security hardening and error handling improvements
  - Timing attack mitigations (constant-time token range checks)
  - Path traversal protection (strict directory validation)
  - DoS prevention (response size, queue depth, connection limits)
  - Null safety fixes (get_current_user() guards)
  - Type annotation cleanup, unused import removal
```

### Unstaged Changes

```
M pyproject.toml              # Added peft, datasets
?? .reaper/                   # Session context (this file)
?? acestep/training/          # New training infrastructure
```

**TODO**: Commit training infrastructure:

```bash
git add acestep/training/ pyproject.toml .reaper/
git commit -m "feat: add LoRA training infrastructure for Romanian

- DiT attention layer targeting for pronunciation adaptation
- CLI interface for easy training runs
- AudioDataset for waveform + transcript pairs
- Checkpoint utilities for save/load
- Ready to train once Romanian dataset provided"
git push
```

---

## Architecture Deep Dive

### Two-Stage Pipeline

```
Text Input
    ↓
[LLM: Qwen 4B + Audio Codes]
    ↓
Audio Code Sequence (e.g., [49152, 65000, 81234, ...])
    ↓
[DiT: Diffusion Transformer]
    ↓
Waveform Output
```

### Why LoRA on DiT (Not LLM)

**LLM Stage**: Converts text → audio codes
- Already trained on massive multilingual data
- Audio code output is language-agnostic (codes represent phonemes, not language)
- Romanian text → audio codes should work reasonably (universal phoneme representation)

**DiT Stage**: Converts audio codes → waveform
- Learns phoneme-to-sound mappings (how code 49152 sounds)
- English-biased during original training
- Romanian phonetics differ (ă, â, î, ș, ț)
- **LoRA here adapts sound synthesis for Romanian phonemes**

### DiT Attention Layers (LoRA Targets)

```python
class AcestepDiT(nn.Module):
    def __init__(self):
        self.blocks = nn.ModuleList([
            DiTBlock(...)  # Contains:
            # - attn.qkv (query/key/value projections) ← LoRA
            # - attn.proj (output projection) ← LoRA
            # - mlp.fc1, mlp.fc2 (feedforward) ← LoRA
        ])
```

**Why these layers**:
- Attention learns cross-timestep phoneme relationships
- MLP learns phoneme-specific sound features
- LoRA injects Romanian-specific acoustic knowledge

---

## Dataset Recommendations

### Audio Quality

- **Sample rate**: 16kHz minimum, 22.05kHz ideal
- **Format**: WAV (lossless)
- **Duration**: 3-15 seconds per clip (too short = no context, too long = memory issues)
- **Quality**: Clean recordings, minimal background noise
- **Speakers**: Multiple speakers improves generalization

### Transcript Quality

- **Accuracy**: Exact match to audio (critical)
- **Text normalization**: Consistent (e.g., numbers as words: "10" → "zece")
- **Punctuation**: Include for prosody cues
- **Encoding**: UTF-8 for Romanian diacritics

### Quantity

| Dataset Size | Expected Quality |
|--------------|------------------|
| 100 samples | Proof of concept, poor generalization |
| 500 samples | Noticeable improvement, some errors |
| 1000+ samples | Good pronunciation, production-ready |
| 5000+ samples | Excellent quality, minimal errors |

---

## Troubleshooting

### Training Issues

**GPU OOM**:
```bash
# Reduce batch size
--batch_size 2

# Reduce LoRA rank
--lora_r 8 --lora_alpha 16
```

**Slow training**:
- Check GPU utilization (`nvidia-smi`)
- Ensure CUDA available (`torch.cuda.is_available()`)
- Use smaller dataset for testing

**Loss not decreasing**:
- Increase learning rate (5e-4)
- Increase epochs (20+)
- Check data quality (transcripts match audio?)

### Inference Issues

**Poor pronunciation after LoRA**:
1. Train longer (more epochs)
2. Increase LoRA rank (r=32)
3. Check dataset quality
4. Consider lyric encoder fine-tuning (see Future Work)

**Model loading errors**:
```python
# Ensure paths are correct
base_model_path="checkpoints/acestep-v15-turbo"  # Not the .safetensors file
lora_checkpoint="lora_checkpoints/romanian/final"  # Directory, not file
```

---

## Contact Context

**User**: hauhaut (hauhaut901@gmail.com)
**Repo**: Fork of ace-step/ACE-Step-1.5
**Task**: Romanian pronunciation improvement via LoRA fine-tuning

**User awaiting**: Romanian audio dataset preparation
**Next agent**: Run training once dataset provided, iterate on quality

---

*Memory bank updated: 2026-02-04*
*Session: Security hardening + Romanian LoRA infrastructure*
*Status: Ready for training, awaiting dataset*
