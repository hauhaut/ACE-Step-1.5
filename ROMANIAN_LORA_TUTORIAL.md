# Romanian Pronunciation LoRA Training Tutorial

Train a custom LoRA adapter to improve ACE-Step's Romanian pronunciation and musical style.

## What is This?

LoRA (Low-Rank Adaptation) allows you to fine-tune ACE-Step's music generation on a specific style, voice, or language with minimal data and compute. For Romanian music, this means:

- Better pronunciation of Romanian lyrics
- Capturing specific vocal timbres
- Learning traditional Romanian musical patterns
- Maintaining consistent style across generations

**Why LoRA?** Training takes ~1 hour on an RTX 3090 with just 8 songs and uses only 12GB VRAM.

## Prerequisites

### Hardware
- GPU with 12GB+ VRAM (RTX 3090, 4090, or better)
- 20GB+ free disk space

### Software
- ACE-Step 1.5 installed (see main [README.md](./README.md))
- Python 3.11
- CUDA-compatible GPU drivers

### Data Requirements
- **8-20 audio files** with clear Romanian vocals
- Audio format: MP3, WAV, FLAC (16-bit or 24-bit, any sample rate)
- **Lyrics files** (.txt) with same name as audio files
- Each song: 30s to 4 minutes (recommended: 2-3 minutes)

**Quality matters more than quantity.** Clean vocals, minimal background noise, and accurate lyrics produce better results.

## Preparing Your Data

### Directory Structure

Organize your Romanian music like this:

```
romanian_music/
├── cântec_popular.mp3
├── cântec_popular.txt
├── doina.wav
├── doina.txt
├── hora.mp3
├── hora.txt
└── ...
```

**Rules:**
- Audio and lyrics must share the same filename (different extensions)
- Lyrics files are **optional** but improve quality
- Use descriptive filenames (helps with organization)

### Audio File Requirements

| Requirement | Details |
|-------------|---------|
| **Format** | MP3, WAV, FLAC |
| **Quality** | 16kHz+ sample rate, clear vocals |
| **Duration** | 30s - 240s (2-3 min recommended) |
| **Vocals** | Clean, intelligible Romanian pronunciation |
| **Mix** | Vocals prominent, minimal distortion |

**Avoid:**
- Heavy compression or artifacts
- Excessive reverb drowning vocals
- Songs with spoken intros/outros (trim them)
- Multiple languages in one song

### Lyrics File Format

Each `.txt` file contains Romanian lyrics:

```
Codrul verde de brad şi de tei
Codrul verde de brad şi de tei
Îmi aduce aminte de cei
Care-au luptat pentru țară
```

**Guidelines:**
- UTF-8 encoding (supports Romanian diacritics: ă, â, î, ș, ț)
- Plain text, no timestamps or formatting
- Match the actual sung lyrics
- Include repeated sections (choruses)

**Example filename pairs:**
```
doina_din_muntenia.mp3  →  doina_din_muntenia.txt
hora_moldovenească.wav  →  hora_moldovenească.txt
```

## Training Options

### Option A: Single Command (Fastest)

Use the all-in-one training script:

```bash
python train_romanian_lora.py /path/to/romanian_music
```

**What it does:**
1. Loads the DiT model
2. Scans audio directory
3. Sets metadata (BPM, key, etc.)
4. Preprocesses audio to tensors
5. Trains LoRA for 30 epochs
6. Saves to `./romanian_lora/final/`

**Customize settings:**

```bash
python train_romanian_lora.py /path/to/romanian_music \
  --output ./my_lora \
  --epochs 50 \
  --lr 5e-5 \
  --r 4 \
  --alpha 8 \
  --custom-tag "romanian folk music"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `./romanian_lora` | Output directory |
| `--epochs` | 30 | Training epochs |
| `--lr` | 5e-5 | Learning rate |
| `--r` | 4 | LoRA rank (capacity) |
| `--alpha` | 8 | LoRA alpha (scaling) |
| `--dropout` | 0.15 | LoRA dropout |
| `--save-every` | 5 | Save checkpoint every N epochs |
| `--custom-tag` | `"romanian music"` | Style activation tag |

**Time estimate:** 1 hour for 10 songs on RTX 3090

### Option B: Gradio UI (Interactive)

Use the web interface for visual control:

```bash
uv run acestep
```

Navigate to **LoRA Training** tab.

#### Step 1: Prepare Dataset

1. **Data Preparation** section:
   - Enter audio directory path
   - Click **Scan Directory**
   - Review detected files

2. **Custom Tag** (optional):
   - Enter: `romanian folk music`
   - Position: `prepend`

3. **Label Samples**:
   - Click **Label All Samples**
   - Wait for LLM to generate captions/metadata
   - Or use **Skip Metas** for basic metadata

4. **Preview & Edit**:
   - Use slider to select samples
   - Manually edit captions, lyrics, BPM, key
   - Click **Save Changes** per sample

5. **Save Dataset**:
   - Enter path: `./romanian_dataset.json`
   - Click **Save Dataset**

#### Step 2: Preprocess

1. **Preprocessing** section:
   - Load dataset JSON (if saved)
   - Set tensor output: `./romanian_tensors`
   - Click **Preprocess**

This converts audio to VAE latents (5-10 min for 10 songs).

#### Step 3: Train

1. **Train LoRA** tab:
   - **Dataset path**: `./romanian_tensors`
   - Click **Load Dataset**

2. **LoRA Settings**:
   - Rank: `4` (small dataset)
   - Alpha: `8`
   - Dropout: `0.15`

3. **Training Parameters**:
   - Learning Rate: `5e-5`
   - Max Epochs: `30`
   - Batch Size: `1`
   - Gradient Accumulation: `4`
   - Save Every N Epochs: `5`

4. **Start Training**:
   - Click **Start Training**
   - Monitor loss curve
   - Stop early if loss plateaus

5. **Export LoRA**:
   - Enter export path: `./romanian_lora_final`
   - Click **Export LoRA**

### Option C: CLI Step-by-Step

For scripting and automation:

```bash
# 1. Prepare dataset
uv run acestep-prepare \
  --audio-dir ./romanian_music \
  --output ./romanian_dataset.json \
  --custom-tag "romanian folk music"

# 2. Preprocess tensors
uv run acestep-preprocess \
  --dataset ./romanian_dataset.json \
  --output ./romanian_tensors

# 3. Train LoRA
uv run acestep-train \
  --dataset ./romanian_tensors \
  --output ./romanian_lora \
  --r 4 --alpha 8 --dropout 0.15 \
  --lr 5e-5 --epochs 30
```

## Recommended Settings for Small Datasets

For **8-15 Romanian songs**, use these conservative settings to avoid overfitting:

| Setting | Value | Why |
|---------|-------|-----|
| **LoRA Rank (r)** | 4 | Low capacity prevents memorization |
| **LoRA Alpha** | 8 | Standard 2x rank scaling |
| **Dropout** | 0.15 | Regularization for small data |
| **Learning Rate** | 5e-5 | Gentle updates |
| **Epochs** | 20-30 | Stop when loss plateaus |
| **Batch Size** | 1 | GPU memory limit |
| **Gradient Accumulation** | 4 | Effective batch size = 4 |

**If you have 20+ songs**, you can increase:
- Rank to `8-16`
- Epochs to `50`
- Learning rate to `1e-4`

**Red flags (overfitting):**
- Loss still dropping but outputs sound identical to training data
- Model only works with exact training lyrics
- No generalization to new prompts

→ Solution: Lower rank, add dropout, reduce epochs

## Using Your Trained LoRA

### Loading in Gradio UI

1. Start ACE-Step:
   ```bash
   uv run acestep
   ```

2. Navigate to **LoRA** tab

3. **Load LoRA**:
   - Path: `./romanian_lora/final`
   - Click **Load LoRA**
   - Enable: Check **Use LoRA**

4. **Test Generation**:
   - Go to **Text-to-Music** tab
   - Caption: `romanian folk music, traditional doina, female vocals`
   - Lyrics: `Codrul verde de brad și de tei...`
   - Generate

### Loading Programmatically

```python
from acestep.handler import AceStepHandler

handler = AceStepHandler()

# Initialize service
handler.initialize_service(
    project_root=".",
    config_path="acestep-v15-turbo",
    device="cuda"
)

# Load LoRA
handler.load_lora("./romanian_lora/final")
handler.set_use_lora(True)

# Generate with LoRA
result = handler.text_to_music(
    caption="romanian folk music, traditional hora",
    lyrics="Sară, sară, să-nserăm...",
    duration=30.0,
    seed=42
)
```

### Prompt Tips

**Include the activation tag** from training:

```
romanian folk music, traditional doina, female vocals, emotional
```

**Be specific about style:**
```
romanian folk music, hora dance, accordion and violin, upbeat
romanian folk music, doina ballad, melancholic, solo female voice
romanian folk music, colinde christmas carol, choir vocals
```

**Mix with other ACE-Step features:**
- Reference audio: Upload a Romanian song for style transfer
- Cover generation: Apply Romanian style to existing music
- Metadata control: Set BPM (120-140 for hora, 60-80 for doina)

## Troubleshooting

### No Audio Generated

**Symptoms:** Empty output or silence

**Causes:**
- LoRA path incorrect
- Model not initialized
- Use LoRA checkbox not enabled

**Fix:**
```bash
# Verify LoRA directory exists
ls ./romanian_lora/final

# Check for adapter_model.safetensors or adapter_model.bin
# Reload LoRA in UI with correct path
```

### Poor Quality / Garbled Pronunciation

**Symptoms:** Romanian words sound wrong, unnatural accent

**Causes:**
- Training data had noisy vocals
- Lyrics didn't match audio
- Overfitting (too many epochs)
- Rank too high for dataset size

**Fix:**
- Retrain with cleaner audio
- Reduce epochs to 15-20
- Lower rank to 2-4
- Add more training data (15+ songs)

### LoRA Has No Effect

**Symptoms:** Output sounds identical with/without LoRA

**Causes:**
- LoRA not loaded
- Rank too low (underfitting)
- Learning rate too low
- Not enough training

**Fix:**
```python
# Verify LoRA is active
handler.check_lora_status()  # Should show LoRA path

# Increase training:
# - Rank to 8
# - Epochs to 50
# - Learning rate to 1e-4
```

### Out of Memory During Training

**Symptoms:** CUDA OOM error

**Causes:**
- GPU too small
- Batch size too high
- Rank too high

**Fix:**
```bash
# Reduce memory usage
python train_romanian_lora.py ./romanian_music \
  --r 4 \           # Lower rank
  --device cuda \
  --custom-tag "romanian music"

# In Gradio: reduce batch size to 1, gradient accumulation to 1
```

### Training Loss Not Decreasing

**Symptoms:** Loss stuck or increasing

**Causes:**
- Learning rate too high/low
- Bad initialization
- Dataset issues

**Fix:**
- Try learning rate `1e-5` or `1e-4`
- Check preprocessing logs for errors
- Verify audio files are valid

## Tips for Best Results

### Data Quality

1. **Clean vocals** - Use vocal isolation tools if needed (e.g., Demucs, UVR)
2. **Consistent style** - Mix traditional and modern Romanian or keep them separate
3. **Accurate lyrics** - Proofread for typos and diacritics
4. **Trim silence** - Remove long intros/outros from audio files

### Training Strategy

1. **Start small** - 10 songs, rank 4, 20 epochs
2. **Monitor loss** - Stop if it plateaus or starts increasing
3. **Test early** - Generate samples at epoch 10, 20, 30
4. **Keep checkpoints** - Save every 5 epochs to compare versions

### Generation Tips

1. **Use the activation tag** - Always include "romanian folk music" or your custom tag
2. **Combine with reference audio** - Upload a Romanian song + use LoRA for best results
3. **Experiment with seeds** - Try 5-10 seeds and pick the best pronunciation
4. **Adjust LoRA weight** - In code: `handler.set_lora_scale(0.8)` (default 1.0)

### Advanced: Multiple LoRAs

Train separate LoRAs for different styles:

```bash
# Traditional folk
python train_romanian_lora.py ./traditional --custom-tag "romanian traditional folk"

# Modern pop
python train_romanian_lora.py ./modern --custom-tag "romanian pop music"

# Load the appropriate one based on desired output
```

### Iteration

If first results aren't perfect:

1. **Check training loss curve** - Should decrease smoothly
2. **Listen to training samples** - Are they high quality?
3. **Test with/without LoRA** - Is there a clear difference?
4. **Adjust hyperparameters** - Try different rank/lr combinations
5. **Add more data** - 20+ songs significantly improve quality

---

**Next Steps:**
- Train your first LoRA with 8-10 songs
- Test with various Romanian prompts
- Share your results on [Discord](https://discord.gg/PeWDxrkdj7)
- See [GRADIO_GUIDE.md](./docs/en/GRADIO_GUIDE.md) for full UI documentation

**Need help?** Ask in the ACE-Step Discord #lora-training channel.
