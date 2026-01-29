"""
Dataset Builder for LoRA Training

Provides functionality to:
1. Scan directories for audio files
2. Auto-label audio using LLM
3. Preview and edit metadata
4. Save datasets in JSON format
"""

import os
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torchaudio
from loguru import logger


# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.opus'}


@dataclass
class AudioSample:
    """Represents a single audio sample with its metadata.
    
    Attributes:
        id: Unique identifier for the sample
        audio_path: Path to the audio file
        filename: Original filename
        caption: Generated or user-provided caption describing the music
        lyrics: Lyrics or "[Instrumental]" for instrumental tracks
        bpm: Beats per minute
        keyscale: Musical key (e.g., "C Major", "Am")
        timesignature: Time signature (e.g., "4" for 4/4)
        duration: Duration in seconds
        language: Vocal language or "instrumental"
        is_instrumental: Whether the track is instrumental
        custom_tag: User-defined activation tag for LoRA
        labeled: Whether the sample has been labeled
    """
    id: str = ""
    audio_path: str = ""
    filename: str = ""
    caption: str = ""
    lyrics: str = "[Instrumental]"
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    duration: int = 0
    language: str = "unknown"
    is_instrumental: bool = True
    custom_tag: str = ""
    labeled: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioSample":
        """Create from dictionary."""
        return cls(**data)
    
    def get_full_caption(self, tag_position: str = "prepend") -> str:
        """Get caption with custom tag applied.
        
        Args:
            tag_position: Where to place the custom tag ("prepend", "append", "replace")
            
        Returns:
            Caption with custom tag applied
        """
        if not self.custom_tag:
            return self.caption
        
        if tag_position == "prepend":
            return f"{self.custom_tag}, {self.caption}" if self.caption else self.custom_tag
        elif tag_position == "append":
            return f"{self.caption}, {self.custom_tag}" if self.caption else self.custom_tag
        elif tag_position == "replace":
            return self.custom_tag
        else:
            return self.caption


@dataclass
class DatasetMetadata:
    """Metadata for the entire dataset.
    
    Attributes:
        name: Dataset name
        custom_tag: Default custom tag for all samples
        tag_position: Where to place custom tag ("prepend", "append", "replace")
        created_at: Creation timestamp
        num_samples: Number of samples in the dataset
        all_instrumental: Whether all tracks are instrumental
    """
    name: str = "untitled_dataset"
    custom_tag: str = ""
    tag_position: str = "prepend"
    created_at: str = ""
    num_samples: int = 0
    all_instrumental: bool = True
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DatasetBuilder:
    """Builder for creating training datasets from audio files.
    
    This class handles:
    - Scanning directories for audio files
    - Auto-labeling using LLM
    - Managing sample metadata
    - Saving/loading datasets
    """
    
    def __init__(self):
        """Initialize the dataset builder."""
        self.samples: List[AudioSample] = []
        self.metadata = DatasetMetadata()
        self._current_dir: str = ""
    
    def scan_directory(self, directory: str) -> Tuple[List[AudioSample], str]:
        """Scan a directory for audio files.
        
        Args:
            directory: Path to directory containing audio files
            
        Returns:
            Tuple of (list of AudioSample objects, status message)
        """
        if not os.path.exists(directory):
            return [], f"❌ Directory not found: {directory}"
        
        if not os.path.isdir(directory):
            return [], f"❌ Not a directory: {directory}"
        
        self._current_dir = directory
        self.samples = []
        
        # Scan for audio files
        audio_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_AUDIO_FORMATS:
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            return [], f"❌ No audio files found in {directory}\nSupported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        
        # Sort files by name
        audio_files.sort()
        
        # Create AudioSample objects
        for audio_path in audio_files:
            try:
                # Get duration
                duration = self._get_audio_duration(audio_path)
                
                sample = AudioSample(
                    audio_path=audio_path,
                    filename=os.path.basename(audio_path),
                    duration=duration,
                    is_instrumental=self.metadata.all_instrumental,
                    custom_tag=self.metadata.custom_tag,
                )
                self.samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
        
        self.metadata.num_samples = len(self.samples)
        
        status = f"✅ Found {len(self.samples)} audio files in {directory}"
        return self.samples, status
    
    def _get_audio_duration(self, audio_path: str) -> int:
        """Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds (integer)
        """
        try:
            info = torchaudio.info(audio_path)
            return int(info.num_frames / info.sample_rate)
        except Exception as e:
            logger.warning(f"Failed to get duration for {audio_path}: {e}")
            return 0
    
    def label_sample(
        self,
        sample_idx: int,
        dit_handler,
        llm_handler,
        progress_callback=None,
    ) -> Tuple[AudioSample, str]:
        """Label a single sample using the LLM.
        
        Args:
            sample_idx: Index of sample to label
            dit_handler: DiT handler for audio encoding
            llm_handler: LLM handler for caption generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (updated AudioSample, status message)
        """
        if sample_idx < 0 or sample_idx >= len(self.samples):
            return None, f"❌ Invalid sample index: {sample_idx}"
        
        sample = self.samples[sample_idx]
        
        try:
            if progress_callback:
                progress_callback(f"Processing: {sample.filename}")
            
            # Step 1: Load and encode audio to get audio codes
            audio_codes = self._get_audio_codes(sample.audio_path, dit_handler)
            
            if not audio_codes:
                return sample, f"❌ Failed to encode audio: {sample.filename}"
            
            if progress_callback:
                progress_callback(f"Generating metadata for: {sample.filename}")
            
            # Step 2: Use LLM to understand the audio
            metadata, status = llm_handler.understand_audio_from_codes(
                audio_codes=audio_codes,
                temperature=0.7,
                use_constrained_decoding=True,
            )
            
            if not metadata:
                return sample, f"❌ LLM labeling failed: {status}"
            
            # Step 3: Update sample with generated metadata
            sample.caption = metadata.get('caption', '')
            sample.bpm = self._parse_int(metadata.get('bpm'))
            sample.keyscale = metadata.get('keyscale', '')
            sample.timesignature = metadata.get('timesignature', '')
            sample.language = metadata.get('vocal_language', 'unknown')
            
            # Handle lyrics based on instrumental flag
            if sample.is_instrumental:
                sample.lyrics = "[Instrumental]"
                sample.language = "unknown"
            else:
                sample.lyrics = metadata.get('lyrics', '')
            
            # NOTE: Duration is NOT overwritten from LM metadata.
            # We keep the real audio duration obtained from torchaudio during scan.
            
            sample.labeled = True
            self.samples[sample_idx] = sample
            
            return sample, f"✅ Labeled: {sample.filename}"
            
        except Exception as e:
            logger.exception(f"Error labeling sample {sample.filename}")
            return sample, f"❌ Error: {str(e)}"
    
    def label_all_samples(
        self,
        dit_handler,
        llm_handler,
        progress_callback=None,
    ) -> Tuple[List[AudioSample], str]:
        """Label all samples in the dataset.
        
        Args:
            dit_handler: DiT handler for audio encoding
            llm_handler: LLM handler for caption generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (list of updated samples, status message)
        """
        if not self.samples:
            return [], "❌ No samples to label. Please scan a directory first."
        
        success_count = 0
        fail_count = 0
        
        for i, sample in enumerate(self.samples):
            if progress_callback:
                progress_callback(f"Labeling {i+1}/{len(self.samples)}: {sample.filename}")
            
            _, status = self.label_sample(i, dit_handler, llm_handler, progress_callback)
            
            if "✅" in status:
                success_count += 1
            else:
                fail_count += 1
        
        status_msg = f"✅ Labeled {success_count}/{len(self.samples)} samples"
        if fail_count > 0:
            status_msg += f" ({fail_count} failed)"
        
        return self.samples, status_msg
    
    def _get_audio_codes(self, audio_path: str, dit_handler) -> Optional[str]:
        """Encode audio to get semantic codes for LLM understanding.
        
        Args:
            audio_path: Path to audio file
            dit_handler: DiT handler with VAE and tokenizer
            
        Returns:
            Audio codes string or None if failed
        """
        try:
            # Check if handler has required methods
            if not hasattr(dit_handler, 'convert_src_audio_to_codes'):
                logger.error("DiT handler missing convert_src_audio_to_codes method")
                return None
            
            # Use handler's method to convert audio to codes
            codes_string = dit_handler.convert_src_audio_to_codes(audio_path)
            
            if codes_string and not codes_string.startswith("❌"):
                return codes_string
            else:
                logger.warning(f"Failed to convert audio to codes: {codes_string}")
                return None
                
        except Exception as e:
            logger.exception(f"Error encoding audio {audio_path}")
            return None
    
    def _parse_int(self, value: Any) -> Optional[int]:
        """Safely parse an integer value."""
        if value is None or value == "N/A" or value == "":
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def update_sample(self, sample_idx: int, **kwargs) -> Tuple[AudioSample, str]:
        """Update a sample's metadata.
        
        Args:
            sample_idx: Index of sample to update
            **kwargs: Fields to update
            
        Returns:
            Tuple of (updated sample, status message)
        """
        if sample_idx < 0 or sample_idx >= len(self.samples):
            return None, f"❌ Invalid sample index: {sample_idx}"
        
        sample = self.samples[sample_idx]
        
        for key, value in kwargs.items():
            if hasattr(sample, key):
                setattr(sample, key, value)
        
        self.samples[sample_idx] = sample
        return sample, f"✅ Updated: {sample.filename}"
    
    def set_custom_tag(self, custom_tag: str, tag_position: str = "prepend"):
        """Set the custom tag for all samples.
        
        Args:
            custom_tag: Custom activation tag
            tag_position: Where to place tag ("prepend", "append", "replace")
        """
        self.metadata.custom_tag = custom_tag
        self.metadata.tag_position = tag_position
        
        for sample in self.samples:
            sample.custom_tag = custom_tag
    
    def set_all_instrumental(self, is_instrumental: bool):
        """Set instrumental flag for all samples.
        
        Args:
            is_instrumental: Whether all tracks are instrumental
        """
        self.metadata.all_instrumental = is_instrumental
        
        for sample in self.samples:
            sample.is_instrumental = is_instrumental
            if is_instrumental:
                sample.lyrics = "[Instrumental]"
                sample.language = "unknown"
    
    def get_sample_count(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.samples)
    
    def get_labeled_count(self) -> int:
        """Get the number of labeled samples."""
        return sum(1 for s in self.samples if s.labeled)
    
    def save_dataset(self, output_path: str, dataset_name: str = None) -> str:
        """Save the dataset to a JSON file.
        
        Args:
            output_path: Path to save the dataset JSON
            dataset_name: Optional name for the dataset
            
        Returns:
            Status message
        """
        if not self.samples:
            return "❌ No samples to save"
        
        if dataset_name:
            self.metadata.name = dataset_name
        
        self.metadata.num_samples = len(self.samples)
        self.metadata.created_at = datetime.now().isoformat()
        
        # Build dataset with captions that include custom tags
        dataset = {
            "metadata": self.metadata.to_dict(),
            "samples": []
        }
        
        for sample in self.samples:
            sample_dict = sample.to_dict()
            # Apply custom tag to caption based on position
            sample_dict["caption"] = sample.get_full_caption(self.metadata.tag_position)
            dataset["samples"].append(sample_dict)
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            return f"✅ Dataset saved to {output_path}\n{len(self.samples)} samples, tag: '{self.metadata.custom_tag}'"
        except Exception as e:
            logger.exception("Error saving dataset")
            return f"❌ Failed to save dataset: {str(e)}"
    
    def load_dataset(self, dataset_path: str) -> Tuple[List[AudioSample], str]:
        """Load a dataset from a JSON file.
        
        Args:
            dataset_path: Path to the dataset JSON file
            
        Returns:
            Tuple of (list of samples, status message)
        """
        if not os.path.exists(dataset_path):
            return [], f"❌ Dataset not found: {dataset_path}"
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load metadata
            if "metadata" in data:
                meta_dict = data["metadata"]
                self.metadata = DatasetMetadata(
                    name=meta_dict.get("name", "untitled"),
                    custom_tag=meta_dict.get("custom_tag", ""),
                    tag_position=meta_dict.get("tag_position", "prepend"),
                    created_at=meta_dict.get("created_at", ""),
                    num_samples=meta_dict.get("num_samples", 0),
                    all_instrumental=meta_dict.get("all_instrumental", True),
                )
            
            # Load samples
            self.samples = []
            for sample_dict in data.get("samples", []):
                sample = AudioSample.from_dict(sample_dict)
                self.samples.append(sample)
            
            return self.samples, f"✅ Loaded {len(self.samples)} samples from {dataset_path}"
            
        except Exception as e:
            logger.exception("Error loading dataset")
            return [], f"❌ Failed to load dataset: {str(e)}"
    
    def get_samples_dataframe_data(self) -> List[List[Any]]:
        """Get samples data in a format suitable for Gradio DataFrame.
        
        Returns:
            List of rows for DataFrame display
        """
        rows = []
        for i, sample in enumerate(self.samples):
            rows.append([
                i,
                sample.filename,
                f"{sample.duration:.1f}s",
                "✅" if sample.labeled else "❌",
                sample.bpm or "-",
                sample.keyscale or "-",
                sample.caption[:50] + "..." if len(sample.caption) > 50 else sample.caption or "-",
            ])
        return rows
    
    def to_training_format(self) -> List[Dict[str, Any]]:
        """Convert dataset to format suitable for training.
        
        Returns:
            List of training sample dictionaries
        """
        training_samples = []
        
        for sample in self.samples:
            if not sample.labeled:
                continue
            
            training_sample = {
                "audio_path": sample.audio_path,
                "caption": sample.get_full_caption(self.metadata.tag_position),
                "lyrics": sample.lyrics,
                "bpm": sample.bpm,
                "keyscale": sample.keyscale,
                "timesignature": sample.timesignature,
                "duration": sample.duration,
                "language": sample.language,
                "is_instrumental": sample.is_instrumental,
            }
            training_samples.append(training_sample)
        
        return training_samples
    
    def preprocess_to_tensors(
        self,
        dit_handler,
        output_dir: str,
        max_duration: float = 240.0,
        progress_callback=None,
    ) -> Tuple[List[str], str]:
        """Preprocess all labeled samples to tensor files for efficient training.
        
        This method pre-computes all tensors needed by the DiT decoder:
        - target_latents: VAE-encoded audio
        - encoder_hidden_states: Condition encoder output
        - context_latents: Source context (silence_latent + zeros for text2music)
        
        Args:
            dit_handler: Initialized DiT handler with model, VAE, and text encoder
            output_dir: Directory to save preprocessed .pt files
            max_duration: Maximum audio duration in seconds (default 240s = 4 min)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (list of output paths, status message)
        """
        if not self.samples:
            return [], "❌ No samples to preprocess"
        
        labeled_samples = [s for s in self.samples if s.labeled]
        if not labeled_samples:
            return [], "❌ No labeled samples to preprocess"
        
        # Validate handler
        if dit_handler is None or dit_handler.model is None:
            return [], "❌ Model not initialized. Please initialize the service first."
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        success_count = 0
        fail_count = 0
        
        # Get model and components
        model = dit_handler.model
        vae = dit_handler.vae
        text_encoder = dit_handler.text_encoder
        text_tokenizer = dit_handler.text_tokenizer
        silence_latent = dit_handler.silence_latent
        device = dit_handler.device
        dtype = dit_handler.dtype
        
        target_sample_rate = 48000
        
        for i, sample in enumerate(labeled_samples):
            try:
                if progress_callback:
                    progress_callback(f"Preprocessing {i+1}/{len(labeled_samples)}: {sample.filename}")
                
                # Step 1: Load and preprocess audio to stereo @ 48kHz
                audio, sr = torchaudio.load(sample.audio_path)
                
                # Resample if needed
                if sr != target_sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                    audio = resampler(audio)
                
                # Convert to stereo
                if audio.shape[0] == 1:
                    audio = audio.repeat(2, 1)
                elif audio.shape[0] > 2:
                    audio = audio[:2, :]
                
                # Truncate to max duration
                max_samples = int(max_duration * target_sample_rate)
                if audio.shape[1] > max_samples:
                    audio = audio[:, :max_samples]
                
                # Add batch dimension: [2, T] -> [1, 2, T]
                audio = audio.unsqueeze(0).to(device).to(vae.dtype)
                
                # Step 2: VAE encode audio to get target_latents
                with torch.no_grad():
                    latent = vae.encode(audio).latent_dist.sample()
                    # [1, 64, T_latent] -> [1, T_latent, 64]
                    target_latents = latent.transpose(1, 2).to(dtype)
                
                latent_length = target_latents.shape[1]
                
                # Step 3: Create attention mask (all ones for valid audio)
                attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)
                
                # Step 4: Encode caption text
                caption = sample.get_full_caption(self.metadata.tag_position)
                text_inputs = text_tokenizer(
                    caption,
                    padding="max_length",
                    max_length=256,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                text_attention_mask = text_inputs.attention_mask.to(device).to(dtype)
                
                with torch.no_grad():
                    text_outputs = text_encoder(text_input_ids)
                    text_hidden_states = text_outputs.last_hidden_state.to(dtype)
                
                # Step 5: Encode lyrics
                lyrics = sample.lyrics if sample.lyrics else "[Instrumental]"
                lyric_inputs = text_tokenizer(
                    lyrics,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                lyric_input_ids = lyric_inputs.input_ids.to(device)
                lyric_attention_mask = lyric_inputs.attention_mask.to(device).to(dtype)
                
                with torch.no_grad():
                    lyric_hidden_states = text_encoder.embed_tokens(lyric_input_ids).to(dtype)
                
                # Step 6: Prepare refer_audio (empty for text2music)
                # Create minimal refer_audio placeholder
                refer_audio_hidden = torch.zeros(1, 1, 64, device=device, dtype=dtype)
                refer_audio_order_mask = torch.zeros(1, device=device, dtype=torch.long)
                
                # Step 7: Run model.encoder to get encoder_hidden_states
                with torch.no_grad():
                    encoder_hidden_states, encoder_attention_mask = model.encoder(
                        text_hidden_states=text_hidden_states,
                        text_attention_mask=text_attention_mask,
                        lyric_hidden_states=lyric_hidden_states,
                        lyric_attention_mask=lyric_attention_mask,
                        refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
                        refer_audio_order_mask=refer_audio_order_mask,
                    )
                
                # Step 8: Build context_latents for text2music
                # For text2music: src_latents = silence_latent, is_covers = 0
                # chunk_masks: 1 = generate, 0 = keep original
                # IMPORTANT: chunk_masks must have same shape as src_latents [B, T, 64]
                # For text2music, we want to generate the entire audio, so chunk_masks = all 1s
                src_latents = silence_latent[:, :latent_length, :].to(dtype)
                if src_latents.shape[0] < 1:
                    src_latents = src_latents.expand(1, -1, -1)
                
                # Pad or truncate silence_latent to match latent_length
                if src_latents.shape[1] < latent_length:
                    pad_len = latent_length - src_latents.shape[1]
                    src_latents = torch.cat([
                        src_latents,
                        silence_latent[:, :pad_len, :].expand(1, -1, -1).to(dtype)
                    ], dim=1)
                elif src_latents.shape[1] > latent_length:
                    src_latents = src_latents[:, :latent_length, :]
                
                # chunk_masks = 1 means "generate this region", 0 = keep original
                # Shape must match src_latents: [B, T, 64] (NOT [B, T, 1])
                # For text2music, generate everything -> all 1s with shape [1, T, 64]
                chunk_masks = torch.ones(1, latent_length, 64, device=device, dtype=dtype)
                # context_latents = [src_latents, chunk_masks] -> [B, T, 128]
                context_latents = torch.cat([src_latents, chunk_masks], dim=-1)
                
                # Step 9: Save all tensors to .pt file (squeeze batch dimension for storage)
                output_data = {
                    "target_latents": target_latents.squeeze(0).cpu(),  # [T, 64]
                    "attention_mask": attention_mask.squeeze(0).cpu(),  # [T]
                    "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),  # [L, D]
                    "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),  # [L]
                    "context_latents": context_latents.squeeze(0).cpu(),  # [T, 65]
                    "metadata": {
                        "audio_path": sample.audio_path,
                        "filename": sample.filename,
                        "caption": caption,
                        "lyrics": lyrics,
                        "duration": sample.duration,
                        "bpm": sample.bpm,
                        "keyscale": sample.keyscale,
                        "timesignature": sample.timesignature,
                        "language": sample.language,
                        "is_instrumental": sample.is_instrumental,
                    }
                }
                
                # Save with sample ID as filename
                output_path = os.path.join(output_dir, f"{sample.id}.pt")
                torch.save(output_data, output_path)
                output_paths.append(output_path)
                success_count += 1
                
            except Exception as e:
                logger.exception(f"Error preprocessing {sample.filename}")
                fail_count += 1
                if progress_callback:
                    progress_callback(f"❌ Failed: {sample.filename}: {str(e)}")
        
        # Save manifest file listing all preprocessed samples
        manifest = {
            "metadata": self.metadata.to_dict(),
            "samples": output_paths,
            "num_samples": len(output_paths),
        }
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        status = f"✅ Preprocessed {success_count}/{len(labeled_samples)} samples to {output_dir}"
        if fail_count > 0:
            status += f" ({fail_count} failed)"
        
        return output_paths, status
