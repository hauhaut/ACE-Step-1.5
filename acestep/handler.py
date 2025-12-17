"""
Business Logic Handler
Encapsulates all data processing and business logic as a bridge between model and UI
"""
import os
import math
import glob
import tempfile
import traceback
import re
import random
from typing import Optional, Dict, Any, Tuple, List, Union

import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
import time

from transformers import AutoTokenizer, AutoModel
from diffusers.models import AutoencoderOobleck


class AceStepHandler:
    """ACE-Step Business Logic Handler"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.device = "cpu"
        self.dtype = torch.float32  # Will be set based on device in initialize_service
        self.temp_dir = tempfile.mkdtemp()
        
        # VAE for audio encoding/decoding
        self.vae = None
        
        # Text encoder and tokenizer
        self.text_encoder = None
        self.text_tokenizer = None
        
        # Silence latent for initialization
        self.silence_latent = None
        
        # Sample rate
        self.sample_rate = 48000
        
        # 5Hz LM related
        self.lm_model = None
        self.lm_tokenizer = None
        self.lm_initialized = False
        
        # Reward model (temporarily disabled)
        self.reward_model = None
        
        # Dataset related (temporarily disabled)
        self.dataset = None
        self.dataset_imported = False
        
        # Batch size
        self.batch_size = 2
        
        # Custom layers config
        self.custom_layers_config = {
            2: [6, 7],
            3: [10, 11],
            4: [3],
            5: [8, 9, 11],
            6: [8]
        }
    
    def get_available_checkpoints(self) -> str:
        """Return project root directory path"""
        # Get project root (handler.py is in acestep/, so go up two levels to project root)
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        # default checkpoints
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        if os.path.exists(checkpoint_dir):
            return [checkpoint_dir]
        else:
            return []
    
    def get_available_acestep_v15_models(self) -> List[str]:
        """Scan and return all model directory names starting with 'acestep-v15-'"""
        # Get project root
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        
        models = []
        if os.path.exists(checkpoint_dir):
            # Scan all directories starting with 'acestep-v15-' in checkpoints folder
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(item_path) and item.startswith("acestep-v15-"):
                    models.append(item)
        
        # Sort by name
        models.sort()
        return models
    
    def get_available_5hz_lm_models(self) -> List[str]:
        """Scan and return all model directory names starting with 'acestep-5Hz-lm-'"""
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        
        models = []
        if os.path.exists(checkpoint_dir):
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(item_path) and item.startswith("acestep-5Hz-lm-"):
                    models.append(item)
        
        models.sort()
        return models
    
    def is_flash_attention_available(self) -> bool:
        """Check if flash attention is available on the system"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def initialize_service(
        self, 
        project_root: str,
        config_path: str,
        device: str = "auto",
        init_llm: bool = False,
        lm_model_path: str = "acestep-5Hz-lm-0.6B",
        use_flash_attention: bool = False,
    ) -> Tuple[str, bool]:
        """
        Initialize model service
        
        Args:
            project_root: Project root path (may be checkpoints directory, will be handled automatically)
            config_path: Model config directory name (e.g., "acestep-v15-turbo")
            device: Device type
            init_llm: Whether to initialize 5Hz LM model
            lm_model_path: 5Hz LM model path
            use_flash_attention: Whether to use flash attention (requires flash_attn package)
        
        Returns:
            (status_message, enable_generate_button)
        """
        try:
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.device = device
            # Set dtype based on device: bfloat16 for cuda, float32 for cpu
            self.dtype = torch.bfloat16 if device == "cuda" else torch.float32

            # Auto-detect project root (independent of passed project_root parameter)
            current_file = os.path.abspath(__file__)
            actual_project_root = os.path.dirname(os.path.dirname(current_file))
            checkpoint_dir = os.path.join(actual_project_root, "checkpoints")

            # 1. Load main model
            # config_path is relative path (e.g., "acestep-v15-turbo"), concatenate to checkpoints directory
            acestep_v15_checkpoint_path = os.path.join(checkpoint_dir, config_path)
            if os.path.exists(acestep_v15_checkpoint_path):
                # Determine attention implementation
                attn_implementation = "flash_attention_2" if use_flash_attention and self.is_flash_attention_available() else "eager"
                self.model = AutoModel.from_pretrained(
                    acestep_v15_checkpoint_path, 
                    trust_remote_code=True,
                    attn_implementation=attn_implementation
                )
                self.config = self.model.config
                # Move model to device and set dtype
                self.model = self.model.to(device).to(self.dtype)
                self.model.eval()
                silence_latent_path = os.path.join(acestep_v15_checkpoint_path, "silence_latent.pt")
                if os.path.exists(silence_latent_path):
                    self.silence_latent = torch.load(silence_latent_path).transpose(1, 2).squeeze(0)  # [L, C]
                    self.silence_latent = self.silence_latent.to(device).to(self.dtype)
                else:
                    raise FileNotFoundError(f"Silence latent not found at {silence_latent_path}")
            else:
                raise FileNotFoundError(f"ACE-Step V1.5 checkpoint not found at {acestep_v15_checkpoint_path}")
            
            # 2. Load VAE
            vae_checkpoint_path = os.path.join(checkpoint_dir, "vae")
            if os.path.exists(vae_checkpoint_path):
                self.vae = AutoencoderOobleck.from_pretrained(vae_checkpoint_path)
                self.vae = self.vae.to(device).to(self.dtype)
                self.vae.eval()
            else:
                raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint_path}")
            
            # 3. Load text encoder and tokenizer
            text_encoder_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
            if os.path.exists(text_encoder_path):
                self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
                self.text_encoder = AutoModel.from_pretrained(text_encoder_path)
                self.text_encoder = self.text_encoder.to(device).to(self.dtype)
                self.text_encoder.eval()
            else:
                raise FileNotFoundError(f"Text encoder not found at {text_encoder_path}")
            
            # 4. Load 5Hz LM model (optional, only if init_llm is True)
            if init_llm:
                full_lm_model_path = os.path.join(checkpoint_dir, lm_model_path)
                if os.path.exists(full_lm_model_path):
                    if device == "cuda":
                        status_msg = self._initialize_5hz_lm_cuda(full_lm_model_path)
                        if not self.llm_initialized:
                            return status_msg, False
                    self.llm = AutoModel.from_pretrained(full_lm_model_path)
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(full_lm_model_path)
                else:
                    # 5Hz LM path not found
                    return f"❌ 5Hz LM model not found at {full_lm_model_path}", False

            # Determine actual attention implementation used
            actual_attn = "flash_attention_2" if use_flash_attention and self.is_flash_attention_available() else "eager"
            
            status_msg = f"✅ Model initialized successfully on {device}\n"
            status_msg += f"Main model: {acestep_v15_checkpoint_path}\n"
            status_msg += f"VAE: {vae_checkpoint_path}\n"
            status_msg += f"Text encoder: {text_encoder_path}\n"
            if init_llm and hasattr(self, 'llm') and self.llm is not None:
                status_msg += f"5Hz LM model: {os.path.join(checkpoint_dir, lm_model_path)}\n"
            else:
                status_msg += f"5Hz LM model: Not loaded (checkbox not selected)\n"
            status_msg += f"Dtype: {self.dtype}\n"
            status_msg += f"Attention: {actual_attn}"
            
            return status_msg, True
            
        except Exception as e:
            error_msg = f"❌ Error initializing model: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return error_msg, False
    
    def import_dataset(self, dataset_type: str) -> str:
        """Import dataset (temporarily disabled)"""
        self.dataset_imported = False
        return f"⚠️ Dataset import is currently disabled. Text2MusicDataset dependency not available."
    
    def get_item_data(self, *args, **kwargs):
        """Get dataset item (temporarily disabled)"""
        return "", "", "", "", "", None, None, None, "❌ Dataset not available", "", 0, "", None, None, None, {}, "text2music"

    def get_gpu_memory_utilization(self, minimal_gpu: float = 8, min_ratio: float = 0.2, max_ratio: float = 0.9) -> float:
        """Get GPU memory utilization ratio"""
        try:
            device = torch.device("cuda:0")
            total_gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
            allocated_mem_bytes = torch.cuda.memory_allocated(device)
            reserved_mem_bytes = torch.cuda.memory_reserved(device)
            
            total_gpu = total_gpu_mem_bytes / 1024**3
            allocated_gpu = allocated_mem_bytes / 1024**3
            reserved_gpu = reserved_mem_bytes / 1024**3
            available_gpu = total_gpu - reserved_gpu
            
            if available_gpu >= minimal_gpu:
                ratio = min(max_ratio, max(min_ratio, minimal_gpu / total_gpu))
            else:
                ratio = min(max_ratio, max(min_ratio, (available_gpu * 0.8) / total_gpu))
            
            return ratio
        except Exception as e:
            return 0.9
    
    def _initialize_5hz_lm_cuda(self, model_path: str) -> str:
        """Initialize 5Hz LM model"""
        try:
            from nanovllm import LLM, SamplingParams
            
            if not torch.cuda.is_available():
                return "❌ CUDA is not available. Please check your GPU setup."
            
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            torch.cuda.empty_cache()
            gpu_memory_utilization = self.get_gpu_memory_utilization(
                minimal_gpu=8, 
                min_ratio=0.2, 
                max_ratio=0.9
            )
            
            self.llm = LLM(
                model=model_path,
                enforce_eager=False,
                tensor_parallel_size=1,
                max_model_len=4096,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            self.llm_tokenizer = self.llm.tokenizer
            self.llm_initialized = True
            return f"✅ 5Hz LM initialized successfully\nModel: {model_path}\nDevice: {device_name}\nGPU Memory Utilization: {gpu_memory_utilization:.2f}"
        except Exception as e:
            self.llm_initialized = False
            error_msg = f"❌ Error initializing 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return error_msg
    
    def generate_with_5hz_lm(self, caption: str, lyrics: str, temperature: float = 0.6) -> Tuple[Dict[str, Any], str, str]:
        """Generate metadata and audio codes using 5Hz LM"""
        if not self.lm_initialized or self.llm is None:
            return {}, "", "❌ 5Hz LM not initialized. Please initialize it first."
        
        try:
            from nanovllm import SamplingParams
            
            prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
            
            formatted_prompt = self.lm_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"},
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            
            sampling_params = SamplingParams(max_tokens=3072, temperature=temperature)
            outputs = self.llm.generate([formatted_prompt], sampling_params)
            
            if isinstance(outputs, list) and len(outputs) > 0:
                if hasattr(outputs[0], 'outputs') and len(outputs[0].outputs) > 0:
                    output_text = outputs[0].outputs[0].text
                elif hasattr(outputs[0], 'text'):
                    output_text = outputs[0].text
                else:
                    output_text = str(outputs[0])
            else:
                output_text = str(outputs)
            
            metadata, audio_codes = self.parse_lm_output(output_text)
            codes_count = len(audio_codes.split('<|audio_code_')) - 1 if audio_codes else 0
            return metadata, audio_codes, f"✅ Generated successfully\nOutput length: {len(output_text)} chars\nCodes count: {codes_count}"
            
        except Exception as e:
            error_msg = f"❌ Error generating with 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return {}, "", error_msg
    
    def parse_lm_output(self, output_text: str) -> Tuple[Dict[str, Any], str]:
        """Parse LM output"""
        metadata = {}
        audio_codes = ""
        
        import re
        
        # Extract audio codes
        code_pattern = r'<\|audio_code_\d+\|>'
        code_matches = re.findall(code_pattern, output_text)
        if code_matches:
            audio_codes = "".join(code_matches)
        
        # Extract metadata
        reasoning_patterns = [
            r'<think>(.*?)</think>',
            r'<reasoning>(.*?)</reasoning>',
        ]
        
        reasoning_text = None
        for pattern in reasoning_patterns:
            match = re.search(pattern, output_text, re.DOTALL)
            if match:
                reasoning_text = match.group(1).strip()
                break
        
        if not reasoning_text:
            lines_before_codes = output_text.split('<|audio_code_')[0] if '<|audio_code_' in output_text else output_text
            reasoning_text = lines_before_codes.strip()
        
        # Parse metadata fields
        if reasoning_text:
            for line in reasoning_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('<'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()
                        
                        if key == 'bpm':
                            try:
                                metadata['bpm'] = int(value)
                            except:
                                metadata['bpm'] = value
                        elif key == 'duration':
                            try:
                                metadata['duration'] = int(value)
                            except:
                                metadata['duration'] = value
                        elif key in ['genres', 'keyscale', 'timesignature']:
                            metadata[key] = value
        
        return metadata, audio_codes
    
    def process_reference_audio(self, audio_file) -> Optional[torch.Tensor]:
        """Process reference audio"""
        if audio_file is None:
            return None
        
        try:
            # Load audio using soundfile
            audio_np, sr = sf.read(audio_file, dtype='float32')
            # Convert to torch: [samples, channels] or [samples] -> [channels, samples]
            if audio_np.ndim == 1:
                audio = torch.from_numpy(audio_np).unsqueeze(0)
            else:
                audio = torch.from_numpy(audio_np.T)
            
            if audio.shape[0] == 1:
                audio = torch.cat([audio, audio], dim=0)
            
            audio = audio[:2]
            
            # Resample if needed
            if sr != 48000:
                import torch.nn.functional as F
                # Simple resampling using interpolate
                ratio = 48000 / sr
                new_length = int(audio.shape[-1] * ratio)
                audio = F.interpolate(audio.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
            
            audio = torch.clamp(audio, -1.0, 1.0)
            
            target_frames = 30 * 48000
            if audio.shape[-1] > target_frames:
                start_frame = (audio.shape[-1] - target_frames) // 2
                audio = audio[:, start_frame:start_frame + target_frames]
            elif audio.shape[-1] < target_frames:
                audio = torch.nn.functional.pad(
                    audio, (0, target_frames - audio.shape[-1]), 'constant', 0
                )
            
            return audio
        except Exception as e:
            print(f"Error processing reference audio: {e}")
            return None
    
    def process_target_audio(self, audio_file) -> Optional[torch.Tensor]:
        """Process target audio"""
        if audio_file is None:
            return None
        
        try:
            # Load audio using soundfile
            audio_np, sr = sf.read(audio_file, dtype='float32')
            # Convert to torch: [samples, channels] or [samples] -> [channels, samples]
            if audio_np.ndim == 1:
                audio = torch.from_numpy(audio_np).unsqueeze(0)
            else:
                audio = torch.from_numpy(audio_np.T)
            
            if audio.shape[0] == 1:
                audio = torch.cat([audio, audio], dim=0)
            
            audio = audio[:2]
            
            # Resample if needed
            if sr != 48000:
                import torch.nn.functional as F
                ratio = 48000 / sr
                new_length = int(audio.shape[-1] * ratio)
                audio = F.interpolate(audio.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
            
            audio = torch.clamp(audio, -1.0, 1.0)
            
            return audio
        except Exception as e:
            print(f"Error processing target audio: {e}")
            return None
    
    def _parse_audio_code_string(self, code_str: str) -> List[int]:
        """Extract integer audio codes from prompt tokens like <|audio_code_123|>."""
        if not code_str:
            return []
        try:
            return [int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str)]
        except Exception:
            return []
    
    def _decode_audio_codes_to_latents(self, code_str: str) -> Optional[torch.Tensor]:
        """
        Convert serialized audio code string into 25Hz latents using model quantizer/detokenizer.
        """
        if not self.model or not hasattr(self.model, 'tokenizer') or not hasattr(self.model, 'detokenizer'):
            return None
        
        code_ids = self._parse_audio_code_string(code_str)
        if len(code_ids) == 0:
            return None
        
        quantizer = self.model.tokenizer.quantizer
        detokenizer = self.model.detokenizer
        
        num_quantizers = getattr(quantizer, "num_quantizers", 1)
        indices = torch.tensor(code_ids, device=self.device, dtype=torch.long).unsqueeze(0)  # [1, T_5Hz]
        
        # Expand to include quantizer dimension: [1, T_5Hz, num_quantizers]
        if indices.dim() == 2:
            indices = indices.unsqueeze(-1).expand(-1, -1, num_quantizers)
        
        # Get quantized representation from indices: [1, T_5Hz, dim]
        quantized = quantizer.get_output_from_indices(indices)
        if quantized.dtype != self.dtype:
            quantized = quantized.to(self.dtype)
        
        # Detokenize to 25Hz: [1, T_5Hz, dim] -> [1, T_25Hz, dim]
        lm_hints_25hz = detokenizer(quantized)
        return lm_hints_25hz
    
    def _create_default_meta(self) -> str:
        """Create default metadata string."""
        return (
            "- bpm: N/A\n"
            "- timesignature: N/A\n" 
            "- keyscale: N/A\n"
            "- duration: 30 seconds\n"
        )
    
    def _dict_to_meta_string(self, meta_dict: Dict[str, Any]) -> str:
        """Convert metadata dict to formatted string."""
        bpm = meta_dict.get('bpm', meta_dict.get('tempo', 'N/A'))
        timesignature = meta_dict.get('timesignature', meta_dict.get('time_signature', 'N/A'))
        keyscale = meta_dict.get('keyscale', meta_dict.get('key', meta_dict.get('scale', 'N/A')))
        duration = meta_dict.get('duration', meta_dict.get('length', 30))

        # Format duration
        if isinstance(duration, (int, float)):
            duration = f"{int(duration)} seconds"
        elif not isinstance(duration, str):
            duration = "30 seconds"
        
        return (
            f"- bpm: {bpm}\n"
            f"- timesignature: {timesignature}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {duration}\n"
        )
    
    def _parse_metas(self, metas: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """Parse and normalize metadata with fallbacks."""
        parsed_metas = []
        for meta in metas:
            if meta is None:
                parsed_meta = self._create_default_meta()
            elif isinstance(meta, str):
                parsed_meta = meta
            elif isinstance(meta, dict):
                parsed_meta = self._dict_to_meta_string(meta)
            else:
                parsed_meta = self._create_default_meta()
            parsed_metas.append(parsed_meta)
        return parsed_metas
    
    def _get_text_hidden_states(self, text_prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get text hidden states from text encoder."""
        if self.text_tokenizer is None or self.text_encoder is None:
            raise ValueError("Text encoder not initialized")
        
        # Tokenize
        text_inputs = self.text_tokenizer(
            text_prompt,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        text_attention_mask = text_inputs.attention_mask.to(self.device).bool()
        
        # Encode
        with torch.no_grad():
            text_outputs = self.text_encoder(text_input_ids)
            if hasattr(text_outputs, 'last_hidden_state'):
                text_hidden_states = text_outputs.last_hidden_state
            elif isinstance(text_outputs, tuple):
                text_hidden_states = text_outputs[0]
            else:
                text_hidden_states = text_outputs
        
        text_hidden_states = text_hidden_states.to(self.dtype)
        
        return text_hidden_states, text_attention_mask
    
    def extract_caption_from_sft_format(self, caption: str) -> str:
        """Extract caption from SFT format if needed."""
        # Simple extraction - can be enhanced if needed
        if caption and isinstance(caption, str):
            return caption.strip()
        return caption if caption else ""
    
    def generate_music(
        self,
        captions: str,
        lyrics: str,
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        vocal_language: str = "en",
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        use_random_seed: bool = True,
        seed: Optional[Union[str, float, int]] = -1,
        reference_audio=None,
        audio_duration: Optional[float] = None,
        batch_size: Optional[int] = None,
        src_audio=None,
        audio_code_string: str = "",
        repainting_start: float = 0.0,
        repainting_end: Optional[float] = None,
        instruction: str = "Fill the audio semantic mask based on the given conditions:",
        audio_cover_strength: float = 1.0,
        task_type: str = "text2music",
        use_adg: bool = False,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        audio_format: str = "mp3",
        lm_temperature: float = 0.6,
        progress=None
    ) -> Tuple[Optional[str], Optional[str], List[str], str, str, str, str, str, Optional[Any], str, str, Optional[Any]]:
        """
        Main interface for music generation
        
        Returns:
            (first_audio, second_audio, all_audio_paths, generation_info, status_message, 
             seed_value_for_ui, align_score_1, align_text_1, align_plot_1, 
             align_score_2, align_text_2, align_plot_2)
        """
        if self.model is None or self.vae is None or self.text_tokenizer is None or self.text_encoder is None:
            return None, None, [], "", "❌ Model not fully initialized. Please initialize all components first.", "-1", "", "", None, "", "", None
        
        try:
            print("[generate_music] Starting generation...")
            if progress:
                progress(0.05, desc="Preparing inputs...")
            print("[generate_music] Preparing inputs...")
            
            # Determine actual batch size
            actual_batch_size = batch_size if batch_size is not None else self.batch_size
            actual_batch_size = max(1, min(actual_batch_size, 8))  # Limit to 8 for memory safety
            
            # Process seeds
            if use_random_seed:
                seed_list = [random.randint(0, 2**32 - 1) for _ in range(actual_batch_size)]
            else:
                # Parse seed input
                if isinstance(seed, str):
                    seed_parts = [s.strip() for s in seed.split(",")]
                    seed_list = [int(float(s)) if s != "-1" and s else random.randint(0, 2**32 - 1) for s in seed_parts[:actual_batch_size]]
                elif isinstance(seed, (int, float)) and seed >= 0:
                    seed_list = [int(seed)] * actual_batch_size
                else:
                    seed_list = [random.randint(0, 2**32 - 1) for _ in range(actual_batch_size)]
                
                # Pad if needed
                while len(seed_list) < actual_batch_size:
                    seed_list.append(random.randint(0, 2**32 - 1))
            
            seed_value_for_ui = ", ".join(str(s) for s in seed_list)
            
            # Process audio inputs
            processed_ref_audio = self.process_reference_audio(reference_audio) if reference_audio else None
            processed_src_audio = self.process_target_audio(src_audio) if src_audio else None
            
            # Extract caption
            pure_caption = self.extract_caption_from_sft_format(captions)
            
            # Determine task type and update instruction if needed
            if task_type == "text2music" and audio_code_string and str(audio_code_string).strip():
                task_type = "cover"
                instruction = "Generate audio semantic tokens based on the given conditions:"
            
            # Build metadata
            metadata_dict = {
                "bpm": bpm if bpm else "N/A",
                "keyscale": key_scale if key_scale else "N/A",
                "timesignature": time_signature if time_signature else "N/A",
            }
            
            # Calculate duration
            if processed_src_audio is not None:
                calculated_duration = processed_src_audio.shape[-1] / self.sample_rate
            elif audio_duration is not None and audio_duration > 0:
                calculated_duration = audio_duration
            else:
                calculated_duration = 30.0  # Default 30 seconds
            
            metadata_dict["duration"] = f"{int(calculated_duration)} seconds"
            
            if progress:
                progress(0.1, desc="Processing audio inputs...")
            print("[generate_music] Processing audio inputs...")
            
            # Prepare batch data
            captions_batch = [pure_caption] * actual_batch_size
            lyrics_batch = [lyrics] * actual_batch_size
            vocal_languages_batch = [vocal_language] * actual_batch_size
            instructions_batch = [instruction] * actual_batch_size
            metas_batch = [metadata_dict.copy()] * actual_batch_size
            audio_code_hints_batch = [audio_code_string if audio_code_string else None] * actual_batch_size
            
            # Process reference audios
            if processed_ref_audio is not None:
                refer_audios = [[processed_ref_audio] for _ in range(actual_batch_size)]
            else:
                # Create silence as fallback
                silence_frames = 30 * self.sample_rate
                silence = torch.zeros(2, silence_frames)
                refer_audios = [[silence] for _ in range(actual_batch_size)]
            
            # Process target wavs (src_audio)
            if processed_src_audio is not None:
                target_wavs_list = [processed_src_audio.clone() for _ in range(actual_batch_size)]
            else:
                # Create silence based on duration
                target_frames = int(calculated_duration * self.sample_rate)
                silence = torch.zeros(2, target_frames)
                target_wavs_list = [silence for _ in range(actual_batch_size)]
            
            # Pad target_wavs to consistent length
            max_target_frames = max(wav.shape[-1] for wav in target_wavs_list)
            target_wavs = torch.stack([
                torch.nn.functional.pad(wav, (0, max_target_frames - wav.shape[-1]), 'constant', 0)
                for wav in target_wavs_list
            ])
            
            if progress:
                progress(0.2, desc="Encoding audio to latents...")
            print("[generate_music] Encoding audio to latents...")
            
            # Encode target_wavs to latents using VAE
            target_latents_list = []
            latent_lengths = []
            
            with torch.no_grad():
                for i in range(actual_batch_size):
                    # Check if audio codes are provided
                    code_hint = audio_code_hints_batch[i]
                    if code_hint:
                        decoded_latents = self._decode_audio_codes_to_latents(code_hint)
                        if decoded_latents is not None:
                            decoded_latents = decoded_latents.squeeze(0)  # Remove batch dim
                            target_latents_list.append(decoded_latents)
                            latent_lengths.append(decoded_latents.shape[0])
                            continue
                    
                    # If no src_audio provided, use silence_latent directly (skip VAE)
                    if processed_src_audio is None:
                        # Calculate required latent length based on duration
                        # VAE downsample ratio is 1920 (2*4*4*6*10), so latent rate is 48000/1920 = 25Hz
                        latent_length = int(calculated_duration * 25)  # 25Hz latent rate
                        latent_length = max(128, latent_length)  # Minimum 128
                        
                        # Tile silence_latent to required length
                        if self.silence_latent.shape[0] >= latent_length:
                            target_latent = self.silence_latent[:latent_length].to(self.device).to(self.dtype)
                        else:
                            repeat_times = (latent_length // self.silence_latent.shape[0]) + 1
                            target_latent = self.silence_latent.repeat(repeat_times, 1)[:latent_length].to(self.device).to(self.dtype)
                        target_latents_list.append(target_latent)
                        latent_lengths.append(target_latent.shape[0])
                        continue
                    
                    # Encode from audio using VAE
                    current_wav = target_wavs[i].unsqueeze(0).to(self.device).to(self.dtype)
                    target_latent = self.vae.encode(current_wav)
                    target_latent = target_latent.squeeze(0).transpose(0, 1)  # [latent_length, latent_dim]
                    target_latents_list.append(target_latent)
                    latent_lengths.append(target_latent.shape[0])
                
                # Pad latents to same length
                max_latent_length = max(latent_lengths)
                max_latent_length = max(128, max_latent_length)  # Minimum 128
                
                padded_latents = []
                for i, latent in enumerate(target_latents_list):
                    if latent.shape[0] < max_latent_length:
                        pad_length = max_latent_length - latent.shape[0]
                        # Tile silence_latent to pad_length (silence_latent is [L, C])
                        if self.silence_latent.shape[0] >= pad_length:
                            pad_latent = self.silence_latent[:pad_length]
                        else:
                            repeat_times = (pad_length // self.silence_latent.shape[0]) + 1
                            pad_latent = self.silence_latent.repeat(repeat_times, 1)[:pad_length]
                        latent = torch.cat([latent, pad_latent.to(self.device).to(self.dtype)], dim=0)
                    padded_latents.append(latent)
                
                target_latents = torch.stack(padded_latents).to(self.device).to(self.dtype)
                latent_masks = torch.stack([
                    torch.cat([
                        torch.ones(l, dtype=torch.long, device=self.device),
                        torch.zeros(max_latent_length - l, dtype=torch.long, device=self.device)
                    ])
                    for l in latent_lengths
                ])
            
            if progress:
                progress(0.3, desc="Preparing conditions...")
            print("[generate_music] Preparing conditions...")
            
            # Determine task type and create chunk masks
            is_covers = []
            chunk_masks = []
            repainting_ranges = {}
            
            for i in range(actual_batch_size):
                has_code_hint = audio_code_hints_batch[i] is not None
                has_repainting = (repainting_end is not None and repainting_end > repainting_start)
                
                if has_repainting:
                    # Repainting mode
                    start_sec = max(0, repainting_start)
                    end_sec = repainting_end if repainting_end is not None else calculated_duration
                    
                    start_latent = int(start_sec * self.sample_rate // 1920)
                    end_latent = int(end_sec * self.sample_rate // 1920)
                    start_latent = max(0, min(start_latent, max_latent_length - 1))
                    end_latent = max(start_latent + 1, min(end_latent, max_latent_length))
                    
                    mask = torch.zeros(max_latent_length, dtype=torch.bool, device=self.device)
                    mask[start_latent:end_latent] = True
                    chunk_masks.append(mask)
                    repainting_ranges[i] = (start_latent, end_latent)
                    is_covers.append(False)
                else:
                    # Full generation or cover
                    chunk_masks.append(torch.ones(max_latent_length, dtype=torch.bool, device=self.device))
                    # Check if cover task
                    instruction_lower = instructions_batch[i].lower()
                    is_cover = ("generate audio semantic tokens" in instruction_lower and 
                               "based on the given conditions" in instruction_lower) or has_code_hint
                    is_covers.append(is_cover)
            
            chunk_masks = torch.stack(chunk_masks).unsqueeze(-1).expand(-1, -1, 64)  # [batch, length, 64]
            is_covers = torch.tensor(is_covers, dtype=torch.bool, device=self.device)
            
            # Create src_latents
            # Tile silence_latent to max_latent_length (silence_latent is now [L, C])
            if self.silence_latent.shape[0] >= max_latent_length:
                silence_latent_tiled = self.silence_latent[:max_latent_length].to(self.device).to(self.dtype)
            else:
                repeat_times = (max_latent_length // self.silence_latent.shape[0]) + 1
                silence_latent_tiled = self.silence_latent.repeat(repeat_times, 1)[:max_latent_length].to(self.device).to(self.dtype)
            src_latents_list = []
            
            for i in range(actual_batch_size):
                has_target_audio = (target_wavs[i].abs().sum() > 1e-6) or (audio_code_hints_batch[i] is not None)
                
                if has_target_audio:
                    if i in repainting_ranges:
                        # Repaint: replace inpainting region with silence
                        src_latent = target_latents[i].clone()
                        start_latent, end_latent = repainting_ranges[i]
                        src_latent[start_latent:end_latent] = silence_latent_tiled[start_latent:end_latent]
                        src_latents_list.append(src_latent)
                    else:
                        # Cover/extract/complete/lego: use target_latents
                        src_latents_list.append(target_latents[i].clone())
                else:
                    # Text2music: use silence
                    src_latents_list.append(silence_latent_tiled.clone())
            
            src_latents = torch.stack(src_latents_list)  # [batch, length, channels]
            
            if progress:
                progress(0.4, desc="Tokenizing text inputs...")
            print("[generate_music] Tokenizing text inputs...")
            
            # Prepare text and lyric hidden states
            SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""
            
            text_hidden_states_list = []
            text_attention_masks_list = []
            lyric_hidden_states_list = []
            lyric_attention_masks_list = []
            
            with torch.no_grad():
                for i in range(actual_batch_size):
                    # Format text prompt
                    inst = instructions_batch[i]
                    if not inst.endswith(":"):
                        inst = inst + ":"
                    
                    meta_str = self._dict_to_meta_string(metas_batch[i])
                    text_prompt = SFT_GEN_PROMPT.format(inst, captions_batch[i], meta_str)
                    
                    # Tokenize and encode text
                    text_hidden, text_mask = self._get_text_hidden_states(text_prompt)
                    text_hidden_states_list.append(text_hidden.squeeze(0))
                    text_attention_masks_list.append(text_mask.squeeze(0))
                    
                    # Format and tokenize lyrics
                    lyrics_text = f"# Languages\n{vocal_languages_batch[i]}\n\n# Lyric\n{lyrics_batch[i]}<|endoftext|>"
                    lyric_hidden, lyric_mask = self._get_text_hidden_states(lyrics_text)
                    lyric_hidden_states_list.append(lyric_hidden.squeeze(0))
                    lyric_attention_masks_list.append(lyric_mask.squeeze(0))
                
                # Pad sequences
                max_text_length = max(h.shape[0] for h in text_hidden_states_list)
                max_lyric_length = max(h.shape[0] for h in lyric_hidden_states_list)
                
                text_hidden_states = torch.stack([
                    torch.nn.functional.pad(h, (0, 0, 0, max_text_length - h.shape[0]), 'constant', 0)
                    for h in text_hidden_states_list
                ]).to(self.device).to(self.dtype)
                
                text_attention_mask = torch.stack([
                    torch.nn.functional.pad(m, (0, max_text_length - m.shape[0]), 'constant', 0)
                    for m in text_attention_masks_list
                ]).to(self.device)
                
                lyric_hidden_states = torch.stack([
                    torch.nn.functional.pad(h, (0, 0, 0, max_lyric_length - h.shape[0]), 'constant', 0)
                    for h in lyric_hidden_states_list
                ]).to(self.device).to(self.dtype)
                
                lyric_attention_mask = torch.stack([
                    torch.nn.functional.pad(m, (0, max_lyric_length - m.shape[0]), 'constant', 0)
                    for m in lyric_attention_masks_list
                ]).to(self.device)
            
            if progress:
                progress(0.5, desc="Processing reference audio...")
            print("[generate_music] Processing reference audio...")
            
            # Process reference audio for timbre
            # Model expects: refer_audio_acoustic_hidden_states_packed [N, timbre_fix_frame, audio_acoustic_hidden_dim]
            #                refer_audio_order_mask [N] indicating batch assignment
            timbre_fix_frame = getattr(self.config, 'timbre_fix_frame', 750)
            refer_audio_acoustic_hidden_states_packed_list = []
            refer_audio_order_mask_list = []
            
            with torch.no_grad():
                for i, ref_audio_list in enumerate(refer_audios):
                    if ref_audio_list and len(ref_audio_list) > 0 and ref_audio_list[0].abs().sum() > 1e-6:
                        # Encode reference audio: [channels, samples] -> [1, latent_dim, T] -> [T, latent_dim]
                        ref_audio = ref_audio_list[0].unsqueeze(0).to(self.device).to(self.dtype)
                        ref_latent = self.vae.encode(ref_audio).latent_dist.sample()  # [1, latent_dim, T]
                        ref_latent = ref_latent.squeeze(0).transpose(0, 1)  # [T, latent_dim]
                        # Ensure dimension matches audio_acoustic_hidden_dim (64)
                        if ref_latent.shape[-1] != self.config.audio_acoustic_hidden_dim:
                            ref_latent = ref_latent[:, :self.config.audio_acoustic_hidden_dim]
                        # Pad or truncate to timbre_fix_frame
                        if ref_latent.shape[0] < timbre_fix_frame:
                            pad_length = timbre_fix_frame - ref_latent.shape[0]
                            padding = torch.zeros(pad_length, ref_latent.shape[1], device=self.device, dtype=self.dtype)
                            ref_latent = torch.cat([ref_latent, padding], dim=0)
                        else:
                            ref_latent = ref_latent[:timbre_fix_frame]
                        refer_audio_acoustic_hidden_states_packed_list.append(ref_latent)
                        refer_audio_order_mask_list.append(i)
                    else:
                        # Use silence_latent directly instead of running VAE
                        if self.silence_latent.shape[0] >= timbre_fix_frame:
                            silence_ref = self.silence_latent[:timbre_fix_frame, :self.config.audio_acoustic_hidden_dim]
                        else:
                            repeat_times = (timbre_fix_frame // self.silence_latent.shape[0]) + 1
                            silence_ref = self.silence_latent.repeat(repeat_times, 1)[:timbre_fix_frame, :self.config.audio_acoustic_hidden_dim]
                        refer_audio_acoustic_hidden_states_packed_list.append(silence_ref.to(self.device).to(self.dtype))
                        refer_audio_order_mask_list.append(i)
                
                # Stack all reference audios: [N, timbre_fix_frame, audio_acoustic_hidden_dim]
                refer_audio_acoustic_hidden_states_packed = torch.stack(refer_audio_acoustic_hidden_states_packed_list, dim=0).to(self.device).to(self.dtype)
                # Order mask: [N] indicating which batch item each reference belongs to
                refer_audio_order_mask = torch.tensor(refer_audio_order_mask_list, dtype=torch.long, device=self.device)
            
            if progress:
                progress(0.6, desc="Generating audio...")
            print("[generate_music] Calling model.generate_audio()...")
            print(f"  - text_hidden_states: {text_hidden_states.shape}, dtype={text_hidden_states.dtype}")
            print(f"  - text_attention_mask: {text_attention_mask.shape}, dtype={text_attention_mask.dtype}")
            print(f"  - lyric_hidden_states: {lyric_hidden_states.shape}, dtype={lyric_hidden_states.dtype}")
            print(f"  - lyric_attention_mask: {lyric_attention_mask.shape}, dtype={lyric_attention_mask.dtype}")
            print(f"  - refer_audio_acoustic_hidden_states_packed: {refer_audio_acoustic_hidden_states_packed.shape}, dtype={refer_audio_acoustic_hidden_states_packed.dtype}")
            print(f"  - refer_audio_order_mask: {refer_audio_order_mask.shape}, dtype={refer_audio_order_mask.dtype}")
            print(f"  - src_latents: {src_latents.shape}, dtype={src_latents.dtype}")
            print(f"  - chunk_masks: {chunk_masks.shape}, dtype={chunk_masks.dtype}")
            print(f"  - is_covers: {is_covers.shape}, dtype={is_covers.dtype}")
            print(f"  - silence_latent: {self.silence_latent.unsqueeze(0).shape}")
            print(f"  - seed: {seed_list[0] if len(seed_list) > 0 else None}")
            print(f"  - fix_nfe: {inference_steps}")
            
            # Call model to generate
            with torch.no_grad():
                outputs = self.model.generate_audio(
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                    refer_audio_order_mask=refer_audio_order_mask,
                    src_latents=src_latents,
                    chunk_masks=chunk_masks,
                    is_covers=is_covers,
                    silence_latent=self.silence_latent.unsqueeze(0),  # [1, L, C]
                    seed=seed_list[0] if len(seed_list) > 0 else None,
                    fix_nfe=inference_steps,
                    infer_method="ode",
                    use_cache=True,
                )
            
            print("[generate_music] Model generation completed. Decoding latents...")
            pred_latents = outputs["target_latents"]  # [batch, latent_length, latent_dim]
            time_costs = outputs["time_costs"]
            print(f"  - pred_latents: {pred_latents.shape}, dtype={pred_latents.dtype} {pred_latents.min()=}, {pred_latents.max()=}, {pred_latents.mean()=} {pred_latents.std()=}")
            print(f"  - time_costs: {time_costs}")
            if progress:
                progress(0.8, desc="Decoding audio...")
            print("[generate_music] Decoding latents with VAE...")
            
            # Decode latents to audio
            start_time = time.time()
            with torch.no_grad():
                # Transpose for VAE decode: [batch, latent_length, latent_dim] -> [batch, latent_dim, latent_length]
                pred_latents_for_decode = pred_latents.transpose(1, 2)
                pred_wavs = self.vae.decode(pred_latents_for_decode).sample  # [batch, channels, samples]
            end_time = time.time()
            time_costs["vae_decode_time_cost"] = end_time - start_time
            time_costs["total_time_cost"] = time_costs["total_time_cost"] + time_costs["vae_decode_time_cost"]
            
            print("[generate_music] VAE decode completed. Saving audio files...")
            if progress:
                progress(0.9, desc="Saving audio files...")
            
            # Save audio files using soundfile (supports wav, flac, mp3 via format param)
            audio_format_lower = audio_format.lower() if audio_format else "wav"
            if audio_format_lower not in ["wav", "flac", "mp3"]:
                audio_format_lower = "wav"
            
            saved_files = []
            for i in range(actual_batch_size):
                audio_file = os.path.join(self.temp_dir, f"generated_{i}_{seed_list[i]}.{audio_format_lower}")
                # Convert to numpy: [channels, samples] -> [samples, channels]
                audio_np = pred_wavs[i].cpu().float().numpy().T
                sf.write(audio_file, audio_np, self.sample_rate)
                saved_files.append(audio_file)
            
            # Prepare return values
            first_audio = saved_files[0] if len(saved_files) > 0 else None
            second_audio = saved_files[1] if len(saved_files) > 1 else None
            
            # Format time costs if available
            time_costs_str = ""
            if time_costs:
                if isinstance(time_costs, dict):
                    time_costs_str = "\n\n**⏱️ Time Costs:**\n"
                    for key, value in time_costs.items():
                        # Format key: encoder_time_cost -> Encoder
                        formatted_key = key.replace("_time_cost", "").replace("_", " ").title()
                        time_costs_str += f"  - {formatted_key}: {value:.2f}s\n"
                elif isinstance(time_costs, (int, float)):
                    time_costs_str = f"\n\n**⏱️ Time Cost:** {time_costs:.2f}s"
            
            generation_info = f"""**🎵 Generation Complete**

**Seeds:** {seed_value_for_ui}
**Duration:** {calculated_duration:.1f}s
**Steps:** {inference_steps}
**Files:** {len(saved_files)} audio(s){time_costs_str}"""
            status_message = f"✅ Generation completed successfully!"
            print(f"[generate_music] Done! Generated {len(saved_files)} audio files.")
            
            # Alignment scores and plots (placeholder for now)
            align_score_1 = ""
            align_text_1 = ""
            align_plot_1 = None
            align_score_2 = ""
            align_text_2 = ""
            align_plot_2 = None
            
            return (
                first_audio,
                second_audio,
                saved_files,
                generation_info,
                status_message,
                seed_value_for_ui,
                align_score_1,
                align_text_1,
                align_plot_1,
                align_score_2,
                align_text_2,
                align_plot_2,
            )
            
        except Exception as e:
            error_msg = f"❌ Error generating music: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return None, None, [], "", error_msg, "-1", "", "", None, "", "", None

