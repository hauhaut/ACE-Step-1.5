"""
5Hz LM (Language Model) Handler
Handles all LM-related operations including initialization and generation
"""
import os
import traceback
import time
from typing import Optional, Dict, Any, Tuple, List
from contextlib import contextmanager

import torch
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    LogitsProcessor,
)
from .constrained_logits_processor import MetadataConstrainedLogitsProcessor


class LLMHandler:
    """5Hz LM Handler for audio code generation"""

    STOP_REASONING_TAG = "</think>"
    
    def __init__(self):
        """Initialize LLMHandler with default values"""
        self.llm = None
        self.llm_tokenizer = None
        self.llm_initialized = False
        self.llm_backend = None
        self.max_model_len = 4096
        self.device = "cpu"
        self.dtype = torch.float32
        self.offload_to_cpu = False
        
        # Shared constrained decoding processor (initialized once when LLM is loaded)
        self.constrained_processor: Optional[MetadataConstrainedLogitsProcessor] = None
    
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
    
    def get_gpu_memory_utilization(self, minimal_gpu: float = 8, min_ratio: float = 0.2, max_ratio: float = 0.9) -> Tuple[float, bool]:
        """Get GPU memory utilization ratio"""
        try:
            device = torch.device("cuda:0")
            total_gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
            allocated_mem_bytes = torch.cuda.memory_allocated(device)
            reserved_mem_bytes = torch.cuda.memory_reserved(device)
            
            total_gpu = total_gpu_mem_bytes / 1024**3
            low_gpu_memory_mode = False
            if total_gpu < minimal_gpu:
                minimal_gpu = 0.5 * total_gpu
                low_gpu_memory_mode = True
            allocated_gpu = allocated_mem_bytes / 1024**3
            reserved_gpu = reserved_mem_bytes / 1024**3
            available_gpu = total_gpu - reserved_gpu
            
            if available_gpu >= minimal_gpu:
                ratio = min(max_ratio, max(min_ratio, minimal_gpu / total_gpu))
            else:
                ratio = min(max_ratio, max(min_ratio, (available_gpu * 0.8) / total_gpu))
            
            return ratio, low_gpu_memory_mode
        except Exception as e:
            return 0.9, False
    
    def initialize(
        self,
        checkpoint_dir: str,
        lm_model_path: str,
        backend: str = "vllm",
        device: str = "auto",
        offload_to_cpu: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[str, bool]:
        """
        Initialize 5Hz LM model
        
        Args:
            checkpoint_dir: Checkpoint directory path
            lm_model_path: LM model path (relative to checkpoint_dir)
            backend: Backend type ("vllm" or "pt")
            device: Device type ("auto", "cuda", or "cpu")
            offload_to_cpu: Whether to offload to CPU
            dtype: Data type (if None, auto-detect based on device)
        
        Returns:
            (status_message, success)
        """
        try:
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.device = device
            self.offload_to_cpu = offload_to_cpu
            # Set dtype based on device: bfloat16 for cuda, float32 for cpu
            if dtype is None:
                self.dtype = torch.bfloat16 if device in ["cuda", "xpu"] else torch.float32
            else:
                self.dtype = dtype
            
            full_lm_model_path = os.path.join(checkpoint_dir, lm_model_path)
            if not os.path.exists(full_lm_model_path):
                return f"❌ 5Hz LM model not found at {full_lm_model_path}", False
            
            logger.info("loading 5Hz LM tokenizer...")
            start_time = time.time()
            llm_tokenizer = AutoTokenizer.from_pretrained(full_lm_model_path, use_fast=True)
            logger.info(f"5Hz LM tokenizer loaded successfully in {time.time() - start_time:.2f} seconds")
            self.llm_tokenizer = llm_tokenizer
            
            # Initialize shared constrained decoding processor (one-time initialization)
            logger.info("Initializing constrained decoding processor...")
            processor_start = time.time()
            self.constrained_processor = MetadataConstrainedLogitsProcessor(
                tokenizer=self.llm_tokenizer,
                enabled=True,
                debug=False,
            )
            logger.info(f"Constrained processor initialized in {time.time() - processor_start:.2f} seconds")
            
            # Initialize based on user-selected backend
            if backend == "vllm":
                # Try to initialize with vllm
                status_msg = self._initialize_5hz_lm_vllm(full_lm_model_path)
                logger.info(f"5Hz LM status message: {status_msg}")
                # Check if initialization failed (status_msg starts with ❌)
                if status_msg.startswith("❌"):
                    # vllm initialization failed, fallback to PyTorch
                    if not self.llm_initialized:
                        logger.warning("vllm initialization failed, falling back to PyTorch backend")
                        try:
                            self.llm = AutoModelForCausalLM.from_pretrained(full_lm_model_path, trust_remote_code=True)
                            if not self.offload_to_cpu:
                                self.llm = self.llm.to(device).to(self.dtype)
                            else:
                                self.llm = self.llm.to("cpu").to(self.dtype)
                            self.llm.eval()
                            self.llm_backend = "pt"
                            self.llm_initialized = True
                            logger.info("5Hz LM initialized successfully using PyTorch backend (fallback)")
                            status_msg = f"✅ 5Hz LM initialized successfully (PyTorch fallback)\nModel: {full_lm_model_path}\nBackend: PyTorch"
                        except Exception as e:
                            return f"❌ Error initializing 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", False
                # If vllm initialization succeeded, self.llm_initialized should already be True
            else:
                # Use PyTorch backend (pt)
                try:
                    self.llm = AutoModelForCausalLM.from_pretrained(full_lm_model_path, trust_remote_code=True)
                    if not self.offload_to_cpu:
                        self.llm = self.llm.to(device).to(self.dtype)
                    else:
                        self.llm = self.llm.to("cpu").to(self.dtype)
                    self.llm.eval()
                    self.llm_backend = "pt"
                    self.llm_initialized = True
                    logger.info(f"5Hz LM initialized successfully using PyTorch backend on {device}")
                    status_msg = f"✅ 5Hz LM initialized successfully\nModel: {full_lm_model_path}\nBackend: PyTorch\nDevice: {device}"
                except Exception as e:
                    return f"❌ Error initializing 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", False
            
            return status_msg, True
            
        except Exception as e:
            error_msg = f"❌ Error initializing 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return error_msg, False
    
    def _initialize_5hz_lm_vllm(self, model_path: str) -> str:
        """Initialize 5Hz LM model using vllm backend"""
        if not torch.cuda.is_available():
            self.llm_initialized = False
            logger.error("CUDA is not available. Please check your GPU setup.")
            return "❌ CUDA is not available. Please check your GPU setup."
        try:
            from nanovllm import LLM, SamplingParams
        except ImportError:
            self.llm_initialized = False
            logger.error("nano-vllm is not installed. Please install it using 'cd acestep/third_parts/nano-vllm && pip install .")
            return "❌ nano-vllm is not installed. Please install it using 'cd acestep/third_parts/nano-vllm && pip install ."
        
        try:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            torch.cuda.empty_cache()
            gpu_memory_utilization, low_gpu_memory_mode = self.get_gpu_memory_utilization(
                minimal_gpu=8, 
                min_ratio=0.2, 
                max_ratio=0.9
            )
            if low_gpu_memory_mode:
                self.max_model_len = 2048
            else:
                self.max_model_len = 4096
            
            logger.info(f"Initializing 5Hz LM with model: {model_path}, enforce_eager: False, tensor_parallel_size: 1, max_model_len: {self.max_model_len}, gpu_memory_utilization: {gpu_memory_utilization}")
            start_time = time.time()
            self.llm = LLM(
                model=model_path,
                enforce_eager=False,
                tensor_parallel_size=1,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                tokenizer=self.llm_tokenizer,
            )
            logger.info(f"5Hz LM initialized successfully in {time.time() - start_time:.2f} seconds")
            self.llm_initialized = True
            self.llm_backend = "vllm"
            return f"✅ 5Hz LM initialized successfully\nModel: {model_path}\nDevice: {device_name}\nGPU Memory Utilization: {gpu_memory_utilization:.2f}"
        except Exception as e:
            self.llm_initialized = False
            error_msg = f"❌ Error initializing 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return error_msg
    
    def generate_with_5hz_lm_vllm(
        self, 
        caption: str, 
        lyrics: str, 
        temperature: float = 0.6, 
        cfg_scale: float = 1.0, 
        negative_prompt: str = "NO USER INPUT",
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        use_constrained_decoding: bool = True,
        constrained_decoding_debug: bool = False,
        metadata_temperature: Optional[float] = 0.85,
        codes_temperature: Optional[float] = None,
        target_duration: Optional[float] = None,
        user_metadata: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        """Generate metadata and audio codes using 5Hz LM with vllm backend
        
        Args:
            caption: Text caption for music generation
            lyrics: Lyrics for music generation
            temperature: Base sampling temperature (used if phase-specific temps not set)
            cfg_scale: CFG scale (>1.0 enables CFG)
            negative_prompt: Negative prompt for CFG
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty
            use_constrained_decoding: Whether to use FSM-based constrained decoding
            constrained_decoding_debug: Whether to print debug info for constrained decoding
            metadata_temperature: Temperature for metadata generation (lower = more accurate)
                                  If None, uses base temperature
            codes_temperature: Temperature for audio codes generation (higher = more diverse)
                               If None, uses base temperature
            target_duration: Target duration in seconds for codes generation constraint.
                            5 codes = 1 second. If specified, blocks EOS until target reached.
        """
        try:
            from nanovllm import SamplingParams
            
            formatted_prompt = self.build_formatted_prompt(caption, lyrics)
            logger.debug(f"[debug] formatted_prompt: {formatted_prompt}")
            
            # Determine effective temperature for sampler
            # If using phase-specific temperatures, set sampler temp to 1.0 (processor handles it)
            use_phase_temperatures = metadata_temperature is not None or codes_temperature is not None
            effective_sampler_temp = 1.0 if use_phase_temperatures else temperature
            
            # Use shared constrained decoding processor if enabled
            constrained_processor = None
            update_state_fn = None
            if use_constrained_decoding or use_phase_temperatures:
                # Use shared processor, just update caption and settings
                self.constrained_processor.enabled = use_constrained_decoding
                self.constrained_processor.debug = constrained_decoding_debug
                self.constrained_processor.metadata_temperature = metadata_temperature if use_phase_temperatures else None
                self.constrained_processor.codes_temperature = codes_temperature if use_phase_temperatures else None
                self.constrained_processor.update_caption(caption)
                self.constrained_processor.set_target_duration(target_duration)
                # Always call set_user_metadata to ensure previous settings are cleared if None
                self.constrained_processor.set_user_metadata(user_metadata)
                
                constrained_processor = self.constrained_processor
                update_state_fn = constrained_processor.update_state
            
            sampling_params = SamplingParams(
                max_tokens=self.max_model_len-64, 
                temperature=effective_sampler_temp, 
                cfg_scale=cfg_scale,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                logits_processor=constrained_processor,
                logits_processor_update_state=update_state_fn,
            )
            # Use CFG if cfg_scale > 1.0
            if cfg_scale > 1.0:
                # Build unconditional prompt (user input replaced with "NO USER INPUT")
                formatted_unconditional_prompt = self.build_formatted_prompt(negative_prompt, is_negative_prompt=True)
                outputs = self.llm.generate(
                    [formatted_prompt], 
                    sampling_params,
                    unconditional_prompts=[formatted_unconditional_prompt]
                )
            else:
                outputs = self.llm.generate([formatted_prompt], sampling_params)
            # Extract text from output - handle different output formats
            if isinstance(outputs, list) and len(outputs) > 0:
                if hasattr(outputs[0], 'outputs') and len(outputs[0].outputs) > 0:
                    output_text = outputs[0].outputs[0].text
                elif hasattr(outputs[0], 'text'):
                    output_text = outputs[0].text
                elif isinstance(outputs[0], dict) and 'text' in outputs[0]:
                    output_text = outputs[0]['text']
                else:
                    output_text = str(outputs[0])
            else:
                output_text = str(outputs)
            metadata, audio_codes = self.parse_lm_output(output_text)
            print(f"[debug]output_text: {output_text}")
            codes_count = len(audio_codes.split('<|audio_code_')) - 1 if audio_codes else 0
            return metadata, audio_codes, f"✅ Generated successfully\nOutput length: {len(output_text)} chars\nCodes count: {codes_count}"
            
        except Exception as e:
            error_msg = f"❌ Error generating with 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return {}, "", error_msg
    
    def _run_vllm_from_formatted(
        self,
        formatted_prompt: str,
        temperature: float,
        cfg_scale: float,
        negative_prompt: str,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        use_constrained_decoding: bool = True,
        constrained_decoding_debug: bool = False,
        metadata_temperature: Optional[float] = 0.85,
        codes_temperature: Optional[float] = None,
        target_duration: Optional[float] = None,
        user_metadata: Optional[Dict[str, Optional[str]]] = None,
    ) -> str:
        """Shared vllm path: accept prebuilt formatted prompt and return text."""
        from nanovllm import SamplingParams

        # Determine effective temperature for sampler
        use_phase_temperatures = metadata_temperature is not None or codes_temperature is not None
        effective_sampler_temp = 1.0 if use_phase_temperatures else temperature

        # Use shared constrained processor if enabled
        constrained_processor = None
        if use_constrained_decoding or use_phase_temperatures:
            # Use shared processor, just update caption and settings
            self.constrained_processor.enabled = use_constrained_decoding
            self.constrained_processor.debug = constrained_decoding_debug
            self.constrained_processor.metadata_temperature = metadata_temperature if use_phase_temperatures else None
            self.constrained_processor.codes_temperature = codes_temperature if use_phase_temperatures else None
            self.constrained_processor.update_caption(formatted_prompt)  # Use formatted prompt for genre extraction
            self.constrained_processor.set_target_duration(target_duration)
            # Always call set_user_metadata to ensure previous settings are cleared if None
            self.constrained_processor.set_user_metadata(user_metadata)
            
            constrained_processor = self.constrained_processor

        sampling_params = SamplingParams(
            max_tokens=self.max_model_len - 64,
            temperature=effective_sampler_temp,
            cfg_scale=cfg_scale,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            logits_processor=constrained_processor,
            logits_processor_update_state=constrained_processor.update_state if constrained_processor else None,
        )

        if cfg_scale > 1.0:
            formatted_unconditional_prompt = self.build_formatted_prompt(negative_prompt, is_negative_prompt=True)
            outputs = self.llm.generate(
                [formatted_prompt],
                sampling_params,
                unconditional_prompts=[formatted_unconditional_prompt],
            )
        else:
            outputs = self.llm.generate([formatted_prompt], sampling_params)

        # Extract text (retain original selection order/logic)
        if isinstance(outputs, list) and len(outputs) > 0:
            if hasattr(outputs[0], "outputs") and len(outputs[0].outputs) > 0:
                output_text = outputs[0].outputs[0].text
            elif hasattr(outputs[0], "text"):
                output_text = outputs[0].text
            elif isinstance(outputs[0], dict) and "text" in outputs[0]:
                output_text = outputs[0]["text"]
            else:
                output_text = str(outputs[0])
        else:
            output_text = str(outputs)

        return output_text
    
    def generate_with_5hz_lm_pt(
        self, 
        caption: str, 
        lyrics: str, 
        temperature: float = 0.6,
        cfg_scale: float = 1.0,
        negative_prompt: str = "NO USER INPUT",
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        use_constrained_decoding: bool = True,
        constrained_decoding_debug: bool = False,
        metadata_temperature: Optional[float] = 0.85,
        codes_temperature: Optional[float] = None,
        target_duration: Optional[float] = None,
        user_metadata: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        """Generate metadata and audio codes using 5Hz LM with PyTorch backend
        
        Args:
            caption: Text caption for music generation
            lyrics: Lyrics for music generation
            temperature: Base sampling temperature (used if phase-specific temps not set)
            cfg_scale: CFG scale (>1.0 enables CFG)
            negative_prompt: Negative prompt for CFG
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty
            use_constrained_decoding: Whether to use FSM-based constrained decoding
            constrained_decoding_debug: Whether to print debug info for constrained decoding
            metadata_temperature: Temperature for metadata generation (lower = more accurate)
                                  If None, uses base temperature
            codes_temperature: Temperature for audio codes generation (higher = more diverse)
                               If None, uses base temperature
            target_duration: Target duration in seconds for codes generation constraint.
                            5 codes = 1 second. If specified, blocks EOS until target reached.
        """
        try:
            formatted_prompt = self.build_formatted_prompt(caption, lyrics)
            
            # Tokenize the prompt
            inputs = self.llm_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=False,
            )
            
            # Generate with the model
            with self._load_model_context():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get max_new_tokens from model config or use a default
                max_new_tokens = getattr(self.llm.config, 'max_new_tokens', 4096)
                if hasattr(self, 'max_model_len'):
                    max_new_tokens = min(max_new_tokens, self.max_model_len - 64)
                
                # Define custom streamer for tqdm
                class TqdmTokenStreamer(BaseStreamer):
                    def __init__(self, total):
                        self.pbar = tqdm(total=total, desc="Generating 5Hz tokens", unit="token", maxinterval=1)
                        
                    def put(self, value):
                        # value is tensor of token ids
                        if value.dim() > 1:
                            num_tokens = value.numel()
                        else:
                            num_tokens = len(value)
                        self.pbar.update(num_tokens)
                        
                    def end(self):
                        self.pbar.close()

                streamer = TqdmTokenStreamer(total=max_new_tokens)

                # Determine if using phase-specific temperatures
                use_phase_temperatures = metadata_temperature is not None or codes_temperature is not None
                effective_temperature = 1.0 if use_phase_temperatures else temperature

                # Use shared constrained decoding processor if enabled
                constrained_processor = None
                if use_constrained_decoding or use_phase_temperatures:
                    # Use shared processor, just update caption and settings
                    self.constrained_processor.enabled = use_constrained_decoding
                    self.constrained_processor.debug = constrained_decoding_debug
                    self.constrained_processor.metadata_temperature = metadata_temperature if use_phase_temperatures else None
                    self.constrained_processor.codes_temperature = codes_temperature if use_phase_temperatures else None
                    self.constrained_processor.update_caption(caption)
                    self.constrained_processor.set_target_duration(target_duration)
                    # Always call set_user_metadata to ensure previous settings are cleared if None
                    self.constrained_processor.set_user_metadata(user_metadata)
                    
                    constrained_processor = self.constrained_processor

                # Build logits processor list (only for CFG and repetition penalty)
                logits_processor = LogitsProcessorList()
                
                # Add repetition penalty if needed (generate() doesn't support it natively in all versions)
                if repetition_penalty != 1.0:
                    logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
                
                # Handle CFG if cfg_scale > 1.0
                if cfg_scale > 1.0:
                    # Build unconditional prompt
                    formatted_unconditional_prompt = self.build_formatted_prompt(negative_prompt, is_negative_prompt=True)
                    
                    # Tokenize both prompts together to ensure same length (with left padding)
                    # Left padding is important for generation tasks
                    batch_texts = [formatted_prompt, formatted_unconditional_prompt]
                    original_padding_side = self.llm_tokenizer.padding_side
                    self.llm_tokenizer.padding_side = 'left'
                    batch_inputs = self.llm_tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    self.llm_tokenizer.padding_side = original_padding_side
                    batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                    
                    # Extract conditional and unconditional inputs
                    batch_input_ids = batch_inputs['input_ids']  # [2, seq_len]
                    batch_attention_mask = batch_inputs.get('attention_mask', None)
                    
                    # Use custom CFG generation loop
                    outputs = self._generate_with_cfg_custom(
                        batch_input_ids=batch_input_ids,
                        batch_attention_mask=batch_attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=effective_temperature,
                        cfg_scale=cfg_scale,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                        streamer=streamer,
                        constrained_processor=constrained_processor,
                    )
                    
                    # Extract only the conditional output (first in batch)
                    outputs = outputs[0:1]  # Keep only conditional output
                elif use_constrained_decoding or use_phase_temperatures:
                    # Use custom generation loop for constrained decoding or phase temperatures (non-CFG)
                    input_ids = inputs['input_ids']
                    attention_mask = inputs.get('attention_mask', None)
                    
                    outputs = self._generate_with_constrained_decoding(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=effective_temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                        streamer=streamer,
                        constrained_processor=constrained_processor,
                    )
                else:
                    # Generate without CFG using native generate() parameters
                    with torch.no_grad():
                        outputs = self.llm.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=effective_temperature if effective_temperature > 0 else 1.0,
                            do_sample=True if effective_temperature > 0 else False,
                            top_k=top_k if top_k is not None and top_k > 0 else None,
                            top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
                            logits_processor=logits_processor if len(logits_processor) > 0 else None,
                            pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                            streamer=streamer,
                        )
            
            # Decode the generated tokens
            # outputs is a tensor with shape [batch_size, seq_len], extract first sequence
            if isinstance(outputs, torch.Tensor):
                if outputs.dim() == 2:
                    generated_ids = outputs[0]
                else:
                    generated_ids = outputs
            else:
                generated_ids = outputs[0]
            
            # Only decode the newly generated tokens (skip the input prompt)
            # Use the correct input length based on whether CFG was used
            if cfg_scale > 1.0:
                # In CFG case, use batch_inputs length (both sequences have same length due to padding)
                input_length = batch_inputs['input_ids'].shape[1]
            else:
                input_length = inputs['input_ids'].shape[1]
            generated_ids = generated_ids[input_length:]
            
            # Move to CPU for decoding
            if generated_ids.is_cuda:
                generated_ids = generated_ids.cpu()
            
            output_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=False)
            
            metadata, audio_codes = self.parse_lm_output(output_text)
            codes_count = len(audio_codes.split('<|audio_code_')) - 1 if audio_codes else 0
            return metadata, audio_codes, f"✅ Generated successfully\nOutput length: {len(output_text)} chars\nCodes count: {codes_count}"
            
        except Exception as e:
            error_msg = f"❌ Error generating with 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {}, "", error_msg
    
    def _run_pt_from_formatted(
        self,
        formatted_prompt: str,
        temperature: float,
        cfg_scale: float,
        negative_prompt: str,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        use_constrained_decoding: bool = True,
        constrained_decoding_debug: bool = False,
        target_duration: Optional[float] = None,
        user_metadata: Optional[Dict[str, Optional[str]]] = None,
    ) -> str:
        """Shared PyTorch path: accept prebuilt formatted prompt and return text."""
        inputs = self.llm_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )

        # Use shared constrained processor if enabled
        constrained_processor = None
        if use_constrained_decoding:
            # Use shared processor, just update caption and settings
            self.constrained_processor.enabled = use_constrained_decoding
            self.constrained_processor.debug = constrained_decoding_debug
            self.constrained_processor.update_caption(formatted_prompt)  # Use formatted prompt for genre extraction
            self.constrained_processor.set_target_duration(target_duration)
            # Always call set_user_metadata to ensure previous settings are cleared if None
            self.constrained_processor.set_user_metadata(user_metadata)
            
            constrained_processor = self.constrained_processor

        with self._load_model_context():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            max_new_tokens = getattr(self.llm.config, "max_new_tokens", 4096)
            if hasattr(self, "max_model_len"):
                max_new_tokens = min(max_new_tokens, self.max_model_len - 64)

            # Build logits processor list (only for CFG and repetition penalty)
            logits_processor = LogitsProcessorList()
            
            # Add repetition penalty if needed
            if repetition_penalty != 1.0:
                logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))

            if cfg_scale > 1.0:
                formatted_unconditional_prompt = self.build_formatted_prompt(negative_prompt, is_negative_prompt=True)
                
                # Tokenize both prompts together to ensure same length (with left padding)
                # Left padding is important for generation tasks
                batch_texts = [formatted_prompt, formatted_unconditional_prompt]
                original_padding_side = self.llm_tokenizer.padding_side
                self.llm_tokenizer.padding_side = 'left'
                batch_inputs_tokenized = self.llm_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                self.llm_tokenizer.padding_side = original_padding_side
                batch_inputs_tokenized = {k: v.to(self.device) for k, v in batch_inputs_tokenized.items()}
                
                # Extract batch inputs
                batch_input_ids = batch_inputs_tokenized['input_ids']
                batch_attention_mask = batch_inputs_tokenized.get('attention_mask', None)

                # Use custom CFG generation loop with constrained decoding
                outputs = self._generate_with_cfg_custom(
                    batch_input_ids=batch_input_ids,
                    batch_attention_mask=batch_attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                    streamer=None,
                    constrained_processor=constrained_processor,
                )
                
                # Extract only the conditional output (first in batch)
                outputs = outputs[0:1]  # Keep only conditional output
            elif use_constrained_decoding:
                # Use custom constrained decoding loop for non-CFG
                outputs = self._generate_with_constrained_decoding(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                    streamer=None,
                    constrained_processor=constrained_processor,
                )
            else:
                # Generate without CFG using native generate() parameters
                with torch.no_grad():
                    outputs = self.llm.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else 1.0,
                        do_sample=True if temperature > 0 else False,
                        top_k=top_k if top_k is not None and top_k > 0 else None,
                        top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
                        logits_processor=logits_processor if len(logits_processor) > 0 else None,
                        pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                        streamer=None,
                    )

        # Decode the generated tokens
        # outputs is a tensor with shape [batch_size, seq_len], extract first sequence
        if isinstance(outputs, torch.Tensor):
            if outputs.dim() == 2:
                generated_ids = outputs[0]
            else:
                generated_ids = outputs
        else:
            generated_ids = outputs[0]
        
        # Only decode the newly generated tokens (skip the input prompt)
        # Use the original input length (before batch processing for CFG)
        if cfg_scale > 1.0:
            # In CFG case, we need to use the conditional input length from batch_inputs_tokenized
            # Both sequences have the same length due to padding
            input_length = batch_inputs_tokenized['input_ids'].shape[1]
        else:
            input_length = inputs["input_ids"].shape[1]
        
        generated_ids = generated_ids[input_length:]
        
        # Move to CPU for decoding
        if generated_ids.is_cuda:
            generated_ids = generated_ids.cpu()
        
        output_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=False)
        return output_text
    
    def generate_with_5hz_lm(
        self, 
        caption: str, 
        lyrics: str, 
        temperature: float = 0.6, 
        cfg_scale: float = 1.0, 
        negative_prompt: str = "NO USER INPUT",
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        use_constrained_decoding: bool = True,
        constrained_decoding_debug: bool = False,
        metadata_temperature: Optional[float] = 0.85,
        codes_temperature: Optional[float] = None,
        target_duration: Optional[float] = None,
        user_metadata: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        """Generate metadata and audio codes using 5Hz LM
        
        Args:
            caption: Text caption for music generation
            lyrics: Lyrics for music generation
            temperature: Base sampling temperature (used if phase-specific temps not set)
            cfg_scale: CFG scale (>1.0 enables CFG)
            negative_prompt: Negative prompt for CFG
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty
            use_constrained_decoding: Whether to use FSM-based constrained decoding for metadata
            constrained_decoding_debug: Whether to print debug info for constrained decoding
            metadata_temperature: Temperature for metadata generation (lower = more accurate)
                                  Recommended: 0.3-0.5 for accurate metadata
            codes_temperature: Temperature for audio codes generation (higher = more diverse)
                               Recommended: 0.7-1.0 for diverse codes
            target_duration: Target duration in seconds for codes generation constraint.
                            5 codes = 1 second. If specified, blocks EOS until target reached.
        """
        # Check if 5Hz LM is initialized
        if not hasattr(self, 'llm_initialized') or not self.llm_initialized:
            debug_info = f"llm_initialized={getattr(self, 'llm_initialized', 'not set')}, "
            debug_info += f"has_llm={hasattr(self, 'llm')}, "
            debug_info += f"llm_is_none={getattr(self, 'llm', None) is None}, "
            debug_info += f"llm_backend={getattr(self, 'llm_backend', 'not set')}"
            return {}, "", f"❌ 5Hz LM not initialized. Please initialize it first. Debug: {debug_info}"
        
        if not hasattr(self, 'llm') or self.llm is None:
            return {}, "", "❌ 5Hz LM model not loaded. Please initialize it first."
        
        if not hasattr(self, 'llm_backend'):
            return {}, "", "❌ 5Hz LM backend not set. Please initialize it first."
        
        if self.llm_backend == "vllm":
            return self.generate_with_5hz_lm_vllm(
                caption=caption, 
                lyrics=lyrics, 
                temperature=temperature, 
                cfg_scale=cfg_scale, 
                negative_prompt=negative_prompt,
                top_k=top_k, 
                top_p=top_p, 
                repetition_penalty=repetition_penalty,
                use_constrained_decoding=use_constrained_decoding, 
                constrained_decoding_debug=constrained_decoding_debug,
                metadata_temperature=metadata_temperature,
                codes_temperature=codes_temperature,
                target_duration=target_duration,
                user_metadata=user_metadata,
            )
        else:
            return self.generate_with_5hz_lm_pt(
                caption=caption, 
                lyrics=lyrics, 
                temperature=temperature, 
                cfg_scale=cfg_scale, 
                negative_prompt=negative_prompt,
                top_k=top_k, 
                top_p=top_p, 
                repetition_penalty=repetition_penalty,
                use_constrained_decoding=use_constrained_decoding, 
                constrained_decoding_debug=constrained_decoding_debug,
                metadata_temperature=metadata_temperature,
                codes_temperature=codes_temperature,
                target_duration=target_duration,
                user_metadata=user_metadata,
            )

    def generate_with_stop_condition(
        self,
        caption: str,
        lyrics: str,
        infer_type: str,
        temperature: float = 0.6,
        cfg_scale: float = 1.0,
        negative_prompt: str = "NO USER INPUT",
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        use_constrained_decoding: bool = True,
        constrained_decoding_debug: bool = False,
        metadata_temperature: Optional[float] = 0.85,
        codes_temperature: Optional[float] = None,
        target_duration: Optional[float] = None,
        user_metadata: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        """Feishu-compatible LM generation.

        - infer_type='dit': stop at </think> and return metas only (no audio codes)
        - infer_type='llm_dit': normal generation (metas + audio codes)
        
        Args:
            target_duration: Target duration in seconds for codes generation constraint.
                            5 codes = 1 second. If specified, blocks EOS until target reached.
            user_metadata: User-provided metadata fields (e.g. bpm/duration/keyscale/timesignature).
                           If specified, constrained decoding will inject these values directly.
        """
        infer_type = (infer_type or "").strip().lower()
        if infer_type not in {"dit", "llm_dit"}:
            return {}, "", f"❌ invalid infer_type: {infer_type!r} (expected 'dit' or 'llm_dit')"

        if infer_type == "llm_dit":
            return self.generate_with_5hz_lm(
                caption=caption,
                lyrics=lyrics,
                temperature=temperature,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_constrained_decoding=use_constrained_decoding,
                constrained_decoding_debug=constrained_decoding_debug,
                metadata_temperature=metadata_temperature,
                codes_temperature=codes_temperature,
                target_duration=target_duration,
                user_metadata=user_metadata,
            )

        # dit: generate and truncate at reasoning end tag
        formatted_prompt = self.build_formatted_prompt(caption, lyrics)
        output_text, status = self.generate_from_formatted_prompt(
            formatted_prompt,
            cfg={
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "negative_prompt": negative_prompt,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "user_metadata": user_metadata,
            },
            use_constrained_decoding=use_constrained_decoding,
            constrained_decoding_debug=constrained_decoding_debug,
        )
        if not output_text:
            return {}, "", status

        if self.STOP_REASONING_TAG in output_text:
            stop_idx = output_text.find(self.STOP_REASONING_TAG)
            output_text = output_text[: stop_idx + len(self.STOP_REASONING_TAG)]

        metadata, _audio_codes = self.parse_lm_output(output_text)
        return metadata, "", status

    def build_formatted_prompt(self, caption: str, lyrics: str = "", is_negative_prompt: bool = False) -> str:
        """
        Build the chat-formatted prompt for 5Hz LM from caption/lyrics.
        Raises a ValueError if the tokenizer is not initialized.

        Example:
            prompt = handler.build_formatted_prompt("calm piano", "hello world")
        """
        if self.llm_tokenizer is None:
            raise ValueError("LLM tokenizer is not initialized. Call initialize() first.")
        if is_negative_prompt:
            prompt = caption
        else:
            prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
        return self.llm_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_from_formatted_prompt(
        self,
        formatted_prompt: str,
        cfg: Optional[Dict[str, Any]] = None,
        use_constrained_decoding: bool = True,
        constrained_decoding_debug: bool = False,
    ) -> Tuple[str, str]:
        """
        Generate raw LM text output from a pre-built formatted prompt.

        Args:
            formatted_prompt: Prompt that is already formatted by `build_formatted_prompt`.
            cfg: Optional dict supporting keys:
                - temperature (float)
                - cfg_scale (float)
                - negative_prompt (str) used when cfg_scale > 1
                - top_k (int), top_p (float), repetition_penalty (float)
                - target_duration (float): Target duration in seconds for codes generation
            use_constrained_decoding: Whether to use FSM-based constrained decoding
            constrained_decoding_debug: Whether to enable debug logging for constrained decoding

        Returns:
            (output_text, status_message)

        Example:
            prompt = handler.build_formatted_prompt(caption, lyric)
            text, status = handler.generate_from_formatted_prompt(prompt, {"temperature": 0.7})
        """
        if not getattr(self, "llm_initialized", False):
            return "", "❌ 5Hz LM not initialized. Please initialize it first."
        if self.llm is None or self.llm_tokenizer is None:
            return "", "❌ 5Hz LM is missing model or tokenizer."

        cfg = cfg or {}
        temperature = cfg.get("temperature", 0.6)
        cfg_scale = cfg.get("cfg_scale", 1.0)
        negative_prompt = cfg.get("negative_prompt", "NO USER INPUT")
        top_k = cfg.get("top_k")
        top_p = cfg.get("top_p")
        repetition_penalty = cfg.get("repetition_penalty", 1.0)
        target_duration = cfg.get("target_duration")
        user_metadata = cfg.get("user_metadata")  # User-provided metadata fields

        try:
            if self.llm_backend == "vllm":
                output_text = self._run_vllm_from_formatted(
                    formatted_prompt=formatted_prompt,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                    constrained_decoding_debug=constrained_decoding_debug,
                    target_duration=target_duration,
                    user_metadata=user_metadata,
                )
                return output_text, f"✅ Generated successfully (vllm) | length={len(output_text)}"

            # PyTorch backend
            output_text = self._run_pt_from_formatted(
                formatted_prompt=formatted_prompt,
                temperature=temperature,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_constrained_decoding=use_constrained_decoding,
                constrained_decoding_debug=constrained_decoding_debug,
                target_duration=target_duration,
                user_metadata=user_metadata,
            )
            return output_text, f"✅ Generated successfully (pt) | length={len(output_text)}"

        except Exception as e:
            return "", f"❌ Error generating from formatted prompt: {e}"
    
    def _generate_with_constrained_decoding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        pad_token_id: int,
        streamer: Optional[BaseStreamer],
        constrained_processor: Optional[MetadataConstrainedLogitsProcessor] = None,
    ) -> torch.Tensor:
        """
        Custom generation loop with constrained decoding support (non-CFG).
        This allows us to call update_state() after each token generation.
        """
        model = self.llm
        device = self.device
        
        # Initialize generated sequences
        generated_ids = input_ids.clone()
        if attention_mask is not None:
            attn_mask = attention_mask.clone()
        else:
            attn_mask = torch.ones_like(input_ids)
        
        # Prepare model inputs
        model_kwargs = {'attention_mask': attn_mask}
        
        # Past key values for KV cache
        past_key_values = None
        use_cache = hasattr(model, 'generation_config') and getattr(model.generation_config, 'use_cache', True)
        
        # Get EOS token ID
        eos_token_id = self.llm_tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = pad_token_id
        
        # Build logits processor for repetition penalty
        logits_processor = LogitsProcessorList()
        if repetition_penalty != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                if past_key_values is None:
                    outputs = model(
                        input_ids=generated_ids,
                        **model_kwargs,
                        use_cache=use_cache,
                    )
                else:
                    outputs = model(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=past_key_values,
                        **model_kwargs,
                        use_cache=use_cache,
                    )
                
                # Get logits for the last position
                next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Apply constrained processor FIRST (modifies logits based on FSM state)
                if constrained_processor is not None:
                    next_token_logits = constrained_processor(generated_ids, next_token_logits)
                
                # Apply other logits processors (repetition penalty)
                for processor in logits_processor:
                    next_token_logits = processor(generated_ids, next_token_logits)
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply temperature and sample
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Update constrained processor state
                if constrained_processor is not None:
                    for b in range(next_tokens.shape[0]):
                        constrained_processor.update_state(next_tokens[b].item())
                
                # Check for EOS token
                should_stop = False
                if torch.any(next_tokens == eos_token_id):
                    should_stop = True
                elif pad_token_id is not None and pad_token_id != eos_token_id:
                    if torch.any(next_tokens == pad_token_id):
                        should_stop = True
                
                # Append token to sequence
                next_tokens_unsqueezed = next_tokens.unsqueeze(1)
                generated_ids = torch.cat([generated_ids, next_tokens_unsqueezed], dim=1)
                attn_mask = torch.cat([attn_mask, torch.ones((input_ids.shape[0], 1), device=device, dtype=attn_mask.dtype)], dim=1)
                model_kwargs['attention_mask'] = attn_mask
                
                # Update KV cache
                if use_cache and hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
                
                # Update streamer
                if streamer is not None:
                    streamer.put(next_tokens_unsqueezed)
                
                if should_stop:
                    break
        
        if streamer is not None:
            streamer.end()
        
        return generated_ids
    
    def _generate_with_cfg_custom(
        self,
        batch_input_ids: torch.Tensor,
        batch_attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        cfg_scale: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        pad_token_id: int,
        streamer: Optional[BaseStreamer],
        constrained_processor: Optional[MetadataConstrainedLogitsProcessor] = None,
    ) -> torch.Tensor:
        """
        Custom CFG generation loop that:
        1. Processes both conditional and unconditional sequences in parallel
        2. Applies CFG formula to logits
        3. Samples tokens only for conditional sequences
        4. Applies the same sampled tokens to both conditional and unconditional sequences
        5. Optionally applies constrained decoding via FSM-based logits processor
        
        Batch format: [cond_input, uncond_input]
        """
        model = self.llm
        device = self.device
        batch_size = batch_input_ids.shape[0] // 2  # Half are conditional, half are unconditional
        cond_start_idx = 0
        uncond_start_idx = batch_size
        
        # Initialize generated sequences
        generated_ids = batch_input_ids.clone()
        if batch_attention_mask is not None:
            attention_mask = batch_attention_mask.clone()
        else:
            attention_mask = torch.ones_like(batch_input_ids)
        
        # Prepare model inputs
        model_kwargs = {}
        if batch_attention_mask is not None:
            model_kwargs['attention_mask'] = attention_mask
        
        # Past key values for KV cache (if model supports it)
        past_key_values = None
        use_cache = hasattr(model, 'generation_config') and getattr(model.generation_config, 'use_cache', True)
        
        # Get EOS token ID for stopping condition
        eos_token_id = self.llm_tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = pad_token_id
        
        # Build logits processor for non-CFG operations (repetition penalty, top_k, top_p)
        logits_processor = LogitsProcessorList()
        if repetition_penalty != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass for the entire batch (conditional + unconditional)
                if past_key_values is None:
                    # First step: full forward pass
                    outputs = model(
                        input_ids=generated_ids,
                        **model_kwargs,
                        use_cache=use_cache,
                    )
                else:
                    # Subsequent steps: only forward the last token (utilizing KV cache)
                    outputs = model(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=past_key_values,
                        **model_kwargs,
                        use_cache=use_cache,
                    )
                
                # Get logits for the last position
                next_token_logits = outputs.logits[:, -1, :]  # [batch_size*2, vocab_size]
                
                # Split conditional and unconditional logits
                cond_logits = next_token_logits[cond_start_idx:cond_start_idx+batch_size]
                uncond_logits = next_token_logits[uncond_start_idx:uncond_start_idx+batch_size]
                
                # Apply CFG formula: cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
                cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
                
                # Apply constrained processor FIRST (modifies logits based on FSM state)
                if constrained_processor is not None:
                    current_input_ids = generated_ids[cond_start_idx:cond_start_idx+batch_size]
                    cfg_logits = constrained_processor(current_input_ids, cfg_logits)
                
                # Apply logits processors (repetition penalty, top-k, top-p)
                # Get current input_ids for repetition penalty (only conditional part)
                current_input_ids = generated_ids[cond_start_idx:cond_start_idx+batch_size]
                for processor in logits_processor:
                    cfg_logits = processor(current_input_ids, cfg_logits)
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = cfg_logits < torch.topk(cfg_logits, top_k)[0][..., -1, None]
                    cfg_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(cfg_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    cfg_logits[indices_to_remove] = float('-inf')
                
                # Apply temperature and sample
                if temperature > 0:
                    cfg_logits = cfg_logits / temperature
                    probs = torch.softmax(cfg_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(cfg_logits, dim=-1)
                
                # Update constrained processor state AFTER sampling
                if constrained_processor is not None:
                    for b in range(next_tokens.shape[0]):
                        constrained_processor.update_state(next_tokens[b].item())
                
                # Check for EOS token in conditional sequences BEFORE unsqueezing
                # Stop if any conditional sequence generates EOS token
                # next_tokens shape: [batch_size] (only conditional tokens)
                should_stop = False
                if torch.any(next_tokens == eos_token_id):
                    should_stop = True
                elif pad_token_id is not None and pad_token_id != eos_token_id:
                    if torch.any(next_tokens == pad_token_id):
                        should_stop = True
                
                # Apply the same sampled tokens to both conditional and unconditional sequences
                next_tokens_unsqueezed = next_tokens.unsqueeze(1)
                generated_ids = torch.cat([generated_ids, next_tokens_unsqueezed.repeat(2, 1)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size*2, 1), device=device, dtype=attention_mask.dtype)], dim=1)
                model_kwargs['attention_mask'] = attention_mask
                
                # Update past_key_values for next iteration
                if use_cache and hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
                
                # Update streamer
                if streamer is not None:
                    streamer.put(next_tokens_unsqueezed)  # Stream conditional tokens
                
                # Stop generation if EOS token detected
                if should_stop:
                    break
        
        if streamer is not None:
            streamer.end()
        
        # Return the full batch (both conditional and unconditional)
        # The caller will extract only the conditional output
        return generated_ids
    
    def parse_lm_output(self, output_text: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse LM output to extract metadata and audio codes.
        
        Expected format:
        <think>
        bpm: 73
        duration: 273
        genres: Chinese folk
        keyscale: G major
        timesignature: 4
        </think>
        
        <|audio_code_56535|><|audio_code_62918|>...
        
        Returns:
            Tuple of (metadata_dict, audio_codes_string)
        """
        debug_output_text = output_text.split("</think>")[0]
        logger.debug(f"Debug output text: {debug_output_text}")
        metadata = {}
        audio_codes = ""
        
        import re
        
        # Extract audio codes - find all <|audio_code_XXX|> patterns
        code_pattern = r'<\|audio_code_\d+\|>'
        code_matches = re.findall(code_pattern, output_text)
        if code_matches:
            audio_codes = "".join(code_matches)
        
        # Extract metadata from reasoning section
        # Try different reasoning tag patterns
        reasoning_patterns = [
            r'<think>(.*?)</think>',
            r'<think>(.*?)</think>',
            r'<reasoning>(.*?)</reasoning>',
        ]
        
        reasoning_text = None
        for pattern in reasoning_patterns:
            match = re.search(pattern, output_text, re.DOTALL)
            if match:
                reasoning_text = match.group(1).strip()
                break
        
        # If no reasoning tags found, try to parse metadata from the beginning of output
        if not reasoning_text:
            # Look for metadata lines before audio codes
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
                        elif key == 'genres':
                            metadata['genres'] = value
                        elif key == 'keyscale':
                            metadata['keyscale'] = value
                        elif key == 'timesignature':
                            metadata['timesignature'] = value
        
        return metadata, audio_codes
    
    @contextmanager
    def _load_model_context(self):
        """
        Context manager to load a model to GPU and offload it back to CPU after use.
        Only used for PyTorch backend when offload_to_cpu is True.
        """
        if not self.offload_to_cpu:
            yield
            return
        
        # If using nanovllm, do not offload (it stays on GPU)
        if self.llm_backend == "vllm":
            yield
            return
        
        model = self.llm
        if model is None:
            yield
            return
        
        # Load to GPU
        logger.info(f"Loading LLM to {self.device}")
        start_time = time.time()
        if hasattr(model, "to"):
            model.to(self.device).to(self.dtype)
        load_time = time.time() - start_time
        logger.info(f"Loaded LLM to {self.device} in {load_time:.4f}s")

        try:
            yield
        finally:
            # Offload to CPU
            logger.info(f"Offloading LLM to CPU")
            start_time = time.time()
            if hasattr(model, "to"):
                model.to("cpu")
            torch.cuda.empty_cache()
            offload_time = time.time() - start_time
            logger.info(f"Offloaded LLM to CPU in {offload_time:.4f}s")

