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
from transformers import AutoTokenizer, AutoModelForCausalLM, ClassifierFreeGuidanceLogitsProcessor
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    LogitsProcessor,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
)




class LLMHandler:
    """5Hz LM Handler for audio code generation"""
    
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
    ) -> Tuple[Dict[str, Any], str, str]:
        """Generate metadata and audio codes using 5Hz LM with vllm backend"""
        try:
            from nanovllm import SamplingParams
            
            prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
            
            formatted_prompt = self.llm_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"},
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.debug(f"[debug] formatted_prompt: {formatted_prompt}")
            
            sampling_params = SamplingParams(
                max_tokens=self.max_model_len-64, 
                temperature=temperature, 
                cfg_scale=cfg_scale,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            # Use CFG if cfg_scale > 1.0
            if cfg_scale > 1.0:
                # Build unconditional prompt (user input replaced with "NO USER INPUT")
                formatted_unconditional_prompt = self.llm_tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"},
                        {"role": "user", "content": negative_prompt}
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
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
    ) -> Tuple[Dict[str, Any], str, str]:
        """Generate metadata and audio codes using 5Hz LM with PyTorch backend"""
        try:
            prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
            
            formatted_prompt = self.llm_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"},
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Tokenize the prompt
            inputs = self.llm_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
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

                # Build logits processor list
                logits_processor = LogitsProcessorList()
                
                # Add repetition penalty if needed
                if repetition_penalty != 1.0:
                    logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
                
                # Add temperature warper if needed (temperature is handled separately in generate, but we can also use warper)
                # Note: temperature is passed directly to generate(), but we can use TemperatureLogitsWarper for consistency
                if temperature != 1.0:
                    logits_processor.append(TemperatureLogitsWarper(temperature=temperature))
                
                # Add top-k warper if specified
                if top_k is not None and top_k > 0:
                    logits_processor.append(TopKLogitsWarper(top_k=top_k))
                
                # Add top-p warper if specified
                if top_p is not None and top_p > 0.0 and top_p < 1.0:
                    logits_processor.append(TopPLogitsWarper(top_p=top_p))
                
                # Handle CFG if cfg_scale > 1.0
                if cfg_scale > 1.0:
                    # Build unconditional prompt
                    formatted_unconditional_prompt = self.llm_tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"},
                            {"role": "user", "content": negative_prompt}
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    
                    # Tokenize unconditional prompt
                    uncond_inputs = self.llm_tokenizer(
                        formatted_unconditional_prompt,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                    )
                    uncond_inputs = {k: v.to(self.device) for k, v in uncond_inputs.items()}
                    
                    # Use custom CFG generation with batch processing
                    # Combine conditional and unconditional inputs into a batch
                    # Format: [cond_input, uncond_input]
                    batch_input_ids = torch.cat([inputs['input_ids'], uncond_inputs['input_ids']], dim=0)
                    batch_attention_mask = None
                    if 'attention_mask' in inputs:
                        batch_attention_mask = torch.cat([inputs['attention_mask'], uncond_inputs.get('attention_mask', torch.ones_like(uncond_inputs['input_ids']))], dim=0)
                    
                    # Custom CFG generation loop
                    outputs = self._generate_with_cfg(
                        batch_input_ids=batch_input_ids,
                        batch_attention_mask=batch_attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        logits_processor=logits_processor,
                        pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                        streamer=streamer,
                    )
                else:
                    # Generate without CFG
                    with torch.no_grad():
                        outputs = self.llm.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature if temperature > 0 else 1.0,
                            do_sample=True if temperature > 0 else False,
                            logits_processor=logits_processor if len(logits_processor) > 0 else None,
                            pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                            streamer=streamer,
                        )
            
            # Decode the generated tokens
            # Only decode the newly generated tokens (skip the input prompt)
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            output_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=False)
            
            metadata, audio_codes = self.parse_lm_output(output_text)
            codes_count = len(audio_codes.split('<|audio_code_')) - 1 if audio_codes else 0
            return metadata, audio_codes, f"✅ Generated successfully\nOutput length: {len(output_text)} chars\nCodes count: {codes_count}"
            
        except Exception as e:
            error_msg = f"❌ Error generating with 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return {}, "", error_msg
    
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
    ) -> Tuple[Dict[str, Any], str, str]:
        """Generate metadata and audio codes using 5Hz LM"""
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
                caption, lyrics, temperature, cfg_scale, negative_prompt,
                top_k, top_p, repetition_penalty
            )
        else:
            return self.generate_with_5hz_lm_pt(
                caption, lyrics, temperature, cfg_scale, negative_prompt,
                top_k, top_p, repetition_penalty
            )
    
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
    
    def _generate_with_cfg(
        self,
        batch_input_ids: torch.Tensor,
        batch_attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        cfg_scale: float,
        logits_processor: Optional[LogitsProcessorList],
        pad_token_id: int,
        streamer: Optional[BaseStreamer],
    ) -> torch.Tensor:
        """
        Custom generation loop with CFG support using batch processing.
        Batch format: [conditional_input, unconditional_input]
        This properly utilizes KV cache by processing both sequences in parallel.
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
                
                # Get logits
                next_token_logits = outputs.logits[:, -1, :]  # [batch_size*2, vocab_size]
                
                # Split conditional and unconditional logits
                cond_logits = next_token_logits[cond_start_idx:cond_start_idx+batch_size]
                uncond_logits = next_token_logits[uncond_start_idx:uncond_start_idx+batch_size]
                
                # Apply CFG formula: logits_cfg = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
                
                # Apply logits processors (temperature, top-k, top-p, repetition penalty)
                if logits_processor is not None:
                    # Get current input_ids for repetition penalty (only conditional part)
                    current_input_ids = generated_ids[cond_start_idx:cond_start_idx+batch_size]
                    for processor in logits_processor:
                        cfg_logits = processor(current_input_ids, cfg_logits)
                
                # Apply temperature and sample
                if temperature > 0:
                    cfg_logits = cfg_logits / temperature
                    probs = torch.softmax(cfg_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(cfg_logits, dim=-1)
                
                # Update generated sequences (apply same token to both conditional and unconditional)
                next_tokens = next_tokens.unsqueeze(1)
                generated_ids = torch.cat([generated_ids, next_tokens.repeat(2, 1)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size*2, 1), device=device, dtype=attention_mask.dtype)], dim=1)
                model_kwargs['attention_mask'] = attention_mask
                
                # Update past_key_values for next iteration
                if use_cache and hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
                
                # Update streamer
                if streamer is not None:
                    streamer.put(next_tokens[0])  # Only stream conditional tokens
                
                # Check for EOS (simplified - you may want to check model's eos_token_id)
                if (next_tokens[0] == pad_token_id).all():
                    break
        
        if streamer is not None:
            streamer.end()
        
        # Return only conditional output
        return generated_ids[cond_start_idx:cond_start_idx+batch_size]
    
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

