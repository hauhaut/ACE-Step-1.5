"""
ACE-Step Inference API Module

This module provides a standardized inference interface for music generation,
designed for third-party integration. It offers both a simplified API and 
backward-compatible Gradio UI support.
"""

import math
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger
import time as time_module


@dataclass
class GenerationConfig:
    """Configuration for music generation.
    
    Attributes:
        # Text Inputs
        caption: Text description of the desired music
        lyrics: Lyrics text for vocal music (use "[Instrumental]" for instrumental)
        
        # Music Metadata
        bpm: Beats per minute (e.g., 120). None for auto-detection
        key_scale: Musical key (e.g., "C Major", "Am"). Empty for auto-detection
        time_signature: Time signature (e.g., "4/4", "3/4"). Empty for auto-detection
        vocal_language: Language code for vocals (e.g., "en", "zh", "ja")
        audio_duration: Duration in seconds. None for auto-detection
        
        # Generation Parameters
        inference_steps: Number of denoising steps (8 for turbo, 32-100 for base)
        guidance_scale: Classifier-free guidance scale (higher = more adherence to prompt)
        use_random_seed: Whether to use random seed (True) or fixed seed
        seed: Random seed for reproducibility (-1 for random)
        batch_size: Number of samples to generate (1-8)
        
        # Advanced DiT Parameters
        use_adg: Use Adaptive Dual Guidance (base model only)
        cfg_interval_start: CFG application start ratio (0.0-1.0)
        cfg_interval_end: CFG application end ratio (0.0-1.0)
        audio_format: Output audio format ("mp3", "wav", "flac")
        
        # Task-Specific Parameters
        task_type: Generation task type ("text2music", "cover", "repaint", "lego", "extract", "complete")
        reference_audio: Path to reference audio file (for style transfer)
        src_audio: Path to source audio file (for audio-to-audio tasks)
        audio_code_string: Pre-extracted audio codes (advanced use)
        repainting_start: Repainting start time in seconds (for repaint/lego tasks)
        repainting_end: Repainting end time in seconds (-1 for end of audio)
        audio_cover_strength: Strength of audio cover/codes influence (0.0-1.0)
        instruction: Task-specific instruction prompt (auto-generated if empty)
        
        # 5Hz Language Model Parameters (CoT Reasoning)
        use_llm_thinking: Enable LM-based Chain-of-Thought reasoning for metadata/codes
        lm_temperature: LM sampling temperature (0.0-2.0, higher = more creative)
        lm_cfg_scale: LM classifier-free guidance scale
        lm_top_k: LM top-k sampling (0 = disabled)
        lm_top_p: LM nucleus sampling (1.0 = disabled)
        lm_negative_prompt: Negative prompt for LM guidance
        use_cot_metas: Generate metadata using LM CoT
        use_cot_caption: Refine caption using LM CoT
        use_cot_language: Detect language using LM CoT
        is_format_caption: Whether caption is already formatted
        constrained_decoding_debug: Enable debug logging for constrained decoding
        
        # Batch LM Generation
        allow_lm_batch: Allow batch LM code generation (faster for batch_size >= 2)
        lm_batch_chunk_size: Maximum batch size per LM inference chunk (GPU memory constraint)
    """
    
    # Text Inputs
    caption: str = ""
    lyrics: str = ""
    
    # Music Metadata
    bpm: Optional[int] = None
    key_scale: str = ""
    time_signature: str = ""
    vocal_language: str = "unknown"
    audio_duration: Optional[float] = None
    
    # Generation Parameters
    inference_steps: int = 8
    guidance_scale: float = 7.0
    use_random_seed: bool = True
    seed: int = -1
    batch_size: int = 1
    
    # Advanced DiT Parameters
    use_adg: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    audio_format: str = "mp3"
    
    # Task-Specific Parameters
    task_type: str = "text2music"
    reference_audio: Optional[str] = None
    src_audio: Optional[str] = None
    audio_code_string: Union[str, List[str]] = ""
    repainting_start: float = 0.0
    repainting_end: float = -1
    audio_cover_strength: float = 1.0
    instruction: str = ""
    
    # 5Hz Language Model Parameters
    use_llm_thinking: bool = False
    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.0
    lm_top_k: int = 0
    lm_top_p: float = 0.9
    lm_negative_prompt: str = "NO USER INPUT"
    use_cot_metas: bool = True
    use_cot_caption: bool = True
    use_cot_language: bool = True
    is_format_caption: bool = False
    constrained_decoding_debug: bool = False
    
    # Batch LM Generation
    allow_lm_batch: bool = False
    lm_batch_chunk_size: int = 4


@dataclass
class GenerationResult:
    """Result of music generation.
    
    Attributes:
        # Audio Outputs
        audio_paths: List of paths to generated audio files
        first_audio: Path to first generated audio (backward compatibility)
        second_audio: Path to second generated audio (backward compatibility)
        
        # Generation Information
        generation_info: Markdown-formatted generation information
        status_message: Status message from generation
        seed_value: Actual seed value used for generation
        
        # LM-Generated Metadata (if applicable)
        lm_metadata: Metadata generated by language model (dict or None)
        
        # Audio-Text Alignment Scores (if available)
        align_score_1: First alignment score
        align_text_1: First alignment text description  
        align_plot_1: First alignment plot image
        align_score_2: Second alignment score
        align_text_2: Second alignment text description
        align_plot_2: Second alignment plot image
        
        # Success Status
        success: Whether generation completed successfully
        error: Error message if generation failed
    """
    
    # Audio Outputs
    audio_paths: List[str] = field(default_factory=list)
    first_audio: Optional[str] = None
    second_audio: Optional[str] = None
    
    # Generation Information
    generation_info: str = ""
    status_message: str = ""
    seed_value: str = ""
    
    # LM-Generated Metadata
    lm_metadata: Optional[Dict[str, Any]] = None
    
    # Audio-Text Alignment Scores
    align_score_1: Optional[float] = None
    align_text_1: Optional[str] = None
    align_plot_1: Optional[Any] = None
    align_score_2: Optional[float] = None
    align_text_2: Optional[str] = None
    align_plot_2: Optional[Any] = None
    
    # Success Status
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return asdict(self)


def generate_music(
    dit_handler,
    llm_handler,
    config: GenerationConfig,
) -> GenerationResult:
    """Generate music using ACE-Step model with optional LM reasoning.
    
    This is the main inference API for music generation. It supports various task types
    (text2music, cover, repaint, etc.) and can optionally use a 5Hz Language Model for
    Chain-of-Thought reasoning to generate metadata and audio codes.
    
    Args:
        dit_handler: Initialized DiT model handler (AceStepHandler instance)
        llm_handler: Initialized LLM handler (LLMHandler instance)
        config: Generation configuration (GenerationConfig instance)
        
    Returns:
        GenerationResult: Generation result containing audio paths and metadata
        
    Example:
        >>> from acestep.handler import AceStepHandler
        >>> from acestep.llm_inference import LLMHandler
        >>> from acestep.inference import GenerationConfig, generate_music
        >>> 
        >>> # Initialize handlers
        >>> dit_handler = AceStepHandler()
        >>> llm_handler = LLMHandler()
        >>> dit_handler.initialize_service(...)
        >>> llm_handler.initialize(...)
        >>> 
        >>> # Configure generation
        >>> config = GenerationConfig(
        ...     caption="upbeat electronic dance music",
        ...     bpm=128,
        ...     audio_duration=30,
        ...     batch_size=2,
        ... )
        >>> 
        >>> # Generate music
        >>> result = generate_music(dit_handler, llm_handler, config)
        >>> print(f"Generated {len(result.audio_paths)} audio files")
        >>> for path in result.audio_paths:
        ...     print(f"Audio: {path}")
    """
    
    try:
        # Phase 1: LM-based metadata and code generation (if enabled)
        audio_code_string_to_use = config.audio_code_string
        lm_generated_metadata = None
        lm_generated_audio_codes = None
        lm_generated_audio_codes_list = []
        
        # Extract mutable copies of metadata (will be updated by LM if needed)
        bpm = config.bpm
        key_scale = config.key_scale
        time_signature = config.time_signature
        audio_duration = config.audio_duration
        
        # Determine if we should use batch LM generation
        should_use_lm_batch = (
            config.use_llm_thinking 
            and llm_handler.llm_initialized 
            and config.use_cot_metas 
            and config.allow_lm_batch
            and config.batch_size >= 2
        )
        
        # LM-based Chain-of-Thought reasoning
        if config.use_llm_thinking and llm_handler.llm_initialized and config.use_cot_metas:
            # Convert sampling parameters
            top_k_value = None if config.lm_top_k == 0 else int(config.lm_top_k)
            top_p_value = None if config.lm_top_p >= 1.0 else config.lm_top_p
            
            # Build user_metadata from user-provided values
            user_metadata = {}
            if bpm is not None:
                try:
                    bpm_value = float(bpm)
                    if bpm_value > 0:
                        user_metadata['bpm'] = str(int(bpm_value))
                except (ValueError, TypeError):
                    pass
                    
            if key_scale and key_scale.strip():
                key_scale_clean = key_scale.strip()
                if key_scale_clean.lower() not in ["n/a", ""]:
                    user_metadata['keyscale'] = key_scale_clean
                    
            if time_signature and time_signature.strip():
                time_sig_clean = time_signature.strip()
                if time_sig_clean.lower() not in ["n/a", ""]:
                    user_metadata['timesignature'] = time_sig_clean
                    
            if audio_duration is not None:
                try:
                    duration_value = float(audio_duration)
                    if duration_value > 0:
                        user_metadata['duration'] = str(int(duration_value))
                except (ValueError, TypeError):
                    pass
            
            user_metadata_to_pass = user_metadata if user_metadata else None
            
            # Batch LM generation (faster for multiple samples)
            if should_use_lm_batch:
                actual_seed_list, _ = dit_handler.prepare_seeds(
                    config.batch_size, config.seed, config.use_random_seed
                )
                
                max_inference_batch_size = int(config.lm_batch_chunk_size)
                num_chunks = math.ceil(config.batch_size / max_inference_batch_size)
                
                all_metadata_list = []
                all_audio_codes_list = []
                
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * max_inference_batch_size
                    chunk_end = min(chunk_start + max_inference_batch_size, config.batch_size)
                    chunk_size = chunk_end - chunk_start
                    chunk_seeds = actual_seed_list[chunk_start:chunk_end]
                    
                    logger.info(
                        f"LM batch chunk {chunk_idx+1}/{num_chunks} "
                        f"(size: {chunk_size}, seeds: {chunk_seeds})"
                    )
                    
                    metadata_list, audio_codes_list, status = llm_handler.generate_with_stop_condition_batch(
                        caption=config.caption or "",
                        lyrics=config.lyrics or "",
                        batch_size=chunk_size,
                        infer_type="llm_dit",
                        temperature=config.lm_temperature,
                        cfg_scale=config.lm_cfg_scale,
                        negative_prompt=config.lm_negative_prompt,
                        top_k=top_k_value,
                        top_p=top_p_value,
                        user_metadata=user_metadata_to_pass,
                        use_cot_caption=config.use_cot_caption,
                        use_cot_language=config.use_cot_language,
                        is_format_caption=config.is_format_caption,
                        constrained_decoding_debug=config.constrained_decoding_debug,
                        seeds=chunk_seeds,
                    )
                    
                    all_metadata_list.extend(metadata_list)
                    all_audio_codes_list.extend(audio_codes_list)
                
                lm_generated_metadata = all_metadata_list[0] if all_metadata_list else None
                lm_generated_audio_codes_list = all_audio_codes_list
                audio_code_string_to_use = all_audio_codes_list
                
                # Update metadata from LM if not provided by user
                if lm_generated_metadata:
                    bpm, key_scale, time_signature, audio_duration = _update_metadata_from_lm(
                        lm_generated_metadata, bpm, key_scale, time_signature, audio_duration
                    )
                    
            else:
                # Sequential LM generation (current behavior)
                # Phase 1: Generate CoT metadata
                phase1_start = time_module.time()
                metadata, _, status = llm_handler.generate_with_stop_condition(
                    caption=config.caption or "",
                    lyrics=config.lyrics or "",
                    infer_type="dit",
                    temperature=config.lm_temperature,
                    cfg_scale=config.lm_cfg_scale,
                    negative_prompt=config.lm_negative_prompt,
                    top_k=top_k_value,
                    top_p=top_p_value,
                    user_metadata=user_metadata_to_pass,
                    use_cot_caption=config.use_cot_caption,
                    use_cot_language=config.use_cot_language,
                    is_format_caption=config.is_format_caption,
                    constrained_decoding_debug=config.constrained_decoding_debug,
                )
                lm_phase1_time = time_module.time() - phase1_start
                logger.info(f"LM Phase 1 (CoT) completed in {lm_phase1_time:.2f}s")
                
                # Phase 2: Generate audio codes
                phase2_start = time_module.time()
                metadata, audio_codes, status = llm_handler.generate_with_stop_condition(
                    caption=config.caption or "",
                    lyrics=config.lyrics or "",
                    infer_type="llm_dit",
                    temperature=config.lm_temperature,
                    cfg_scale=config.lm_cfg_scale,
                    negative_prompt=config.lm_negative_prompt,
                    top_k=top_k_value,
                    top_p=top_p_value,
                    user_metadata=user_metadata_to_pass,
                    use_cot_caption=config.use_cot_caption,
                    use_cot_language=config.use_cot_language,
                    is_format_caption=config.is_format_caption,
                    constrained_decoding_debug=config.constrained_decoding_debug,
                )
                lm_phase2_time = time_module.time() - phase2_start
                logger.info(f"LM Phase 2 (Codes) completed in {lm_phase2_time:.2f}s")
                
                lm_generated_metadata = metadata
                if audio_codes:
                    audio_code_string_to_use = audio_codes
                    lm_generated_audio_codes = audio_codes
                    
                    # Update metadata from LM if not provided by user
                    bpm, key_scale, time_signature, audio_duration = _update_metadata_from_lm(
                        metadata, bpm, key_scale, time_signature, audio_duration
                    )
        
        # Phase 2: DiT music generation
        result = dit_handler.generate_music(
            captions=config.caption,
            lyrics=config.lyrics,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            vocal_language=config.vocal_language,
            inference_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
            use_random_seed=config.use_random_seed,
            seed=config.seed,
            reference_audio=config.reference_audio,
            audio_duration=audio_duration,
            batch_size=config.batch_size,
            src_audio=config.src_audio,
            audio_code_string=audio_code_string_to_use,
            repainting_start=config.repainting_start,
            repainting_end=config.repainting_end,
            instruction=config.instruction,
            audio_cover_strength=config.audio_cover_strength,
            task_type=config.task_type,
            use_adg=config.use_adg,
            cfg_interval_start=config.cfg_interval_start,
            cfg_interval_end=config.cfg_interval_end,
            audio_format=config.audio_format,
            lm_temperature=config.lm_temperature,
        )
        
        # Extract results
        (first_audio, second_audio, all_audio_paths, generation_info, status_message, 
         seed_value, align_score_1, align_text_1, align_plot_1, 
         align_score_2, align_text_2, align_plot_2) = result
        
        # Append LM metadata to generation info
        if lm_generated_metadata:
            generation_info = _append_lm_metadata_to_info(generation_info, lm_generated_metadata)
        
        # Create result object
        return GenerationResult(
            audio_paths=all_audio_paths or [],
            first_audio=first_audio,
            second_audio=second_audio,
            generation_info=generation_info,
            status_message=status_message,
            seed_value=seed_value,
            lm_metadata=lm_generated_metadata,
            align_score_1=align_score_1,
            align_text_1=align_text_1,
            align_plot_1=align_plot_1,
            align_score_2=align_score_2,
            align_text_2=align_text_2,
            align_plot_2=align_plot_2,
            success=True,
            error=None,
        )
        
    except Exception as e:
        logger.exception("Music generation failed")
        return GenerationResult(
            success=False,
            error=str(e),
            generation_info=f"❌ Generation failed: {str(e)}",
            status_message=f"Error: {str(e)}",
        )


def _update_metadata_from_lm(
    metadata: Dict[str, Any],
    bpm: Optional[int],
    key_scale: str,
    time_signature: str,
    audio_duration: Optional[float],
) -> Tuple[Optional[int], str, str, Optional[float]]:
    """Update metadata fields from LM output if not provided by user."""
    
    if bpm is None and metadata.get('bpm'):
        bpm_value = metadata.get('bpm')
        if bpm_value not in ["N/A", ""]:
            try:
                bpm = int(bpm_value)
            except (ValueError, TypeError):
                pass
    
    if not key_scale and metadata.get('keyscale'):
        key_scale_value = metadata.get('keyscale', metadata.get('key_scale', ""))
        if key_scale_value != "N/A":
            key_scale = key_scale_value
    
    if not time_signature and metadata.get('timesignature'):
        time_signature_value = metadata.get('timesignature', metadata.get('time_signature', ""))
        if time_signature_value != "N/A":
            time_signature = time_signature_value
    
    if audio_duration is None or audio_duration <= 0:
        audio_duration_value = metadata.get('duration', -1)
        if audio_duration_value not in ["N/A", ""]:
            try:
                audio_duration = float(audio_duration_value)
            except (ValueError, TypeError):
                pass
    
    return bpm, key_scale, time_signature, audio_duration


def _append_lm_metadata_to_info(generation_info: str, metadata: Dict[str, Any]) -> str:
    """Append LM-generated metadata to generation info string."""
    
    metadata_lines = []
    if metadata.get('bpm'):
        metadata_lines.append(f"- **BPM:** {metadata['bpm']}")
    if metadata.get('caption'):
        metadata_lines.append(f"- **Refined Caption:** {metadata['caption']}")
    if metadata.get('duration'):
        metadata_lines.append(f"- **Duration:** {metadata['duration']} seconds")
    if metadata.get('keyscale'):
        metadata_lines.append(f"- **Key Scale:** {metadata['keyscale']}")
    if metadata.get('language'):
        metadata_lines.append(f"- **Language:** {metadata['language']}")
    if metadata.get('timesignature'):
        metadata_lines.append(f"- **Time Signature:** {metadata['timesignature']}")
    
    if metadata_lines:
        metadata_section = "\n\n**🤖 LM-Generated Metadata:**\n" + "\n\n".join(metadata_lines)
        return metadata_section + "\n\n" + generation_info
    
    return generation_info


# ============================================================================
# LEGACY GRADIO UI COMPATIBILITY LAYER
# ============================================================================

def generate(
    dit_handler,
    llm_handler,
    captions,
    lyrics,
    bpm,
    key_scale,
    time_signature,
    vocal_language,
    inference_steps,
    guidance_scale,
    random_seed_checkbox,
    seed,
    reference_audio,
    audio_duration,
    batch_size_input,
    src_audio,
    text2music_audio_code_string,
    repainting_start,
    repainting_end,
    instruction_display_gen,
    audio_cover_strength,
    task_type,
    use_adg,
    cfg_interval_start,
    cfg_interval_end,
    audio_format,
    lm_temperature,
    think_checkbox,
    lm_cfg_scale,
    lm_top_k,
    lm_top_p,
    lm_negative_prompt,
    use_cot_metas,
    use_cot_caption,
    use_cot_language,
    is_format_caption,
    constrained_decoding_debug,
    allow_lm_batch,
    lm_batch_chunk_size,
):
    """Legacy Gradio UI compatibility wrapper.
    
    This function maintains backward compatibility with the Gradio UI.
    For new integrations, use generate_music() with GenerationConfig instead.
    
    Returns:
        Tuple with 28 elements for Gradio UI component updates
    """
    
    # Convert legacy parameters to new config
    config = GenerationConfig(
        caption=captions,
        lyrics=lyrics,
        bpm=bpm,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=vocal_language,
        audio_duration=audio_duration,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        use_random_seed=random_seed_checkbox,
        seed=seed,
        batch_size=batch_size_input,
        use_adg=use_adg,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        audio_format=audio_format,
        task_type=task_type,
        reference_audio=reference_audio,
        src_audio=src_audio,
        audio_code_string=text2music_audio_code_string,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        audio_cover_strength=audio_cover_strength,
        instruction=instruction_display_gen,
        use_llm_thinking=think_checkbox,
        lm_temperature=lm_temperature,
        lm_cfg_scale=lm_cfg_scale,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        lm_negative_prompt=lm_negative_prompt,
        use_cot_metas=use_cot_metas,
        use_cot_caption=use_cot_caption,
        use_cot_language=use_cot_language,
        is_format_caption=is_format_caption,
        constrained_decoding_debug=constrained_decoding_debug,
        allow_lm_batch=allow_lm_batch,
        lm_batch_chunk_size=lm_batch_chunk_size,
    )
    
    # Call new API
    result = generate_music(dit_handler, llm_handler, config)
    
    # Determine which codes to update in UI
    if config.allow_lm_batch and result.lm_metadata:
        # Batch mode: extract codes from metadata if available
        lm_codes_list = result.lm_metadata.get('audio_codes_list', [])
        updated_audio_codes = lm_codes_list[0] if lm_codes_list else text2music_audio_code_string
        codes_outputs = (lm_codes_list + [""] * 8)[:8]
    else:
        # Single mode
        lm_codes = result.lm_metadata.get('audio_codes', '') if result.lm_metadata else ''
        updated_audio_codes = lm_codes if lm_codes else text2music_audio_code_string
        codes_outputs = [""] * 8
    
    # Prepare audio outputs (up to 8)
    audio_outputs = (result.audio_paths + [None] * 8)[:8]
    
    # Return tuple for Gradio UI (28 elements)
    return (
        audio_outputs[0],  # generated_audio_1
        audio_outputs[1],  # generated_audio_2
        audio_outputs[2],  # generated_audio_3
        audio_outputs[3],  # generated_audio_4
        audio_outputs[4],  # generated_audio_5
        audio_outputs[5],  # generated_audio_6
        audio_outputs[6],  # generated_audio_7
        audio_outputs[7],  # generated_audio_8
        result.audio_paths,  # generated_audio_batch
        result.generation_info,
        result.status_message,
        result.seed_value,
        result.align_score_1,
        result.align_text_1,
        result.align_plot_1,
        result.align_score_2,
        result.align_text_2,
        result.align_plot_2,
        updated_audio_codes,  # Update main audio codes in UI
        codes_outputs[0],  # text2music_audio_code_string_1
        codes_outputs[1],  # text2music_audio_code_string_2
        codes_outputs[2],  # text2music_audio_code_string_3
        codes_outputs[3],  # text2music_audio_code_string_4
        codes_outputs[4],  # text2music_audio_code_string_5
        codes_outputs[5],  # text2music_audio_code_string_6
        codes_outputs[6],  # text2music_audio_code_string_7
        codes_outputs[7],  # text2music_audio_code_string_8
        result.lm_metadata,  # Store metadata for "Send to src audio" buttons
        is_format_caption,  # Keep is_format_caption unchanged
    )


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Test suite for the inference API.
    Demonstrates various usage scenarios and validates functionality.
    
    Usage:
        python -m acestep.inference
    """
    
    import os
    import json
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    
    # Initialize paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    
    print("=" * 80)
    print("ACE-Step Inference API Test Suite")
    print("=" * 80)
    
    # ========================================================================
    # Initialize Handlers
    # ========================================================================
    print("\n[1/3] Initializing handlers...")
    dit_handler = AceStepHandler(save_root="./")
    llm_handler = LLMHandler()
    
    try:
        # Initialize DiT handler
        print("  - Initializing DiT model...")
        status_dit, success_dit = dit_handler.initialize_service(
            project_root=project_root,
            config_path="acestep-v15-turbo-rl",
            device="cuda",
        )
        if not success_dit:
            print(f"  ❌ DiT initialization failed: {status_dit}")
            exit(1)
        print(f"  ✓ DiT model initialized successfully")
        
        # Initialize LLM handler
        print("  - Initializing 5Hz LM model...")
        status_llm, success_llm = llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path="acestep-5Hz-lm-0.6B-v3",
            backend="vllm",
            device="cuda",
        )
        if success_llm:
            print(f"  ✓ LM model initialized successfully")
        else:
            print(f"  ⚠ LM initialization failed (will skip LM tests): {status_llm}")
            
    except Exception as e:
        print(f"  ❌ Initialization error: {e}")
        exit(1)
    
    # ========================================================================
    # Helper Functions
    # ========================================================================
    def load_example_config(example_file: str) -> GenerationConfig:
        """Load configuration from an example JSON file."""
        try:
            with open(example_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert example format to GenerationConfig
            # Handle time signature format (example uses "4" instead of "4/4")
            time_sig = data.get('timesignature', '')
            if time_sig and '/' not in time_sig:
                time_sig = f"{time_sig}/4"  # Default to /4 if only numerator given
            
            config = GenerationConfig(
                caption=data.get('caption', ''),
                lyrics=data.get('lyrics', ''),
                bpm=data.get('bpm'),
                key_scale=data.get('keyscale', ''),
                time_signature=time_sig,
                vocal_language=data.get('language', 'unknown'),
                audio_duration=data.get('duration'),
                use_llm_thinking=data.get('think', False),
                batch_size=data.get('batch_size', 1),
                inference_steps=data.get('inference_steps', 8),
            )
            return config
            
        except Exception as e:
            print(f"  ⚠ Failed to load example file: {e}")
            return None
    
    # ========================================================================
    # Test Cases
    # ========================================================================
    test_results = []
    
    def run_test(test_name: str, config: GenerationConfig, expected_outputs: int = 1):
        """Run a single test case and collect results."""
        print(f"\n{'=' * 80}")
        print(f"Test: {test_name}")
        print(f"{'=' * 80}")
        
        # Display configuration
        print("\nConfiguration:")
        print(f"  Task Type: {config.task_type}")
        print(f"  Caption: {config.caption[:60]}..." if len(config.caption) > 60 else f"  Caption: {config.caption}")
        if config.lyrics:
            print(f"  Lyrics: {config.lyrics[:60]}..." if len(config.lyrics) > 60 else f"  Lyrics: {config.lyrics}")
        if config.bpm:
            print(f"  BPM: {config.bpm}")
        if config.key_scale:
            print(f"  Key Scale: {config.key_scale}")
        if config.time_signature:
            print(f"  Time Signature: {config.time_signature}")
        if config.audio_duration:
            print(f"  Duration: {config.audio_duration}s")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Inference Steps: {config.inference_steps}")
        print(f"  Use LLM Thinking: {config.use_llm_thinking}")
        
        # Run generation
        print("\nGenerating...")
        import time
        start_time = time.time()
        
        result = generate_music(dit_handler, llm_handler, config)
        
        elapsed_time = time.time() - start_time
        
        # Display results
        print("\nResults:")
        print(f"  Success: {'✓' if result.success else '✗'}")
        
        if result.success:
            print(f"  Generated Files: {len(result.audio_paths)}")
            for i, path in enumerate(result.audio_paths, 1):
                if os.path.exists(path):
                    file_size = os.path.getsize(path) / (1024 * 1024)  # MB
                    print(f"    [{i}] {os.path.basename(path)} ({file_size:.2f} MB)")
                else:
                    print(f"    [{i}] {os.path.basename(path)} (file not found)")
            
            print(f"  Seed: {result.seed_value}")
            print(f"  Generation Time: {elapsed_time:.2f}s")
            
            # Display LM metadata if available
            if result.lm_metadata:
                print(f"\n  LM-Generated Metadata:")
                for key, value in result.lm_metadata.items():
                    if key not in ['audio_codes', 'audio_codes_list']:  # Skip large code strings
                        print(f"    {key}: {value}")
            
            # Validate outputs
            if len(result.audio_paths) != expected_outputs:
                print(f"  ⚠ Warning: Expected {expected_outputs} outputs, got {len(result.audio_paths)}")
                success = False
            else:
                success = True
                
        else:
            print(f"  Error: {result.error}")
            success = False
        
        # Store test result
        test_results.append({
            "test_name": test_name,
            "success": success,
            "generation_success": result.success,
            "num_outputs": len(result.audio_paths) if result.success else 0,
            "expected_outputs": expected_outputs,
            "elapsed_time": elapsed_time,
            "error": result.error if not result.success else None,
        })
        
        return result
    
    # ========================================================================
    # Test: Production Example (from examples directory)
    # ========================================================================
    print("\n[2/3] Running Test...")
    
    # Load production example (J-Rock song from examples/text2music/example_05.json)
    example_file = os.path.join(project_root, "examples", "text2music", "example_05.json")
    
    if not os.path.exists(example_file):
        print(f"\n  ❌ Example file not found: {example_file}")
        print("     Please ensure the examples directory exists.")
        exit(1)
    
    print(f"  Loading example: {os.path.basename(example_file)}")
    config = load_example_config(example_file)
    
    if not config:
        print("  ❌ Failed to load example configuration")
        exit(1)
    
    # Reduce duration for faster testing (original is 200s)
    print(f"  Original duration: {config.audio_duration}s")
    config.audio_duration = 30
    config.use_random_seed = False
    config.seed = 42
    print(f"  Test duration: {config.audio_duration}s (reduced for testing)")
    
    run_test("Production Example (J-Rock Song)", config, expected_outputs=1)
    
    # ========================================================================
    # Test Summary
    # ========================================================================
    print("\n[3/3] Test Summary")
    print("=" * 80)
    
    if len(test_results) == 0:
        print("No tests were run.")
        exit(1)
    
    result = test_results[0]
    
    print(f"\nTest: {result['test_name']}")
    print(f"Status: {'✓ PASS' if result['success'] else '✗ FAIL'}")
    print(f"Generation: {'Success' if result['generation_success'] else 'Failed'}")
    print(f"Outputs: {result['num_outputs']}/{result['expected_outputs']}")
    print(f"Time: {result['elapsed_time']:.2f}s")
    
    if result["error"]:
        print(f"Error: {result['error']}")
    
    # Save test results to JSON
    results_file = os.path.join(project_root, "test_results.json")
    try:
        with open(results_file, "w") as f:
            json.dump({
                "test_name": result['test_name'],
                "success": result['success'],
                "generation_success": result['generation_success'],
                "num_outputs": result['num_outputs'],
                "expected_outputs": result['expected_outputs'],
                "elapsed_time": result['elapsed_time'],
                "error": result['error'],
            }, f, indent=2)
        print(f"\n✓ Test results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠ Failed to save test results: {e}")
    
    # Exit with appropriate code
    print("\n" + "=" * 80)
    if result['success']:
        print("Test passed! ✓")
        print("=" * 80)
        exit(0)
    else:
        print("Test failed! ✗")
        print("=" * 80)
        exit(1)