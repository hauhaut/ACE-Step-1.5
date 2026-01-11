"""
Results Handlers Module
Contains event handlers and helper functions related to result display, scoring, and batch management
"""
import os
import json
import datetime
import math
import tempfile
import shutil
import zipfile
import time as time_module
from typing import Dict, Any, Optional
import gradio as gr
from loguru import logger
from acestep.gradio_ui.i18n import t
from acestep.inference import generate_music, GenerationParams, GenerationConfig
from acestep.audio_utils import save_audio


def _build_generation_info(
    lm_metadata: Optional[Dict[str, Any]],
    time_costs: Dict[str, float],
    seed_value: str,
    inference_steps: int,
    num_audios: int,
) -> str:
    """Build generation info string from result data.
    
    Args:
        lm_metadata: LM-generated metadata dictionary
        time_costs: Unified time costs dictionary
        seed_value: Seed value string
        inference_steps: Number of inference steps
        num_audios: Number of generated audios
        
    Returns:
        Formatted generation info string
    """
    info_parts = []
    
    # Part 1: LM-generated metadata (if available)
    if lm_metadata:
        metadata_lines = []
        if lm_metadata.get('bpm'):
            metadata_lines.append(f"- **BPM:** {lm_metadata['bpm']}")
        if lm_metadata.get('caption'):
            metadata_lines.append(f"- **Refined Caption:** {lm_metadata['caption']}")
        if lm_metadata.get('lyrics'):
            metadata_lines.append(f"- **Refined Lyrics:** {lm_metadata['lyrics']}")
        if lm_metadata.get('duration'):
            metadata_lines.append(f"- **Duration:** {lm_metadata['duration']} seconds")
        if lm_metadata.get('keyscale'):
            metadata_lines.append(f"- **Key Scale:** {lm_metadata['keyscale']}")
        if lm_metadata.get('language'):
            metadata_lines.append(f"- **Language:** {lm_metadata['language']}")
        if lm_metadata.get('timesignature'):
            metadata_lines.append(f"- **Time Signature:** {lm_metadata['timesignature']}")
        
        if metadata_lines:
            metadata_section = "**🤖 LM-Generated Metadata:**\n" + "\n".join(metadata_lines)
            info_parts.append(metadata_section)
    
    # Part 2: Time costs (formatted and beautified)
    if time_costs:
        time_lines = []
        
        # LM time costs
        lm_phase1 = time_costs.get('lm_phase1_time', 0.0)
        lm_phase2 = time_costs.get('lm_phase2_time', 0.0)
        lm_total = time_costs.get('lm_total_time', 0.0)
        
        if lm_total > 0:
            time_lines.append("**🧠 LM Time:**")
            if lm_phase1 > 0:
                time_lines.append(f"  - Phase 1 (CoT): {lm_phase1:.2f}s")
            if lm_phase2 > 0:
                time_lines.append(f"  - Phase 2 (Codes): {lm_phase2:.2f}s")
            time_lines.append(f"  - Total: {lm_total:.2f}s")
        
        # DiT time costs
        dit_encoder = time_costs.get('dit_encoder_time_cost', 0.0)
        dit_model = time_costs.get('dit_model_time_cost', 0.0)
        dit_vae_decode = time_costs.get('dit_vae_decode_time_cost', 0.0)
        dit_offload = time_costs.get('dit_offload_time_cost', 0.0)
        dit_total = time_costs.get('dit_total_time_cost', 0.0)
        if dit_total > 0:
            time_lines.append("\n**🎵 DiT Time:**")
            if dit_encoder > 0:
                time_lines.append(f"  - Encoder: {dit_encoder:.2f}s")
            if dit_model > 0:
                time_lines.append(f"  - Model: {dit_model:.2f}s")
            if dit_vae_decode > 0:
                time_lines.append(f"  - VAE Decode: {dit_vae_decode:.2f}s")
            if dit_offload > 0:
                time_lines.append(f"  - Offload: {dit_offload:.2f}s")
            time_lines.append(f"  - Total: {dit_total:.2f}s")
        
        # Post-processing time costs
        audio_conversion_time = time_costs.get('audio_conversion_time', 0.0)
        auto_score_time = time_costs.get('auto_score_time', 0.0)
        
        if audio_conversion_time > 0 or auto_score_time > 0:
            time_lines.append("\n**🔧 Post-processing Time:**")
            if audio_conversion_time > 0:
                time_lines.append(f"  - Audio Conversion: {audio_conversion_time:.2f}s")
            if auto_score_time > 0:
                time_lines.append(f"  - Auto Score: {auto_score_time:.2f}s")
        
        # Pipeline total
        pipeline_total = time_costs.get('pipeline_total_time', 0.0)
        if pipeline_total > 0:
            time_lines.append(f"\n**⏱️ Pipeline Total: {pipeline_total:.2f}s**")
        
        if time_lines:
            time_section = "\n".join(time_lines)
            info_parts.append(time_section)
    
    # Part 3: Generation summary
    summary_lines = [
        "**🎵 Generation Complete**",
        f"  - **Seeds:** {seed_value}",
        f"  - **Steps:** {inference_steps}",
        f"  - **Audio Count:** {num_audios} audio(s)",
    ]
    info_parts.append("\n".join(summary_lines))
    
    # Combine all parts
    return "\n\n".join(info_parts)


def store_batch_in_queue(
    batch_queue,
    batch_index,
    audio_paths,
    generation_info,
    seeds,
    codes=None,
    scores=None,
    allow_lm_batch=False,
    batch_size=2,
    generation_params=None,
    lm_generated_metadata=None,
    status="completed"
):
    """Store batch results in queue with ALL generation parameters
    
    Args:
        codes: Audio codes used for generation (list for batch mode, string for single mode)
        scores: List of score displays for each audio (optional)
        allow_lm_batch: Whether batch LM mode was used for this batch
        batch_size: Batch size used for this batch
        generation_params: Complete dictionary of ALL generation parameters used
        lm_generated_metadata: LM-generated metadata for scoring (optional)
    """
    batch_queue[batch_index] = {
        "status": status,
        "audio_paths": audio_paths,
        "generation_info": generation_info,
        "seeds": seeds,
        "codes": codes,  # Store codes used for this batch
        "scores": scores if scores else [""] * 8,  # Store scores, default to empty
        "allow_lm_batch": allow_lm_batch,  # Store batch mode setting
        "batch_size": batch_size,  # Store batch size
        "generation_params": generation_params if generation_params else {},  # Store ALL parameters
        "lm_generated_metadata": lm_generated_metadata,  # Store LM metadata for scoring
        "timestamp": datetime.datetime.now().isoformat()
    }
    return batch_queue


def update_batch_indicator(current_batch, total_batches):
    """Update batch indicator text"""
    return t("results.batch_indicator", current=current_batch + 1, total=total_batches)


def update_navigation_buttons(current_batch, total_batches):
    """Determine navigation button states"""
    can_go_previous = current_batch > 0
    can_go_next = current_batch < total_batches - 1
    return can_go_previous, can_go_next

def send_audio_to_src_with_metadata(audio_file, lm_metadata):
    """Send generated audio file to src_audio input and populate metadata fields
    
    Args:
        audio_file: Audio file path
        lm_metadata: Dictionary containing LM-generated metadata
        
    Returns:
        Tuple of (audio_file, bpm, caption, lyrics, duration, key_scale, language, time_signature, is_format_caption)
    """
    if audio_file is None:
        return None, None, None, None, None, None, None, None, True  # Keep is_format_caption as True
    
    # Extract metadata fields if available
    bpm_value = None
    caption_value = None
    lyrics_value = None
    duration_value = None
    key_scale_value = None
    language_value = None
    time_signature_value = None
    
    if lm_metadata:
        # BPM
        if lm_metadata.get('bpm'):
            bpm_str = lm_metadata.get('bpm')
            if bpm_str and bpm_str != "N/A":
                try:
                    bpm_value = int(bpm_str)
                except (ValueError, TypeError):
                    pass
        
        # Caption (Rewritten Caption)
        if lm_metadata.get('caption'):
            caption_value = lm_metadata.get('caption')
        
        # Lyrics
        if lm_metadata.get('lyrics'):
            lyrics_value = lm_metadata.get('lyrics')
        
        # Duration
        if lm_metadata.get('duration'):
            duration_str = lm_metadata.get('duration')
            if duration_str and duration_str != "N/A":
                try:
                    duration_value = float(duration_str)
                except (ValueError, TypeError):
                    pass
        
        # KeyScale
        if lm_metadata.get('keyscale'):
            key_scale_str = lm_metadata.get('keyscale')
            if key_scale_str and key_scale_str != "N/A":
                key_scale_value = key_scale_str
        
        # Language
        if lm_metadata.get('language'):
            language_str = lm_metadata.get('language')
            if language_str and language_str != "N/A":
                language_value = language_str
        
        # Time Signature
        if lm_metadata.get('timesignature'):
            time_sig_str = lm_metadata.get('timesignature')
            if time_sig_str and time_sig_str != "N/A":
                time_signature_value = time_sig_str
    
    return (
        audio_file,
        bpm_value,
        caption_value,
        lyrics_value,
        duration_value,
        key_scale_value,
        language_value,
        time_signature_value,
        True  # Set is_format_caption to True (from LM-generated metadata)
    )


def generate_with_progress(
    dit_handler, llm_handler,
    captions, lyrics, bpm, key_scale, time_signature, vocal_language,
    inference_steps, guidance_scale, random_seed_checkbox, seed,
    reference_audio, audio_duration, batch_size_input, src_audio,
    text2music_audio_code_string, repainting_start, repainting_end,
    instruction_display_gen, audio_cover_strength, task_type,
    use_adg, cfg_interval_start, cfg_interval_end, shift, audio_format, lm_temperature,
    think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
    use_cot_metas, use_cot_caption, use_cot_language, is_format_caption,
    constrained_decoding_debug,
    allow_lm_batch,
    auto_score,
    score_scale,
    lm_batch_chunk_size,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate audio with progress tracking"""
    
    # step 1: prepare inputs
    # generate_music, GenerationParams, GenerationConfig
    gen_params = GenerationParams(
        task_type=task_type,
        instruction=instruction_display_gen,
        reference_audio=reference_audio,
        src_audio=src_audio,
        audio_codes=text2music_audio_code_string if not think_checkbox else "",
        caption=captions or "",
        lyrics=lyrics or "",
        instrumental=False,
        vocal_language=vocal_language,
        bpm=bpm,
        keyscale=key_scale,
        timesignature=time_signature,
        duration=audio_duration,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        use_adg=use_adg,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        shift=shift,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        audio_cover_strength=audio_cover_strength,
        thinking=think_checkbox,
        lm_temperature=lm_temperature,
        lm_cfg_scale=lm_cfg_scale,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        lm_negative_prompt=lm_negative_prompt,
        use_cot_metas=use_cot_metas,
        use_cot_caption=use_cot_caption,
        use_cot_language=use_cot_language,
        use_constrained_decoding=True,
    )
    # seed string to list
    if isinstance(seed, str) and seed.strip():
        if "," in seed:
            seed_list = [int(s.strip()) for s in seed.split(",")]
        else:
            seed_list = [int(seed.strip())]
    else:
        seed_list = None
    gen_config = GenerationConfig(
        batch_size=batch_size_input,
        allow_lm_batch=allow_lm_batch,
        use_random_seed=random_seed_checkbox,
        seeds=seed_list,
        lm_batch_chunk_size=lm_batch_chunk_size,
        constrained_decoding_debug=constrained_decoding_debug,
        audio_format=audio_format,
    )
    result = generate_music(
        dit_handler,
        llm_handler,
        params=gen_params,
        config=gen_config,
        progress=progress,
    )
    
    audio_outputs = [None] * 8
    all_audio_paths = []
    final_codes_list = [""] * 8
    final_scores_list = [""] * 8
    
    # Build generation_info from result data
    status_message = result.status_message
    seed_value_for_ui = result.extra_outputs.get("seed_value", "")
    lm_generated_metadata = result.extra_outputs.get("lm_metadata", {})
    time_costs = result.extra_outputs.get("time_costs", {}).copy()
    
    # Initialize post-processing timing
    audio_conversion_start_time = time_module.time()
    total_auto_score_time = 0.0
    
    align_score_1 = ""
    align_text_1 = ""
    align_plot_1 = None
    align_score_2 = ""
    align_text_2 = ""
    align_plot_2 = None
    updated_audio_codes = text2music_audio_code_string if not think_checkbox else ""
    
    # Build initial generation_info (will be updated with post-processing times at the end)
    generation_info = _build_generation_info(
        lm_metadata=lm_generated_metadata,
        time_costs=time_costs,
        seed_value=seed_value_for_ui,
        inference_steps=inference_steps,
        num_audios=len(result.audios) if result.success else 0,
    )
    
    if not result.success:
        yield (None,) * 8 + (None, generation_info, result.status_message) + (gr.skip(),) * 26
        return
    
    audios = result.audios
    progress(0.99, "Converting audio to mp3...")
    for i in range(8):
        if i < len(audios):
            key = audios[i]["key"]
            audio_tensor = audios[i]["tensor"]
            sample_rate = audios[i]["sample_rate"]
            audio_params = audios[i]["params"]
            temp_dir = tempfile.mkdtemp(f"acestep_gradio_results/")
            os.makedirs(temp_dir, exist_ok=True)
            json_path = os.path.join(temp_dir, f"{key}.json")
            audio_path = os.path.join(temp_dir, f"{key}.{audio_format}")
            save_audio(audio_data=audio_tensor, output_path=audio_path, sample_rate=sample_rate, format=audio_format, channels_first=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(audio_params, f, indent=2, ensure_ascii=False)
            audio_outputs[i] = audio_path
            all_audio_paths.append(audio_path)
            all_audio_paths.append(json_path)
            
            code_str = audio_params.get("audio_codes", "")
            final_codes_list[i] = code_str
            
            scores_ui_updates = [gr.skip()] * 8
            score_str = "Done!"
            if auto_score:
                auto_score_start = time_module.time()
                score_str = calculate_score_handler(llm_handler, code_str, captions, lyrics, lm_generated_metadata, bpm, key_scale, time_signature, audio_duration, vocal_language, score_scale)
                auto_score_end = time_module.time()
                total_auto_score_time += (auto_score_end - auto_score_start)
            scores_ui_updates[i] = score_str
            final_scores_list[i] = score_str
            
            status_message = f"Encoding & Ready: {i+1}/{len(audios)}"
            current_audio_updates = [gr.skip()] * 8
            current_audio_updates[i] = audio_path

            audio_codes_ui_updates = [gr.skip()] * 8
            audio_codes_ui_updates[i] = code_str
            yield (
                current_audio_updates[0], current_audio_updates[1], current_audio_updates[2], current_audio_updates[3],
                current_audio_updates[4], current_audio_updates[5], current_audio_updates[6], current_audio_updates[7],
                all_audio_paths,   # Real-time update of Batch File list
                generation_info,
                status_message,
                seed_value_for_ui,
                # Align plot placeholders (assume no need to update in real time)
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                # Scores
                scores_ui_updates[0], scores_ui_updates[1], scores_ui_updates[2], scores_ui_updates[3], scores_ui_updates[4], scores_ui_updates[5], scores_ui_updates[6], scores_ui_updates[7],
                updated_audio_codes,
                # Codes
                audio_codes_ui_updates[0], audio_codes_ui_updates[1], audio_codes_ui_updates[2], audio_codes_ui_updates[3],
                audio_codes_ui_updates[4], audio_codes_ui_updates[5], audio_codes_ui_updates[6], audio_codes_ui_updates[7],
                lm_generated_metadata,
                is_format_caption,
            )
        else:
            # If i exceeds the generated count (e.g., batch=2, i=2..7), do not yield
            pass
        time_module.sleep(0.1)
    
    # Record audio conversion time
    audio_conversion_end_time = time_module.time()
    audio_conversion_time = audio_conversion_end_time - audio_conversion_start_time
    
    # Add post-processing times to time_costs
    if audio_conversion_time > 0:
        time_costs['audio_conversion_time'] = audio_conversion_time
    if total_auto_score_time > 0:
        time_costs['auto_score_time'] = total_auto_score_time
    
    # Update pipeline total time to include post-processing
    if 'pipeline_total_time' in time_costs:
        time_costs['pipeline_total_time'] += audio_conversion_time + total_auto_score_time
    
    # Rebuild generation_info with complete timing information
    generation_info = _build_generation_info(
        lm_metadata=lm_generated_metadata,
        time_costs=time_costs,
        seed_value=seed_value_for_ui,
        inference_steps=inference_steps,
        num_audios=len(result.audios),
    )
    
    yield (
        gr.skip(), gr.skip(), gr.skip(), gr.skip(), # Audio 1-4: SKIP
        gr.skip(), gr.skip(), gr.skip(), gr.skip(), # Audio 5-8: SKIP
        all_audio_paths,
        generation_info,
        "Generation Complete",
        seed_value_for_ui,
        align_score_1, align_text_1, align_plot_1, align_score_2, align_text_2, align_plot_2,
        final_scores_list[0], final_scores_list[1], final_scores_list[2], final_scores_list[3],
        final_scores_list[4], final_scores_list[5], final_scores_list[6], final_scores_list[7],
        updated_audio_codes,
        final_codes_list[0], final_codes_list[1], final_codes_list[2], final_codes_list[3],
        final_codes_list[4], final_codes_list[5], final_codes_list[6], final_codes_list[7],
        lm_generated_metadata,
        is_format_caption,
    )



def calculate_score_handler(llm_handler, audio_codes_str, caption, lyrics, lm_metadata, bpm, key_scale, time_signature, audio_duration, vocal_language, score_scale):
    """
    Calculate PMI-based quality score for generated audio.
    
    PMI (Pointwise Mutual Information) removes condition bias:
    score = log P(condition|codes) - log P(condition)
    
    Args:
        llm_handler: LLM handler instance
        audio_codes_str: Generated audio codes string
        caption: Caption text used for generation
        lyrics: Lyrics text used for generation
        lm_metadata: LM-generated metadata dictionary (from CoT generation)
        bpm: BPM value
        key_scale: Key scale value
        time_signature: Time signature value
        audio_duration: Audio duration value
        vocal_language: Vocal language value
        score_scale: Sensitivity scale parameter
        
    Returns:
        Score display string
    """
    from acestep.test_time_scaling import calculate_pmi_score_per_condition
    
    if not llm_handler.llm_initialized:
        return t("messages.lm_not_initialized")
    
    if not audio_codes_str or not audio_codes_str.strip():
        return t("messages.no_codes")
    
    try:
        # Build metadata dictionary from both LM metadata and user inputs
        metadata = {}
        
        # Priority 1: Use LM-generated metadata if available
        if lm_metadata and isinstance(lm_metadata, dict):
            metadata.update(lm_metadata)
        
        # Priority 2: Add user-provided metadata (if not already in LM metadata)
        if bpm is not None and 'bpm' not in metadata:
            try:
                metadata['bpm'] = int(bpm)
            except:
                pass
        
        if caption and 'caption' not in metadata:
            metadata['caption'] = caption
        
        if audio_duration is not None and audio_duration > 0 and 'duration' not in metadata:
            try:
                metadata['duration'] = int(audio_duration)
            except:
                pass
        
        if key_scale and key_scale.strip() and 'keyscale' not in metadata:
            metadata['keyscale'] = key_scale.strip()
        
        if vocal_language and vocal_language.strip() and 'language' not in metadata:
            metadata['language'] = vocal_language.strip()
        
        if time_signature and time_signature.strip() and 'timesignature' not in metadata:
            metadata['timesignature'] = time_signature.strip()
        
        # Calculate per-condition scores with appropriate metrics
        # - Metadata fields (bpm, duration, etc.): Top-k recall
        # - Caption and lyrics: PMI (normalized)
        scores_per_condition, global_score, status = calculate_pmi_score_per_condition(
            llm_handler=llm_handler,
            audio_codes=audio_codes_str,
            caption=caption or "",
            lyrics=lyrics or "",
            metadata=metadata if metadata else None,
            temperature=1.0,
            topk=10,
            score_scale=score_scale
        )
        
        # Format display string with per-condition breakdown
        if global_score == 0.0 and not scores_per_condition:
            return t("messages.score_failed", error=status)
        else:
            # Build per-condition scores display
            condition_lines = []
            for condition_name, score_value in sorted(scores_per_condition.items()):
                condition_lines.append(
                    f"  • {condition_name}: {score_value:.4f}"
                )
            
            conditions_display = "\n".join(condition_lines) if condition_lines else "  (no conditions)"
            
            return (
                f"✅ Global Quality Score: {global_score:.4f} (0-1, higher=better)\n\n"
                f"📊 Per-Condition Scores (0-1):\n{conditions_display}\n\n"
                f"Note: Metadata uses Top-k Recall, Caption/Lyrics use PMI\n"
            )
            
    except Exception as e:
        import traceback
        error_msg = t("messages.score_error", error=str(e)) + f"\n{traceback.format_exc()}"
        return error_msg


def calculate_score_handler_with_selection(llm_handler, sample_idx, score_scale, current_batch_index, batch_queue):
    """
    Calculate PMI-based quality score - REFACTORED to read from batch_queue only.
    This ensures scoring uses the actual generation parameters, not current UI values.
    
    Args:
        llm_handler: LLM handler instance
        sample_idx: Which sample to score (1-8)
        score_scale: Sensitivity scale parameter (tool setting, can be from UI)
        current_batch_index: Current batch index
        batch_queue: Batch queue containing historical generation data
    """
    if current_batch_index not in batch_queue:
        return t("messages.scoring_failed"), batch_queue
    
    batch_data = batch_queue[current_batch_index]
    params = batch_data.get("generation_params", {})
    
    # Read ALL parameters from historical batch data
    caption = params.get("captions", "")
    lyrics = params.get("lyrics", "")
    bpm = params.get("bpm")
    key_scale = params.get("key_scale", "")
    time_signature = params.get("time_signature", "")
    audio_duration = params.get("audio_duration", -1)
    vocal_language = params.get("vocal_language", "")
    
    # Get LM metadata from batch_data (if it was saved during generation)
    lm_metadata = batch_data.get("lm_generated_metadata", None)
    
    # Get codes from batch_data
    stored_codes = batch_data.get("codes", "")
    stored_allow_lm_batch = batch_data.get("allow_lm_batch", False)
    
    # Select correct codes for this sample
    audio_codes_str = ""
    if stored_allow_lm_batch and isinstance(stored_codes, list):
        # Batch mode: use specific sample's codes
        if 0 <= sample_idx - 1 < len(stored_codes):
            code_item = stored_codes[sample_idx - 1]
            # Ensure it's a string (handle cases where dict was mistakenly stored)
            audio_codes_str = code_item if isinstance(code_item, str) else ""
    else:
        # Single mode: all samples use same codes
        audio_codes_str = stored_codes if isinstance(stored_codes, str) else ""
    
    # Calculate score using historical parameters
    score_display = calculate_score_handler(
        llm_handler,
        audio_codes_str, caption, lyrics, lm_metadata,
        bpm, key_scale, time_signature, audio_duration, vocal_language,
        score_scale
    )
    
    # Update batch_queue with the calculated score
    if current_batch_index in batch_queue:
        if "scores" not in batch_queue[current_batch_index]:
            batch_queue[current_batch_index]["scores"] = [""] * 8
        batch_queue[current_batch_index]["scores"][sample_idx - 1] = score_display
    
    return score_display, batch_queue


def capture_current_params(
    captions, lyrics, bpm, key_scale, time_signature, vocal_language,
    inference_steps, guidance_scale, random_seed_checkbox, seed,
    reference_audio, audio_duration, batch_size_input, src_audio,
    text2music_audio_code_string, repainting_start, repainting_end,
    instruction_display_gen, audio_cover_strength, task_type,
    use_adg, cfg_interval_start, cfg_interval_end, shift, audio_format, lm_temperature,
    think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
    use_cot_metas, use_cot_caption, use_cot_language,
    constrained_decoding_debug, allow_lm_batch, auto_score, score_scale, lm_batch_chunk_size,
    track_name, complete_track_classes
):
    """Capture current UI parameters for next batch generation
    
    IMPORTANT: For AutoGen batches, we clear audio codes to ensure:
    - Thinking mode: LM generates NEW codes for each batch
    - Non-thinking mode: DiT generates with different random seeds
    """
    return {
        "captions": captions,
        "lyrics": lyrics,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "vocal_language": vocal_language,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "random_seed_checkbox": True,  # Always use random for AutoGen batches
        "seed": seed,
        "reference_audio": reference_audio,
        "audio_duration": audio_duration,
        "batch_size_input": batch_size_input,
        "src_audio": src_audio,
        "text2music_audio_code_string": "",  # CLEAR codes for next batch! Let LM regenerate or DiT use new seeds
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "instruction_display_gen": instruction_display_gen,
        "audio_cover_strength": audio_cover_strength,
        "task_type": task_type,
        "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "shift": shift,
        "audio_format": audio_format,
        "lm_temperature": lm_temperature,
        "think_checkbox": think_checkbox,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption,
        "use_cot_language": use_cot_language,
        "constrained_decoding_debug": constrained_decoding_debug,
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score,
        "score_scale": score_scale,
        "lm_batch_chunk_size": lm_batch_chunk_size,
        "track_name": track_name,
        "complete_track_classes": complete_track_classes,
    }


def generate_with_batch_management(
    dit_handler, llm_handler,
    captions, lyrics, bpm, key_scale, time_signature, vocal_language,
    inference_steps, guidance_scale, random_seed_checkbox, seed,
    reference_audio, audio_duration, batch_size_input, src_audio,
    text2music_audio_code_string, repainting_start, repainting_end,
    instruction_display_gen, audio_cover_strength, task_type,
    use_adg, cfg_interval_start, cfg_interval_end, shift, audio_format, lm_temperature,
    think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
    use_cot_metas, use_cot_caption, use_cot_language, is_format_caption,
    constrained_decoding_debug,
    allow_lm_batch,
    auto_score,
    score_scale,
    lm_batch_chunk_size,
    track_name,
    complete_track_classes,
    autogen_checkbox,
    current_batch_index,
    total_batches,
    batch_queue,
    generation_params_state,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Wrapper for generate_with_progress that adds batch queue management
    """
    # Call the original generation function
    generator = generate_with_progress(
        dit_handler, llm_handler,
        captions, lyrics, bpm, key_scale, time_signature, vocal_language,
        inference_steps, guidance_scale, random_seed_checkbox, seed,
        reference_audio, audio_duration, batch_size_input, src_audio,
        text2music_audio_code_string, repainting_start, repainting_end,
        instruction_display_gen, audio_cover_strength, task_type,
        use_adg, cfg_interval_start, cfg_interval_end, shift, audio_format, lm_temperature,
        think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
        use_cot_metas, use_cot_caption, use_cot_language, is_format_caption,
        constrained_decoding_debug,
        allow_lm_batch,
        auto_score,
        score_scale,
        lm_batch_chunk_size,
        progress
    )
    final_result_from_inner = None
    for partial_result in generator:
        final_result_from_inner = partial_result
        # current_batch_index, total_batches, batch_queue, next_params, 
        # batch_indicator_text, prev_btn, next_btn, next_status, restore_btn
        yield partial_result + (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), 
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        )
    result = final_result_from_inner
    all_audio_paths = result[8]

    if all_audio_paths is None:
        
        yield result + (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), 
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        )
        return

    # Extract results from generation (使用 result 下标访问)
    generation_info = result[9]
    seed_value_for_ui = result[11]
    lm_generated_metadata = result[35]  # Fixed: lm_metadata is at index 35, not 34
    
    # Extract codes
    generated_codes_single = result[26]
    generated_codes_batch = [result[27], result[28], result[29], result[30], result[31], result[32], result[33], result[34]]

    # Determine which codes to store based on mode
    if allow_lm_batch and batch_size_input >= 2:
        codes_to_store = generated_codes_batch[:int(batch_size_input)]
    else:
        codes_to_store = generated_codes_single

    # Save parameters for history
    saved_params = {
        "captions": captions,
        "lyrics": lyrics,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "vocal_language": vocal_language,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "random_seed_checkbox": random_seed_checkbox,
        "seed": seed,
        "reference_audio": reference_audio,
        "audio_duration": audio_duration,
        "batch_size_input": batch_size_input,
        "src_audio": src_audio,
        "text2music_audio_code_string": text2music_audio_code_string,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "instruction_display_gen": instruction_display_gen,
        "audio_cover_strength": audio_cover_strength,
        "task_type": task_type,
        "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "shift": shift,
        "audio_format": audio_format,
        "lm_temperature": lm_temperature,
        "think_checkbox": think_checkbox,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption,
        "use_cot_language": use_cot_language,
        "constrained_decoding_debug": constrained_decoding_debug,
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score,
        "score_scale": score_scale,
        "lm_batch_chunk_size": lm_batch_chunk_size,
        "track_name": track_name,
        "complete_track_classes": complete_track_classes,
    }
    
    # Next batch parameters (with cleared codes & random seed)
    # Next batch parameters
    next_params = saved_params.copy()
    next_params["text2music_audio_code_string"] = ""
    next_params["random_seed_checkbox"] = True
    
    # Store current batch in queue
    batch_queue = store_batch_in_queue(
        batch_queue,
        current_batch_index,
        all_audio_paths,
        generation_info,
        seed_value_for_ui,
        codes=codes_to_store,
        allow_lm_batch=allow_lm_batch,
        batch_size=int(batch_size_input),
        generation_params=saved_params,
        lm_generated_metadata=lm_generated_metadata,
        status="completed"
    )
    
    # Update batch counters
    total_batches = max(total_batches, current_batch_index + 1)
    
    # Update batch indicator
    batch_indicator_text = update_batch_indicator(current_batch_index, total_batches)
    
    # Update navigation button states
    can_go_previous, can_go_next = update_navigation_buttons(current_batch_index, total_batches)
    
    # Prepare next batch status message
    next_batch_status_text = ""
    if autogen_checkbox:
        next_batch_status_text = t("messages.autogen_enabled")

    # 4. Yield final result (includes Batch UI updates)
    # The result here is already a tuple structure
    yield result + (
        current_batch_index,
        total_batches,
        batch_queue,
        next_params,
        batch_indicator_text,
        gr.update(interactive=can_go_previous),
        gr.update(interactive=can_go_next),
        next_batch_status_text,
        gr.update(interactive=True),
    )


def generate_next_batch_background(
    dit_handler,
    llm_handler,
    autogen_enabled,
    generation_params,
    current_batch_index,
    total_batches,
    batch_queue,
    is_format_caption,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Generate next batch in background if AutoGen is enabled
    """
    # Early return if AutoGen not enabled
    if not autogen_enabled:
        return (
            batch_queue,
            total_batches,
            "",
            gr.update(interactive=False),
        )
    
    # Calculate next batch index
    next_batch_idx = current_batch_index + 1
    
    # Check if next batch already exists
    if next_batch_idx in batch_queue and batch_queue[next_batch_idx].get("status") == "completed":
        return (
            batch_queue,
            total_batches,
            t("messages.batch_ready", n=next_batch_idx + 1),
            gr.update(interactive=True),
        )
    
    # Update total batches count
    total_batches = next_batch_idx + 1
    
    gr.Info(t("messages.batch_generating", n=next_batch_idx + 1))
    
    # Generate next batch using stored parameters
    params = generation_params.copy()
    
    # DEBUG LOGGING: Log all parameters used for background generation
    logger.info(f"========== BACKGROUND GENERATION BATCH {next_batch_idx + 1} ==========")
    logger.info(f"Parameters used for background generation:")
    logger.info(f"  - captions: {params.get('captions', 'N/A')}")
    logger.info(f"  - lyrics: {params.get('lyrics', 'N/A')[:50]}..." if params.get('lyrics') else "  - lyrics: N/A")
    logger.info(f"  - bpm: {params.get('bpm')}")
    logger.info(f"  - batch_size_input: {params.get('batch_size_input')}")
    logger.info(f"  - allow_lm_batch: {params.get('allow_lm_batch')}")
    logger.info(f"  - think_checkbox: {params.get('think_checkbox')}")
    logger.info(f"  - lm_temperature: {params.get('lm_temperature')}")
    logger.info(f"  - track_name: {params.get('track_name')}")
    logger.info(f"  - complete_track_classes: {params.get('complete_track_classes')}")
    logger.info(f"  - text2music_audio_code_string: {'<CLEARED>' if params.get('text2music_audio_code_string') == '' else 'HAS_VALUE'}")
    logger.info(f"=========================================================")
    
    # Add error handling for background generation
    try:
        # Ensure all parameters have default values to prevent None errors
        params.setdefault("captions", "")
        params.setdefault("lyrics", "")
        params.setdefault("bpm", None)
        params.setdefault("key_scale", "")
        params.setdefault("time_signature", "")
        params.setdefault("vocal_language", "unknown")
        params.setdefault("inference_steps", 8)
        params.setdefault("guidance_scale", 7.0)
        params.setdefault("random_seed_checkbox", True)
        params.setdefault("seed", "-1")
        params.setdefault("reference_audio", None)
        params.setdefault("audio_duration", -1)
        params.setdefault("batch_size_input", 2)
        params.setdefault("src_audio", None)
        params.setdefault("text2music_audio_code_string", "")
        params.setdefault("repainting_start", 0.0)
        params.setdefault("repainting_end", -1)
        params.setdefault("instruction_display_gen", "")
        params.setdefault("audio_cover_strength", 1.0)
        params.setdefault("task_type", "text2music")
        params.setdefault("use_adg", False)
        params.setdefault("cfg_interval_start", 0.0)
        params.setdefault("cfg_interval_end", 1.0)
        params.setdefault("shift", 1.0)
        params.setdefault("audio_format", "mp3")
        params.setdefault("lm_temperature", 0.85)
        params.setdefault("think_checkbox", True)
        params.setdefault("lm_cfg_scale", 2.0)
        params.setdefault("lm_top_k", 0)
        params.setdefault("lm_top_p", 0.9)
        params.setdefault("lm_negative_prompt", "NO USER INPUT")
        params.setdefault("use_cot_metas", True)
        params.setdefault("use_cot_caption", True)
        params.setdefault("use_cot_language", True)
        params.setdefault("constrained_decoding_debug", False)
        params.setdefault("allow_lm_batch", True)
        params.setdefault("auto_score", False)
        params.setdefault("score_scale", 0.5)
        params.setdefault("lm_batch_chunk_size", 8)
        params.setdefault("track_name", None)
        params.setdefault("complete_track_classes", [])
        
        # Call generate_with_progress with the saved parameters
        # Note: generate_with_progress is a generator, need to iterate through it
        generator = generate_with_progress(
            dit_handler,
            llm_handler,
            captions=params.get("captions"),
            lyrics=params.get("lyrics"),
            bpm=params.get("bpm"),
            key_scale=params.get("key_scale"),
            time_signature=params.get("time_signature"),
            vocal_language=params.get("vocal_language"),
            inference_steps=params.get("inference_steps"),
            guidance_scale=params.get("guidance_scale"),
            random_seed_checkbox=params.get("random_seed_checkbox"),
            seed=params.get("seed"),
            reference_audio=params.get("reference_audio"),
            audio_duration=params.get("audio_duration"),
            batch_size_input=params.get("batch_size_input"),
            src_audio=params.get("src_audio"),
            text2music_audio_code_string=params.get("text2music_audio_code_string"),
            repainting_start=params.get("repainting_start"),
            repainting_end=params.get("repainting_end"),
            instruction_display_gen=params.get("instruction_display_gen"),
            audio_cover_strength=params.get("audio_cover_strength"),
            task_type=params.get("task_type"),
            use_adg=params.get("use_adg"),
            cfg_interval_start=params.get("cfg_interval_start"),
            cfg_interval_end=params.get("cfg_interval_end"),
            shift=params.get("shift"),
            audio_format=params.get("audio_format"),
            lm_temperature=params.get("lm_temperature"),
            think_checkbox=params.get("think_checkbox"),
            lm_cfg_scale=params.get("lm_cfg_scale"),
            lm_top_k=params.get("lm_top_k"),
            lm_top_p=params.get("lm_top_p"),
            lm_negative_prompt=params.get("lm_negative_prompt"),
            use_cot_metas=params.get("use_cot_metas"),
            use_cot_caption=params.get("use_cot_caption"),
            use_cot_language=params.get("use_cot_language"),
            is_format_caption=is_format_caption,
            constrained_decoding_debug=params.get("constrained_decoding_debug"),
            allow_lm_batch=params.get("allow_lm_batch"),
            auto_score=params.get("auto_score"),
            score_scale=params.get("score_scale"),
            lm_batch_chunk_size=params.get("lm_batch_chunk_size"),
            progress=progress
        )
        
        # Consume generator to get final result (similar to generate_with_batch_management)
        final_result = None
        for partial_result in generator:
            final_result = partial_result
        
        # Extract results from final_result
        all_audio_paths = final_result[8]  # generated_audio_batch
        generation_info = final_result[9]
        seed_value_for_ui = final_result[11]
        lm_generated_metadata = final_result[35]  # Fixed: lm_metadata is at index 35, not 34
        
        # Extract codes
        generated_codes_single = final_result[26]
        generated_codes_batch = [final_result[27], final_result[28], final_result[29], final_result[30], final_result[31], final_result[32], final_result[33], final_result[34]]
        
        # Determine which codes to store
        batch_size = params.get("batch_size_input", 2)
        allow_lm_batch = params.get("allow_lm_batch", False)
        if allow_lm_batch and batch_size >= 2:
            codes_to_store = generated_codes_batch[:int(batch_size)]
        else:
            codes_to_store = generated_codes_single
        
        # DEBUG LOGGING: Log codes extraction and storage
        logger.info(f"Codes extraction for Batch {next_batch_idx + 1}:")
        logger.info(f"  - allow_lm_batch: {allow_lm_batch}")
        logger.info(f"  - batch_size: {batch_size}")
        logger.info(f"  - generated_codes_single exists: {bool(generated_codes_single)}")
        if isinstance(codes_to_store, list):
            logger.info(f"  - codes_to_store: LIST with {len(codes_to_store)} items")
            for idx, code in enumerate(codes_to_store):
                logger.info(f"    * Sample {idx + 1}: {len(code) if code else 0} chars")
        else:
            logger.info(f"  - codes_to_store: STRING with {len(codes_to_store) if codes_to_store else 0} chars")
        
        # Store next batch in queue with codes, batch settings, and ALL generation params
        batch_queue = store_batch_in_queue(
            batch_queue,
            next_batch_idx,
            all_audio_paths,
            generation_info,
            seed_value_for_ui,
            codes=codes_to_store,
            allow_lm_batch=allow_lm_batch,
            batch_size=int(batch_size),
            generation_params=params,
            lm_generated_metadata=lm_generated_metadata,
            status="completed"
        )
        
        logger.info(f"Batch {next_batch_idx + 1} stored in queue successfully")
        
        # Success message
        next_batch_status = t("messages.batch_ready", n=next_batch_idx + 1)
        
        # Enable next button now that batch is ready
        return (
            batch_queue,
            total_batches,
            next_batch_status,
            gr.update(interactive=True),
        )
    except Exception as e:
        # Handle generation errors
        import traceback
        error_msg = t("messages.batch_failed", error=str(e))
        gr.Warning(error_msg)
        
        # Mark batch as failed in queue
        batch_queue[next_batch_idx] = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        return (
            batch_queue,
            total_batches,
            error_msg,
            gr.update(interactive=False),
        )


def navigate_to_previous_batch(current_batch_index, batch_queue):
    """Navigate to previous batch (Result View Only - Never touches Input UI)"""
    if current_batch_index <= 0:
        gr.Warning(t("messages.at_first_batch"))
        return [gr.update()] * 24
    
    # Move to previous batch
    new_batch_index = current_batch_index - 1
    
    # Load batch data from queue
    if new_batch_index not in batch_queue:
        gr.Warning(t("messages.batch_not_found", n=new_batch_index + 1))
        return [gr.update()] * 24
    
    batch_data = batch_queue[new_batch_index]
    audio_paths = batch_data.get("audio_paths", [])
    generation_info_text = batch_data.get("generation_info", "")
    
    # Prepare audio outputs (up to 8)
    audio_outputs = [None] * 8
    real_audio_paths = [p for p in audio_paths if not p.lower().endswith('.json')]
    for idx in range(min(len(real_audio_paths), 8)):
        audio_outputs[idx] = real_audio_paths[idx]
    
    # Update batch indicator
    total_batches = len(batch_queue)
    batch_indicator_text = update_batch_indicator(new_batch_index, total_batches)
    
    # Update button states
    can_go_previous, can_go_next = update_navigation_buttons(new_batch_index, total_batches)
    
    # Restore score displays from batch queue
    stored_scores = batch_data.get("scores", [""] * 8)
    score_displays = stored_scores if stored_scores else [""] * 8
    
    return (
        audio_outputs[0], audio_outputs[1], audio_outputs[2], audio_outputs[3],
        audio_outputs[4], audio_outputs[5], audio_outputs[6], audio_outputs[7],
        audio_paths, generation_info_text, new_batch_index, batch_indicator_text,
        gr.update(interactive=can_go_previous), gr.update(interactive=can_go_next),
        t("messages.viewing_batch", n=new_batch_index + 1),
        score_displays[0], score_displays[1], score_displays[2], score_displays[3],
        score_displays[4], score_displays[5], score_displays[6], score_displays[7],
        gr.update(interactive=True),
    )


def navigate_to_next_batch(autogen_enabled, current_batch_index, total_batches, batch_queue):
    """Navigate to next batch (Result View Only - Never touches Input UI)"""
    if current_batch_index >= total_batches - 1:
        gr.Warning(t("messages.at_last_batch"))
        return [gr.update()] * 25
    
    # Move to next batch
    new_batch_index = current_batch_index + 1
    
    # Load batch data from queue
    if new_batch_index not in batch_queue:
        gr.Warning(t("messages.batch_not_found", n=new_batch_index + 1))
        return [gr.update()] * 25
    
    batch_data = batch_queue[new_batch_index]
    audio_paths = batch_data.get("audio_paths", [])
    generation_info_text = batch_data.get("generation_info", "")
    
    # Prepare audio outputs (up to 8)
    audio_outputs = [None] * 8
    real_audio_paths = [p for p in audio_paths if not p.lower().endswith('.json')]
    for idx in range(min(len(real_audio_paths), 8)):
        audio_outputs[idx] = real_audio_paths[idx]
    
    # Update batch indicator
    batch_indicator_text = update_batch_indicator(new_batch_index, total_batches)
    
    # Update button states
    can_go_previous, can_go_next = update_navigation_buttons(new_batch_index, total_batches)
    
    # Prepare next batch status message
    next_batch_status_text = ""
    is_latest_view = (new_batch_index == total_batches - 1)
    if autogen_enabled and is_latest_view:
        next_batch_status_text = "🔄 AutoGen will generate next batch in background..."
    
    # Restore score displays from batch queue
    stored_scores = batch_data.get("scores", [""] * 8)
    score_displays = stored_scores if stored_scores else [""] * 8
    
    return (
        audio_outputs[0], audio_outputs[1], audio_outputs[2], audio_outputs[3],
        audio_outputs[4], audio_outputs[5], audio_outputs[6], audio_outputs[7],
        audio_paths, generation_info_text, new_batch_index, batch_indicator_text,
        gr.update(interactive=can_go_previous), gr.update(interactive=can_go_next),
        t("messages.viewing_batch", n=new_batch_index + 1), next_batch_status_text,
        score_displays[0], score_displays[1], score_displays[2], score_displays[3],
        score_displays[4], score_displays[5], score_displays[6], score_displays[7],
        gr.update(interactive=True),
    )


def restore_batch_parameters(current_batch_index, batch_queue):
    """
    Restore parameters from currently viewed batch to Input UI.
    This is the bridge allowing users to "reuse" historical settings.
    """
    if current_batch_index not in batch_queue:
        gr.Warning(t("messages.no_batch_data"))
        return [gr.update()] * 29
    
    batch_data = batch_queue[current_batch_index]
    params = batch_data.get("generation_params", {})
    
    # Extract all parameters with defaults
    captions = params.get("captions", "")
    lyrics = params.get("lyrics", "")
    bpm = params.get("bpm", None)
    key_scale = params.get("key_scale", "")
    time_signature = params.get("time_signature", "")
    vocal_language = params.get("vocal_language", "unknown")
    audio_duration = params.get("audio_duration", -1)
    batch_size_input = params.get("batch_size_input", 2)
    inference_steps = params.get("inference_steps", 8)
    lm_temperature = params.get("lm_temperature", 0.85)
    lm_cfg_scale = params.get("lm_cfg_scale", 2.0)
    lm_top_k = params.get("lm_top_k", 0)
    lm_top_p = params.get("lm_top_p", 0.9)
    think_checkbox = params.get("think_checkbox", True)
    use_cot_caption = params.get("use_cot_caption", True)
    use_cot_language = params.get("use_cot_language", True)
    allow_lm_batch = params.get("allow_lm_batch", True)
    track_name = params.get("track_name", None)
    complete_track_classes = params.get("complete_track_classes", [])
    
    # Extract and process codes
    stored_codes = batch_data.get("codes", "")
    stored_allow_lm_batch = params.get("allow_lm_batch", False)
    
    codes_outputs = [""] * 9  # [Main, 1-8]
    if stored_codes:
        if stored_allow_lm_batch and isinstance(stored_codes, list):
            # Batch mode: populate codes 1-8, main shows first
            codes_outputs[0] = stored_codes[0] if stored_codes else ""
            for idx in range(min(len(stored_codes), 8)):
                codes_outputs[idx + 1] = stored_codes[idx]
        else:
            # Single mode: populate main, clear 1-8
            codes_outputs[0] = stored_codes if isinstance(stored_codes, str) else (stored_codes[0] if stored_codes else "")
    
    gr.Info(t("messages.params_restored", n=current_batch_index + 1))
    
    return (
        codes_outputs[0], codes_outputs[1], codes_outputs[2], codes_outputs[3],
        codes_outputs[4], codes_outputs[5], codes_outputs[6], codes_outputs[7],
        codes_outputs[8], captions, lyrics, bpm, key_scale, time_signature,
        vocal_language, audio_duration, batch_size_input, inference_steps,
        lm_temperature, lm_cfg_scale, lm_top_k, lm_top_p, think_checkbox,
        use_cot_caption, use_cot_language, allow_lm_batch,
        track_name, complete_track_classes
    )