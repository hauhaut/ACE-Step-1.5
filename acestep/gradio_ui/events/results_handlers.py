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
import gradio as gr
from loguru import logger
from acestep.gradio_ui.i18n import t


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


def save_audio_and_metadata(
    audio_path, task_type, captions, lyrics, vocal_language, bpm, key_scale, time_signature, audio_duration,
    batch_size_input, inference_steps, guidance_scale, seed, random_seed_checkbox,
    use_adg, cfg_interval_start, cfg_interval_end, audio_format,
    lm_temperature, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
    use_cot_caption, use_cot_language, audio_cover_strength,
    think_checkbox, text2music_audio_code_string, repainting_start, repainting_end,
    track_name, complete_track_classes, lm_metadata
):
    """Save audio file and its metadata as a zip package"""
    if audio_path is None:
        gr.Warning(t("messages.no_audio_to_save"))
        return None
    
    try:
        # Create metadata dictionary
        metadata = {
            "saved_at": datetime.datetime.now().isoformat(),
            "task_type": task_type,
            "caption": captions or "",
            "lyrics": lyrics or "",
            "vocal_language": vocal_language,
            "bpm": bpm if bpm is not None else None,
            "keyscale": key_scale or "",
            "timesignature": time_signature or "",
            "duration": audio_duration if audio_duration is not None else -1,
            "batch_size": batch_size_input,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "random_seed": False,  # Disable random seed for reproducibility
            "use_adg": use_adg,
            "cfg_interval_start": cfg_interval_start,
            "cfg_interval_end": cfg_interval_end,
            "audio_format": audio_format,
            "lm_temperature": lm_temperature,
            "lm_cfg_scale": lm_cfg_scale,
            "lm_top_k": lm_top_k,
            "lm_top_p": lm_top_p,
            "lm_negative_prompt": lm_negative_prompt,
            "use_cot_caption": use_cot_caption,
            "use_cot_language": use_cot_language,
            "audio_cover_strength": audio_cover_strength,
            "think": think_checkbox,
            "audio_codes": text2music_audio_code_string or "",
            "repainting_start": repainting_start,
            "repainting_end": repainting_end,
            "track_name": track_name,
            "complete_track_classes": complete_track_classes or [],
        }
        
        # Add LM-generated metadata if available
        if lm_metadata:
            metadata["lm_generated_metadata"] = lm_metadata
        
        # Generate timestamp and base name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract audio filename extension
        audio_ext = os.path.splitext(audio_path)[1]
        
        # Create temporary directory for packaging
        temp_dir = tempfile.mkdtemp()
        
        # Save JSON metadata
        json_path = os.path.join(temp_dir, f"metadata_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Copy audio file
        audio_copy_path = os.path.join(temp_dir, f"audio_{timestamp}{audio_ext}")
        shutil.copy2(audio_path, audio_copy_path)
        
        # Create zip file
        zip_path = os.path.join(tempfile.gettempdir(), f"music_package_{timestamp}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(audio_copy_path, os.path.basename(audio_copy_path))
            zipf.write(json_path, os.path.basename(json_path))
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        gr.Info(t("messages.save_success", filename=os.path.basename(zip_path)))
        return zip_path
        
    except Exception as e:
        gr.Warning(t("messages.save_failed", error=str(e)))
        import traceback
        traceback.print_exc()
        return None


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
    use_adg, cfg_interval_start, cfg_interval_end, audio_format, lm_temperature,
    think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
    use_cot_metas, use_cot_caption, use_cot_language, is_format_caption,
    constrained_decoding_debug,
    allow_lm_batch,
    auto_score,
    score_scale,
    lm_batch_chunk_size,
    progress=gr.Progress(track_tqdm=True)
):
    """Generate audio with progress tracking"""
    # If think is enabled (llm_dit mode) and use_cot_metas is True, generate audio codes using LM first
    audio_code_string_to_use = text2music_audio_code_string
    lm_generated_metadata = None  # Store LM-generated metadata for display
    lm_generated_audio_codes = None  # Store LM-generated audio codes for display
    lm_generated_audio_codes_list = []  # Store list of audio codes for batch processing
    
    # Determine if we should use batch LM generation
    should_use_lm_batch = (
        think_checkbox and
        llm_handler.llm_initialized and
        use_cot_metas and
        allow_lm_batch and
        batch_size_input >= 2
    )
    
    if think_checkbox and llm_handler.llm_initialized and use_cot_metas:
        # Convert top_k: 0 means None (disabled)
        top_k_value = None if lm_top_k == 0 else int(lm_top_k)
        # Convert top_p: 1.0 means None (disabled)
        top_p_value = None if lm_top_p >= 1.0 else lm_top_p
        
        # Build user_metadata from user-provided values (only include non-empty values)
        user_metadata = {}
        # Handle bpm: gr.Number can be None, int, float, or string
        if bpm is not None:
            try:
                bpm_value = float(bpm)
                if bpm_value > 0:
                    user_metadata['bpm'] = str(int(bpm_value))
            except (ValueError, TypeError):
                # If bpm is not a valid number, skip it
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
                # If audio_duration is not a valid number, skip it
                pass
        
        # Only pass user_metadata if user provided any values, otherwise let LM generate
        user_metadata_to_pass = user_metadata if user_metadata else None
        
        if should_use_lm_batch:
            # BATCH LM GENERATION
            logger.info(f"Using LM batch generation for {batch_size_input} items...")
            
            # Prepare seeds for batch items
            actual_seed_list, _ = dit_handler.prepare_seeds(batch_size_input, seed, random_seed_checkbox)
            
            # Split batch into chunks (GPU memory constraint)
            max_inference_batch_size = int(lm_batch_chunk_size)
            num_chunks = math.ceil(batch_size_input / max_inference_batch_size)
            
            all_metadata_list = []
            all_audio_codes_list = []
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * max_inference_batch_size
                chunk_end = min(chunk_start + max_inference_batch_size, batch_size_input)
                chunk_size = chunk_end - chunk_start
                chunk_seeds = actual_seed_list[chunk_start:chunk_end]
                
                logger.info(f"Generating LM batch chunk {chunk_idx+1}/{num_chunks} (size: {chunk_size}, seeds: {chunk_seeds})...")
                
                # Generate batch
                metadata_list, audio_codes_list, status = llm_handler.generate_with_stop_condition(
                    caption=captions or "",
                    lyrics=lyrics or "",
                    infer_type="llm_dit",
                    temperature=lm_temperature,
                    cfg_scale=lm_cfg_scale,
                    negative_prompt=lm_negative_prompt,
                    top_k=top_k_value,
                    top_p=top_p_value,
                    user_metadata=user_metadata_to_pass,
                    use_cot_caption=use_cot_caption,
                    use_cot_language=use_cot_language,
                    is_format_caption=is_format_caption,
                    constrained_decoding_debug=constrained_decoding_debug,
                    batch_size=chunk_size,
                    seeds=chunk_seeds,
                )
                
                all_metadata_list.extend(metadata_list)
                all_audio_codes_list.extend(audio_codes_list)
            
            # Use first metadata as representative (all are same)
            lm_generated_metadata = all_metadata_list[0] if all_metadata_list else None
            
            # Store audio codes list for later use
            lm_generated_audio_codes_list = all_audio_codes_list
            
            # Prepare audio codes for DiT (list of codes, one per batch item)
            audio_code_string_to_use = all_audio_codes_list
            
            # Update metadata fields from LM if not provided by user
            if lm_generated_metadata:
                if bpm is None and lm_generated_metadata.get('bpm'):
                    bpm_value = lm_generated_metadata.get('bpm')
                    if bpm_value != "N/A" and bpm_value != "":
                        try:
                            bpm = int(bpm_value)
                        except:
                            pass
                if not key_scale and lm_generated_metadata.get('keyscale'):
                    key_scale_value = lm_generated_metadata.get('keyscale', lm_generated_metadata.get('key_scale', ""))
                    if key_scale_value != "N/A":
                        key_scale = key_scale_value
                if not time_signature and lm_generated_metadata.get('timesignature'):
                    time_signature_value = lm_generated_metadata.get('timesignature', lm_generated_metadata.get('time_signature', ""))
                    if time_signature_value != "N/A":
                        time_signature = time_signature_value
                if audio_duration is None or audio_duration <= 0:
                    audio_duration_value = lm_generated_metadata.get('duration', -1)
                    if audio_duration_value != "N/A" and audio_duration_value != "":
                        try:
                            audio_duration = float(audio_duration_value)
                        except:
                            pass
        else:
            # SEQUENTIAL LM GENERATION (current behavior, when allow_lm_batch is False)
            # Phase 1: Generate CoT metadata
            phase1_start = time_module.time()
            metadata, _, status = llm_handler.generate_with_stop_condition(
                caption=captions or "",
                lyrics=lyrics or "",
                infer_type="dit",  # Only generate metadata in Phase 1
                temperature=lm_temperature,
                cfg_scale=lm_cfg_scale,
                negative_prompt=lm_negative_prompt,
                top_k=top_k_value,
                top_p=top_p_value,
                user_metadata=user_metadata_to_pass,
                use_cot_caption=use_cot_caption,
                use_cot_language=use_cot_language,
                is_format_caption=is_format_caption,
                constrained_decoding_debug=constrained_decoding_debug,
            )
            lm_phase1_time = time_module.time() - phase1_start
            logger.info(f"LM Phase 1 (CoT) completed in {lm_phase1_time:.2f}s")
            
            # Phase 2: Generate audio codes
            phase2_start = time_module.time()
            metadata, audio_codes, status = llm_handler.generate_with_stop_condition(
                caption=captions or "",
                lyrics=lyrics or "",
                infer_type="llm_dit",  # Generate both metadata and codes
                temperature=lm_temperature,
                cfg_scale=lm_cfg_scale,
                negative_prompt=lm_negative_prompt,
                top_k=top_k_value,
                top_p=top_p_value,
                user_metadata=user_metadata_to_pass,
                use_cot_caption=use_cot_caption,
                use_cot_language=use_cot_language,
                is_format_caption=is_format_caption,
                constrained_decoding_debug=constrained_decoding_debug,
            )
            lm_phase2_time = time_module.time() - phase2_start
            logger.info(f"LM Phase 2 (Codes) completed in {lm_phase2_time:.2f}s")
            
            # Store LM-generated metadata and audio codes for display
            lm_generated_metadata = metadata
            if audio_codes:
                audio_code_string_to_use = audio_codes
                lm_generated_audio_codes = audio_codes
                # Update metadata fields only if they are empty/None (user didn't provide them)
                if bpm is None and metadata.get('bpm'):
                    bpm_value = metadata.get('bpm')
                    if bpm_value != "N/A" and bpm_value != "":
                        try:
                            bpm = int(bpm_value)
                        except:
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
                    if audio_duration_value != "N/A" and audio_duration_value != "":
                        try:
                            audio_duration = float(audio_duration_value)
                        except:
                            pass
    
    # Call generate_music and get results
    result = dit_handler.generate_music(
        captions=captions, lyrics=lyrics, bpm=bpm, key_scale=key_scale,
        time_signature=time_signature, vocal_language=vocal_language,
        inference_steps=inference_steps, guidance_scale=guidance_scale,
        use_random_seed=random_seed_checkbox, seed=seed,
        reference_audio=reference_audio, audio_duration=audio_duration,
        batch_size=batch_size_input, src_audio=src_audio,
        audio_code_string=audio_code_string_to_use,
        repainting_start=repainting_start, repainting_end=repainting_end,
        instruction=instruction_display_gen, audio_cover_strength=audio_cover_strength,
        task_type=task_type, use_adg=use_adg,
        cfg_interval_start=cfg_interval_start, cfg_interval_end=cfg_interval_end,
        audio_format=audio_format, lm_temperature=lm_temperature,
        progress=progress
    )
    
    # Extract results from new dict structure
    if not isinstance(result, dict):
        # Fallback for old tuple format (should not happen)
        first_audio, second_audio, all_audio_paths, generation_info, status_message, seed_value_for_ui, \
            align_score_1, align_text_1, align_plot_1, align_score_2, align_text_2, align_plot_2 = result
    else:
        audios = result.get("audios", [])
        all_audio_paths = [audio.get("path") for audio in audios]
        first_audio = all_audio_paths[0] if len(all_audio_paths) > 0 else None
        second_audio = all_audio_paths[1] if len(all_audio_paths) > 1 else None
        generation_info = result.get("generation_info", "")
        status_message = result.get("status_message", "")
        seed_value_for_ui = result.get("extra_outputs", {}).get("seed_value", "")
        # Legacy alignment fields (no longer used)
        align_score_1 = ""
        align_text_1 = ""
        align_plot_1 = None
        align_score_2 = ""
        align_text_2 = ""
        align_plot_2 = None
    
    # Extract LM timing from status if available and prepend to generation_info
    if status:
        import re
        # Try to extract timing info from status using regex
        # Expected format: "Phase1: X.XXs" and "Phase2: X.XXs"
        phase1_match = re.search(r'Phase1:\s*([\d.]+)s', status)
        phase2_match = re.search(r'Phase2:\s*([\d.]+)s', status)
        
        if phase1_match or phase2_match:
            lm_timing_section = "\n\n**🤖 LM Timing:**\n"
            lm_total = 0.0
            if phase1_match:
                phase1_time = float(phase1_match.group(1))
                lm_timing_section += f"  - Phase 1 (CoT Metadata): {phase1_time:.2f}s\n"
                lm_total += phase1_time
            if phase2_match:
                phase2_time = float(phase2_match.group(1))
                lm_timing_section += f"  - Phase 2 (Audio Codes): {phase2_time:.2f}s\n"
                lm_total += phase2_time
            if lm_total > 0:
                lm_timing_section += f"  - Total LM Time: {lm_total:.2f}s\n"
            generation_info = lm_timing_section + "\n" + generation_info
    
    # Append LM-generated metadata to generation_info if available
    if lm_generated_metadata:
        metadata_lines = []
        if lm_generated_metadata.get('bpm'):
            metadata_lines.append(f"- **BPM:** {lm_generated_metadata['bpm']}")
        if lm_generated_metadata.get('caption'):
            metadata_lines.append(f"- **User Query Rewritten Caption:** {lm_generated_metadata['caption']}")
        if lm_generated_metadata.get('duration'):
            metadata_lines.append(f"- **Duration:** {lm_generated_metadata['duration']} seconds")
        if lm_generated_metadata.get('keyscale'):
            metadata_lines.append(f"- **KeyScale:** {lm_generated_metadata['keyscale']}")
        if lm_generated_metadata.get('language'):
            metadata_lines.append(f"- **Language:** {lm_generated_metadata['language']}")
        if lm_generated_metadata.get('timesignature'):
            metadata_lines.append(f"- **Time Signature:** {lm_generated_metadata['timesignature']}")
        
        if metadata_lines:
            metadata_section = "\n\n**🤖 LM-Generated Metadata:**\n" + "\n\n".join(metadata_lines)
            generation_info = metadata_section + "\n\n" + generation_info
    
    # Update audio codes in UI if LM generated them
    codes_outputs = [""] * 8  # Codes for 8 components
    if should_use_lm_batch and lm_generated_audio_codes_list:
        # Batch mode: update individual codes inputs
        for idx in range(min(len(lm_generated_audio_codes_list), 8)):
            codes_outputs[idx] = lm_generated_audio_codes_list[idx]
        # For single codes input, show first one
        updated_audio_codes = lm_generated_audio_codes_list[0] if lm_generated_audio_codes_list else text2music_audio_code_string
    else:
        # Single mode: update main codes input
        updated_audio_codes = lm_generated_audio_codes if lm_generated_audio_codes else text2music_audio_code_string
    
    # AUTO-SCORING
    score_displays = [""] * 8  # Scores for 8 components
    if auto_score and all_audio_paths:
        logger.info(f"Auto-scoring enabled, calculating quality scores for {batch_size_input} generated audios...")
        
        # Determine which audio codes to use for scoring
        if should_use_lm_batch and lm_generated_audio_codes_list:
            codes_list = lm_generated_audio_codes_list
        elif audio_code_string_to_use and isinstance(audio_code_string_to_use, list):
            codes_list = audio_code_string_to_use
        else:
            # Single code string, replicate for all audios
            codes_list = [audio_code_string_to_use] * len(all_audio_paths)
        
        # Calculate scores only for actually generated audios (up to batch_size_input)
        # Don't score beyond the actual batch size to avoid duplicates
        actual_audios_to_score = min(len(all_audio_paths), int(batch_size_input))
        for idx in range(actual_audios_to_score):
            if idx < len(codes_list) and codes_list[idx]:
                try:
                    score_display = calculate_score_handler(
                        llm_handler,
                        codes_list[idx],
                        captions,
                        lyrics,
                        lm_generated_metadata,
                        bpm, key_scale, time_signature, audio_duration, vocal_language,
                        score_scale
                    )
                    score_displays[idx] = score_display
                    logger.info(f"Auto-scored audio {idx+1}")
                except Exception as e:
                    logger.error(f"Auto-scoring failed for audio {idx+1}: {e}")
                    score_displays[idx] = f"❌ Auto-scoring failed: {str(e)}"
    
    # Prepare audio outputs (up to 8)
    audio_outputs = [None] * 8
    for idx in range(min(len(all_audio_paths), 8)):
        audio_outputs[idx] = all_audio_paths[idx]
    
    return (
        audio_outputs[0],  # generated_audio_1
        audio_outputs[1],  # generated_audio_2
        audio_outputs[2],  # generated_audio_3
        audio_outputs[3],  # generated_audio_4
        audio_outputs[4],  # generated_audio_5
        audio_outputs[5],  # generated_audio_6
        audio_outputs[6],  # generated_audio_7
        audio_outputs[7],  # generated_audio_8
        all_audio_paths,   # generated_audio_batch
        generation_info,
        status_message,
        seed_value_for_ui,
        align_score_1,
        align_text_1,
        align_plot_1,
        align_score_2,
        align_text_2,
        align_plot_2,
        score_displays[0],  # score_display_1
        score_displays[1],  # score_display_2
        score_displays[2],  # score_display_3
        score_displays[3],  # score_display_4
        score_displays[4],  # score_display_5
        score_displays[5],  # score_display_6
        score_displays[6],  # score_display_7
        score_displays[7],  # score_display_8
        updated_audio_codes,  # Update main audio codes in UI
        codes_outputs[0],  # text2music_audio_code_string_1
        codes_outputs[1],  # text2music_audio_code_string_2
        codes_outputs[2],  # text2music_audio_code_string_3
        codes_outputs[3],  # text2music_audio_code_string_4
        codes_outputs[4],  # text2music_audio_code_string_5
        codes_outputs[5],  # text2music_audio_code_string_6
        codes_outputs[6],  # text2music_audio_code_string_7
        codes_outputs[7],  # text2music_audio_code_string_8
        lm_generated_metadata,  # Store metadata for "Send to src audio" buttons
        is_format_caption,  # Keep is_format_caption unchanged
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
            audio_codes_str = stored_codes[sample_idx - 1]
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
    use_adg, cfg_interval_start, cfg_interval_end, audio_format, lm_temperature,
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
    use_adg, cfg_interval_start, cfg_interval_end, audio_format, lm_temperature,
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
    result = generate_with_progress(
        dit_handler, llm_handler,
        captions, lyrics, bpm, key_scale, time_signature, vocal_language,
        inference_steps, guidance_scale, random_seed_checkbox, seed,
        reference_audio, audio_duration, batch_size_input, src_audio,
        text2music_audio_code_string, repainting_start, repainting_end,
        instruction_display_gen, audio_cover_strength, task_type,
        use_adg, cfg_interval_start, cfg_interval_end, audio_format, lm_temperature,
        think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
        use_cot_metas, use_cot_caption, use_cot_language, is_format_caption,
        constrained_decoding_debug,
        allow_lm_batch,
        auto_score,
        score_scale,
        lm_batch_chunk_size,
        progress
    )
    
    # Extract results from generation
    all_audio_paths = result[8]  # generated_audio_batch
    generation_info = result[9]
    seed_value_for_ui = result[11]
    lm_generated_metadata = result[34]  # Index 34 is lm_metadata_state
    
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
    
    # Return original results plus batch management state updates
    return result + (
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
        result = generate_with_progress(
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
        
        # Extract results
        all_audio_paths = result[8]  # generated_audio_batch
        generation_info = result[9]
        seed_value_for_ui = result[11]
        lm_generated_metadata = result[34]  # Index 34 is lm_metadata_state
        
        # Extract codes
        generated_codes_single = result[26]
        generated_codes_batch = [result[27], result[28], result[29], result[30], result[31], result[32], result[33], result[34]]
        
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
    for idx in range(min(len(audio_paths), 8)):
        audio_outputs[idx] = audio_paths[idx]
    
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
    for idx in range(min(len(audio_paths), 8)):
        audio_outputs[idx] = audio_paths[idx]
    
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