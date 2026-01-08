"""
Gradio UI Event Handlers Module
Contains all event handler definitions and connections
"""
import os
import json
import random
import glob
import time as time_module
import tempfile
import gradio as gr
from typing import Optional
from acestep.constants import (
    TASK_TYPES_TURBO,
    TASK_TYPES_BASE,
)
from acestep.gradio_ui.i18n import t


def setup_event_handlers(demo, dit_handler, llm_handler, dataset_handler, dataset_section, generation_section, results_section):
    """Setup event handlers connecting UI components and business logic"""
    
    # Helper functions for batch queue management
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
        import datetime
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
        import datetime
        import shutil
        import zipfile
        
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
    
    def load_metadata(file_obj):
        """Load generation parameters from a JSON file"""
        if file_obj is None:
            gr.Warning(t("messages.no_file_selected"))
            return [None] * 31 + [False]  # Return None for all fields, False for is_format_caption
        
        try:
            # Read the uploaded file
            if hasattr(file_obj, 'name'):
                filepath = file_obj.name
            else:
                filepath = file_obj
            
            with open(filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Extract all fields
            task_type = metadata.get('task_type', 'text2music')
            captions = metadata.get('caption', '')
            lyrics = metadata.get('lyrics', '')
            vocal_language = metadata.get('vocal_language', 'unknown')
            
            # Convert bpm
            bpm_value = metadata.get('bpm')
            if bpm_value is not None and bpm_value != "N/A":
                try:
                    bpm = int(bpm_value) if bpm_value else None
                except:
                    bpm = None
            else:
                bpm = None
            
            key_scale = metadata.get('keyscale', '')
            time_signature = metadata.get('timesignature', '')
            
            # Convert duration
            duration_value = metadata.get('duration', -1)
            if duration_value is not None and duration_value != "N/A":
                try:
                    audio_duration = float(duration_value)
                except:
                    audio_duration = -1
            else:
                audio_duration = -1
            
            batch_size = metadata.get('batch_size', 2)
            inference_steps = metadata.get('inference_steps', 8)
            guidance_scale = metadata.get('guidance_scale', 7.0)
            seed = metadata.get('seed', '-1')
            random_seed = metadata.get('random_seed', True)
            use_adg = metadata.get('use_adg', False)
            cfg_interval_start = metadata.get('cfg_interval_start', 0.0)
            cfg_interval_end = metadata.get('cfg_interval_end', 1.0)
            audio_format = metadata.get('audio_format', 'mp3')
            lm_temperature = metadata.get('lm_temperature', 0.85)
            lm_cfg_scale = metadata.get('lm_cfg_scale', 2.0)
            lm_top_k = metadata.get('lm_top_k', 0)
            lm_top_p = metadata.get('lm_top_p', 0.9)
            lm_negative_prompt = metadata.get('lm_negative_prompt', 'NO USER INPUT')
            use_cot_caption = metadata.get('use_cot_caption', True)
            use_cot_language = metadata.get('use_cot_language', True)
            audio_cover_strength = metadata.get('audio_cover_strength', 1.0)
            think = metadata.get('think', True)
            audio_codes = metadata.get('audio_codes', '')
            repainting_start = metadata.get('repainting_start', 0.0)
            repainting_end = metadata.get('repainting_end', -1)
            track_name = metadata.get('track_name')
            complete_track_classes = metadata.get('complete_track_classes', [])
            
            gr.Info(t("messages.params_loaded", filename=os.path.basename(filepath)))
            
            return (
                task_type, captions, lyrics, vocal_language, bpm, key_scale, time_signature,
                audio_duration, batch_size, inference_steps, guidance_scale, seed, random_seed,
                use_adg, cfg_interval_start, cfg_interval_end, audio_format,
                lm_temperature, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
                use_cot_caption, use_cot_language, audio_cover_strength,
                think, audio_codes, repainting_start, repainting_end,
                track_name, complete_track_classes,
                True  # Set is_format_caption to True when loading from file
            )
            
        except json.JSONDecodeError as e:
            gr.Warning(t("messages.invalid_json", error=str(e)))
            return [None] * 31 + [False]
        except Exception as e:
            gr.Warning(t("messages.load_error", error=str(e)))
            return [None] * 31 + [False]
    
    def load_random_example(task_type: str):
        """Load a random example from the task-specific examples directory
        
        Args:
            task_type: The task type (e.g., "text2music")
            
        Returns:
            Tuple of (caption, lyrics, think, bpm, duration, keyscale, language, timesignature) for updating UI components
        """
        try:
            # Get the project root directory
            current_file = os.path.abspath(__file__)
            # event.py is in acestep/gradio_ui/, need 3 levels up to reach project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            
            # Construct the examples directory path
            examples_dir = os.path.join(project_root, "examples", task_type)
            
            # Check if directory exists
            if not os.path.exists(examples_dir):
                gr.Warning(f"Examples directory not found: examples/{task_type}/")
                return "", "", True, None, None, "", "", ""
            
            # Find all JSON files in the directory
            json_files = glob.glob(os.path.join(examples_dir, "*.json"))
            
            if not json_files:
                gr.Warning(f"No JSON files found in examples/{task_type}/")
                return "", "", True, None, None, "", "", ""
            
            # Randomly select one file
            selected_file = random.choice(json_files)
            
            # Read and parse JSON
            try:
                with open(selected_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract caption (prefer 'caption', fallback to 'prompt')
                caption_value = data.get('caption', data.get('prompt', ''))
                if not isinstance(caption_value, str):
                    caption_value = str(caption_value) if caption_value else ''
                
                # Extract lyrics
                lyrics_value = data.get('lyrics', '')
                if not isinstance(lyrics_value, str):
                    lyrics_value = str(lyrics_value) if lyrics_value else ''
                
                # Extract think (default to True if not present)
                think_value = data.get('think', True)
                if not isinstance(think_value, bool):
                    think_value = True
                
                # Extract optional metadata fields
                bpm_value = None
                if 'bpm' in data and data['bpm'] not in [None, "N/A", ""]:
                    try:
                        bpm_value = int(data['bpm'])
                    except (ValueError, TypeError):
                        pass
                
                duration_value = None
                if 'duration' in data and data['duration'] not in [None, "N/A", ""]:
                    try:
                        duration_value = float(data['duration'])
                    except (ValueError, TypeError):
                        pass
                
                keyscale_value = data.get('keyscale', '')
                if keyscale_value in [None, "N/A"]:
                    keyscale_value = ''
                
                language_value = data.get('language', '')
                if language_value in [None, "N/A"]:
                    language_value = ''
                
                timesignature_value = data.get('timesignature', '')
                if timesignature_value in [None, "N/A"]:
                    timesignature_value = ''
                
                gr.Info(t("messages.example_loaded", filename=os.path.basename(selected_file)))
                return caption_value, lyrics_value, think_value, bpm_value, duration_value, keyscale_value, language_value, timesignature_value
                
            except json.JSONDecodeError as e:
                gr.Warning(t("messages.example_failed", filename=os.path.basename(selected_file), error=str(e)))
                return "", "", True, None, None, "", "", ""
            except Exception as e:
                gr.Warning(t("messages.example_error", error=str(e)))
                return "", "", True, None, None, "", "", ""
                
        except Exception as e:
            gr.Warning(t("messages.example_error", error=str(e)))
            return "", "", True, None, None, "", "", ""
    
    def sample_example_smart(task_type: str, constrained_decoding_debug: bool = False):
        """Smart sample function that uses LM if initialized, otherwise falls back to examples
        
        Args:
            task_type: The task type (e.g., "text2music")
            constrained_decoding_debug: Whether to enable debug logging for constrained decoding
            
        Returns:
            Tuple of (caption, lyrics, think, bpm, duration, keyscale, language, timesignature) for updating UI components
        """
        # Check if LM is initialized
        if llm_handler.llm_initialized:
            # Use LM to generate example
            try:
                # Generate example using LM with empty input (NO USER INPUT)
                metadata, status = llm_handler.understand_audio_from_codes(
                    audio_codes="NO USER INPUT",
                    use_constrained_decoding=True,
                    temperature=0.85,
                    constrained_decoding_debug=constrained_decoding_debug,
                )
                
                if metadata:
                    caption_value = metadata.get('caption', '')
                    lyrics_value = metadata.get('lyrics', '')
                    think_value = True  # Always enable think when using LM-generated examples
                    
                    # Extract optional metadata fields
                    bpm_value = None
                    if 'bpm' in metadata and metadata['bpm'] not in [None, "N/A", ""]:
                        try:
                            bpm_value = int(metadata['bpm'])
                        except (ValueError, TypeError):
                            pass
                    
                    duration_value = None
                    if 'duration' in metadata and metadata['duration'] not in [None, "N/A", ""]:
                        try:
                            duration_value = float(metadata['duration'])
                        except (ValueError, TypeError):
                            pass
                    
                    keyscale_value = metadata.get('keyscale', '')
                    if keyscale_value in [None, "N/A"]:
                        keyscale_value = ''
                    
                    language_value = metadata.get('language', '')
                    if language_value in [None, "N/A"]:
                        language_value = ''
                    
                    timesignature_value = metadata.get('timesignature', '')
                    if timesignature_value in [None, "N/A"]:
                        timesignature_value = ''
                    
                    gr.Info(t("messages.lm_generated"))
                    return caption_value, lyrics_value, think_value, bpm_value, duration_value, keyscale_value, language_value, timesignature_value
                else:
                    gr.Warning(t("messages.lm_fallback"))
                    return load_random_example(task_type)
                    
            except Exception as e:
                gr.Warning(t("messages.lm_fallback"))
                return load_random_example(task_type)
        else:
            # LM not initialized, use examples directory
            return load_random_example(task_type)
    
    def update_init_status(status_msg, enable_btn):
        """Update initialization status and enable/disable generate button"""
        return status_msg, gr.update(interactive=enable_btn)
    
    # Dataset handlers
    dataset_section["import_dataset_btn"].click(
        fn=dataset_handler.import_dataset,
        inputs=[dataset_section["dataset_type"]],
        outputs=[dataset_section["data_status"]]
    )
    
    # Service initialization - refresh checkpoints
    def refresh_checkpoints():
        choices = dit_handler.get_available_checkpoints()
        return gr.update(choices=choices)
    
    generation_section["refresh_btn"].click(
        fn=refresh_checkpoints,
        outputs=[generation_section["checkpoint_dropdown"]]
    )
    
    # Update UI based on model type (turbo vs base)
    def update_model_type_settings(config_path):
        """Update UI settings based on model type"""
        if config_path is None:
            config_path = ""
        config_path_lower = config_path.lower()
        
        if "turbo" in config_path_lower:
            # Turbo model: max 8 steps, hide CFG/ADG, only show text2music/repaint/cover
            return (
                gr.update(value=8, maximum=8, minimum=1),  # inference_steps
                gr.update(visible=False),  # guidance_scale
                gr.update(visible=False),  # use_adg
                gr.update(visible=False),  # cfg_interval_start
                gr.update(visible=False),  # cfg_interval_end
                gr.update(choices=TASK_TYPES_TURBO),  # task_type
            )
        elif "base" in config_path_lower:
            # Base model: max 100 steps, show CFG/ADG, show all task types
            return (
                gr.update(value=32, maximum=100, minimum=1),  # inference_steps
                gr.update(visible=True),  # guidance_scale
                gr.update(visible=True),  # use_adg
                gr.update(visible=True),  # cfg_interval_start
                gr.update(visible=True),  # cfg_interval_end
                gr.update(choices=TASK_TYPES_BASE),  # task_type
            )
        else:
            # Default to turbo settings
            return (
                gr.update(value=8, maximum=8, minimum=1),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(choices=TASK_TYPES_TURBO),  # task_type
            )
    
    generation_section["config_path"].change(
        fn=update_model_type_settings,
        inputs=[generation_section["config_path"]],
        outputs=[
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["use_adg"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["task_type"],
        ]
    )
    
    # Service initialization
    def init_service_wrapper(checkpoint, config_path, device, init_llm, lm_model_path, backend, use_flash_attention, offload_to_cpu, offload_dit_to_cpu):
        """Wrapper for service initialization, returns status, button state, and accordion state"""
        # Initialize DiT handler
        status, enable = dit_handler.initialize_service(
            checkpoint, config_path, device,
            use_flash_attention=use_flash_attention, compile_model=False, 
            offload_to_cpu=offload_to_cpu, offload_dit_to_cpu=offload_dit_to_cpu
        )
        
        # Initialize LM handler if requested
        if init_llm:
            # Get checkpoint directory
            current_file = os.path.abspath(__file__)
            # event.py is in acestep/gradio_ui/, need 3 levels up to reach project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            checkpoint_dir = os.path.join(project_root, "checkpoints")
            
            lm_status, lm_success = llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend=backend,
                device=device,
                offload_to_cpu=offload_to_cpu,
                dtype=dit_handler.dtype
            )
            
            if lm_success:
                status += f"\n{lm_status}"
            else:
                status += f"\n{lm_status}"
                # Don't fail the entire initialization if LM fails, but log it
                # Keep enable as is (DiT initialization result) even if LM fails
        
        # Check if model is initialized - if so, collapse the accordion
        is_model_initialized = dit_handler.model is not None
        accordion_state = gr.update(open=not is_model_initialized)
        
        return status, gr.update(interactive=enable), accordion_state
    
    # Update negative prompt visibility based on "Initialize 5Hz LM" checkbox
    def update_negative_prompt_visibility(init_llm_checked):
        """Update negative prompt visibility: show if Initialize 5Hz LM checkbox is checked"""
        return gr.update(visible=init_llm_checked)
    
    # Update audio_cover_strength visibility and label based on task type and LM initialization
    def update_audio_cover_strength_visibility(task_type_value, init_llm_checked):
        """Update audio_cover_strength visibility and label"""
        # Show if task is cover OR if LM is initialized
        is_visible = (task_type_value == "cover") or init_llm_checked
        # Change label based on context
        if init_llm_checked and task_type_value != "cover":
            label = "LM codes strength"
            info = "Control how many denoising steps use LM-generated codes"
        else:
            label = "Audio Cover Strength"
            info = "Control how many denoising steps use cover mode"
        
        return gr.update(visible=is_visible, label=label, info=info)
    
    # Update visibility when init_llm_checkbox changes
    generation_section["init_llm_checkbox"].change(
        fn=update_negative_prompt_visibility,
        inputs=[generation_section["init_llm_checkbox"]],
        outputs=[generation_section["lm_negative_prompt"]]
    )
    
    # Update audio_cover_strength visibility and label when init_llm_checkbox changes
    generation_section["init_llm_checkbox"].change(
        fn=update_audio_cover_strength_visibility,
        inputs=[generation_section["task_type"], generation_section["init_llm_checkbox"]],
        outputs=[generation_section["audio_cover_strength"]]
    )
    
    # Also update audio_cover_strength when task_type changes (to handle label changes)
    generation_section["task_type"].change(
        fn=update_audio_cover_strength_visibility,
        inputs=[generation_section["task_type"], generation_section["init_llm_checkbox"]],
        outputs=[generation_section["audio_cover_strength"]]
    )
    
    generation_section["init_btn"].click(
        fn=init_service_wrapper,
        inputs=[
            generation_section["checkpoint_dropdown"],
            generation_section["config_path"],
            generation_section["device"],
            generation_section["init_llm_checkbox"],
            generation_section["lm_model_path"],
            generation_section["backend_dropdown"],
            generation_section["use_flash_attention_checkbox"],
            generation_section["offload_to_cpu_checkbox"],
            generation_section["offload_dit_to_cpu_checkbox"],
        ],
        outputs=[generation_section["init_status"], generation_section["generate_btn"], generation_section["service_config_accordion"]]
    )
    
    # Generation with progress bar
    def generate_with_progress(
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
                import math
                from loguru import logger
                
                logger.info(f"Using LM batch generation for {batch_size_input} items...")
                
                # Prepare seeds for batch items
                from acestep.handler import AceStepHandler
                temp_handler = AceStepHandler()
                actual_seed_list, _ = temp_handler.prepare_seeds(batch_size_input, seed, random_seed_checkbox)
                
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
                    metadata_list, audio_codes_list, status = llm_handler.generate_with_stop_condition_batch(
                        caption=captions or "",
                        lyrics=lyrics or "",
                        batch_size=chunk_size,
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
        
        # Pass LM timing to dit_handler.generate_music via generation_info
        # We'll add it to the result after getting it back
        
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
        
        # Extract results
        first_audio, second_audio, all_audio_paths, generation_info, status_message, seed_value_for_ui, \
            align_score_1, align_text_1, align_plot_1, align_score_2, align_text_2, align_plot_2 = result
        
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
            from loguru import logger
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
    
    # Helper function to capture current UI parameters - NOT NEEDED ANYMORE
    # Parameters are already captured during generate_with_batch_management
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
        track_name, complete_track_classes  # ADDED: missing parameters
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
            "track_name": track_name,  # ADDED
            "complete_track_classes": complete_track_classes,  # ADDED
        }
    
    # Wrapper function with batch queue management
    def generate_with_batch_management(
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
        track_name,  # ADDED: track name for lego/extract tasks
        complete_track_classes,  # ADDED: complete track classes
        autogen_checkbox,  # NEW: AutoGen checkbox state
        current_batch_index,  # NEW: Current batch index
        total_batches,  # NEW: Total batches
        batch_queue,  # NEW: Batch queue
        generation_params_state,  # NEW: Generation parameters state
        progress=gr.Progress(track_tqdm=True)
    ):
        """
        Wrapper for generate_with_progress that adds batch queue management
        """
        # Call the original generation function
        result = generate_with_progress(
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
        
        # --- FIXED: Corrected index offsets for codes extraction ---
        # Index 25 is score_display_8
        # Index 26 is updated_audio_codes (Single)
        # Index 27-34 are codes_outputs[0] through codes_outputs[7] (Batch 1-8)
        generated_codes_single = result[26]
        generated_codes_batch = [result[27], result[28], result[29], result[30], result[31], result[32], result[33], result[34]]
        
        # Determine which codes to store based on mode
        if allow_lm_batch and batch_size_input >= 2:
            # Batch mode: store list of codes
            codes_to_store = generated_codes_batch[:int(batch_size_input)]
        else:
            # Single mode: store single code string
            codes_to_store = generated_codes_single
        
        # --- OPTIMIZATION: Separate "saved params" (for history) and "next params" (for AutoGen) ---
        
        # 1. Real historical parameters (for storage in Queue, for accurate restoration)
        # These record the actual parameter state used for this generation
        saved_params = {
            "captions": captions,
            "lyrics": lyrics,
            "bpm": bpm,
            "key_scale": key_scale,
            "time_signature": time_signature,
            "vocal_language": vocal_language,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "random_seed_checkbox": random_seed_checkbox,  # Save real checkbox state
            "seed": seed,
            "reference_audio": reference_audio,
            "audio_duration": audio_duration,
            "batch_size_input": batch_size_input,
            "src_audio": src_audio,
            "text2music_audio_code_string": text2music_audio_code_string,  # Save real input
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
        
        # 2. Next batch parameters (for background AutoGen)
        # Based on current params, but clear codes and force random seeds to generate new content
        next_params = saved_params.copy()
        next_params["text2music_audio_code_string"] = ""  # CLEAR! Let LM regenerate or DiT use new seeds
        next_params["random_seed_checkbox"] = True        # Always use random for next batch
        
        # Store current batch in queue using saved_params (real historical snapshot)
        batch_queue = store_batch_in_queue(
            batch_queue,
            current_batch_index,
            all_audio_paths,
            generation_info,
            seed_value_for_ui,
            codes=codes_to_store,  # Store the codes used for this batch
            allow_lm_batch=allow_lm_batch,  # Store batch mode setting
            batch_size=int(batch_size_input),  # Store batch size
            generation_params=saved_params,  # <-- Use saved_params for accurate history
            lm_generated_metadata=lm_generated_metadata,  # Store LM metadata for scoring
            status="completed"
        )
        
        # Update batch counters (start with 1 batch)
        # Don't increment total_batches yet - will do that when next batch starts generating
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
            current_batch_index,  # Keep current batch index unchanged (still on batch 0)
            total_batches,  # Updated total batches
            batch_queue,  # Updated batch queue
            next_params,  # Pass next_params for background generation (with cleared codes & random seed)
            batch_indicator_text,  # Update batch indicator
            gr.update(interactive=can_go_previous),  # prev_batch_btn
            gr.update(interactive=can_go_next),  # next_batch_btn
            next_batch_status_text,  # next_batch_status
            gr.update(interactive=True),  # restore_params_btn - Enable after generation
        )
    
    # Background generation function
    def generate_next_batch_background(
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
        from loguru import logger
        
        # Early return if AutoGen not enabled
        if not autogen_enabled:
            return (
                batch_queue,
                total_batches,
                "",  # next_batch_status
                gr.update(interactive=False),  # keep next_batch_btn disabled
            )
        
        # Calculate next batch index
        next_batch_idx = current_batch_index + 1
        
        # Check if next batch already exists
        if next_batch_idx in batch_queue and batch_queue[next_batch_idx].get("status") == "completed":
            # Next batch already generated, enable button
            return (
                batch_queue,
                total_batches,
                t("messages.batch_ready", n=next_batch_idx + 1),
                gr.update(interactive=True),
            )
        
        # Update total batches count
        total_batches = next_batch_idx + 1
        
        # Update status to show generation starting
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
            
            # --- FIXED: Corrected index offsets for codes extraction ---
            # Index 25 is score_display_8
            # Index 26 is updated_audio_codes (Single)
            # Index 27-34 are codes_outputs[0] through codes_outputs[7] (Batch 1-8)
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
                codes=codes_to_store,  # Store codes
                allow_lm_batch=allow_lm_batch,  # Store batch mode setting
                batch_size=int(batch_size),  # Store batch size
                generation_params=params,  # Store ALL generation parameters used
                lm_generated_metadata=lm_generated_metadata,  # Store LM metadata for scoring
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
                gr.update(interactive=True),  # Enable next_batch_btn
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
                gr.update(interactive=False),  # Keep next_batch_btn disabled on error
            )
    
    # Wire up generation button with background generation chaining
    generation_section["generate_btn"].click(
        fn=generate_with_batch_management,
        inputs=[
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["vocal_language"],
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["random_seed_checkbox"],
            generation_section["seed"],
            generation_section["reference_audio"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["src_audio"],
            generation_section["text2music_audio_code_string"],
            generation_section["repainting_start"],
            generation_section["repainting_end"],
            generation_section["instruction_display_gen"],
            generation_section["audio_cover_strength"],
            generation_section["task_type"],
            generation_section["use_adg"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["audio_format"],
            generation_section["lm_temperature"],
            generation_section["think_checkbox"],
            generation_section["lm_cfg_scale"],
            generation_section["lm_top_k"],
            generation_section["lm_top_p"],
            generation_section["lm_negative_prompt"],
            generation_section["use_cot_metas"],
            generation_section["use_cot_caption"],
            generation_section["use_cot_language"],
            results_section["is_format_caption_state"],
            generation_section["constrained_decoding_debug"],
            generation_section["allow_lm_batch"],
            generation_section["auto_score"],
            generation_section["score_scale"],
            generation_section["lm_batch_chunk_size"],
            generation_section["track_name"],  # ADDED: For lego/extract tasks
            generation_section["complete_track_classes"],  # ADDED: For complete task
            generation_section["autogen_checkbox"],  # NEW: AutoGen checkbox
            results_section["current_batch_index"],  #NEW: Current batch index
            results_section["total_batches"],  # NEW: Total batches
            results_section["batch_queue"],  # NEW: Batch queue
            results_section["generation_params_state"],  # NEW: Generation parameters
        ],
        outputs=[
            results_section["generated_audio_1"],
            results_section["generated_audio_2"],
            results_section["generated_audio_3"],
            results_section["generated_audio_4"],
            results_section["generated_audio_5"],
            results_section["generated_audio_6"],
            results_section["generated_audio_7"],
            results_section["generated_audio_8"],
            results_section["generated_audio_batch"],
            results_section["generation_info"],
            results_section["status_output"],
            generation_section["seed"],
            results_section["align_score_1"],
            results_section["align_text_1"],
            results_section["align_plot_1"],
            results_section["align_score_2"],
            results_section["align_text_2"],
            results_section["align_plot_2"],
            results_section["score_display_1"],
            results_section["score_display_2"],
            results_section["score_display_3"],
            results_section["score_display_4"],
            results_section["score_display_5"],
            results_section["score_display_6"],
            results_section["score_display_7"],
            results_section["score_display_8"],
            generation_section["text2music_audio_code_string"],  # Update main audio codes display
            generation_section["text2music_audio_code_string_1"],  # Update codes for sample 1
            generation_section["text2music_audio_code_string_2"],  # Update codes for sample 2
            generation_section["text2music_audio_code_string_3"],  # Update codes for sample 3
            generation_section["text2music_audio_code_string_4"],  # Update codes for sample 4
            generation_section["text2music_audio_code_string_5"],  # Update codes for sample 5
            generation_section["text2music_audio_code_string_6"],  # Update codes for sample 6
            generation_section["text2music_audio_code_string_7"],  # Update codes for sample 7
            generation_section["text2music_audio_code_string_8"],  # Update codes for sample 8
            results_section["lm_metadata_state"],  # Store metadata
            results_section["is_format_caption_state"],  # Update is_format_caption state
            results_section["current_batch_index"],  # NEW: Update current batch index
            results_section["total_batches"],  # NEW: Update total batches
            results_section["batch_queue"],  # NEW: Update batch queue
            results_section["generation_params_state"],  # NEW: Update generation params
            results_section["batch_indicator"],  # NEW: Update batch indicator
            results_section["prev_batch_btn"],  # NEW: Update prev button state
            results_section["next_batch_btn"],  # NEW: Update next button state
            results_section["next_batch_status"],  # NEW: Update next batch status
            results_section["restore_params_btn"],  # NEW: Enable restore button after generation
        ]
    ).then(
        # Chain background generation with parameters already stored by generate_with_batch_management
        # NOTE: No need to capture_current_params again - already stored at generation time
        fn=generate_next_batch_background,
        inputs=[
            generation_section["autogen_checkbox"],
            results_section["generation_params_state"],  # Use params from generate_with_batch_management
            results_section["current_batch_index"],
            results_section["total_batches"],
            results_section["batch_queue"],
            results_section["is_format_caption_state"],
        ],
        outputs=[
            results_section["batch_queue"],
            results_section["total_batches"],
            results_section["next_batch_status"],
            results_section["next_batch_btn"],
        ]
    )
    
    # Update audio components visibility based on batch size
    def update_audio_components_visibility(batch_size):
        """Show/hide individual audio components based on batch size (1-8)
        
        Row 1: Components 1-4 (batch_size 1-4)
        Row 2: Components 5-8 (batch_size 5-8)
        """
        # Clamp batch size to 1-8 range for UI
        batch_size = min(max(int(batch_size), 1), 8)
        
        # Row 1 columns (1-4)
        updates_row1 = (
            gr.update(visible=True),  # audio_col_1: always visible
            gr.update(visible=batch_size >= 2),  # audio_col_2
            gr.update(visible=batch_size >= 3),  # audio_col_3
            gr.update(visible=batch_size >= 4),  # audio_col_4
        )
        
        # Row 2 container and columns (5-8)
        show_row_5_8 = batch_size >= 5
        updates_row2 = (
            gr.update(visible=show_row_5_8),  # audio_row_5_8 (container)
            gr.update(visible=batch_size >= 5),  # audio_col_5
            gr.update(visible=batch_size >= 6),  # audio_col_6
            gr.update(visible=batch_size >= 7),  # audio_col_7
            gr.update(visible=batch_size >= 8),  # audio_col_8
        )
        
        return updates_row1 + updates_row2
    
    generation_section["batch_size_input"].change(
        fn=update_audio_components_visibility,
        inputs=[generation_section["batch_size_input"]],
        outputs=[
            # Row 1 (1-4)
            results_section["audio_col_1"],
            results_section["audio_col_2"],
            results_section["audio_col_3"],
            results_section["audio_col_4"],
            # Row 2 container and columns (5-8)
            results_section["audio_row_5_8"],
            results_section["audio_col_5"],
            results_section["audio_col_6"],
            results_section["audio_col_7"],
            results_section["audio_col_8"],
        ]
    )
    
    # Update LM codes hints display based on src_audio, allow_lm_batch and batch_size
    def update_codes_hints_visibility(src_audio, allow_lm_batch, batch_size):
        """Switch between single/batch codes input based on src_audio presence
        
        When src_audio is present:
            - Show single mode with transcribe button
            - Clear codes (will be filled by transcription)
        
        When src_audio is absent:
            - Hide transcribe button
            - Show batch mode if allow_lm_batch=True and batch_size>=2
            - Show single mode otherwise
        
        Row 1: Codes 1-4
        Row 2: Codes 5-8 (batch_size >= 5)
        """
        batch_size = min(max(int(batch_size), 1), 8)
        has_src_audio = src_audio is not None
        
        if has_src_audio:
            # Has src_audio: show single mode with transcribe button
            return (
                gr.update(visible=True),   # codes_single_row
                gr.update(visible=False),  # codes_batch_row
                gr.update(visible=False),  # codes_batch_row_2
                *[gr.update(visible=False)] * 8,  # Hide all batch columns
                gr.update(visible=True),   # transcribe_btn: show when src_audio present
            )
        else:
            # No src_audio: decide between single/batch mode based on settings
            if allow_lm_batch and batch_size >= 2:
                # Batch mode: hide single, show batch codes with dynamic columns
                show_row_2 = batch_size >= 5
                return (
                    gr.update(visible=False),  # codes_single_row
                    gr.update(visible=True),   # codes_batch_row (row 1)
                    gr.update(visible=show_row_2),  # codes_batch_row_2 (row 2)
                    # Row 1 columns (1-4)
                    gr.update(visible=True),   # codes_col_1: always visible in batch mode
                    gr.update(visible=batch_size >= 2),  # codes_col_2
                    gr.update(visible=batch_size >= 3),  # codes_col_3
                    gr.update(visible=batch_size >= 4),  # codes_col_4
                    # Row 2 columns (5-8)
                    gr.update(visible=batch_size >= 5),  # codes_col_5
                    gr.update(visible=batch_size >= 6),  # codes_col_6
                    gr.update(visible=batch_size >= 7),  # codes_col_7
                    gr.update(visible=batch_size >= 8),  # codes_col_8
                    gr.update(visible=False),  # transcribe_btn: hide when no src_audio
                )
            else:
                # Single mode: show single, hide batch
                return (
                    gr.update(visible=True),   # codes_single_row
                    gr.update(visible=False),  # codes_batch_row
                    gr.update(visible=False),  # codes_batch_row_2
                    *[gr.update(visible=False)] * 8,  # Hide all batch columns
                    gr.update(visible=False),  # transcribe_btn: hide when no src_audio
                )
    
    # Update codes hints when src_audio, allow_lm_batch, or batch_size changes
    generation_section["src_audio"].change(
        fn=update_codes_hints_visibility,
        inputs=[
            generation_section["src_audio"],
            generation_section["allow_lm_batch"],
            generation_section["batch_size_input"]
        ],
        outputs=[
            generation_section["codes_single_row"],
            generation_section["codes_batch_row"],
            generation_section["codes_batch_row_2"],
            # Row 1
            generation_section["codes_col_1"],
            generation_section["codes_col_2"],
            generation_section["codes_col_3"],
            generation_section["codes_col_4"],
            # Row 2
            generation_section["codes_col_5"],
            generation_section["codes_col_6"],
            generation_section["codes_col_7"],
            generation_section["codes_col_8"],
            generation_section["transcribe_btn"],
        ]
    )
    
    generation_section["allow_lm_batch"].change(
        fn=update_codes_hints_visibility,
        inputs=[
            generation_section["src_audio"],
            generation_section["allow_lm_batch"],
            generation_section["batch_size_input"]
        ],
        outputs=[
            generation_section["codes_single_row"],
            generation_section["codes_batch_row"],
            generation_section["codes_batch_row_2"],
            # Row 1
            generation_section["codes_col_1"],
            generation_section["codes_col_2"],
            generation_section["codes_col_3"],
            generation_section["codes_col_4"],
            # Row 2
            generation_section["codes_col_5"],
            generation_section["codes_col_6"],
            generation_section["codes_col_7"],
            generation_section["codes_col_8"],
            generation_section["transcribe_btn"],
        ]
    )
    
    # Also update codes hints when batch_size changes
    generation_section["batch_size_input"].change(
        fn=update_codes_hints_visibility,
        inputs=[
            generation_section["src_audio"],
            generation_section["allow_lm_batch"],
            generation_section["batch_size_input"]
        ],
        outputs=[
            generation_section["codes_single_row"],
            generation_section["codes_batch_row"],
            generation_section["codes_batch_row_2"],
            # Row 1
            generation_section["codes_col_1"],
            generation_section["codes_col_2"],
            generation_section["codes_col_3"],
            generation_section["codes_col_4"],
            # Row 2
            generation_section["codes_col_5"],
            generation_section["codes_col_6"],
            generation_section["codes_col_7"],
            generation_section["codes_col_8"],
            generation_section["transcribe_btn"],
        ]
    )
    
    # Convert src audio to codes
    def convert_src_audio_to_codes_wrapper(src_audio):
        """Wrapper for converting src audio to codes"""
        codes_string = dit_handler.convert_src_audio_to_codes(src_audio)
        return codes_string
    
    generation_section["convert_src_to_codes_btn"].click(
        fn=convert_src_audio_to_codes_wrapper,
        inputs=[generation_section["src_audio"]],
        outputs=[generation_section["text2music_audio_code_string"]]
    )
    
    # Update instruction and UI visibility based on task type
    def update_instruction_ui(
        task_type_value: str, 
        track_name_value: Optional[str], 
        complete_track_classes_value: list, 
        audio_codes_content: str = "",
        init_llm_checked: bool = False
    ) -> tuple:
        """Update instruction and UI visibility based on task type."""
        instruction = dit_handler.generate_instruction(
            task_type=task_type_value,
            track_name=track_name_value,
            complete_track_classes=complete_track_classes_value
        )
        
        # Show track_name for lego and extract
        track_name_visible = task_type_value in ["lego", "extract"]
        # Show complete_track_classes for complete
        complete_visible = task_type_value == "complete"
        # Show audio_cover_strength for cover OR when LM is initialized
        audio_cover_strength_visible = (task_type_value == "cover") or init_llm_checked
        # Determine label and info based on context
        if init_llm_checked and task_type_value != "cover":
            audio_cover_strength_label = "LM codes strength"
            audio_cover_strength_info = "Control how many denoising steps use LM-generated codes"
        else:
            audio_cover_strength_label = "Audio Cover Strength"
            audio_cover_strength_info = "Control how many denoising steps use cover mode"
        # Show repainting controls for repaint and lego
        repainting_visible = task_type_value in ["repaint", "lego"]
        # Show text2music_audio_codes if task is text2music OR if it has content
        # This allows it to stay visible even if user switches task type but has codes
        has_audio_codes = audio_codes_content and str(audio_codes_content).strip()
        text2music_audio_codes_visible = task_type_value == "text2music" or has_audio_codes
        
        return (
            instruction,  # instruction_display_gen
            gr.update(visible=track_name_visible),  # track_name
            gr.update(visible=complete_visible),  # complete_track_classes
            gr.update(visible=audio_cover_strength_visible, label=audio_cover_strength_label, info=audio_cover_strength_info),  # audio_cover_strength
            gr.update(visible=repainting_visible),  # repainting_group
            gr.update(visible=text2music_audio_codes_visible),  # text2music_audio_codes_group
        )
    
    # Bind update_instruction_ui to task_type, track_name, and complete_track_classes changes
    generation_section["task_type"].change(
        fn=update_instruction_ui,
        inputs=[
            generation_section["task_type"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["text2music_audio_code_string"],
            generation_section["init_llm_checkbox"]
        ],
        outputs=[
            generation_section["instruction_display_gen"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["audio_cover_strength"],
            generation_section["repainting_group"],
            generation_section["text2music_audio_codes_group"],
        ]
    )
    
    # Also update instruction when track_name changes (for lego/extract tasks)
    generation_section["track_name"].change(
        fn=update_instruction_ui,
        inputs=[
            generation_section["task_type"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["text2music_audio_code_string"],
            generation_section["init_llm_checkbox"]
        ],
        outputs=[
            generation_section["instruction_display_gen"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["audio_cover_strength"],
            generation_section["repainting_group"],
            generation_section["text2music_audio_codes_group"],
        ]
    )
    
    # Also update instruction when complete_track_classes changes (for complete task)
    generation_section["complete_track_classes"].change(
        fn=update_instruction_ui,
        inputs=[
            generation_section["task_type"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["text2music_audio_code_string"],
            generation_section["init_llm_checkbox"]
        ],
        outputs=[
            generation_section["instruction_display_gen"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["audio_cover_strength"],
            generation_section["repainting_group"],
            generation_section["text2music_audio_codes_group"],
        ]
    )
    
    # Send generated audio to src_audio and populate metadata
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
    
    results_section["send_to_src_btn_1"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[
            results_section["generated_audio_1"],
            results_section["lm_metadata_state"]
        ],
        outputs=[
            generation_section["src_audio"],
            generation_section["bpm"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["audio_duration"],
            generation_section["key_scale"],
            generation_section["vocal_language"],
            generation_section["time_signature"],
            results_section["is_format_caption_state"]
        ]
    )
    
    results_section["send_to_src_btn_2"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[
            results_section["generated_audio_2"],
            results_section["lm_metadata_state"]
        ],
        outputs=[
            generation_section["src_audio"],
            generation_section["bpm"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["audio_duration"],
            generation_section["key_scale"],
            generation_section["vocal_language"],
            generation_section["time_signature"],
            results_section["is_format_caption_state"]
        ]
    )
    
    # Sample button - smart sample (uses LM if initialized, otherwise examples)
    # Need to add is_format_caption return value to sample_example_smart
    def sample_example_smart_with_flag(task_type: str, constrained_decoding_debug: bool):
        """Wrapper for sample_example_smart that adds is_format_caption flag"""
        result = sample_example_smart(task_type, constrained_decoding_debug)
        # Add True at the end to set is_format_caption
        return result + (True,)
    
    generation_section["sample_btn"].click(
        fn=sample_example_smart_with_flag,
        inputs=[
            generation_section["task_type"],
            generation_section["constrained_decoding_debug"]
        ],
        outputs=[
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["think_checkbox"],
            generation_section["bpm"],
            generation_section["audio_duration"],
            generation_section["key_scale"],
            generation_section["vocal_language"],
            generation_section["time_signature"],
            results_section["is_format_caption_state"]  # Set is_format_caption to True (from Sample/LM)
        ]
    )
    
    # Transcribe audio codes to metadata (or generate example if empty)
    def transcribe_audio_codes(audio_code_string, constrained_decoding_debug):
        """
        Transcribe audio codes to metadata using LLM understanding.
        If audio_code_string is empty, generate a sample example instead.
        
        Args:
            audio_code_string: String containing audio codes (or empty for example generation)
            constrained_decoding_debug: Whether to enable debug logging for constrained decoding
            
        Returns:
            Tuple of (status_message, caption, lyrics, bpm, duration, keyscale, language, timesignature)
        """
        if not llm_handler.llm_initialized:
            return t("messages.lm_not_initialized"), "", "", None, None, "", "", ""
        
        # If codes are empty, this becomes a "generate example" task
        # Use "NO USER INPUT" as the input to generate a sample
        if not audio_code_string or not audio_code_string.strip():
            audio_code_string = "NO USER INPUT"
        
        # Call LLM understanding
        metadata, status = llm_handler.understand_audio_from_codes(
            audio_codes=audio_code_string,
            use_constrained_decoding=True,
            constrained_decoding_debug=constrained_decoding_debug,
        )
        
        # Extract fields for UI update
        caption = metadata.get('caption', '')
        lyrics = metadata.get('lyrics', '')
        bpm = metadata.get('bpm')
        duration = metadata.get('duration')
        keyscale = metadata.get('keyscale', '')
        language = metadata.get('language', '')
        timesignature = metadata.get('timesignature', '')
        
        # Convert to appropriate types
        try:
            bpm = int(bpm) if bpm and bpm != 'N/A' else None
        except:
            bpm = None
        
        try:
            duration = float(duration) if duration and duration != 'N/A' else None
        except:
            duration = None
        
        return (
            status,
            caption,
            lyrics,
            bpm,
            duration,
            keyscale,
            language,
            timesignature,
            True  # Set is_format_caption to True (from Transcribe/LM understanding)
        )
    
    # Update transcribe button text based on whether codes are present
    def update_transcribe_button_text(audio_code_string):
        """
        Update the transcribe button text based on input content.
        If empty: "Generate Example"
        If has content: "Transcribe"
        """
        if not audio_code_string or not audio_code_string.strip():
            return gr.update(value="Generate Example")
        else:
            return gr.update(value="Transcribe")
    
    # Update button text when codes change
    generation_section["text2music_audio_code_string"].change(
        fn=update_transcribe_button_text,
        inputs=[generation_section["text2music_audio_code_string"]],
        outputs=[generation_section["transcribe_btn"]]
    )
    
    generation_section["transcribe_btn"].click(
        fn=transcribe_audio_codes,
        inputs=[
            generation_section["text2music_audio_code_string"],
            generation_section["constrained_decoding_debug"]
        ],
        outputs=[
            results_section["status_output"],       # Show status
            generation_section["captions"],         # Update caption field
            generation_section["lyrics"],           # Update lyrics field
            generation_section["bpm"],              # Update BPM field
            generation_section["audio_duration"],   # Update duration field
            generation_section["key_scale"],        # Update keyscale field
            generation_section["vocal_language"],   # Update language field
            generation_section["time_signature"],   # Update time signature field
            results_section["is_format_caption_state"]  # Set is_format_caption to True
        ]
    )
    
    # Reset is_format_caption to False when user manually edits fields
    def reset_format_caption_flag():
        """Reset is_format_caption to False when user manually edits caption/metadata"""
        return False
    
    # Connect reset function to all user-editable metadata fields
    generation_section["captions"].change(
        fn=reset_format_caption_flag,
        inputs=[],
        outputs=[results_section["is_format_caption_state"]]
    )
    
    generation_section["lyrics"].change(
        fn=reset_format_caption_flag,
        inputs=[],
        outputs=[results_section["is_format_caption_state"]]
    )
    
    generation_section["bpm"].change(
        fn=reset_format_caption_flag,
        inputs=[],
        outputs=[results_section["is_format_caption_state"]]
    )
    
    generation_section["key_scale"].change(
        fn=reset_format_caption_flag,
        inputs=[],
        outputs=[results_section["is_format_caption_state"]]
    )
    
    generation_section["time_signature"].change(
        fn=reset_format_caption_flag,
        inputs=[],
        outputs=[results_section["is_format_caption_state"]]
    )
    
    generation_section["vocal_language"].change(
        fn=reset_format_caption_flag,
        inputs=[],
        outputs=[results_section["is_format_caption_state"]]
    )
    
    generation_section["audio_duration"].change(
        fn=reset_format_caption_flag,
        inputs=[],
        outputs=[results_section["is_format_caption_state"]]
    )
    
    # Auto-expand Audio Uploads accordion when audio is uploaded
    def update_audio_uploads_accordion(reference_audio, src_audio):
        """Update Audio Uploads accordion open state based on whether audio files are present"""
        has_audio = (reference_audio is not None) or (src_audio is not None)
        return gr.update(open=has_audio)
    
    # Bind to both audio components' change events
    generation_section["reference_audio"].change(
        fn=update_audio_uploads_accordion,
        inputs=[generation_section["reference_audio"], generation_section["src_audio"]],
        outputs=[generation_section["audio_uploads_accordion"]]
    )
    
    generation_section["src_audio"].change(
        fn=update_audio_uploads_accordion,
        inputs=[generation_section["reference_audio"], generation_section["src_audio"]],
        outputs=[generation_section["audio_uploads_accordion"]]
    )
    
    # Save audio and metadata handlers - downloads as zip package
    results_section["save_btn_1"].click(
        fn=save_audio_and_metadata,
        inputs=[
            results_section["generated_audio_1"],
            generation_section["task_type"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["vocal_language"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["seed"],
            generation_section["random_seed_checkbox"],
            generation_section["use_adg"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["audio_format"],
            generation_section["lm_temperature"],
            generation_section["lm_cfg_scale"],
            generation_section["lm_top_k"],
            generation_section["lm_top_p"],
            generation_section["lm_negative_prompt"],
            generation_section["use_cot_caption"],
            generation_section["use_cot_language"],
            generation_section["audio_cover_strength"],
            generation_section["think_checkbox"],
            generation_section["text2music_audio_code_string"],
            generation_section["repainting_start"],
            generation_section["repainting_end"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            results_section["lm_metadata_state"],
        ],
        outputs=[gr.File(label="Download Package", visible=False)]
    )
    
    results_section["save_btn_2"].click(
        fn=save_audio_and_metadata,
        inputs=[
            results_section["generated_audio_2"],
            generation_section["task_type"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["vocal_language"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["seed"],
            generation_section["random_seed_checkbox"],
            generation_section["use_adg"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["audio_format"],
            generation_section["lm_temperature"],
            generation_section["lm_cfg_scale"],
            generation_section["lm_top_k"],
            generation_section["lm_top_p"],
            generation_section["lm_negative_prompt"],
            generation_section["use_cot_caption"],
            generation_section["use_cot_language"],
            generation_section["audio_cover_strength"],
            generation_section["think_checkbox"],
            generation_section["text2music_audio_code_string"],
            generation_section["repainting_start"],
            generation_section["repainting_end"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            results_section["lm_metadata_state"],
        ],
        outputs=[gr.File(label="Download Package", visible=False)]
    )
    
    # Load metadata handler - triggered when file is uploaded via UploadButton
    generation_section["load_file"].upload(
        fn=load_metadata,
        inputs=[generation_section["load_file"]],
        outputs=[
            generation_section["task_type"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["vocal_language"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["seed"],
            generation_section["random_seed_checkbox"],
            generation_section["use_adg"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["audio_format"],
            generation_section["lm_temperature"],
            generation_section["lm_cfg_scale"],
            generation_section["lm_top_k"],
            generation_section["lm_top_p"],
            generation_section["lm_negative_prompt"],
            generation_section["use_cot_caption"],
            generation_section["use_cot_language"],
            generation_section["audio_cover_strength"],
            generation_section["think_checkbox"],
            generation_section["text2music_audio_code_string"],
            generation_section["repainting_start"],
            generation_section["repainting_end"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            results_section["is_format_caption_state"]
        ]
    )
    
    # Instrumental checkbox handler - auto-fill [Instrumental] when checked
    def handle_instrumental_checkbox(instrumental_checked, current_lyrics):
        """
        Handle instrumental checkbox changes.
        When checked: if no lyrics, fill with [Instrumental]
        When unchecked: if lyrics is [Instrumental], clear it
        """
        if instrumental_checked:
            # If checked and no lyrics, fill with [Instrumental]
            if not current_lyrics or not current_lyrics.strip():
                return "[Instrumental]"
            else:
                # Has lyrics, don't change
                return current_lyrics
        else:
            # If unchecked and lyrics is exactly [Instrumental], clear it
            if current_lyrics and current_lyrics.strip() == "[Instrumental]":
                return ""
            else:
                # Has other lyrics, don't change
                return current_lyrics
    
    generation_section["instrumental_checkbox"].change(
        fn=handle_instrumental_checkbox,
        inputs=[generation_section["instrumental_checkbox"], generation_section["lyrics"]],
        outputs=[generation_section["lyrics"]]
    )
    
    # Score calculation handlers
    def update_batch_score(current_batch_index, batch_queue, sample_idx, score_display):
        """Update score for a specific sample in the current batch"""
        if current_batch_index in batch_queue:
            if "scores" not in batch_queue[current_batch_index]:
                batch_queue[current_batch_index]["scores"] = [""] * 8
            batch_queue[current_batch_index]["scores"][sample_idx - 1] = score_display
        return batch_queue
    
    def calculate_score_handler_with_selection(
        sample_idx,
        score_scale,
        current_batch_index,
        batch_queue
    ):
        """
        Calculate PMI-based quality score - REFACTORED to read from batch_queue only.
        This ensures scoring uses the actual generation parameters, not current UI values.
        
        Args:
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
            audio_codes_str, caption, lyrics, lm_metadata,
            bpm, key_scale, time_signature, audio_duration, vocal_language,
            score_scale
        )
        
        # Update batch_queue with the calculated score
        batch_queue = update_batch_score(current_batch_index, batch_queue, sample_idx, score_display)
        
        return score_display, batch_queue
    
    def calculate_score_handler(audio_codes_str, caption, lyrics, lm_metadata, bpm, key_scale, time_signature, audio_duration, vocal_language, score_scale):
        """
        Calculate PMI-based quality score for generated audio.
        
        PMI (Pointwise Mutual Information) removes condition bias:
        score = log P(condition|codes) - log P(condition)
        
        Args:
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
    
    # Connect score buttons - REFACTORED: Read from batch_queue only, not UI
    def get_score_btn_inputs(sample_idx):
        """Simplified score inputs - only batch data, no UI components"""
        return [
            gr.State(value=sample_idx),
            generation_section["score_scale"],  # Only UI param is the tool setting
            results_section["current_batch_index"],
            results_section["batch_queue"],
        ]
    
    results_section["score_btn_1"].click(
        fn=calculate_score_handler_with_selection,
        inputs=get_score_btn_inputs(1),
        outputs=[results_section["score_display_1"], results_section["batch_queue"]]
    )
    
    results_section["score_btn_2"].click(
        fn=calculate_score_handler_with_selection,
        inputs=get_score_btn_inputs(2),
        outputs=[results_section["score_display_2"], results_section["batch_queue"]]
    )
    
    results_section["score_btn_3"].click(
        fn=calculate_score_handler_with_selection,
        inputs=get_score_btn_inputs(3),
        outputs=[results_section["score_display_3"], results_section["batch_queue"]]
    )
    
    results_section["score_btn_4"].click(
        fn=calculate_score_handler_with_selection,
        inputs=get_score_btn_inputs(4),
        outputs=[results_section["score_display_4"], results_section["batch_queue"]]
    )
    
    results_section["score_btn_5"].click(
        fn=calculate_score_handler_with_selection,
        inputs=get_score_btn_inputs(5),
        outputs=[results_section["score_display_5"], results_section["batch_queue"]]
    )
    
    results_section["score_btn_6"].click(
        fn=calculate_score_handler_with_selection,
        inputs=get_score_btn_inputs(6),
        outputs=[results_section["score_display_6"], results_section["batch_queue"]]
    )
    
    results_section["score_btn_7"].click(
        fn=calculate_score_handler_with_selection,
        inputs=get_score_btn_inputs(7),
        outputs=[results_section["score_display_7"], results_section["batch_queue"]]
    )
    
    results_section["score_btn_8"].click(
        fn=calculate_score_handler_with_selection,
        inputs=get_score_btn_inputs(8),
        outputs=[results_section["score_display_8"], results_section["batch_queue"]]
    )
    
    # Send to src handlers for audio 3 and 4
    results_section["send_to_src_btn_3"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[
            results_section["generated_audio_3"],
            results_section["lm_metadata_state"]
        ],
        outputs=[
            generation_section["src_audio"],
            generation_section["bpm"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["audio_duration"],
            generation_section["key_scale"],
            generation_section["vocal_language"],
            generation_section["time_signature"],
            results_section["is_format_caption_state"]
        ]
    )
    
    results_section["send_to_src_btn_4"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[
            results_section["generated_audio_4"],
            results_section["lm_metadata_state"]
        ],
        outputs=[
            generation_section["src_audio"],
            generation_section["bpm"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["audio_duration"],
            generation_section["key_scale"],
            generation_section["vocal_language"],
            generation_section["time_signature"],
            results_section["is_format_caption_state"]
        ]
    )
    
    # Send to src handlers for audio 5-8
    results_section["send_to_src_btn_5"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[results_section["generated_audio_5"], results_section["lm_metadata_state"]],
        outputs=[
            generation_section["src_audio"], generation_section["bpm"], generation_section["captions"],
            generation_section["lyrics"], generation_section["audio_duration"], generation_section["key_scale"],
            generation_section["vocal_language"], generation_section["time_signature"], results_section["is_format_caption_state"]
        ]
    )
    
    results_section["send_to_src_btn_6"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[results_section["generated_audio_6"], results_section["lm_metadata_state"]],
        outputs=[
            generation_section["src_audio"], generation_section["bpm"], generation_section["captions"],
            generation_section["lyrics"], generation_section["audio_duration"], generation_section["key_scale"],
            generation_section["vocal_language"], generation_section["time_signature"], results_section["is_format_caption_state"]
        ]
    )
    
    results_section["send_to_src_btn_7"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[results_section["generated_audio_7"], results_section["lm_metadata_state"]],
        outputs=[
            generation_section["src_audio"], generation_section["bpm"], generation_section["captions"],
            generation_section["lyrics"], generation_section["audio_duration"], generation_section["key_scale"],
            generation_section["vocal_language"], generation_section["time_signature"], results_section["is_format_caption_state"]
        ]
    )
    
    results_section["send_to_src_btn_8"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[results_section["generated_audio_8"], results_section["lm_metadata_state"]],
        outputs=[
            generation_section["src_audio"], generation_section["bpm"], generation_section["captions"],
            generation_section["lyrics"], generation_section["audio_duration"], generation_section["key_scale"],
            generation_section["vocal_language"], generation_section["time_signature"], results_section["is_format_caption_state"]
        ]
    )
    
    # Navigation button handlers - REFACTORED: Only update results, never touch input UI
    def navigate_to_previous_batch(
        current_batch_index,
        batch_queue,
    ):
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
            audio_outputs[0],  # generated_audio_1
            audio_outputs[1],  # generated_audio_2
            audio_outputs[2],  # generated_audio_3
            audio_outputs[3],  # generated_audio_4
            audio_outputs[4],  # generated_audio_5
            audio_outputs[5],  # generated_audio_6
            audio_outputs[6],  # generated_audio_7
            audio_outputs[7],  # generated_audio_8
            audio_paths,  # generated_audio_batch
            generation_info_text,  # generation_info
            new_batch_index,  # current_batch_index
            batch_indicator_text,  # batch_indicator
            gr.update(interactive=can_go_previous),  # prev_batch_btn
            gr.update(interactive=can_go_next),  # next_batch_btn
            t("messages.viewing_batch", n=new_batch_index + 1),  # status_output
            score_displays[0],  # score_display_1
            score_displays[1],  # score_display_2
            score_displays[2],  # score_display_3
            score_displays[3],  # score_display_4
            score_displays[4],  # score_display_5
            score_displays[5],  # score_display_6
            score_displays[6],  # score_display_7
            score_displays[7],  # score_display_8
            gr.update(interactive=True),  # restore_params_btn - Enable when viewing batch
            # NO generation_section outputs - Input UI remains untouched!
        )
    
    def navigate_to_next_batch(
        autogen_enabled,
        current_batch_index,
        total_batches,
        batch_queue,
    ):
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
            audio_outputs[0],  # generated_audio_1
            audio_outputs[1],  # generated_audio_2
            audio_outputs[2],  # generated_audio_3
            audio_outputs[3],  # generated_audio_4
            audio_outputs[4],  # generated_audio_5
            audio_outputs[5],  # generated_audio_6
            audio_outputs[6],  # generated_audio_7
            audio_outputs[7],  # generated_audio_8
            audio_paths,  # generated_audio_batch
            generation_info_text,  # generation_info
            new_batch_index,  # current_batch_index
            batch_indicator_text,  # batch_indicator
            gr.update(interactive=can_go_previous),  # prev_batch_btn
            gr.update(interactive=can_go_next),  # next_batch_btn
            t("messages.viewing_batch", n=new_batch_index + 1),  # status_output
            next_batch_status_text,  # next_batch_status
            score_displays[0],  # score_display_1
            score_displays[1],  # score_display_2
            score_displays[2],  # score_display_3
            score_displays[3],  # score_display_4
            score_displays[4],  # score_display_5
            score_displays[5],  # score_display_6
            score_displays[6],  # score_display_7
            score_displays[7],  # score_display_8
            gr.update(interactive=True),  # restore_params_btn - Enable when viewing batch
            # NO generation_section outputs - Input UI remains untouched!
        )
    
    def restore_batch_parameters(current_batch_index, batch_queue):
        """
        Restore parameters from currently viewed batch to Input UI.
        This is the bridge allowing users to "reuse" historical settings.
        """
        if current_batch_index not in batch_queue:
            gr.Warning(t("messages.no_batch_data"))
            return [gr.update()] * 29  # Match number of outputs
        
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
        
        # Extract and process codes (prefer actual codes from batch_data over params)
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
            codes_outputs[0],  # text2music_audio_code_string
            codes_outputs[1],  # text2music_audio_code_string_1
            codes_outputs[2],  # text2music_audio_code_string_2
            codes_outputs[3],  # text2music_audio_code_string_3
            codes_outputs[4],  # text2music_audio_code_string_4
            codes_outputs[5],  # text2music_audio_code_string_5
            codes_outputs[6],  # text2music_audio_code_string_6
            codes_outputs[7],  # text2music_audio_code_string_7
            codes_outputs[8],  # text2music_audio_code_string_8
            captions,
            lyrics,
            bpm,
            key_scale,
            time_signature,
            vocal_language,
            audio_duration,
            batch_size_input,
            inference_steps,
            lm_temperature,
            lm_cfg_scale,
            lm_top_k,
            lm_top_p,
            think_checkbox,
            use_cot_caption,
            use_cot_language,
            allow_lm_batch,
            track_name,
            complete_track_classes
        )
    
    # Wire up navigation buttons - REFACTORED: Results-only outputs
    results_section["prev_batch_btn"].click(
        fn=navigate_to_previous_batch,
        inputs=[
            results_section["current_batch_index"],
            results_section["batch_queue"],
        ],
        outputs=[
            results_section["generated_audio_1"],
            results_section["generated_audio_2"],
            results_section["generated_audio_3"],
            results_section["generated_audio_4"],
            results_section["generated_audio_5"],
            results_section["generated_audio_6"],
            results_section["generated_audio_7"],
            results_section["generated_audio_8"],
            results_section["generated_audio_batch"],
            results_section["generation_info"],
            results_section["current_batch_index"],
            results_section["batch_indicator"],
            results_section["prev_batch_btn"],
            results_section["next_batch_btn"],
            results_section["status_output"],
            results_section["score_display_1"],
            results_section["score_display_2"],
            results_section["score_display_3"],
            results_section["score_display_4"],
            results_section["score_display_5"],
            results_section["score_display_6"],
            results_section["score_display_7"],
            results_section["score_display_8"],
            results_section["restore_params_btn"],  # Enable restore button
            # NO generation_section outputs - Input UI preserved across navigation!
        ]
    )
    
    # REFACTORED: Capture->Navigate->Generate chain with Input/Result decoupling
    results_section["next_batch_btn"].click(
        # Step 1: Capture current UI parameters (user's modifications like BS=8)
        fn=capture_current_params,
        inputs=[
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["vocal_language"],
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["random_seed_checkbox"],
            generation_section["seed"],
            generation_section["reference_audio"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["src_audio"],
            generation_section["text2music_audio_code_string"],
            generation_section["repainting_start"],
            generation_section["repainting_end"],
            generation_section["instruction_display_gen"],
            generation_section["audio_cover_strength"],
            generation_section["task_type"],
            generation_section["use_adg"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["audio_format"],
            generation_section["lm_temperature"],
            generation_section["think_checkbox"],
            generation_section["lm_cfg_scale"],
            generation_section["lm_top_k"],
            generation_section["lm_top_p"],
            generation_section["lm_negative_prompt"],
            generation_section["use_cot_metas"],
            generation_section["use_cot_caption"],
            generation_section["use_cot_language"],
            generation_section["constrained_decoding_debug"],
            generation_section["allow_lm_batch"],
            generation_section["auto_score"],
            generation_section["score_scale"],
            generation_section["lm_batch_chunk_size"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
        ],
        outputs=[results_section["generation_params_state"]]
    ).then(
        # Step 2: Navigate to next batch (updates results only, preserves input UI)
        fn=navigate_to_next_batch,
        inputs=[
            generation_section["autogen_checkbox"],
            results_section["current_batch_index"],
            results_section["total_batches"],
            results_section["batch_queue"],
        ],
        outputs=[
            results_section["generated_audio_1"],
            results_section["generated_audio_2"],
            results_section["generated_audio_3"],
            results_section["generated_audio_4"],
            results_section["generated_audio_5"],
            results_section["generated_audio_6"],
            results_section["generated_audio_7"],
            results_section["generated_audio_8"],
            results_section["generated_audio_batch"],
            results_section["generation_info"],
            results_section["current_batch_index"],
            results_section["batch_indicator"],
            results_section["prev_batch_btn"],
            results_section["next_batch_btn"],
            results_section["status_output"],
            results_section["next_batch_status"],
            results_section["score_display_1"],
            results_section["score_display_2"],
            results_section["score_display_3"],
            results_section["score_display_4"],
            results_section["score_display_5"],
            results_section["score_display_6"],
            results_section["score_display_7"],
            results_section["score_display_8"],
            results_section["restore_params_btn"],  # Enable restore button
            # NO generation_section outputs - Input UI preserved across navigation!
        ]
    ).then(
        # Step 3: Generate next batch in background (uses captured params from Step 1)
        fn=generate_next_batch_background,
        inputs=[
            generation_section["autogen_checkbox"],
            results_section["generation_params_state"],  # Uses Step 1 captured params
            results_section["current_batch_index"],
            results_section["total_batches"],
            results_section["batch_queue"],
            results_section["is_format_caption_state"],
        ],
        outputs=[
            results_section["batch_queue"],
            results_section["total_batches"],
            results_section["next_batch_status"],
            results_section["next_batch_btn"],
        ]
    )
    
    # Bind restore parameters button - Bridge between Result View and Input View
    results_section["restore_params_btn"].click(
        fn=restore_batch_parameters,
        inputs=[
            results_section["current_batch_index"],
            results_section["batch_queue"]
        ],
        outputs=[
            generation_section["text2music_audio_code_string"],
            generation_section["text2music_audio_code_string_1"],
            generation_section["text2music_audio_code_string_2"],
            generation_section["text2music_audio_code_string_3"],
            generation_section["text2music_audio_code_string_4"],
            generation_section["text2music_audio_code_string_5"],
            generation_section["text2music_audio_code_string_6"],
            generation_section["text2music_audio_code_string_7"],
            generation_section["text2music_audio_code_string_8"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["vocal_language"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["inference_steps"],
            generation_section["lm_temperature"],
            generation_section["lm_cfg_scale"],
            generation_section["lm_top_k"],
            generation_section["lm_top_p"],
            generation_section["think_checkbox"],
            generation_section["use_cot_caption"],
            generation_section["use_cot_language"],
            generation_section["allow_lm_batch"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
        ]
    )

