"""
Gradio UI Components Module
Contains all Gradio interface component definitions and layouts
"""
import gradio as gr
from typing import Callable, Optional


def create_gradio_interface(handler) -> gr.Blocks:
    """
    Create Gradio interface
    
    Args:
        handler: Business logic handler instance
        
    Returns:
        Gradio Blocks instance
    """
    with gr.Blocks(
        title="ACE-Step V1.5 Demo",
        theme=gr.themes.Soft(),
        css="""
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>♪ACE-Step V1.5 Demo</h1>
            <p>Generate music from text captions and lyrics using diffusion models</p>
        </div>
        """)
        
        # Dataset Explorer Section
        dataset_section = create_dataset_section(handler)
        
        # Generation Section
        generation_section = create_generation_section(handler)
        
        # Results Section
        results_section = create_results_section(handler)
        
        # Connect event handlers
        setup_event_handlers(demo, handler, dataset_section, generation_section, results_section)
    
    return demo


def create_dataset_section(handler) -> dict:
    """Create dataset explorer section"""
    with gr.Group():
        gr.HTML('<div class="section-header"><h3>📊 Dataset Explorer</h3></div>')
        
        with gr.Row(equal_height=True):
            dataset_type = gr.Dropdown(
                choices=["train", "test"],
                value="train",
                label="Dataset",
                info="Choose dataset to explore",
                scale=2
            )
            import_dataset_btn = gr.Button("📥 Import Dataset", variant="primary", scale=1)
            
            search_type = gr.Dropdown(
                choices=["keys", "idx", "random"],
                value="random",
                label="Search Type",
                info="How to find items",
                scale=1
            )
            search_value = gr.Textbox(
                label="Search Value",
                placeholder="Enter keys or index (leave empty for random)",
                info="Keys: exact match, Index: 0 to dataset size-1",
                scale=2
            )

        instruction_display = gr.Textbox(
            label="📝 Instruction",
            interactive=False,
            placeholder="No instruction available",
            lines=1
        )
        
        repaint_viz_plot = gr.Plot()
        
        with gr.Accordion("📋 Item Metadata (JSON)", open=False):
            item_info_json = gr.Code(
                label="Complete Item Information",
                language="json",
                interactive=False,
                lines=15
            )
        
        with gr.Row(equal_height=True):
            item_src_audio = gr.Audio(
                label="Source Audio",
                type="filepath",
                interactive=False,
                scale=8
            )
            get_item_btn = gr.Button("🔍 Get Item", variant="secondary", interactive=False, scale=2)
        
        with gr.Row(equal_height=True):
            item_target_audio = gr.Audio(
                label="Target Audio",
                type="filepath",
                interactive=False,
                scale=8
            )
            item_refer_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                interactive=False,
                scale=2
            )
        
        with gr.Row():
            use_src_checkbox = gr.Checkbox(
                label="Use Source Audio from Dataset",
                value=True,
                info="Check to use the source audio from dataset"
            )

        data_status = gr.Textbox(label="📊 Data Status", interactive=False, value="❌ No dataset imported")
        auto_fill_btn = gr.Button("📋 Auto-fill Generation Form", variant="primary")
    
    return {
        "dataset_type": dataset_type,
        "import_dataset_btn": import_dataset_btn,
        "search_type": search_type,
        "search_value": search_value,
        "instruction_display": instruction_display,
        "repaint_viz_plot": repaint_viz_plot,
        "item_info_json": item_info_json,
        "item_src_audio": item_src_audio,
        "get_item_btn": get_item_btn,
        "item_target_audio": item_target_audio,
        "item_refer_audio": item_refer_audio,
        "use_src_checkbox": use_src_checkbox,
        "data_status": data_status,
        "auto_fill_btn": auto_fill_btn,
    }


def create_generation_section(handler) -> dict:
    """Create generation section"""
    with gr.Group():
        gr.HTML('<div class="section-header"><h3>🎼 ACE-Step V1.5 Demo </h3></div>')
        
        # Service Configuration
        with gr.Accordion("🔧 Service Configuration", open=True) as service_config_accordion:
            with gr.Row():
                with gr.Column(scale=2):
                    checkpoint_dropdown = gr.Dropdown(
                        label="Checkpoint File",
                        choices=handler.get_available_checkpoints(),
                        value=None,
                        info="Select a trained model checkpoint file (full path or filename)"
                    )
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    
            with gr.Row():
                # Get available acestep-v15- model list
                available_models = handler.get_available_acestep_v15_models()
                default_model = "acestep-v15-turbo" if "acestep-v15-turbo" in available_models else (available_models[0] if available_models else None)
                
                config_path = gr.Dropdown(
                    label="Main Model Path", 
                    choices=available_models,
                    value=default_model,
                    info="Select the model configuration directory (auto-scanned from checkpoints)"
                )
                device = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    label="Device",
                    info="Processing device (auto-detect recommended)"
                )
            
            with gr.Row():
                # Get available 5Hz LM model list
                available_lm_models = handler.get_available_5hz_lm_models()
                default_lm_model = "acestep-5Hz-lm-0.6B" if "acestep-5Hz-lm-0.6B" in available_lm_models else (available_lm_models[0] if available_lm_models else None)
                
                lm_model_path = gr.Dropdown(
                    label="5Hz LM Model Path",
                    choices=available_lm_models,
                    value=default_lm_model,
                    info="Select the 5Hz LM model checkpoint (auto-scanned from checkpoints)"
                )
                init_llm_checkbox = gr.Checkbox(
                    label="Initialize 5Hz LM",
                    value=False,
                    info="Check to initialize 5Hz LM during service initialization",
                    interactive=False
                )
            
            with gr.Row():
                # Auto-detect flash attention availability
                flash_attn_available = handler.is_flash_attention_available()
                use_flash_attention_checkbox = gr.Checkbox(
                    label="Use Flash Attention",
                    value=flash_attn_available,
                    interactive=flash_attn_available,
                    info="Enable flash attention for faster inference (requires flash_attn package)" if flash_attn_available else "Flash attention not available (flash_attn package not installed)"
                )
            
            init_btn = gr.Button("Initialize Service", variant="primary", size="lg")
            init_status = gr.Textbox(label="Status", interactive=False, lines=3)
        
        # Inputs
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("📝 Required Inputs", open=True):
                    # Task type
                    # Determine initial task_type choices based on default model
                    default_model_lower = (default_model or "").lower()
                    if "turbo" in default_model_lower:
                        initial_task_choices = ["text2music", "repaint", "cover"]
                    else:
                        initial_task_choices = ["text2music", "repaint", "cover", "extract", "lego", "complete"]
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            task_type = gr.Dropdown(
                                choices=initial_task_choices,
                                value="text2music",
                                label="Task Type",
                                info="Select the task type for generation",
                            )
                        with gr.Column(scale=8):
                            instruction_display_gen = gr.Textbox(
                                label="Instruction",
                                value="Fill the audio semantic mask based on the given conditions:",
                                interactive=False,
                                lines=1,
                                info="Instruction is automatically generated based on task type",
                            )
                    
                    track_name = gr.Dropdown(
                        choices=["woodwinds", "brass", "fx", "synth", "strings", "percussion", 
                                "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"],
                        value=None,
                        label="Track Name",
                        info="Select track name for lego/extract tasks",
                        visible=False
                    )
                    
                    complete_track_classes = gr.CheckboxGroup(
                        choices=["woodwinds", "brass", "fx", "synth", "strings", "percussion", 
                                "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"],
                        label="Track Names",
                        info="Select multiple track classes for complete task",
                        visible=False
                    )
                    
                    # Audio uploads
                    with gr.Accordion("🎵 Audio Uploads", open=False):
                        with gr.Row():
                            with gr.Column(scale=2):
                                reference_audio = gr.Audio(
                                    label="Reference Audio (optional)",
                                    type="filepath",
                                )
                            with gr.Column(scale=8):
                                src_audio = gr.Audio(
                                    label="Source Audio (optional)",
                                    type="filepath",
                                )
                        
                        audio_code_string = gr.Textbox(
                            label="Audio Codes (optional)",
                            placeholder="<|audio_code_10695|><|audio_code_54246|>...",
                            lines=4,
                            visible=False,
                            info="Paste precomputed audio code tokens"
                        )
                    
                    # Audio Codes for text2music
                    with gr.Accordion("🎼 Audio Codes (for text2music)", open=True, visible=True) as text2music_audio_codes_group:
                        text2music_audio_code_string = gr.Textbox(
                            label="Audio Codes",
                            placeholder="<|audio_code_10695|><|audio_code_54246|>...",
                            lines=6,
                            info="Paste precomputed audio code tokens for text2music generation"
                        )
                    
                    # 5Hz LM
                    with gr.Row(visible=False) as use_5hz_lm_row:
                        use_5hz_lm_btn = gr.Button(
                            "Generate LM Hints",
                            variant="secondary",
                            size="lg",
                        )
                        lm_temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            scale=2,
                            info="Temperature for 5Hz LM sampling"
                        )
                    
                    # Repainting controls
                    with gr.Group(visible=False) as repainting_group:
                        gr.HTML("<h5>🎨 Repainting Controls (seconds) </h5>")
                        with gr.Row():
                            repainting_start = gr.Number(
                                label="Repainting Start",
                                value=0.0,
                                step=0.1,
                            )
                            repainting_end = gr.Number(
                                label="Repainting End",
                                value=-1,
                                minimum=-1,
                                step=0.1,
                            )
                    
                    # Audio Cover Strength
                    audio_cover_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.01,
                        label="Audio Cover Strength",
                        info="Control how many denoising steps use cover mode",
                        visible=False
                    )
                
                # Music Caption
                with gr.Accordion("📝 Music Caption", open=True):
                    captions = gr.Textbox(
                        label="Music Caption (optional)",
                        placeholder="A peaceful acoustic guitar melody with soft vocals...",
                        lines=3,
                        info="Describe the style, genre, instruments, and mood"
                    )
                
                # Lyrics
                with gr.Accordion("📝 Lyrics", open=True):
                    lyrics = gr.Textbox(
                        label="Lyrics (optional)",
                        placeholder="[Verse 1]\nUnder the starry night\nI feel so alive...",
                        lines=8,
                        info="Song lyrics with structure"
                    )
                
                # Optional Parameters
                with gr.Accordion("⚙️ Optional Parameters", open=True):
                    with gr.Row():
                        vocal_language = gr.Dropdown(
                            choices=["en", "zh", "ja", "ko", "es", "fr", "de"],
                            value="en",
                            label="Vocal Language (optional)",
                            allow_custom_value=True
                        )
                        bpm = gr.Number(
                            label="BPM (optional)",
                            value=None,
                            step=1,
                            info="leave empty for N/A"
                        )
                        key_scale = gr.Textbox(
                            label="Key/Scale (optional)",
                            placeholder="Leave empty for N/A",
                            value="",
                        )
                        time_signature = gr.Dropdown(
                            choices=["2", "3", "4", "N/A", ""],
                            value="4",
                            label="Time Signature (optional)",
                            allow_custom_value=True
                        )
                        audio_duration = gr.Number(
                            label="Audio Duration (seconds)",
                            value=-1,
                            minimum=-1,
                            maximum=600.0,
                            step=0.1,
                            info="Use -1 for random"
                        )
                        batch_size_input = gr.Number(
                            label="Batch Size",
                            value=1,
                            minimum=1,
                            maximum=8,
                            step=1,
                            info="Number of audio files to parallel generate"
                        )
        
        # Advanced Settings
        with gr.Accordion("🔧 Advanced Settings", open=False):
            with gr.Row():
                inference_steps = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=8,
                    step=1,
                    label="Inference Steps",
                    info="Turbo: max 8, Base: max 100"
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=7.0,
                    step=0.1,
                    label="Guidance Scale",
                    info="Higher values follow text more closely",
                    visible=False
                )
                seed = gr.Textbox(
                    label="Seed",
                    value="-1",
                    info="Use comma-separated values for batches"
                )
                random_seed_checkbox = gr.Checkbox(
                    label="Random Seed",
                    value=True,
                    info="Enable to auto-generate seeds"
                )
            
            with gr.Row():
                use_adg = gr.Checkbox(
                    label="Use ADG",
                    value=False,
                    info="Enable Angle Domain Guidance",
                    visible=False
                )
            
            with gr.Row():
                cfg_interval_start = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    label="CFG Interval Start",
                    visible=False
                )
                cfg_interval_end = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label="CFG Interval End",
                    visible=False
                )
            
            with gr.Row():
                audio_format = gr.Dropdown(
                    choices=["mp3", "flac"],
                    value="mp3",
                    label="Audio Format",
                    info="Audio format for saved files"
                )
            
            with gr.Row():
                output_alignment_preference = gr.Checkbox(
                    label="Output Attention Focus Score (disabled)",
                    value=False,
                    info="Output attention focus score analysis",
                    interactive=False
                )
        
        generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg", interactive=False)
    
    return {
        "checkpoint_dropdown": checkpoint_dropdown,
        "refresh_btn": refresh_btn,
        "config_path": config_path,
        "device": device,
        "init_btn": init_btn,
        "init_status": init_status,
        "lm_model_path": lm_model_path,
        "init_llm_checkbox": init_llm_checkbox,
        "use_flash_attention_checkbox": use_flash_attention_checkbox,
        "task_type": task_type,
        "instruction_display_gen": instruction_display_gen,
        "track_name": track_name,
        "complete_track_classes": complete_track_classes,
        "reference_audio": reference_audio,
        "src_audio": src_audio,
        "audio_code_string": audio_code_string,
        "text2music_audio_code_string": text2music_audio_code_string,
        "text2music_audio_codes_group": text2music_audio_codes_group,
        "use_5hz_lm_row": use_5hz_lm_row,
        "use_5hz_lm_btn": use_5hz_lm_btn,
        "lm_temperature": lm_temperature,
        "repainting_group": repainting_group,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "audio_cover_strength": audio_cover_strength,
        "captions": captions,
        "lyrics": lyrics,
        "vocal_language": vocal_language,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "audio_duration": audio_duration,
        "batch_size_input": batch_size_input,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "random_seed_checkbox": random_seed_checkbox,
        "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "audio_format": audio_format,
        "output_alignment_preference": output_alignment_preference,
        "generate_btn": generate_btn,
    }


def create_results_section(handler) -> dict:
    """Create results display section"""
    with gr.Group():
        gr.HTML('<div class="section-header"><h3>🎧 Generated Results</h3></div>')
        
        status_output = gr.Textbox(label="Generation Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                generated_audio_1 = gr.Audio(
                    label="🎵 Generated Music (Sample 1)",
                    type="filepath",
                    interactive=False
                )
            with gr.Column():
                generated_audio_2 = gr.Audio(
                    label="🎵 Generated Music (Sample 2)",
                    type="filepath",
                    interactive=False
                )

        with gr.Accordion("📁 Batch Results & Generation Details", open=False):
            generated_audio_batch = gr.File(
                label="📁 All Generated Files (Download)",
                file_count="multiple",
                interactive=False
            )
            generation_info = gr.Markdown(label="Generation Details")

        with gr.Accordion("⚖️ Attention Focus Score Analysis", open=False):
            with gr.Row():
                with gr.Column():
                    align_score_1 = gr.Textbox(label="Attention Focus Score (Sample 1)", interactive=False)
                    align_text_1 = gr.Textbox(label="Lyric Timestamps (Sample 1)", interactive=False, lines=10)
                    align_plot_1 = gr.Plot(label="Attention Focus Score Heatmap (Sample 1)")
                with gr.Column():
                    align_score_2 = gr.Textbox(label="Attention Focus Score (Sample 2)", interactive=False)
                    align_text_2 = gr.Textbox(label="Lyric Timestamps (Sample 2)", interactive=False, lines=10)
                    align_plot_2 = gr.Plot(label="Attention Focus Score Heatmap (Sample 2)")
    
    return {
        "status_output": status_output,
        "generated_audio_1": generated_audio_1,
        "generated_audio_2": generated_audio_2,
        "generated_audio_batch": generated_audio_batch,
        "generation_info": generation_info,
        "align_score_1": align_score_1,
        "align_text_1": align_text_1,
        "align_plot_1": align_plot_1,
        "align_score_2": align_score_2,
        "align_text_2": align_text_2,
        "align_plot_2": align_plot_2,
    }


def setup_event_handlers(demo, handler, dataset_section, generation_section, results_section):
    """Setup event handlers connecting UI components and business logic"""
    
    def update_init_status(status_msg, enable_btn):
        """Update initialization status and enable/disable generate button"""
        return status_msg, gr.update(interactive=enable_btn)
    
    # Dataset handlers
    dataset_section["import_dataset_btn"].click(
        fn=handler.import_dataset,
        inputs=[dataset_section["dataset_type"]],
        outputs=[dataset_section["data_status"]]
    )
    
    # Service initialization - refresh checkpoints
    def refresh_checkpoints():
        choices = handler.get_available_checkpoints()
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
                gr.update(choices=["text2music", "repaint", "cover"]),  # task_type
            )
        elif "base" in config_path_lower:
            # Base model: max 100 steps, show CFG/ADG, show all task types
            return (
                gr.update(value=32, maximum=100, minimum=1),  # inference_steps
                gr.update(visible=True),  # guidance_scale
                gr.update(visible=True),  # use_adg
                gr.update(visible=True),  # cfg_interval_start
                gr.update(visible=True),  # cfg_interval_end
                gr.update(choices=["text2music", "repaint", "cover", "extract", "lego", "complete"]),  # task_type
            )
        else:
            # Default to turbo settings
            return (
                gr.update(value=8, maximum=8, minimum=1),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(choices=["text2music", "repaint", "cover"]),  # task_type
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
    def init_service_wrapper(checkpoint, config_path, device, init_llm, lm_model_path, use_flash_attention):
        """Wrapper for service initialization, returns status and button state"""
        status, enable = handler.initialize_service(checkpoint, config_path, device, init_llm, lm_model_path, use_flash_attention)
        return status, gr.update(interactive=enable)
    
    generation_section["init_btn"].click(
        fn=init_service_wrapper,
        inputs=[
            generation_section["checkpoint_dropdown"],
            generation_section["config_path"],
            generation_section["device"],
            generation_section["init_llm_checkbox"],
            generation_section["lm_model_path"],
            generation_section["use_flash_attention_checkbox"],
        ],
        outputs=[generation_section["init_status"], generation_section["generate_btn"]]
    )
    
    # Generation with progress bar
    def generate_with_progress(
        captions, lyrics, bpm, key_scale, time_signature, vocal_language,
        inference_steps, guidance_scale, random_seed_checkbox, seed,
        reference_audio, audio_duration, batch_size_input, src_audio,
        text2music_audio_code_string, repainting_start, repainting_end,
        instruction_display_gen, audio_cover_strength, task_type,
        use_adg, cfg_interval_start, cfg_interval_end, audio_format, lm_temperature,
        progress=gr.Progress(track_tqdm=True)
    ):
        return handler.generate_music(
            captions=captions, lyrics=lyrics, bpm=bpm, key_scale=key_scale,
            time_signature=time_signature, vocal_language=vocal_language,
            inference_steps=inference_steps, guidance_scale=guidance_scale,
            use_random_seed=random_seed_checkbox, seed=seed,
            reference_audio=reference_audio, audio_duration=audio_duration,
            batch_size=batch_size_input, src_audio=src_audio,
            audio_code_string=text2music_audio_code_string,
            repainting_start=repainting_start, repainting_end=repainting_end,
            instruction=instruction_display_gen, audio_cover_strength=audio_cover_strength,
            task_type=task_type, use_adg=use_adg,
            cfg_interval_start=cfg_interval_start, cfg_interval_end=cfg_interval_end,
            audio_format=audio_format, lm_temperature=lm_temperature,
            progress=progress
        )
    
    generation_section["generate_btn"].click(
        fn=generate_with_progress,
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
            generation_section["lm_temperature"]
        ],
        outputs=[
            results_section["generated_audio_1"],
            results_section["generated_audio_2"],
            results_section["generated_audio_batch"],
            results_section["generation_info"],
            results_section["status_output"],
            generation_section["seed"],
            results_section["align_score_1"],
            results_section["align_text_1"],
            results_section["align_plot_1"],
            results_section["align_score_2"],
            results_section["align_text_2"],
            results_section["align_plot_2"]
        ]
    )
    
    # 5Hz LM generation (simplified version, can be extended as needed)
    def generate_lm_hints_wrapper(caption, lyrics, temperature):
        """Wrapper for 5Hz LM generation"""
        metadata, audio_codes, status = handler.generate_with_5hz_lm(caption, lyrics, temperature)
        # 返回格式化的结果，可以根据需要调整
        result_text = f"Status: {status}\n\nMetadata: {metadata}\n\nAudio Codes: {audio_codes[:200]}..." if len(audio_codes) > 200 else f"Status: {status}\n\nMetadata: {metadata}\n\nAudio Codes: {audio_codes}"
        return result_text
    
    generation_section["use_5hz_lm_btn"].click(
        fn=generate_lm_hints_wrapper,
        inputs=[
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["lm_temperature"]
        ],
        outputs=[generation_section["text2music_audio_code_string"]]
    )
    
    # Update instruction and UI visibility based on task type
    def update_instruction_ui(
        task_type_value: str, 
        track_name_value: Optional[str], 
        complete_track_classes_value: list, 
        audio_codes_content: str = ""
    ) -> tuple:
        """Update instruction and UI visibility based on task type."""
        instruction = handler.generate_instruction(
            task_type=task_type_value,
            track_name=track_name_value,
            complete_track_classes=complete_track_classes_value
        )
        
        # Show track_name for lego and extract
        track_name_visible = task_type_value in ["lego", "extract"]
        # Show complete_track_classes for complete
        complete_visible = task_type_value == "complete"
        # Show audio_cover_strength for cover
        audio_cover_strength_visible = task_type_value == "cover"
        # Show audio_code_string for cover
        audio_code_visible = task_type_value == "cover"
        # Show repainting controls for repaint and lego
        repainting_visible = task_type_value in ["repaint", "lego"]
        # Show use_5hz_lm, lm_temperature for text2music
        use_5hz_lm_visible = task_type_value == "text2music"
        # Show text2music_audio_codes if task is text2music OR if it has content
        # This allows it to stay visible even if user switches task type but has codes
        has_audio_codes = audio_codes_content and str(audio_codes_content).strip()
        text2music_audio_codes_visible = task_type_value == "text2music" or has_audio_codes
        
        return (
            instruction,  # instruction_display_gen
            gr.update(visible=track_name_visible),  # track_name
            gr.update(visible=complete_visible),  # complete_track_classes
            gr.update(visible=audio_cover_strength_visible),  # audio_cover_strength
            gr.update(visible=repainting_visible),  # repainting_group
            gr.update(visible=audio_code_visible),  # audio_code_string
            gr.update(visible=use_5hz_lm_visible),  # use_5hz_lm_row
            gr.update(visible=text2music_audio_codes_visible),  # text2music_audio_codes_group
        )
    
    # Bind update_instruction_ui to task_type, track_name, and complete_track_classes changes
    generation_section["task_type"].change(
        fn=update_instruction_ui,
        inputs=[
            generation_section["task_type"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["text2music_audio_code_string"]
        ],
        outputs=[
            generation_section["instruction_display_gen"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["audio_cover_strength"],
            generation_section["repainting_group"],
            generation_section["audio_code_string"],
            generation_section["use_5hz_lm_row"],
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
            generation_section["text2music_audio_code_string"]
        ],
        outputs=[
            generation_section["instruction_display_gen"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["audio_cover_strength"],
            generation_section["repainting_group"],
            generation_section["audio_code_string"],
            generation_section["use_5hz_lm_row"],
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
            generation_section["text2music_audio_code_string"]
        ],
        outputs=[
            generation_section["instruction_display_gen"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["audio_cover_strength"],
            generation_section["repainting_group"],
            generation_section["audio_code_string"],
            generation_section["use_5hz_lm_row"],
            generation_section["text2music_audio_codes_group"],
        ]
    )

