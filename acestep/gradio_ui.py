"""
Gradio UI Components Module
Contains all Gradio interface component definitions and layouts
"""
import os
import json
import random
import glob
import gradio as gr
from typing import Callable, Optional, Tuple
from acestep.constants import (
    VALID_LANGUAGES,
    TRACK_NAMES,
    TASK_TYPES,
    TASK_TYPES_TURBO,
    TASK_TYPES_BASE,
    DEFAULT_DIT_INSTRUCTION,
)


def create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=None) -> gr.Blocks:
    """
    Create Gradio interface
    
    Args:
        dit_handler: DiT handler instance
        llm_handler: LM handler instance
        dataset_handler: Dataset handler instance
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
        
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
        .lm-hints-row {
            align-items: stretch;
        }
        .lm-hints-col {
            display: flex;
        }
        .lm-hints-col > div {
            flex: 1;
            display: flex;
        }
        .lm-hints-btn button {
            height: 100%;
            width: 100%;
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
        dataset_section = create_dataset_section(dataset_handler)
        
        # Generation Section (pass init_params to support pre-initialization)
        generation_section = create_generation_section(dit_handler, llm_handler, init_params=init_params)
        
        # Results Section
        results_section = create_results_section(dit_handler)
        
        # Connect event handlers
        setup_event_handlers(demo, dit_handler, llm_handler, dataset_handler, dataset_section, generation_section, results_section)
    
    return demo


def create_dataset_section(dataset_handler) -> dict:
    """Create dataset explorer section"""
    with gr.Accordion("📊 Dataset Explorer", open=False):
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


def create_generation_section(dit_handler, llm_handler, init_params=None) -> dict:
    """Create generation section
    
    Args:
        dit_handler: DiT handler instance
        llm_handler: LM handler instance
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
    """
    # Check if service is pre-initialized
    service_pre_initialized = init_params is not None and init_params.get('pre_initialized', False)
    
    with gr.Group():
        gr.HTML('<div class="section-header"><h3>🎼 ACE-Step V1.5 Demo </h3></div>')
        
        # Service Configuration - collapse if pre-initialized
        accordion_open = not service_pre_initialized
        with gr.Accordion("🔧 Service Configuration", open=accordion_open) as service_config_accordion:
            # Dropdown options section - all dropdowns grouped together
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    # Set checkpoint value from init_params if pre-initialized
                    checkpoint_value = init_params.get('checkpoint') if service_pre_initialized else None
                    checkpoint_dropdown = gr.Dropdown(
                        label="Checkpoint File",
                        choices=dit_handler.get_available_checkpoints(),
                        value=checkpoint_value,
                        info="Select a trained model checkpoint file (full path or filename)"
                    )
                with gr.Column(scale=1, min_width=90):
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            with gr.Row():
                # Get available acestep-v15- model list
                available_models = dit_handler.get_available_acestep_v15_models()
                default_model = "acestep-v15-turbo" if "acestep-v15-turbo" in available_models else (available_models[0] if available_models else None)
                
                # Set config_path value from init_params if pre-initialized
                config_path_value = init_params.get('config_path', default_model) if service_pre_initialized else default_model
                config_path = gr.Dropdown(
                    label="Main Model Path", 
                    choices=available_models,
                    value=config_path_value,
                    info="Select the model configuration directory (auto-scanned from checkpoints)"
                )
                # Set device value from init_params if pre-initialized
                device_value = init_params.get('device', 'auto') if service_pre_initialized else 'auto'
                device = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value=device_value,
                    label="Device",
                    info="Processing device (auto-detect recommended)"
                )
            
            with gr.Row():
                # Get available 5Hz LM model list
                available_lm_models = llm_handler.get_available_5hz_lm_models()
                default_lm_model = "acestep-5Hz-lm-0.6B" if "acestep-5Hz-lm-0.6B" in available_lm_models else (available_lm_models[0] if available_lm_models else None)
                
                # Set lm_model_path value from init_params if pre-initialized
                lm_model_path_value = init_params.get('lm_model_path', default_lm_model) if service_pre_initialized else default_lm_model
                lm_model_path = gr.Dropdown(
                    label="5Hz LM Model Path",
                    choices=available_lm_models,
                    value=lm_model_path_value,
                    info="Select the 5Hz LM model checkpoint (auto-scanned from checkpoints)"
                )
                # Set backend value from init_params if pre-initialized
                backend_value = init_params.get('backend', 'vllm') if service_pre_initialized else 'vllm'
                backend_dropdown = gr.Dropdown(
                    choices=["vllm", "pt"],
                    value=backend_value,
                    label="5Hz LM Backend",
                    info="Select backend for 5Hz LM: vllm (faster) or pt (PyTorch, more compatible)"
                )
            
            # Checkbox options section - all checkboxes grouped together
            with gr.Row():
                # Set init_llm value from init_params if pre-initialized
                init_llm_value = init_params.get('init_llm', True) if service_pre_initialized else True
                init_llm_checkbox = gr.Checkbox(
                    label="Initialize 5Hz LM",
                    value=init_llm_value,
                    info="Check to initialize 5Hz LM during service initialization",
                )
                # Auto-detect flash attention availability
                flash_attn_available = dit_handler.is_flash_attention_available()
                # Set use_flash_attention value from init_params if pre-initialized
                use_flash_attention_value = init_params.get('use_flash_attention', flash_attn_available) if service_pre_initialized else flash_attn_available
                use_flash_attention_checkbox = gr.Checkbox(
                    label="Use Flash Attention",
                    value=use_flash_attention_value,
                    interactive=flash_attn_available,
                    info="Enable flash attention for faster inference (requires flash_attn package)" if flash_attn_available else "Flash attention not available (flash_attn package not installed)"
                )
                # Set offload_to_cpu value from init_params if pre-initialized
                offload_to_cpu_value = init_params.get('offload_to_cpu', False) if service_pre_initialized else False
                offload_to_cpu_checkbox = gr.Checkbox(
                    label="Offload to CPU",
                    value=offload_to_cpu_value,
                    info="Offload models to CPU when not in use to save GPU memory"
                )
                # Set offload_dit_to_cpu value from init_params if pre-initialized
                offload_dit_to_cpu_value = init_params.get('offload_dit_to_cpu', False) if service_pre_initialized else False
                offload_dit_to_cpu_checkbox = gr.Checkbox(
                    label="Offload DiT to CPU",
                    value=offload_dit_to_cpu_value,
                    info="Offload DiT to CPU (needs Offload to CPU)"
                )
            
            init_btn = gr.Button("Initialize Service", variant="primary", size="lg")
            # Set init_status value from init_params if pre-initialized
            init_status_value = init_params.get('init_status', '') if service_pre_initialized else ''
            init_status = gr.Textbox(label="Status", interactive=False, lines=3, value=init_status_value)
        
        # Inputs
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("📝 Required Inputs", open=True):
                    # Task type
                    # Determine initial task_type choices based on default model
                    default_model_lower = (default_model or "").lower()
                    if "turbo" in default_model_lower:
                        initial_task_choices = TASK_TYPES_TURBO
                    else:
                        initial_task_choices = TASK_TYPES_BASE
                    
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            task_type = gr.Dropdown(
                                choices=initial_task_choices,
                                value="text2music",
                                label="Task Type",
                                info="Select the task type for generation",
                            )
                        with gr.Column(scale=7):
                            instruction_display_gen = gr.Textbox(
                                label="Instruction",
                                value=DEFAULT_DIT_INSTRUCTION,
                                interactive=False,
                                lines=1,
                                info="Instruction is automatically generated based on task type",
                            )
                        with gr.Column(scale=1, min_width=100):
                            load_file = gr.UploadButton(
                                "Load",
                                file_types=[".json"],
                                file_count="single",
                                variant="secondary",
                                size="sm",
                            )
                    
                    track_name = gr.Dropdown(
                        choices=TRACK_NAMES,
                        value=None,
                        label="Track Name",
                        info="Select track name for lego/extract tasks",
                        visible=False
                    )
                    
                    complete_track_classes = gr.CheckboxGroup(
                        choices=TRACK_NAMES,
                        label="Track Names",
                        info="Select multiple track classes for complete task",
                        visible=False
                    )
                    
                    # Audio uploads
                    audio_uploads_accordion = gr.Accordion("🎵 Audio Uploads", open=False)
                    with audio_uploads_accordion:
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=2):
                                reference_audio = gr.Audio(
                                    label="Reference Audio (optional)",
                                    type="filepath",
                                )
                            with gr.Column(scale=7):
                                src_audio = gr.Audio(
                                    label="Source Audio (optional)",
                                    type="filepath",
                                )
                            with gr.Column(scale=1, min_width=80):
                                convert_src_to_codes_btn = gr.Button(
                                    "Convert to Codes",
                                    variant="secondary",
                                    size="sm"
                                )
                        
                    # Audio Codes for text2music
                    with gr.Accordion("🎼 LM Codes Hints", open=False, visible=True) as text2music_audio_codes_group:
                        with gr.Row(equal_height=True):
                            text2music_audio_code_string = gr.Textbox(
                                label="LM Codes Hints",
                                placeholder="<|audio_code_10695|><|audio_code_54246|>...",
                                lines=6,
                                info="Paste LM codes hints for text2music generation",
                                scale=9,
                            )
                            transcribe_btn = gr.Button(
                                "Transcribe",
                                variant="secondary",
                                size="sm",
                                scale=1,
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
                
                # Music Caption
                with gr.Accordion("📝 Music Caption", open=True):
                    with gr.Row(equal_height=True):
                        captions = gr.Textbox(
                            label="Music Caption (optional)",
                            placeholder="A peaceful acoustic guitar melody with soft vocals...",
                            lines=3,
                            info="Describe the style, genre, instruments, and mood",
                            scale=9,
                        )
                        sample_btn = gr.Button(
                            "Sample",
                            variant="secondary",
                            size="sm",
                            scale=1,
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
                            choices=VALID_LANGUAGES,
                            value="unknown",
                            label="Vocal Language (optional)",
                            allow_custom_value=True,
                            info="use `unknown` for inst"
                        )
                        bpm = gr.Number(
                            label="BPM (optional)",
                            value=None,
                            step=1,
                            info="leave empty for N/A"
                        )
                        key_scale = gr.Textbox(
                            label="KeyScale (optional)",
                            placeholder="Leave empty for N/A",
                            value="",
                            info="A-G, #/♭, major/minor"
                        )
                        time_signature = gr.Dropdown(
                            choices=["2", "3", "4", "N/A", ""],
                            value="",
                            label="Time Signature (optional)",
                            allow_custom_value=True,
                            info="2/4, 3/4, 4/4..."
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
                            value=2,
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
                    label="DiT Inference Steps",
                    info="Turbo: max 8, Base: max 100"
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=7.0,
                    step=0.1,
                    label="DiT Guidance Scale (Only support for base model)",
                    info="Higher values follow text more closely",
                    visible=False
                )
                with gr.Column():
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
                audio_format = gr.Dropdown(
                    choices=["mp3", "flac"],
                    value="mp3",
                    label="Audio Format",
                    info="Audio format for saved files"
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

            # LM (Language Model) Parameters
            gr.HTML("<h4>🤖 LM Generation Parameters</h4>")
            with gr.Row():
                lm_temperature = gr.Slider(
                    label="LM Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.85,
                    step=0.1,
                    scale=1,
                    info="5Hz LM temperature (higher = more random)"
                )
                lm_cfg_scale = gr.Slider(
                    label="LM CFG Scale",
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    scale=1,
                    info="5Hz LM CFG (1.0 = no CFG)"
                )
                lm_top_k = gr.Slider(
                    label="LM Top-K",
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    scale=1,
                    info="Top-K (0 = disabled)"
                )
                lm_top_p = gr.Slider(
                    label="LM Top-P",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.01,
                    scale=1,
                    info="Top-P (1.0 = disabled)"
                )
            
            with gr.Row():
                lm_negative_prompt = gr.Textbox(
                    label="LM Negative Prompt",
                    value="NO USER INPUT",
                    placeholder="Enter negative prompt for CFG (default: NO USER INPUT)",
                    info="Negative prompt (use when LM CFG Scale > 1.0)",
                    lines=2,
                    scale=2,
                )
            
            with gr.Row():
                use_cot_metas = gr.Checkbox(
                    label="CoT Metas",
                    value=True,
                    info="Use LM to generate CoT metadata (uncheck to skip LM CoT generation)",
                    scale=1,
                )
                use_cot_caption = gr.Checkbox(
                    label="CoT Caption",
                    value=True,
                    info="Generate caption in CoT (chain-of-thought)",
                    scale=1,
                )
                use_cot_language = gr.Checkbox(
                    label="CoT Language",
                    value=True,
                    info="Generate language in CoT (chain-of-thought)",
                    scale=1,
                )
                constrained_decoding_debug = gr.Checkbox(
                    label="Constrained Decoding Debug",
                    value=False,
                    info="Enable debug logging for constrained decoding (check to see detailed logs)",
                    scale=1,
                )
            
            with gr.Row():
                audio_cover_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label="LM Codes Strength",
                    info="Control how many denoising steps use LM-generated codes",
                    scale=1,
                )
                score_scale = gr.Slider(
                    minimum=1.0,
                    maximum=200.0,
                    value=10.0,
                    step=1.0,
                    label="Quality Score Sensitivity",
                    info="Lower = more sensitive to quality differences (default: 10.0)",
                    scale=1,
                )
                output_alignment_preference = gr.Checkbox(
                    label="Output Attention Focus Score (disabled)",
                    value=False,
                    info="Output attention focus score analysis",
                    interactive=False,
                    visible=False,
                    scale=1,
                )
        
        # Set generate_btn to interactive if service is pre-initialized
        generate_btn_interactive = init_params.get('enable_generate', False) if service_pre_initialized else False
        with gr.Row(equal_height=True):
            think_checkbox = gr.Checkbox(
                label="Think",
                value=True,
                scale=1,
            )
            generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg", interactive=generate_btn_interactive, scale=10)
            instrumental_checkbox = gr.Checkbox(
                label="Instrumental",
                value=False,
                scale=1,
            )
    
    return {
        "service_config_accordion": service_config_accordion,
        "checkpoint_dropdown": checkpoint_dropdown,
        "refresh_btn": refresh_btn,
        "config_path": config_path,
        "device": device,
        "init_btn": init_btn,
        "init_status": init_status,
        "lm_model_path": lm_model_path,
        "init_llm_checkbox": init_llm_checkbox,
        "backend_dropdown": backend_dropdown,
        "use_flash_attention_checkbox": use_flash_attention_checkbox,
        "offload_to_cpu_checkbox": offload_to_cpu_checkbox,
        "offload_dit_to_cpu_checkbox": offload_dit_to_cpu_checkbox,
        "task_type": task_type,
        "instruction_display_gen": instruction_display_gen,
        "track_name": track_name,
        "complete_track_classes": complete_track_classes,
        "audio_uploads_accordion": audio_uploads_accordion,
        "reference_audio": reference_audio,
        "src_audio": src_audio,
        "convert_src_to_codes_btn": convert_src_to_codes_btn,
        "text2music_audio_code_string": text2music_audio_code_string,
        "transcribe_btn": transcribe_btn,
        "text2music_audio_codes_group": text2music_audio_codes_group,
        "lm_temperature": lm_temperature,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption,
        "use_cot_language": use_cot_language,
        "repainting_group": repainting_group,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "audio_cover_strength": audio_cover_strength,
        "captions": captions,
        "sample_btn": sample_btn,
        "load_file": load_file,
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
        "think_checkbox": think_checkbox,
        "generate_btn": generate_btn,
        "instrumental_checkbox": instrumental_checkbox,
        "constrained_decoding_debug": constrained_decoding_debug,
        "score_scale": score_scale,
    }


def create_results_section(dit_handler) -> dict:
    """Create results display section"""
    with gr.Group():
        gr.HTML('<div class="section-header"><h3>🎧 Generated Results</h3></div>')
        
        # Hidden state to store LM-generated metadata
        lm_metadata_state = gr.State(value=None)
        
        # Hidden state to track if caption/metadata is from formatted source (LM/transcription)
        is_format_caption_state = gr.State(value=False)
        
        status_output = gr.Textbox(label="Generation Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                generated_audio_1 = gr.Audio(
                    label="🎵 Generated Music (Sample 1)",
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_1 = gr.Button(
                        "🔗 Send To Src Audio",
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                    save_btn_1 = gr.Button(
                        "💾 Save",
                        variant="primary",
                        size="sm",
                        scale=1
                    )
                    score_btn_1 = gr.Button(
                        "📊 Score",
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                score_display_1 = gr.Textbox(
                    label="Quality Score (Sample 1)",
                    interactive=False,
                    placeholder="Click 'Score' to calculate perplexity-based quality score"
                )
            with gr.Column():
                generated_audio_2 = gr.Audio(
                    label="🎵 Generated Music (Sample 2)",
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_2 = gr.Button(
                        "🔗 Send To Src Audio",
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                    save_btn_2 = gr.Button(
                        "💾 Save",
                        variant="primary",
                        size="sm",
                        scale=1
                    )
                    score_btn_2 = gr.Button(
                        "📊 Score",
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                score_display_2 = gr.Textbox(
                    label="Quality Score (Sample 2)",
                    interactive=False,
                    placeholder="Click 'Score' to calculate perplexity-based quality score"
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
        "lm_metadata_state": lm_metadata_state,
        "is_format_caption_state": is_format_caption_state,
        "status_output": status_output,
        "generated_audio_1": generated_audio_1,
        "generated_audio_2": generated_audio_2,
        "send_to_src_btn_1": send_to_src_btn_1,
        "send_to_src_btn_2": send_to_src_btn_2,
        "save_btn_1": save_btn_1,
        "save_btn_2": save_btn_2,
        "score_btn_1": score_btn_1,
        "score_btn_2": score_btn_2,
        "score_display_1": score_display_1,
        "score_display_2": score_display_2,
        "generated_audio_batch": generated_audio_batch,
        "generation_info": generation_info,
        "align_score_1": align_score_1,
        "align_text_1": align_text_1,
        "align_plot_1": align_plot_1,
        "align_score_2": align_score_2,
        "align_text_2": align_text_2,
        "align_plot_2": align_plot_2,
    }


def setup_event_handlers(demo, dit_handler, llm_handler, dataset_handler, dataset_section, generation_section, results_section):
    """Setup event handlers connecting UI components and business logic"""
    
    def save_metadata(
        task_type, captions, lyrics, vocal_language, bpm, key_scale, time_signature, audio_duration,
        batch_size_input, inference_steps, guidance_scale, seed, random_seed_checkbox,
        use_adg, cfg_interval_start, cfg_interval_end, audio_format,
        lm_temperature, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
        use_cot_caption, use_cot_language, audio_cover_strength,
        think_checkbox, text2music_audio_code_string, repainting_start, repainting_end,
        track_name, complete_track_classes, lm_metadata
    ):
        """Save all generation parameters to a JSON file"""
        import datetime
        
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
            "random_seed": False, # Disable random seed for reproducibility
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
        
        # Save to file
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generation_params_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            gr.Info(f"✅ Parameters saved to {filename}")
            return filename
        except Exception as e:
            gr.Warning(f"❌ Failed to save parameters: {str(e)}")
            return None
    
    def load_metadata(file_obj):
        """Load generation parameters from a JSON file"""
        if file_obj is None:
            gr.Warning("⚠️ No file selected")
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
            
            gr.Info(f"✅ Parameters loaded from {os.path.basename(filepath)}")
            
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
            gr.Warning(f"❌ Invalid JSON file: {str(e)}")
            return [None] * 31 + [False]
        except Exception as e:
            gr.Warning(f"❌ Error loading file: {str(e)}")
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
            project_root = os.path.dirname(os.path.dirname(current_file))
            
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
                
                gr.Info(f"📁 Loaded example from {os.path.basename(selected_file)}")
                return caption_value, lyrics_value, think_value, bpm_value, duration_value, keyscale_value, language_value, timesignature_value
                
            except json.JSONDecodeError as e:
                gr.Warning(f"Failed to parse JSON file {os.path.basename(selected_file)}: {str(e)}")
                return "", "", True, None, None, "", "", ""
            except Exception as e:
                gr.Warning(f"Error reading file {os.path.basename(selected_file)}: {str(e)}")
                return "", "", True, None, None, "", "", ""
                
        except Exception as e:
            gr.Warning(f"Error loading example: {str(e)}")
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
                    
                    gr.Info("🤖 Generated example using LM")
                    return caption_value, lyrics_value, think_value, bpm_value, duration_value, keyscale_value, language_value, timesignature_value
                else:
                    gr.Warning("Failed to generate example using LM, falling back to examples directory")
                    return load_random_example(task_type)
                    
            except Exception as e:
                gr.Warning(f"Error generating example with LM: {str(e)}, falling back to examples directory")
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
            project_root = os.path.dirname(os.path.dirname(current_file))
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
        progress=gr.Progress(track_tqdm=True)
    ):
        # If think is enabled (llm_dit mode) and use_cot_metas is True, generate audio codes using LM first
        audio_code_string_to_use = text2music_audio_code_string
        lm_generated_metadata = None  # Store LM-generated metadata for display
        lm_generated_audio_codes = None  # Store LM-generated audio codes for display
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
            
            # Generate using llm_dit mode (infer_type='llm_dit')
            metadata, audio_codes, status = llm_handler.generate_with_stop_condition(
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
            )
            
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
        
        # Extract results
        first_audio, second_audio, all_audio_paths, generation_info, status_message, seed_value_for_ui, \
            align_score_1, align_text_1, align_plot_1, align_score_2, align_text_2, align_plot_2 = result
        
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
        updated_audio_codes = lm_generated_audio_codes if lm_generated_audio_codes else text2music_audio_code_string
        
        return (
            first_audio,
            second_audio,
            all_audio_paths,
            generation_info,
            status_message,
            seed_value_for_ui,
            align_score_1,
            align_text_1,
            align_plot_1,
            align_score_2,
            align_text_2,
            align_plot_2,
            updated_audio_codes,  # Update audio codes in UI
            lm_generated_metadata,  # Store metadata for "Send to src audio" buttons
            is_format_caption  # Keep is_format_caption unchanged (LM doesn't modify user input fields)
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
            generation_section["constrained_decoding_debug"]
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
            results_section["align_plot_2"],
            generation_section["text2music_audio_code_string"],  # Update audio codes display
            results_section["lm_metadata_state"],  # Store metadata
            results_section["is_format_caption_state"]  # Update is_format_caption state
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
            Tuple of (audio_file, bpm, caption, duration, key_scale, language, time_signature, is_format_caption)
        """
        if audio_file is None:
            return None, None, None, None, None, None, None, True  # Keep is_format_caption as True
        
        # Extract metadata fields if available
        bpm_value = None
        caption_value = None
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
            return "❌ 5Hz LM not initialized. Please initialize it first.", "", "", None, None, "", "", ""
        
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
    
    # Save metadata handlers - use JavaScript to trigger automatic download
    results_section["save_btn_1"].click(
        fn=None,
        inputs=[
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
        outputs=None,
        js="""
        (task_type, captions, lyrics, vocal_language, bpm, key_scale, time_signature, audio_duration,
         batch_size_input, inference_steps, guidance_scale, seed, random_seed_checkbox,
         use_adg, cfg_interval_start, cfg_interval_end, audio_format,
         lm_temperature, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
         use_cot_caption, use_cot_language, audio_cover_strength,
         think_checkbox, text2music_audio_code_string, repainting_start, repainting_end,
         track_name, complete_track_classes, lm_metadata) => {
            // Create metadata object
            const metadata = {
                saved_at: new Date().toISOString(),
                task_type: task_type,
                caption: captions || "",
                lyrics: lyrics || "",
                vocal_language: vocal_language,
                bpm: bpm,
                keyscale: key_scale || "",
                timesignature: time_signature || "",
                duration: audio_duration,
                batch_size: batch_size_input,
                inference_steps: inference_steps,
                guidance_scale: guidance_scale,
                seed: seed,
                random_seed: random_seed_checkbox,
                use_adg: use_adg,
                cfg_interval_start: cfg_interval_start,
                cfg_interval_end: cfg_interval_end,
                audio_format: audio_format,
                lm_temperature: lm_temperature,
                lm_cfg_scale: lm_cfg_scale,
                lm_top_k: lm_top_k,
                lm_top_p: lm_top_p,
                lm_negative_prompt: lm_negative_prompt,
                use_cot_caption: use_cot_caption,
                use_cot_language: use_cot_language,
                audio_cover_strength: audio_cover_strength,
                think: think_checkbox,
                audio_codes: text2music_audio_code_string || "",
                repainting_start: repainting_start,
                repainting_end: repainting_end,
                track_name: track_name,
                complete_track_classes: complete_track_classes || []
            };
            
            if (lm_metadata) {
                metadata.lm_generated_metadata = lm_metadata;
            }
            
            // Create JSON string
            const jsonStr = JSON.stringify(metadata, null, 2);
            
            // Create blob and download
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '_').split('.')[0];
            a.download = `generation_params_${timestamp}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            return Array(32).fill(null);
        }
        """
    )
    
    results_section["save_btn_2"].click(
        fn=None,
        inputs=[
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
        outputs=None,
        js="""
        (task_type, captions, lyrics, vocal_language, bpm, key_scale, time_signature, audio_duration,
         batch_size_input, inference_steps, guidance_scale, seed, random_seed_checkbox,
         use_adg, cfg_interval_start, cfg_interval_end, audio_format,
         lm_temperature, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
         use_cot_caption, use_cot_language, audio_cover_strength,
         think_checkbox, text2music_audio_code_string, repainting_start, repainting_end,
         track_name, complete_track_classes, lm_metadata) => {
            // Create metadata object
            const metadata = {
                saved_at: new Date().toISOString(),
                task_type: task_type,
                caption: captions || "",
                lyrics: lyrics || "",
                vocal_language: vocal_language,
                bpm: bpm,
                keyscale: key_scale || "",
                timesignature: time_signature || "",
                duration: audio_duration,
                batch_size: batch_size_input,
                inference_steps: inference_steps,
                guidance_scale: guidance_scale,
                seed: seed,
                random_seed: random_seed_checkbox,
                use_adg: use_adg,
                cfg_interval_start: cfg_interval_start,
                cfg_interval_end: cfg_interval_end,
                audio_format: audio_format,
                lm_temperature: lm_temperature,
                lm_cfg_scale: lm_cfg_scale,
                lm_top_k: lm_top_k,
                lm_top_p: lm_top_p,
                lm_negative_prompt: lm_negative_prompt,
                use_cot_caption: use_cot_caption,
                use_cot_language: use_cot_language,
                audio_cover_strength: audio_cover_strength,
                think: think_checkbox,
                audio_codes: text2music_audio_code_string || "",
                repainting_start: repainting_start,
                repainting_end: repainting_end,
                track_name: track_name,
                complete_track_classes: complete_track_classes || []
            };
            
            if (lm_metadata) {
                metadata.lm_generated_metadata = lm_metadata;
            }
            
            // Create JSON string
            const jsonStr = JSON.stringify(metadata, null, 2);
            
            // Create blob and download
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '_').split('.')[0];
            a.download = `generation_params_${timestamp}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            return Array(32).fill(null);
        }
        """
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
    def calculate_score_handler(audio_codes_str, caption, lyrics, lm_metadata, bpm, key_scale, time_signature, audio_duration, vocal_language, score_scale):
        """
        Calculate perplexity-based quality score for generated audio.
        
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
            score_scale: Sensitivity scale parameter (lower = more sensitive)
            
        Returns:
            Score display string
        """
        from acestep.test_time_scaling import calculate_perplexity, perplexity_to_score
        
        if not llm_handler.llm_initialized:
            return "❌ LLM not initialized. Please initialize 5Hz LM first."
        
        if not audio_codes_str or not audio_codes_str.strip():
            return "❌ No audio codes available. Please generate music first."
        
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
            
            # Calculate perplexity
            perplexity, status = calculate_perplexity(
                llm_handler=llm_handler,
                audio_codes=audio_codes_str,
                caption=caption or "",
                lyrics=lyrics or "",
                metadata=metadata if metadata else None,
                temperature=1.0
            )
            
            # Convert perplexity to normalized score [0, 1] (higher is better)
            normalized_score = perplexity_to_score(perplexity, scale=score_scale)
            
            # Format display string
            if perplexity == float('inf'):
                return f"❌ Scoring failed: {status}"
            else:
                return f"✅ Quality Score: {normalized_score:.4f} (range: 0-1, higher is better)\nPerplexity: {perplexity:.4f}\nSensitivity: {score_scale}\n{status}"
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error calculating score: {str(e)}\n{traceback.format_exc()}"
            return error_msg
    
    # Connect score buttons to handlers
    results_section["score_btn_1"].click(
        fn=calculate_score_handler,
        inputs=[
            generation_section["text2music_audio_code_string"],
            generation_section["captions"],
            generation_section["lyrics"],
            results_section["lm_metadata_state"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["audio_duration"],
            generation_section["vocal_language"],
            generation_section["score_scale"]
        ],
        outputs=[results_section["score_display_1"]]
    )
    
    results_section["score_btn_2"].click(
        fn=calculate_score_handler,
        inputs=[
            generation_section["text2music_audio_code_string"],
            generation_section["captions"],
            generation_section["lyrics"],
            results_section["lm_metadata_state"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["audio_duration"],
            generation_section["vocal_language"],
            generation_section["score_scale"]
        ],
        outputs=[results_section["score_display_2"]]
    )

