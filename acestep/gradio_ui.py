"""
Gradio UI Components Module
Contains all Gradio interface component definitions and layouts
"""
import os
import json
import random
import glob
import time as time_module
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
                        
                    # Audio Codes for text2music (dynamic display based on batch size and allow_lm_batch)
                    with gr.Accordion("🎼 LM Codes Hints", open=False, visible=True) as text2music_audio_codes_group:
                        # Single codes input (default mode)
                        with gr.Row(equal_height=True, visible=True) as codes_single_row:
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
                        
                        # Multiple codes inputs (batch mode when allow_lm_batch is enabled)
                        with gr.Row(visible=False) as codes_batch_row:
                            with gr.Column(visible=True) as codes_col_1:
                                text2music_audio_code_string_1 = gr.Textbox(
                                    label="LM Codes Hints (Sample 1)",
                                    placeholder="<|audio_code_...|>",
                                    lines=4,
                                    info="Codes for sample 1",
                                )
                            with gr.Column(visible=True) as codes_col_2:
                                text2music_audio_code_string_2 = gr.Textbox(
                                    label="LM Codes Hints (Sample 2)",
                                    placeholder="<|audio_code_...|>",
                                    lines=4,
                                    info="Codes for sample 2",
                                )
                            with gr.Column(visible=False) as codes_col_3:
                                text2music_audio_code_string_3 = gr.Textbox(
                                    label="LM Codes Hints (Sample 3)",
                                    placeholder="<|audio_code_...|>",
                                    lines=4,
                                    info="Codes for sample 3",
                                )
                            with gr.Column(visible=False) as codes_col_4:
                                text2music_audio_code_string_4 = gr.Textbox(
                                    label="LM Codes Hints (Sample 4)",
                                    placeholder="<|audio_code_...|>",
                                    lines=4,
                                    info="Codes for sample 4",
                                )
                        
                        # Additional row for codes 5-8
                        with gr.Row(visible=False) as codes_batch_row_2:
                            with gr.Column() as codes_col_5:
                                text2music_audio_code_string_5 = gr.Textbox(
                                    label="LM Codes Hints (Sample 5)",
                                    placeholder="<|audio_code_...|>",
                                    lines=4,
                                    info="Codes for sample 5",
                                )
                            with gr.Column() as codes_col_6:
                                text2music_audio_code_string_6 = gr.Textbox(
                                    label="LM Codes Hints (Sample 6)",
                                    placeholder="<|audio_code_...|>",
                                    lines=4,
                                    info="Codes for sample 6",
                                )
                            with gr.Column() as codes_col_7:
                                text2music_audio_code_string_7 = gr.Textbox(
                                    label="LM Codes Hints (Sample 7)",
                                    placeholder="<|audio_code_...|>",
                                    lines=4,
                                    info="Codes for sample 7",
                                )
                            with gr.Column() as codes_col_8:
                                text2music_audio_code_string_8 = gr.Textbox(
                                    label="LM Codes Hints (Sample 8)",
                                    placeholder="<|audio_code_...|>",
                                    lines=4,
                                    info="Codes for sample 8",
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
                    instrumental_checkbox = gr.Checkbox(
                        label="Instrumental",
                        value=False,
                        scale=1,
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
                            info="Number of audio files to parallel generate (max 8)"
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
                auto_score = gr.Checkbox(
                    label="Auto Score",
                    value=False,
                    info="Automatically calculate quality scores for all generated audios",
                    scale=1,
                )
                lm_batch_chunk_size = gr.Number(
                    label="LM Batch Chunk Size",
                    value=8,
                    minimum=1,
                    maximum=32,
                    step=1,
                    info="Max items per LM batch chunk (default: 8, limited by GPU memory)",
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
                    minimum=0.01,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="Quality Score Sensitivity",
                    info="Lower = more sensitive (default: 1.0). Adjusts how PMI maps to [0,1]",
                    scale=1,
                )
                output_alignment_preference = gr.Checkbox(
                    label="Output Attention Focus Score (disabled)",
                    value=False,
                    info="Output attention focus score analysis",
                    interactive=False,
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
            allow_lm_batch = gr.Checkbox(
                label="ParallelThinking",
                value=True,
                scale=1,
            )
            generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg", interactive=generate_btn_interactive, scale=9)
            autogen_checkbox = gr.Checkbox(
                label="AutoGen",
                value=True,
                scale=1,
            )
            use_cot_caption = gr.Checkbox(
                label="CaptionRewrite",
                value=True,
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
        "autogen_checkbox": autogen_checkbox,
        "generate_btn": generate_btn,
        "instrumental_checkbox": instrumental_checkbox,
        "constrained_decoding_debug": constrained_decoding_debug,
        "score_scale": score_scale,
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score,
        "lm_batch_chunk_size": lm_batch_chunk_size,
        "codes_single_row": codes_single_row,
        "codes_batch_row": codes_batch_row,
        "codes_batch_row_2": codes_batch_row_2,
        "text2music_audio_code_string_1": text2music_audio_code_string_1,
        "text2music_audio_code_string_2": text2music_audio_code_string_2,
        "text2music_audio_code_string_3": text2music_audio_code_string_3,
        "text2music_audio_code_string_4": text2music_audio_code_string_4,
        "text2music_audio_code_string_5": text2music_audio_code_string_5,
        "text2music_audio_code_string_6": text2music_audio_code_string_6,
        "text2music_audio_code_string_7": text2music_audio_code_string_7,
        "text2music_audio_code_string_8": text2music_audio_code_string_8,
        "codes_col_1": codes_col_1,
        "codes_col_2": codes_col_2,
        "codes_col_3": codes_col_3,
        "codes_col_4": codes_col_4,
        "codes_col_5": codes_col_5,
        "codes_col_6": codes_col_6,
        "codes_col_7": codes_col_7,
        "codes_col_8": codes_col_8,
    }


def create_results_section(dit_handler) -> dict:
    """Create results display section"""
    with gr.Group():
        gr.HTML('<div class="section-header"><h3>🎧 Generated Results</h3></div>')
        
        # Hidden state to store LM-generated metadata
        lm_metadata_state = gr.State(value=None)
        
        # Hidden state to track if caption/metadata is from formatted source (LM/transcription)
        is_format_caption_state = gr.State(value=False)
        
        # Batch management states
        current_batch_index = gr.State(value=0)  # Currently displayed batch index
        total_batches = gr.State(value=1)  # Total number of batches generated
        batch_queue = gr.State(value={})  # Dictionary storing all batch data
        generation_params_state = gr.State(value={})  # Store generation parameters for next batches
        is_generating_background = gr.State(value=False)  # Background generation flag
        
        status_output = gr.Textbox(label="Generation Status", interactive=False)
        
        # Batch navigation controls
        with gr.Row(equal_height=True):
            prev_batch_btn = gr.Button(
                "◀ Previous",
                variant="secondary",
                interactive=False,
                scale=1,
                size="sm"
            )
            batch_indicator = gr.Textbox(
                label="Current Batch",
                value="Batch 1 / 1",
                interactive=False,
                scale=3
            )
            next_batch_status = gr.Textbox(
                label="Next Batch Status",
                value="",
                interactive=False,
                scale=3
            )
            next_batch_btn = gr.Button(
                "Next ▶",
                variant="primary",
                interactive=False,
                scale=1,
                size="sm"
            )
        
        # All audio components in one row with dynamic visibility
        with gr.Row():
            with gr.Column(visible=True) as audio_col_1:
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
            with gr.Column(visible=True) as audio_col_2:
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
            with gr.Column(visible=False) as audio_col_3:
                generated_audio_3 = gr.Audio(
                    label="🎵 Generated Music (Sample 3)",
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_3 = gr.Button(
                        "🔗 Send To Src Audio",
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                    save_btn_3 = gr.Button(
                        "💾 Save",
                        variant="primary",
                        size="sm",
                        scale=1
                    )
                    score_btn_3 = gr.Button(
                        "📊 Score",
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                score_display_3 = gr.Textbox(
                    label="Quality Score (Sample 3)",
                    interactive=False,
                    placeholder="Click 'Score' to calculate perplexity-based quality score"
                )
            with gr.Column(visible=False) as audio_col_4:
                generated_audio_4 = gr.Audio(
                    label="🎵 Generated Music (Sample 4)",
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_4 = gr.Button(
                        "🔗 Send To Src Audio",
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                    save_btn_4 = gr.Button(
                        "💾 Save",
                        variant="primary",
                        size="sm",
                        scale=1
                    )
                    score_btn_4 = gr.Button(
                        "📊 Score",
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                score_display_4 = gr.Textbox(
                    label="Quality Score (Sample 4)",
                    interactive=False,
                    placeholder="Click 'Score' to calculate perplexity-based quality score"
                )
        
        # Second row for batch size 5-8 (initially hidden)
        with gr.Row(visible=False) as audio_row_5_8:
            with gr.Column() as audio_col_5:
                generated_audio_5 = gr.Audio(
                    label="🎵 Generated Music (Sample 5)",
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_5 = gr.Button("🔗 Send To Src Audio", variant="secondary", size="sm", scale=1)
                    save_btn_5 = gr.Button("💾 Save", variant="primary", size="sm", scale=1)
                    score_btn_5 = gr.Button("📊 Score", variant="secondary", size="sm", scale=1)
                score_display_5 = gr.Textbox(
                    label="Quality Score (Sample 5)",
                    interactive=False,
                    placeholder="Click 'Score' to calculate perplexity-based quality score"
                )
            with gr.Column() as audio_col_6:
                generated_audio_6 = gr.Audio(
                    label="🎵 Generated Music (Sample 6)",
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_6 = gr.Button("🔗 Send To Src Audio", variant="secondary", size="sm", scale=1)
                    save_btn_6 = gr.Button("💾 Save", variant="primary", size="sm", scale=1)
                    score_btn_6 = gr.Button("📊 Score", variant="secondary", size="sm", scale=1)
                score_display_6 = gr.Textbox(
                    label="Quality Score (Sample 6)",
                    interactive=False,
                    placeholder="Click 'Score' to calculate perplexity-based quality score"
                )
            with gr.Column() as audio_col_7:
                generated_audio_7 = gr.Audio(
                    label="🎵 Generated Music (Sample 7)",
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_7 = gr.Button("🔗 Send To Src Audio", variant="secondary", size="sm", scale=1)
                    save_btn_7 = gr.Button("💾 Save", variant="primary", size="sm", scale=1)
                    score_btn_7 = gr.Button("📊 Score", variant="secondary", size="sm", scale=1)
                score_display_7 = gr.Textbox(
                    label="Quality Score (Sample 7)",
                    interactive=False,
                    placeholder="Click 'Score' to calculate perplexity-based quality score"
                )
            with gr.Column() as audio_col_8:
                generated_audio_8 = gr.Audio(
                    label="🎵 Generated Music (Sample 8)",
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_8 = gr.Button("🔗 Send To Src Audio", variant="secondary", size="sm", scale=1)
                    save_btn_8 = gr.Button("💾 Save", variant="primary", size="sm", scale=1)
                    score_btn_8 = gr.Button("📊 Score", variant="secondary", size="sm", scale=1)
                score_display_8 = gr.Textbox(
                    label="Quality Score (Sample 8)",
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
        "current_batch_index": current_batch_index,
        "total_batches": total_batches,
        "batch_queue": batch_queue,
        "generation_params_state": generation_params_state,
        "is_generating_background": is_generating_background,
        "status_output": status_output,
        "prev_batch_btn": prev_batch_btn,
        "batch_indicator": batch_indicator,
        "next_batch_btn": next_batch_btn,
        "next_batch_status": next_batch_status,
        "generated_audio_1": generated_audio_1,
        "generated_audio_2": generated_audio_2,
        "generated_audio_3": generated_audio_3,
        "generated_audio_4": generated_audio_4,
        "generated_audio_5": generated_audio_5,
        "generated_audio_6": generated_audio_6,
        "generated_audio_7": generated_audio_7,
        "generated_audio_8": generated_audio_8,
        "audio_row_5_8": audio_row_5_8,
        "audio_col_1": audio_col_1,
        "audio_col_2": audio_col_2,
        "audio_col_3": audio_col_3,
        "audio_col_4": audio_col_4,
        "audio_col_5": audio_col_5,
        "audio_col_6": audio_col_6,
        "audio_col_7": audio_col_7,
        "audio_col_8": audio_col_8,
        "send_to_src_btn_1": send_to_src_btn_1,
        "send_to_src_btn_2": send_to_src_btn_2,
        "send_to_src_btn_3": send_to_src_btn_3,
        "send_to_src_btn_4": send_to_src_btn_4,
        "send_to_src_btn_5": send_to_src_btn_5,
        "send_to_src_btn_6": send_to_src_btn_6,
        "send_to_src_btn_7": send_to_src_btn_7,
        "send_to_src_btn_8": send_to_src_btn_8,
        "save_btn_1": save_btn_1,
        "save_btn_2": save_btn_2,
        "save_btn_3": save_btn_3,
        "save_btn_4": save_btn_4,
        "save_btn_5": save_btn_5,
        "save_btn_6": save_btn_6,
        "save_btn_7": save_btn_7,
        "save_btn_8": save_btn_8,
        "score_btn_1": score_btn_1,
        "score_btn_2": score_btn_2,
        "score_btn_3": score_btn_3,
        "score_btn_4": score_btn_4,
        "score_btn_5": score_btn_5,
        "score_btn_6": score_btn_6,
        "score_btn_7": score_btn_7,
        "score_btn_8": score_btn_8,
        "score_display_1": score_display_1,
        "score_display_2": score_display_2,
        "score_display_3": score_display_3,
        "score_display_4": score_display_4,
        "score_display_5": score_display_5,
        "score_display_6": score_display_6,
        "score_display_7": score_display_7,
        "score_display_8": score_display_8,
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
        status="completed"
    ):
        """Store batch results in queue with ALL generation parameters
        
        Args:
            codes: Audio codes used for generation (list for batch mode, string for single mode)
            scores: List of score displays for each audio (optional)
            allow_lm_batch: Whether batch LM mode was used for this batch
            batch_size: Batch size used for this batch
            generation_params: Complete dictionary of ALL generation parameters used
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
            "timestamp": datetime.datetime.now().isoformat()
        }
        return batch_queue
    
    def update_batch_indicator(current_batch, total_batches):
        """Update batch indicator text"""
        return f"Batch {current_batch + 1} / {total_batches}"
    
    def update_navigation_buttons(current_batch, total_batches):
        """Determine navigation button states"""
        can_go_previous = current_batch > 0
        can_go_next = current_batch < total_batches - 1
        return can_go_previous, can_go_next
    
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
            next_batch_status_text = "🔄 AutoGen enabled - next batch will generate after this"
        
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
                f"✅ Batch {next_batch_idx + 1} already ready!",
                gr.update(interactive=True),
            )
        
        # Update total batches count
        total_batches = next_batch_idx + 1
        
        # Update status to show generation starting
        gr.Info(f"🔄 Starting background generation for Batch {next_batch_idx + 1}...")
        
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
                status="completed"
            )
            
            logger.info(f"Batch {next_batch_idx + 1} stored in queue successfully")
            
            # Success message
            next_batch_status = f"✅ Batch {next_batch_idx + 1} ready! Click 'Next' to view."
            
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
            error_msg = f"❌ Background generation failed: {str(e)}"
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
    def update_batch_score(current_batch_index, batch_queue, sample_idx, score_display):
        """Update score for a specific sample in the current batch"""
        if current_batch_index in batch_queue:
            if "scores" not in batch_queue[current_batch_index]:
                batch_queue[current_batch_index]["scores"] = [""] * 8
            batch_queue[current_batch_index]["scores"][sample_idx - 1] = score_display
        return batch_queue
    
    def calculate_score_handler_with_selection(
        allow_lm_batch,
        codes_single, codes_1, codes_2, codes_3, codes_4, codes_5, codes_6, codes_7, codes_8,
        sample_idx,
        caption, lyrics, lm_metadata, bpm, key_scale, time_signature, audio_duration, vocal_language, score_scale,
        current_batch_index, batch_queue
    ):
        """
        Calculate PMI-based quality score for generated audio.
        Intelligently selects the correct codes based on mode and sample index.
        
        Args:
            allow_lm_batch: Whether batch mode is enabled (from UI, may not match batch settings)
            codes_single: Codes from single input (when allow_lm_batch=False)
            codes_1/2/3/4/5/6/7/8: Codes from batch inputs (when allow_lm_batch=True)
            sample_idx: Which sample to score (1-8)
            current_batch_index: Current batch index
            batch_queue: Batch queue to update with score
            ... (other parameters same as before)
        """
        # CRITICAL FIX: Use the stored batch settings, not the current UI values
        # This ensures that when we switch between batches, we use the correct codes
        batch_allow_lm_batch = allow_lm_batch  # Default to UI value
        if current_batch_index in batch_queue:
            batch_data = batch_queue[current_batch_index]
            # Use stored batch setting if available
            if "allow_lm_batch" in batch_data:
                batch_allow_lm_batch = batch_data["allow_lm_batch"]
        
        # Select the correct audio codes based on the STORED batch mode setting
        if batch_allow_lm_batch:
            # Batch mode: use corresponding batch codes
            codes_map = {
                1: codes_1, 2: codes_2, 3: codes_3, 4: codes_4,
                5: codes_5, 6: codes_6, 7: codes_7, 8: codes_8
            }
            audio_codes_str = codes_map.get(sample_idx, codes_single)
        else:
            # Single mode: all use same codes
            audio_codes_str = codes_single
        
        score_display = calculate_score_handler(audio_codes_str, caption, lyrics, lm_metadata, bpm, key_scale, time_signature, audio_duration, vocal_language, score_scale)
        
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
                return f"❌ Scoring failed: {status}"
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
            error_msg = f"❌ Error calculating score: {str(e)}\n{traceback.format_exc()}"
            return error_msg
    
    # Connect score buttons to handlers with correct codes selection
    # Define common inputs template for score buttons
    def get_score_btn_inputs(sample_idx):
        return [
            generation_section["allow_lm_batch"],
            generation_section["text2music_audio_code_string"],
            generation_section["text2music_audio_code_string_1"],
            generation_section["text2music_audio_code_string_2"],
            generation_section["text2music_audio_code_string_3"],
            generation_section["text2music_audio_code_string_4"],
            generation_section["text2music_audio_code_string_5"],
            generation_section["text2music_audio_code_string_6"],
            generation_section["text2music_audio_code_string_7"],
            generation_section["text2music_audio_code_string_8"],
            gr.State(value=sample_idx),
            generation_section["captions"],
            generation_section["lyrics"],
            results_section["lm_metadata_state"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["audio_duration"],
            generation_section["vocal_language"],
            generation_section["score_scale"],
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
            generation_section["audio_duration"], generation_section["key_scale"], generation_section["vocal_language"],
            generation_section["time_signature"], results_section["is_format_caption_state"]
        ]
    )
    
    results_section["send_to_src_btn_6"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[results_section["generated_audio_6"], results_section["lm_metadata_state"]],
        outputs=[
            generation_section["src_audio"], generation_section["bpm"], generation_section["captions"],
            generation_section["audio_duration"], generation_section["key_scale"], generation_section["vocal_language"],
            generation_section["time_signature"], results_section["is_format_caption_state"]
        ]
    )
    
    results_section["send_to_src_btn_7"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[results_section["generated_audio_7"], results_section["lm_metadata_state"]],
        outputs=[
            generation_section["src_audio"], generation_section["bpm"], generation_section["captions"],
            generation_section["audio_duration"], generation_section["key_scale"], generation_section["vocal_language"],
            generation_section["time_signature"], results_section["is_format_caption_state"]
        ]
    )
    
    results_section["send_to_src_btn_8"].click(
        fn=send_audio_to_src_with_metadata,
        inputs=[results_section["generated_audio_8"], results_section["lm_metadata_state"]],
        outputs=[
            generation_section["src_audio"], generation_section["bpm"], generation_section["captions"],
            generation_section["audio_duration"], generation_section["key_scale"], generation_section["vocal_language"],
            generation_section["time_signature"], results_section["is_format_caption_state"]
        ]
    )
    
    # Navigation button handlers
    def navigate_to_previous_batch(
        current_batch_index,
        batch_queue,
        allow_lm_batch,
        batch_size_input
    ):
        """Navigate to previous batch and restore ALL its parameters"""
        if current_batch_index <= 0:
            gr.Warning("Already at first batch")
            return [None] * 56  # Extended to include all restored parameters
        
        # Move to previous batch
        new_batch_index = current_batch_index - 1
        
        # Load batch data from queue
        if new_batch_index not in batch_queue:
            gr.Warning(f"Batch {new_batch_index + 1} not found in queue")
            return [None] * 56
        
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
        
        # Restore score displays from batch queue (if available)
        stored_scores = batch_data.get("scores", [""] * 8)
        score_displays = stored_scores if stored_scores else [""] * 8
        
        # CRITICAL FIX: Restore codes based on STORED batch settings, not current UI settings
        stored_codes = batch_data.get("codes", "")
        stored_allow_lm_batch = batch_data.get("allow_lm_batch", False)
        codes_outputs = [""] * 9  # 1 single + 8 batch codes
        
        if stored_codes:
            if stored_allow_lm_batch and isinstance(stored_codes, list):
                # This batch was generated in batch mode: restore to batch codes inputs
                codes_outputs[0] = stored_codes[0] if stored_codes else ""  # Main display shows first
                for idx in range(min(len(stored_codes), 8)):
                    codes_outputs[idx + 1] = stored_codes[idx]
            else:
                # This batch was generated in single mode: restore to single code input
                codes_outputs[0] = stored_codes if isinstance(stored_codes, str) else (stored_codes[0] if stored_codes else "")
                # Clear batch codes since this wasn't a batch mode generation
                for idx in range(1, 9):
                    codes_outputs[idx] = ""
        
        # NEW: Restore ALL generation parameters from stored state
        stored_params = batch_data.get("generation_params", {})
        
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
            f"✅ Viewing Batch {new_batch_index + 1}",  # status_output
            score_displays[0],  # score_display_1 - restored
            score_displays[1],  # score_display_2 - restored
            score_displays[2],  # score_display_3 - restored
            score_displays[3],  # score_display_4 - restored
            score_displays[4],  # score_display_5 - restored
            score_displays[5],  # score_display_6 - restored
            score_displays[6],  # score_display_7 - restored
            score_displays[7],  # score_display_8 - restored
            codes_outputs[0],  # text2music_audio_code_string - restored
            codes_outputs[1],  # text2music_audio_code_string_1 - restored
            codes_outputs[2],  # text2music_audio_code_string_2 - restored
            codes_outputs[3],  # text2music_audio_code_string_3 - restored
            codes_outputs[4],  # text2music_audio_code_string_4 - restored
            codes_outputs[5],  # text2music_audio_code_string_5 - restored
            codes_outputs[6],  # text2music_audio_code_string_6 - restored
            codes_outputs[7],  # text2music_audio_code_string_7 - restored
            codes_outputs[8],  # text2music_audio_code_string_8 - restored
            # NEW: Restore ALL generation parameters
            stored_params.get("captions", ""),  # captions
            stored_params.get("lyrics", ""),  # lyrics
            stored_params.get("bpm", None),  # bpm
            stored_params.get("key_scale", ""),  # key_scale
            stored_params.get("time_signature", ""),  # time_signature
            stored_params.get("vocal_language", "unknown"),  # vocal_language
            stored_params.get("audio_duration", -1),  # audio_duration
            stored_params.get("batch_size_input", 2),  # batch_size_input
            stored_params.get("inference_steps", 8),  # inference_steps
            stored_params.get("lm_temperature", 0.85),  # lm_temperature
            stored_params.get("lm_cfg_scale", 2.0),  # lm_cfg_scale
            stored_params.get("lm_top_k", 0),  # lm_top_k
            stored_params.get("lm_top_p", 0.9),  # lm_top_p
            stored_params.get("think_checkbox", True),  # think_checkbox
            stored_params.get("use_cot_caption", True),  # use_cot_caption
            stored_params.get("use_cot_language", True),  # use_cot_language
            stored_params.get("allow_lm_batch", True),  # allow_lm_batch (restore UI checkbox)
            stored_params.get("track_name", None),  # track_name - ADDED
            stored_params.get("complete_track_classes", []),  # complete_track_classes - ADDED
        )
    
    def navigate_to_next_batch(
        autogen_enabled,
        current_batch_index,
        total_batches,
        batch_queue,
        generation_params,
        is_format_caption,
        allow_lm_batch,
        batch_size_input
    ):
        """Navigate to next batch and restore ALL its parameters"""
        if current_batch_index >= total_batches - 1:
            gr.Warning("No next batch available")
            return [None] * 57  # Extended to include all restored parameters + next_batch_status
        
        # Move to next batch
        new_batch_index = current_batch_index + 1
        
        # Load batch data from queue
        if new_batch_index not in batch_queue:
            gr.Warning(f"Batch {new_batch_index + 1} not found in queue")
            return [None] * 57
        
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
        if autogen_enabled and new_batch_index == total_batches - 1:
            # User is viewing the latest batch, indicate next generation will start
            next_batch_status_text = "🔄 AutoGen will generate next batch in background..."
        
        # Restore score displays from batch queue (if available)
        stored_scores = batch_data.get("scores", [""] * 8)
        score_displays = stored_scores if stored_scores else [""] * 8
        
        # CRITICAL FIX: Restore codes based on STORED batch settings, not current UI settings
        stored_codes = batch_data.get("codes", "")
        stored_allow_lm_batch = batch_data.get("allow_lm_batch", False)
        codes_outputs = [""] * 9  # 1 single + 8 batch codes
        
        if stored_codes:
            if stored_allow_lm_batch and isinstance(stored_codes, list):
                # This batch was generated in batch mode: restore to batch codes inputs
                codes_outputs[0] = stored_codes[0] if stored_codes else ""  # Main display shows first
                for idx in range(min(len(stored_codes), 8)):
                    codes_outputs[idx + 1] = stored_codes[idx]
            else:
                # This batch was generated in single mode: restore to single code input
                codes_outputs[0] = stored_codes if isinstance(stored_codes, str) else (stored_codes[0] if stored_codes else "")
                # Clear batch codes since this wasn't a batch mode generation
                for idx in range(1, 9):
                    codes_outputs[idx] = ""
        
        # NEW: Restore ALL generation parameters from stored state
        stored_params = batch_data.get("generation_params", {})
        
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
            gr.update(interactive=can_go_next),  # next_batch_btn - will be disabled if at latest
            f"✅ Viewing Batch {new_batch_index + 1}",  # status_output
            next_batch_status_text,  # next_batch_status
            score_displays[0],  # score_display_1 - restored
            score_displays[1],  # score_display_2 - restored
            score_displays[2],  # score_display_3 - restored
            score_displays[3],  # score_display_4 - restored
            score_displays[4],  # score_display_5 - restored
            score_displays[5],  # score_display_6 - restored
            score_displays[6],  # score_display_7 - restored
            score_displays[7],  # score_display_8 - restored
            codes_outputs[0],  # text2music_audio_code_string - restored
            codes_outputs[1],  # text2music_audio_code_string_1 - restored
            codes_outputs[2],  # text2music_audio_code_string_2 - restored
            codes_outputs[3],  # text2music_audio_code_string_3 - restored
            codes_outputs[4],  # text2music_audio_code_string_4 - restored
            codes_outputs[5],  # text2music_audio_code_string_5 - restored
            codes_outputs[6],  # text2music_audio_code_string_6 - restored
            codes_outputs[7],  # text2music_audio_code_string_7 - restored
            codes_outputs[8],  # text2music_audio_code_string_8 - restored
            # NEW: Restore ALL generation parameters
            stored_params.get("captions", ""),  # captions
            stored_params.get("lyrics", ""),  # lyrics
            stored_params.get("bpm", None),  # bpm
            stored_params.get("key_scale", ""),  # key_scale
            stored_params.get("time_signature", ""),  # time_signature
            stored_params.get("vocal_language", "unknown"),  # vocal_language
            stored_params.get("audio_duration", -1),  # audio_duration
            stored_params.get("batch_size_input", 2),  # batch_size_input
            stored_params.get("inference_steps", 8),  # inference_steps
            stored_params.get("lm_temperature", 0.85),  # lm_temperature
            stored_params.get("lm_cfg_scale", 2.0),  # lm_cfg_scale
            stored_params.get("lm_top_k", 0),  # lm_top_k
            stored_params.get("lm_top_p", 0.9),  # lm_top_p
            stored_params.get("think_checkbox", True),  # think_checkbox
            stored_params.get("use_cot_caption", True),  # use_cot_caption
            stored_params.get("use_cot_language", True),  # use_cot_language
            stored_params.get("allow_lm_batch", True),  # allow_lm_batch (restore UI checkbox)
            stored_params.get("track_name", None),  # track_name - ADDED
            stored_params.get("complete_track_classes", []),  # complete_track_classes - ADDED
        )
    
    # Wire up navigation buttons
    results_section["prev_batch_btn"].click(
        fn=navigate_to_previous_batch,
        inputs=[
            results_section["current_batch_index"],
            results_section["batch_queue"],
            generation_section["allow_lm_batch"],
            generation_section["batch_size_input"],
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
            generation_section["text2music_audio_code_string"],
            generation_section["text2music_audio_code_string_1"],
            generation_section["text2music_audio_code_string_2"],
            generation_section["text2music_audio_code_string_3"],
            generation_section["text2music_audio_code_string_4"],
            generation_section["text2music_audio_code_string_5"],
            generation_section["text2music_audio_code_string_6"],
            generation_section["text2music_audio_code_string_7"],
            generation_section["text2music_audio_code_string_8"],
            # NEW: Restore all generation parameters
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
            generation_section["track_name"],  # ADDED
            generation_section["complete_track_classes"],  # ADDED
        ]
    )
    
    results_section["next_batch_btn"].click(
        fn=navigate_to_next_batch,
        inputs=[
            generation_section["autogen_checkbox"],
            results_section["current_batch_index"],
            results_section["total_batches"],
            results_section["batch_queue"],
            results_section["generation_params_state"],
            results_section["is_format_caption_state"],
            generation_section["allow_lm_batch"],
            generation_section["batch_size_input"],
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
            generation_section["text2music_audio_code_string"],
            generation_section["text2music_audio_code_string_1"],
            generation_section["text2music_audio_code_string_2"],
            generation_section["text2music_audio_code_string_3"],
            generation_section["text2music_audio_code_string_4"],
            generation_section["text2music_audio_code_string_5"],
            generation_section["text2music_audio_code_string_6"],
            generation_section["text2music_audio_code_string_7"],
            generation_section["text2music_audio_code_string_8"],
            # NEW: Restore all generation parameters
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
            generation_section["track_name"],  # ADDED
            generation_section["complete_track_classes"],  # ADDED
        ]
    ).then(
        # First capture current UI parameters (in case user modified them after navigation)
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
        ],
        outputs=[results_section["generation_params_state"]]
    ).then(
        # Then chain background generation with updated parameters
        fn=generate_next_batch_background,
        inputs=[
            generation_section["autogen_checkbox"],
            results_section["generation_params_state"],  # Now contains updated params
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

