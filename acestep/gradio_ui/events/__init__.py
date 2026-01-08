"""
Gradio UI Event Handlers Module
Main entry point for setting up all event handlers
"""
import gradio as gr
from typing import Optional

# Import handler modules
from . import generation_handlers as gen_h
from . import results_handlers as res_h
from acestep.gradio_ui.i18n import t


def setup_event_handlers(demo, dit_handler, llm_handler, dataset_handler, dataset_section, generation_section, results_section):
    """Setup event handlers connecting UI components and business logic"""
    
    # ========== Dataset Handlers ==========
    dataset_section["import_dataset_btn"].click(
        fn=dataset_handler.import_dataset,
        inputs=[dataset_section["dataset_type"]],
        outputs=[dataset_section["data_status"]]
    )
    
    # ========== Service Initialization ==========
    generation_section["refresh_btn"].click(
        fn=lambda: gen_h.refresh_checkpoints(dit_handler),
        outputs=[generation_section["checkpoint_dropdown"]]
    )
    
    generation_section["config_path"].change(
        fn=gen_h.update_model_type_settings,
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
    
    generation_section["init_btn"].click(
        fn=lambda *args: gen_h.init_service_wrapper(dit_handler, llm_handler, *args),
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
    
    # ========== UI Visibility Updates ==========
    generation_section["init_llm_checkbox"].change(
        fn=gen_h.update_negative_prompt_visibility,
        inputs=[generation_section["init_llm_checkbox"]],
        outputs=[generation_section["lm_negative_prompt"]]
    )
    
    generation_section["init_llm_checkbox"].change(
        fn=gen_h.update_audio_cover_strength_visibility,
        inputs=[generation_section["task_type"], generation_section["init_llm_checkbox"]],
        outputs=[generation_section["audio_cover_strength"]]
    )
    
    generation_section["task_type"].change(
        fn=gen_h.update_audio_cover_strength_visibility,
        inputs=[generation_section["task_type"], generation_section["init_llm_checkbox"]],
        outputs=[generation_section["audio_cover_strength"]]
    )
    
    generation_section["batch_size_input"].change(
        fn=gen_h.update_audio_components_visibility,
        inputs=[generation_section["batch_size_input"]],
        outputs=[
            results_section["audio_col_1"],
            results_section["audio_col_2"],
            results_section["audio_col_3"],
            results_section["audio_col_4"],
            results_section["audio_row_5_8"],
            results_section["audio_col_5"],
            results_section["audio_col_6"],
            results_section["audio_col_7"],
            results_section["audio_col_8"],
        ]
    )
    
    # Update codes hints visibility
    for trigger in [generation_section["src_audio"], generation_section["allow_lm_batch"], generation_section["batch_size_input"]]:
        trigger.change(
            fn=gen_h.update_codes_hints_visibility,
            inputs=[
                generation_section["src_audio"],
                generation_section["allow_lm_batch"],
                generation_section["batch_size_input"]
            ],
            outputs=[
                generation_section["codes_single_row"],
                generation_section["codes_batch_row"],
                generation_section["codes_batch_row_2"],
                generation_section["codes_col_1"],
                generation_section["codes_col_2"],
                generation_section["codes_col_3"],
                generation_section["codes_col_4"],
                generation_section["codes_col_5"],
                generation_section["codes_col_6"],
                generation_section["codes_col_7"],
                generation_section["codes_col_8"],
                generation_section["transcribe_btn"],
            ]
        )
    
    # ========== Audio Conversion ==========
    generation_section["convert_src_to_codes_btn"].click(
        fn=lambda src: gen_h.convert_src_audio_to_codes_wrapper(dit_handler, src),
        inputs=[generation_section["src_audio"]],
        outputs=[generation_section["text2music_audio_code_string"]]
    )
    
    # ========== Instruction UI Updates ==========
    for trigger in [generation_section["task_type"], generation_section["track_name"], generation_section["complete_track_classes"]]:
        trigger.change(
            fn=lambda *args: gen_h.update_instruction_ui(dit_handler, *args),
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
    
    # ========== Sample/Transcribe Handlers ==========
    generation_section["sample_btn"].click(
        fn=lambda task, debug: gen_h.sample_example_smart(llm_handler, task, debug) + (True,),
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
            results_section["is_format_caption_state"]
        ]
    )
    
    generation_section["text2music_audio_code_string"].change(
        fn=gen_h.update_transcribe_button_text,
        inputs=[generation_section["text2music_audio_code_string"]],
        outputs=[generation_section["transcribe_btn"]]
    )
    
    generation_section["transcribe_btn"].click(
        fn=lambda codes, debug: gen_h.transcribe_audio_codes(llm_handler, codes, debug),
        inputs=[
            generation_section["text2music_audio_code_string"],
            generation_section["constrained_decoding_debug"]
        ],
        outputs=[
            results_section["status_output"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["bpm"],
            generation_section["audio_duration"],
            generation_section["key_scale"],
            generation_section["vocal_language"],
            generation_section["time_signature"],
            results_section["is_format_caption_state"]
        ]
    )
    
    # ========== Reset Format Caption Flag ==========
    for trigger in [generation_section["captions"], generation_section["lyrics"], generation_section["bpm"],
                    generation_section["key_scale"], generation_section["time_signature"],
                    generation_section["vocal_language"], generation_section["audio_duration"]]:
        trigger.change(
            fn=gen_h.reset_format_caption_flag,
            inputs=[],
            outputs=[results_section["is_format_caption_state"]]
        )
    
    # ========== Audio Uploads Accordion ==========
    for trigger in [generation_section["reference_audio"], generation_section["src_audio"]]:
        trigger.change(
            fn=gen_h.update_audio_uploads_accordion,
            inputs=[generation_section["reference_audio"], generation_section["src_audio"]],
            outputs=[generation_section["audio_uploads_accordion"]]
        )
    
    # ========== Instrumental Checkbox ==========
    generation_section["instrumental_checkbox"].change(
        fn=gen_h.handle_instrumental_checkbox,
        inputs=[generation_section["instrumental_checkbox"], generation_section["lyrics"]],
        outputs=[generation_section["lyrics"]]
    )
    
    # ========== Load/Save Metadata ==========
    generation_section["load_file"].upload(
        fn=gen_h.load_metadata,
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
    
    # Save buttons for audio 1 and 2
    for btn_idx, btn_key in [(1, "save_btn_1"), (2, "save_btn_2")]:
        results_section[btn_key].click(
            fn=res_h.save_audio_and_metadata,
            inputs=[
                results_section[f"generated_audio_{btn_idx}"],
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
    
    # ========== Send to SRC Handlers ==========
    for btn_idx in range(1, 9):
        results_section[f"send_to_src_btn_{btn_idx}"].click(
            fn=res_h.send_audio_to_src_with_metadata,
            inputs=[
                results_section[f"generated_audio_{btn_idx}"],
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
    
    # ========== Score Calculation Handlers ==========
    for btn_idx in range(1, 9):
        results_section[f"score_btn_{btn_idx}"].click(
            fn=lambda sample_idx, scale, batch_idx, queue: res_h.calculate_score_handler_with_selection(
                llm_handler, sample_idx, scale, batch_idx, queue
            ),
            inputs=[
                gr.State(value=btn_idx),
                generation_section["score_scale"],
                results_section["current_batch_index"],
                results_section["batch_queue"],
            ],
            outputs=[results_section[f"score_display_{btn_idx}"], results_section["batch_queue"]]
        )
    
    # ========== Generation Handler ==========
    generation_section["generate_btn"].click(
        fn=lambda *args: res_h.generate_with_batch_management(dit_handler, llm_handler, *args),
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
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["autogen_checkbox"],
            results_section["current_batch_index"],
            results_section["total_batches"],
            results_section["batch_queue"],
            results_section["generation_params_state"],
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
            generation_section["text2music_audio_code_string"],
            generation_section["text2music_audio_code_string_1"],
            generation_section["text2music_audio_code_string_2"],
            generation_section["text2music_audio_code_string_3"],
            generation_section["text2music_audio_code_string_4"],
            generation_section["text2music_audio_code_string_5"],
            generation_section["text2music_audio_code_string_6"],
            generation_section["text2music_audio_code_string_7"],
            generation_section["text2music_audio_code_string_8"],
            results_section["lm_metadata_state"],
            results_section["is_format_caption_state"],
            results_section["current_batch_index"],
            results_section["total_batches"],
            results_section["batch_queue"],
            results_section["generation_params_state"],
            results_section["batch_indicator"],
            results_section["prev_batch_btn"],
            results_section["next_batch_btn"],
            results_section["next_batch_status"],
            results_section["restore_params_btn"],
        ]
    ).then(
        fn=lambda *args: res_h.generate_next_batch_background(dit_handler, llm_handler, *args),
        inputs=[
            generation_section["autogen_checkbox"],
            results_section["generation_params_state"],
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
    
    # ========== Batch Navigation Handlers ==========
    results_section["prev_batch_btn"].click(
        fn=res_h.navigate_to_previous_batch,
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
            results_section["restore_params_btn"],
        ]
    )
    
    results_section["next_batch_btn"].click(
        fn=res_h.capture_current_params,
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
        fn=res_h.navigate_to_next_batch,
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
            results_section["restore_params_btn"],
        ]
    ).then(
        fn=lambda *args: res_h.generate_next_batch_background(dit_handler, llm_handler, *args),
        inputs=[
            generation_section["autogen_checkbox"],
            results_section["generation_params_state"],
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
    
    # ========== Restore Parameters Handler ==========
    results_section["restore_params_btn"].click(
        fn=res_h.restore_batch_parameters,
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
