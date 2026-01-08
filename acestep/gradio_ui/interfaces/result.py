"""
Gradio UI Results Section Module
Contains results display section component definitions
"""
import gradio as gr
from acestep.gradio_ui.i18n import t


def create_results_section(dit_handler) -> dict:
    """Create results display section"""
    with gr.Accordion(t("results.title"), open=True):
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

        # All audio components in one row with dynamic visibility
        with gr.Row():
            with gr.Column(visible=True) as audio_col_1:
                generated_audio_1 = gr.Audio(
                    label=t("results.generated_music", n=1),
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_1 = gr.Button(
                        t("results.send_to_src_btn"),
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                    save_btn_1 = gr.Button(
                        t("results.save_btn"),
                        variant="primary",
                        size="sm",
                        scale=1
                    )
                    score_btn_1 = gr.Button(
                        t("results.score_btn"),
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                score_display_1 = gr.Textbox(
                    label=t("results.quality_score_label", n=1),
                    interactive=False,
                    placeholder=t("results.quality_score_placeholder")
                )
            with gr.Column(visible=True) as audio_col_2:
                generated_audio_2 = gr.Audio(
                    label=t("results.generated_music", n=2),
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_2 = gr.Button(
                        t("results.send_to_src_btn"),
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                    save_btn_2 = gr.Button(
                        t("results.save_btn"),
                        variant="primary",
                        size="sm",
                        scale=1
                    )
                    score_btn_2 = gr.Button(
                        t("results.score_btn"),
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                score_display_2 = gr.Textbox(
                    label=t("results.quality_score_label", n=2),
                    interactive=False,
                    placeholder=t("results.quality_score_placeholder")
                )
            with gr.Column(visible=False) as audio_col_3:
                generated_audio_3 = gr.Audio(
                    label=t("results.generated_music", n=3),
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_3 = gr.Button(
                        t("results.send_to_src_btn"),
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                    save_btn_3 = gr.Button(
                        t("results.save_btn"),
                        variant="primary",
                        size="sm",
                        scale=1
                    )
                    score_btn_3 = gr.Button(
                        t("results.score_btn"),
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                score_display_3 = gr.Textbox(
                    label=t("results.quality_score_label", n=3),
                    interactive=False,
                    placeholder=t("results.quality_score_placeholder")
                )
            with gr.Column(visible=False) as audio_col_4:
                generated_audio_4 = gr.Audio(
                    label=t("results.generated_music", n=4),
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_4 = gr.Button(
                        t("results.send_to_src_btn"),
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                    save_btn_4 = gr.Button(
                        t("results.save_btn"),
                        variant="primary",
                        size="sm",
                        scale=1
                    )
                    score_btn_4 = gr.Button(
                        t("results.score_btn"),
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                score_display_4 = gr.Textbox(
                    label=t("results.quality_score_label", n=4),
                    interactive=False,
                    placeholder=t("results.quality_score_placeholder")
                )
        
        # Second row for batch size 5-8 (initially hidden)
        with gr.Row(visible=False) as audio_row_5_8:
            with gr.Column() as audio_col_5:
                generated_audio_5 = gr.Audio(
                    label=t("results.generated_music", n=5),
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_5 = gr.Button(t("results.send_to_src_btn"), variant="secondary", size="sm", scale=1)
                    save_btn_5 = gr.Button(t("results.save_btn"), variant="primary", size="sm", scale=1)
                    score_btn_5 = gr.Button(t("results.score_btn"), variant="secondary", size="sm", scale=1)
                score_display_5 = gr.Textbox(
                    label=t("results.quality_score_label", n=5),
                    interactive=False,
                    placeholder=t("results.quality_score_placeholder")
                )
            with gr.Column() as audio_col_6:
                generated_audio_6 = gr.Audio(
                    label=t("results.generated_music", n=6),
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_6 = gr.Button(t("results.send_to_src_btn"), variant="secondary", size="sm", scale=1)
                    save_btn_6 = gr.Button(t("results.save_btn"), variant="primary", size="sm", scale=1)
                    score_btn_6 = gr.Button(t("results.score_btn"), variant="secondary", size="sm", scale=1)
                score_display_6 = gr.Textbox(
                    label=t("results.quality_score_label", n=6),
                    interactive=False,
                    placeholder=t("results.quality_score_placeholder")
                )
            with gr.Column() as audio_col_7:
                generated_audio_7 = gr.Audio(
                    label=t("results.generated_music", n=7),
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_7 = gr.Button(t("results.send_to_src_btn"), variant="secondary", size="sm", scale=1)
                    save_btn_7 = gr.Button(t("results.save_btn"), variant="primary", size="sm", scale=1)
                    score_btn_7 = gr.Button(t("results.score_btn"), variant="secondary", size="sm", scale=1)
                score_display_7 = gr.Textbox(
                    label=t("results.quality_score_label", n=7),
                    interactive=False,
                    placeholder=t("results.quality_score_placeholder")
                )
            with gr.Column() as audio_col_8:
                generated_audio_8 = gr.Audio(
                    label=t("results.generated_music", n=8),
                    type="filepath",
                    interactive=False
                )
                with gr.Row(equal_height=True):
                    send_to_src_btn_8 = gr.Button(t("results.send_to_src_btn"), variant="secondary", size="sm", scale=1)
                    save_btn_8 = gr.Button(t("results.save_btn"), variant="primary", size="sm", scale=1)
                    score_btn_8 = gr.Button(t("results.score_btn"), variant="secondary", size="sm", scale=1)
                score_display_8 = gr.Textbox(
                    label=t("results.quality_score_label", n=8),
                    interactive=False,
                    placeholder=t("results.quality_score_placeholder")
                )
        
        status_output = gr.Textbox(label=t("results.generation_status"), interactive=False)
        
        # Batch navigation controls
        with gr.Row(equal_height=True):
            prev_batch_btn = gr.Button(
                t("results.prev_btn"),
                variant="secondary",
                interactive=False,
                scale=1,
                size="sm"
            )
            batch_indicator = gr.Textbox(
                label=t("results.current_batch"),
                value=t("results.batch_indicator", current=1, total=1),
                interactive=False,
                scale=3
            )
            next_batch_status = gr.Textbox(
                label=t("results.next_batch_status"),
                value="",
                interactive=False,
                scale=3
            )
            next_batch_btn = gr.Button(
                t("results.next_btn"),
                variant="primary",
                interactive=False,
                scale=1,
                size="sm"
            )
        
        # One-click restore parameters button
        restore_params_btn = gr.Button(
            t("results.restore_params_btn"),
            variant="secondary",
            interactive=False,  # Initially disabled, enabled after generation
            size="sm"
        )
        
        with gr.Accordion(t("results.batch_results_title"), open=False):
            generated_audio_batch = gr.File(
                label=t("results.all_files_label"),
                file_count="multiple",
                interactive=False
            )
            generation_info = gr.Markdown(label=t("results.generation_details"))

        with gr.Accordion(t("results.attention_analysis"), open=False):
            with gr.Row():
                with gr.Column():
                    align_score_1 = gr.Textbox(label=t("results.attention_score", n=1), interactive=False)
                    align_text_1 = gr.Textbox(label=t("results.lyric_timestamps", n=1), interactive=False, lines=10)
                    align_plot_1 = gr.Plot(label=t("results.attention_heatmap", n=1))
                with gr.Column():
                    align_score_2 = gr.Textbox(label=t("results.attention_score", n=2), interactive=False)
                    align_text_2 = gr.Textbox(label=t("results.lyric_timestamps", n=2), interactive=False, lines=10)
                    align_plot_2 = gr.Plot(label=t("results.attention_heatmap", n=2))
    
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
        "restore_params_btn": restore_params_btn,
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

