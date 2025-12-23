import os
import sys
import torch
import shutil
if sys.platform != 'win32':
    import resource
else:
    import psutil
from acestep.handler import AceStepHandler


def main():
    print("Initializing AceStepHandler...")
    handler = AceStepHandler()
    
    # Find checkpoints
    checkpoints = handler.get_available_checkpoints()
    if checkpoints:
        project_root = checkpoints[0]
    else:
        # Fallback
        current_file = os.path.abspath(__file__)
        project_root = os.path.join(os.path.dirname(current_file), "checkpoints")
        
    print(f"Project root (checkpoints dir): {project_root}")
    
    # Find models
    models = handler.get_available_acestep_v15_models()
    if not models:
        print("No models found. Using default 'acestep-v15-turbo'.")
        model_name = "./acestep-v15-turbo"
    else:
        model_name = models[0]
        print(f"Found models: {models}")
        print(f"Using model: {model_name}")
        
    # Initialize service
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    use_llm = False

    status, enabled = handler.initialize_service(
        project_root=project_root,
        config_path=model_name,
        device=device,
        init_llm=use_llm,
        use_flash_attention=True, # Default in UI
        compile_model=True,
        offload_to_cpu=True,
        offload_dit_to_cpu=False, # Keep DiT on GPU
        quantization="int8_weight_only", # Enable FP8 weight-only quantization
    )
    
    if not enabled:
        print(f"Error initializing service: {status}")
        return
    
    print(status)
    print("Service initialized successfully.")
    
    # Prepare inputs
    captions = "A soft pop arrangement led by light, fingerpicked guitar sets a gentle foundation, Airy keys subtly fill the background, while delicate percussion adds warmth, The sweet female voice floats above, blending naturally with minimal harmonies in the chorus for an intimate, uplifting sound"
    
    lyrics = """[Intro]

[Verse 1]
风吹动那年仲夏
翻开谁青涩喧哗
白枫书架
第七页码

[Verse 2]
珍藏谁的长发
星夜似手中花洒
淋湿旧忆木篱笆
木槿花下
天蓝发夹
她默认了他

[Bridge]
时光将青春的薄荷红蜡
匆匆地融化
她却沉入人海再无应答
隐没在天涯

[Chorus]
燕子在窗前飞掠
寻不到的花被季节带回
拧不干的思念如月
初恋颜色才能够描绘

木槿在窗外落雪
倾泻道别的滋味
闭上眼听见微咸的泪水
到后来才知那故梦珍贵

[Outro]"""

    seeds = "320145306, 1514681811"
    
    print("Starting generation...")

    # Generate hints using 5Hz LLM
    if use_llm:
        print("Generating hints using 5Hz LLM...")
        lm_temperature = 0.6
        metadata, audio_codes, lm_status = handler.generate_with_5hz_lm(captions, lyrics, lm_temperature)
        print(f"5Hz LLM Status: {lm_status}")
        print(f"Generated Metadata: {metadata}")
        print(f"Generated Audio Codes (first 50 chars): {audio_codes[:50]}...")
    else:
        print("Skipping 5Hz LLM generation...")
        metadata = {
            'bpm': 90,
            'keyscale': 'A major',
            'timesignature': '4',
            'duration': 240,
        }
        audio_codes = None
        lm_status = "Skipped"

    # Use generated metadata if available
    bpm = metadata.get('bpm', 90)
    if bpm == "N/A" or bpm == "":
        bpm = 90
    else:
        try:
            bpm = int(float(bpm))
        except:
            bpm = 90
            
    key_scale = metadata.get('keyscale', metadata.get('key_scale', "A major"))
    if key_scale == "N/A":
        key_scale = "A major"
        
    time_signature = metadata.get('timesignature', metadata.get('time_signature', "4"))
    if time_signature == "N/A":
        time_signature = "4"
        
    audio_duration = metadata.get('duration', 120)
    if audio_duration == "N/A":
        audio_duration = 120
    else:
        try:
            audio_duration = float(audio_duration)
        except:
            audio_duration = 120

    print(f"Using parameters: BPM={bpm}, Key={key_scale}, Time Sig={time_signature}, Duration={audio_duration}")

    # Reset peak memory stats
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.reset_peak_memory_stats()
    elif torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Call generate_music
    results = handler.generate_music(
        captions=captions,
        lyrics=lyrics,
        bpm=bpm,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language="zh",
        inference_steps=8,
        guidance_scale=7.0,
        use_random_seed=False,
        seed=seeds,
        audio_duration=audio_duration,
        batch_size=1,
        task_type="text2music",
        cfg_interval_start=0.0,
        cfg_interval_end=0.95,
        audio_format="wav",
        use_tiled_decode=True,
        audio_code_string=audio_codes,
    )
    
    # Unpack results
    (audio1, audio2, saved_files, info, status_msg, seed_val, 
     align_score1, align_text1, align_plot1, 
     align_score2, align_text2, align_plot2) = results
     
    print("\nGeneration Complete!")

    # Print memory stats
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        peak_vram = torch.xpu.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM usage: {peak_vram:.2f} GB")
    elif torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM usage: {peak_vram:.2f} GB")
        
    if sys.platform != 'win32':
        peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    else:
        process = psutil.Process()
        peak_ram = process.memory_info().rss / (1024 ** 3)

    print(f"Peak RAM usage: {peak_ram:.2f} GB")
    print(f"Status: {status_msg}")
    print(f"Info: {info}")
    print(f"Seeds used: {seed_val}")
    print(f"Saved files: {saved_files}")
    
    # Copy files
    for f in saved_files:
        if os.path.exists(f):
            dst = os.path.basename(f)
            shutil.copy(f, dst)
            print(f"Saved output to: {os.path.abspath(dst)}")

if __name__ == "__main__":
    main()
