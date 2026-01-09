#!/usr/bin/env python3
"""
Profiling script for acestep/inference.py using cProfile

Usage:
    python profile_inference.py
    python profile_inference.py --warmup
"""

import cProfile
import pstats
import io
import time
import argparse
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.inference import generate_music, GenerationParams, GenerationConfig
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
import json
from typing import Tuple


def profile_with_cprofile(dit_handler, llm_handler, params, config, warmup=False):
    """Profile using Python's built-in cProfile.
    
    Args:
        warmup: If True, run once for warmup before profiling (default: False)
    """
    print("=" * 80)
    print("Profiling with cProfile")
    print("=" * 80)
    
    # Warmup run (to exclude PyTorch compilation overhead)
    if warmup:
        print("\n[Warmup] Running first generation to warm up (PyTorch compilation, etc.)...")
        warmup_start = time.time()
        params.use_cot_metas = False
        config.is_format_caption = True
        config.use_constrained_decoding = False
        warmup_result = generate_music(dit_handler, llm_handler, params, config, save_dir="./")
        warmup_time = time.time() - warmup_start
        print(f"[Warmup] Completed in {warmup_time:.2f}s")
        if not warmup_result.success:
            print(f"[Warmup] ⚠ Warmup generation failed: {warmup_result.error}")
            return warmup_result
    
    # Actual profiling run (first inference)
    print("\n[Profiling] Running first generation for profiling...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    profiling_start = time.time()
    try:
        result = generate_music(dit_handler, llm_handler, params, config, save_dir="./")
    finally:
        profiler.disable()
    profiling_time = time.time() - profiling_start
    
    # Create stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    
    print(f"\n[Profiling] Completed in {profiling_time:.2f}s")
    print("\nTop 30 functions by cumulative time:")
    print("-" * 80)
    ps.print_stats(30)
    
    print("\nTop 30 functions by total time:")
    print("-" * 80)
    ps.sort_stats('tottime')
    ps.print_stats(30)
    
    # Save detailed report to file
    output_file = "profile_cprofile.txt"
    with open(output_file, 'w') as f:
        # Create a new Stats object with file as stream
        ps_file = pstats.Stats(profiler, stream=f)
        ps_file.sort_stats('cumulative')
        ps_file.print_stats()
    print(f"\nDetailed profile saved to: {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Profile acestep/inference.py")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Path to checkpoints directory"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="acestep-v15-turbo-rl",
        help="Model config path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--lm-model",
        type=str,
        default="acestep-5Hz-lm-0.6B-v3",
        help="LM model path"
    )
    parser.add_argument(
        "--lm-backend",
        type=str,
        default="vllm",
        help="LM backend"
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Enable warmup run before profiling (default: False, profile first run)"
    )
    
    args = parser.parse_args()
    
    # Initialize handlers
    print("Initializing handlers...")
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()
    
    # Initialize DiT
    print("  - Initializing DiT model...")
    status_dit, success_dit = dit_handler.initialize_service(
        project_root=project_root,
        config_path=args.config_path,
        device=args.device,
    )
    if not success_dit:
        print(f"  ❌ DiT initialization failed: {status_dit}")
        sys.exit(1)
    print("  ✓ DiT model initialized")
    
    # Initialize LLM
    print("  - Initializing LLM model...")
    status_llm, success_llm = llm_handler.initialize(
        checkpoint_dir=args.checkpoint_dir,
        lm_model_path=args.lm_model,
        backend=args.lm_backend,
        device=args.device,
    )
    if success_llm:
        print("  ✓ LM model initialized")
    else:
        print(f"  ⚠ LM initialization failed: {status_llm}")
    
    # Load test parameters from example file (same as acestep/inference.py)
    def load_example_config(example_file: str) -> Tuple[GenerationParams, GenerationConfig]:
        """Load configuration from an example JSON file."""
        try:
            with open(example_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert example format to GenerationParams and GenerationConfig
            # Handle time signature format (example uses "4" instead of "4/4")
            time_sig = data.get('timesignature', '')
            
            params = GenerationParams(
                caption=data.get('caption', ''),
                lyrics=data.get('lyrics', ''),
                bpm=data.get('bpm'),
                keyscale=data.get('keyscale', ''),
                timesignature=time_sig,
                vocal_language=data.get('language', 'unknown'),
                duration=data.get('duration'),
                thinking=data.get('think', False),
                inference_steps=data.get('inference_steps', 8),
                seed=42,
            )
            
            config = GenerationConfig()
            config.batch_size = data.get('batch_size', 1)
            
            return params, config
            
        except Exception as e:
            print(f"  ⚠ Failed to load example file: {e}")
            return None, None
    
    # Load production example (same as acestep/inference.py)
    example_file = os.path.join(project_root, "examples", "text2music", "example_05.json")
    
    if not os.path.exists(example_file):
        print(f"\n  ❌ Example file not found: {example_file}")
        print("     Please ensure the examples directory exists.")
        sys.exit(1)
    
    print(f"\n  Loading example: {os.path.basename(example_file)}")
    params, config = load_example_config(example_file)
    
    if not params or not config:
        print("  ❌ Failed to load example configuration")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Starting profiling...")
    print("=" * 80)
    
    result = profile_with_cprofile(dit_handler, llm_handler, params, config, warmup=args.warmup)
    
    if result and not result.success:
        print(f"\n⚠ Generation failed: {result.error}")


if __name__ == "__main__":
    main()

