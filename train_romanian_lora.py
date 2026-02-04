#!/usr/bin/env python
"""
Romanian LoRA Training Script for ACE-Step
Single-command training from audio directory to LoRA adapter.

Usage:
    python train_romanian_lora.py /path/to/romanian/audio

Expected directory structure:
    /path/to/romanian/audio/
        song1.mp3
        song1.txt   # lyrics (optional)
        song2.wav
        song2.txt   # lyrics (optional)
        ...

Output:
    ./romanian_lora/final/  # trained LoRA adapter
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

import torch
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Train Romanian LoRA for ACE-Step")
    parser.add_argument("audio_dir", type=str, help="Path to directory containing Romanian audio files")
    parser.add_argument("--output", type=str, default="./romanian_lora", help="Output directory for LoRA")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.15, help="LoRA dropout")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--custom-tag", type=str, default="romanian music", help="Custom tag for training")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("ACE-Step Romanian LoRA Training")
    print(f"{'='*60}")
    print(f"Audio directory: {audio_dir}")
    print(f"Output: {args.output}")
    print(f"Config: r={args.r}, alpha={args.alpha}, dropout={args.dropout}")
    print(f"Training: lr={args.lr}, epochs={args.epochs}")
    print(f"{'='*60}\n")

    # Step 1: Initialize DiT handler
    print("[1/5] Initializing DiT model...")
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    project_root = Path(__file__).parent.resolve()

    status, success = handler.initialize_service(
        project_root=str(project_root),
        config_path="acestep-v15-turbo",
        device=args.device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )

    if not success:
        print(f"Error initializing model: {status}")
        sys.exit(1)
    print(f"  Model loaded: {status}")

    # Step 2: Scan and build dataset
    print("\n[2/5] Scanning audio directory...")
    from acestep.training.dataset_builder import DatasetBuilder

    builder = DatasetBuilder()
    builder.set_custom_tag(args.custom_tag, tag_position="prepend")
    builder.set_all_instrumental(False)  # Romanian music likely has vocals

    samples, scan_status = builder.scan_directory(str(audio_dir))
    print(f"  {scan_status}")

    if not samples:
        print("Error: No audio files found")
        sys.exit(1)

    # Step 3: Label samples with basic metadata
    # Note: LLM labeling requires additional setup. For simplicity, use basic metadata.
    print("\n[3/5] Setting up sample metadata...")

    for i, sample in enumerate(samples):
        sample.labeled = True
        sample.caption = args.custom_tag
        sample.genre = "romanian music, folk, traditional"
        if sample.bpm is None:
            sample.bpm = 120
        if not sample.keyscale:
            sample.keyscale = "C Major"
        if not sample.timesignature:
            sample.timesignature = "4"
        # Keep lyrics from .txt files if present
        if not sample.lyrics or sample.lyrics == "[Instrumental]":
            if sample.raw_lyrics:
                sample.lyrics = sample.raw_lyrics

    print(f"  Prepared {len(samples)} samples with metadata")

    # Step 4: Preprocess to tensors
    print("\n[4/5] Preprocessing audio to tensors...")

    tensor_dir = tempfile.mkdtemp(prefix="acestep_tensors_")

    def progress_cb(msg):
        print(f"  {msg}")

    output_paths, preprocess_status = builder.preprocess_to_tensors(
        dit_handler=handler,
        output_dir=tensor_dir,
        max_duration=240.0,
        progress_callback=progress_cb,
    )
    print(f"  {preprocess_status}")

    if not output_paths:
        print("Error: No samples preprocessed")
        sys.exit(1)

    # Step 5: Train LoRA
    print(f"\n[5/5] Training LoRA ({args.epochs} epochs)...")
    from acestep.training.configs import LoRAConfig, TrainingConfig
    from acestep.training.trainer import LoRATrainer

    lora_config = LoRAConfig(
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=1,
        gradient_accumulation_steps=4,
        max_epochs=args.epochs,
        save_every_n_epochs=args.save_every,
        warmup_steps=50,
        weight_decay=0.01,
        max_grad_norm=1.0,
        output_dir=args.output,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
        log_every_n_steps=5,
    )

    trainer = LoRATrainer(
        dit_handler=handler,
        lora_config=lora_config,
        training_config=training_config,
    )

    # Run training
    final_loss = 0.0
    for step, loss, status in trainer.train_from_preprocessed(tensor_dir):
        print(f"  {status}")
        final_loss = loss

    # Cleanup temp directory
    import shutil
    shutil.rmtree(tensor_dir, ignore_errors=True)

    # Print usage instructions
    final_lora_path = Path(args.output) / "final"
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"LoRA saved to: {final_lora_path.resolve()}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"\nTo use the trained LoRA:")
    print(f"  1. Start the ACE-Step UI")
    print(f"  2. Go to the 'LoRA' tab")
    print(f"  3. Enter path: {final_lora_path.resolve()}")
    print(f"  4. Click 'Load LoRA'")
    print(f"\nOr use programmatically:")
    print(f'  handler.load_lora("{final_lora_path.resolve()}")')
    print(f'  handler.set_use_lora(True)')
    print(f"\nPrompt tip: Include '{args.custom_tag}' in your prompts")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
