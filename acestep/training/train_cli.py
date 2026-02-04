#!/usr/bin/env python3
"""CLI for ACE-Step LoRA training."""

import argparse
import os
import sys
import signal

import torch
from loguru import logger

from acestep.training.configs import LoRAConfig, TrainingConfig
from acestep.training.trainer import LoRATrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA adapters for ACE-Step",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tensor-dir", required=True, help="Preprocessed tensor directory")
    parser.add_argument("--output-dir", default="./lora_output", help="Output directory")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", default="acestep-v15-turbo", help="Model name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate tensor directory
    if not os.path.exists(args.tensor_dir):
        logger.error(f"Tensor directory not found: {args.tensor_dir}")
        sys.exit(1)

    # Initialize handler
    from acestep.handler import AceStepHandler
    handler = AceStepHandler()

    logger.info("Initializing model...")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    status, success = handler.initialize_service(
        project_root=project_root,
        config_path=args.model,
        device="auto",
    )
    if not success:
        logger.error(f"Failed to initialize: {status}")
        sys.exit(1)
    logger.info(status)

    # Create configs
    lora_config = LoRAConfig(
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_epochs=args.max_epochs,
        save_every_n_epochs=args.save_every,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Training state for interrupt handling
    training_state = {"should_stop": False}

    def handle_interrupt(signum, frame):
        logger.warning("Interrupt received, stopping training...")
        training_state["should_stop"] = True

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    # Create trainer and run
    trainer = LoRATrainer(
        dit_handler=handler,
        lora_config=lora_config,
        training_config=training_config,
    )

    logger.info(f"Starting training from {args.tensor_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"Training: lr={args.learning_rate}, batch={args.batch_size}, epochs={args.max_epochs}")

    for step, loss, status in trainer.train_from_preprocessed(
        tensor_dir=args.tensor_dir,
        training_state=training_state,
        resume_from=args.resume_from,
    ):
        logger.info(f"[Step {step}] loss={loss:.4f} | {status}")
        if training_state["should_stop"]:
            break

    logger.info("Training finished")


if __name__ == "__main__":
    main()
