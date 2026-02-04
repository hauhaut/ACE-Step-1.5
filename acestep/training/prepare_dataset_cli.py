#!/usr/bin/env python3
"""CLI for preparing LoRA training datasets."""
import argparse
import sys
from pathlib import Path


def cmd_scan(args):
    """Scan directory and create dataset JSON."""
    from acestep.training import DatasetBuilder

    builder = DatasetBuilder()
    if args.custom_tag:
        builder.set_custom_tag(args.custom_tag)
    builder.metadata.name = args.dataset_name

    samples, status = builder.scan_directory(args.audio_dir)
    print(status)

    if not samples:
        sys.exit(1)

    result = builder.save_dataset(args.output, args.dataset_name)
    print(result)


def cmd_preprocess(args):
    """Convert dataset JSON to training tensors."""
    from acestep.training import DatasetBuilder
    from acestep.handler import AceStepHandler

    # Load dataset
    builder = DatasetBuilder()
    samples, status = builder.load_dataset(args.dataset)
    print(status)
    if not samples:
        sys.exit(1)

    # Initialize handler
    handler = AceStepHandler()
    project_root = Path(__file__).parent.parent.parent
    checkpoint_dir = project_root / "checkpoints"

    # Find first available model
    models = [d.name for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("acestep-v15-")]
    if not models:
        print("No acestep-v15-* model found in checkpoints/")
        sys.exit(1)

    model_name = models[0]
    print(f"Using model: {model_name}")

    status_msg, ok = handler.initialize_service(
        project_root=str(project_root),
        config_path=model_name,
        device="auto",
    )
    print(status_msg)
    if not ok:
        sys.exit(1)

    # Preprocess
    paths, status = builder.preprocess_to_tensors(
        dit_handler=handler,
        output_dir=args.output_dir,
        max_duration=args.max_duration,
        progress_callback=lambda msg: print(f"  {msg}"),
    )
    print(status)


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for LoRA training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan subcommand
    scan_parser = subparsers.add_parser("scan", help="Scan directory and create dataset JSON")
    scan_parser.add_argument("--audio-dir", required=True, help="Directory with audio + lyrics files")
    scan_parser.add_argument("--output", required=True, help="Output JSON path")
    scan_parser.add_argument("--dataset-name", default="custom_dataset", help="Dataset name")
    scan_parser.add_argument("--custom-tag", help="LoRA activation tag (e.g., romanian_song)")
    scan_parser.set_defaults(func=cmd_scan)

    # preprocess subcommand
    preprocess_parser = subparsers.add_parser("preprocess", help="Convert dataset JSON to training tensors")
    preprocess_parser.add_argument("--dataset", required=True, help="Dataset JSON from scan step")
    preprocess_parser.add_argument("--output-dir", required=True, help="Tensor output directory")
    preprocess_parser.add_argument("--max-duration", type=float, default=240, help="Max audio duration in seconds")
    preprocess_parser.set_defaults(func=cmd_preprocess)

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)


if __name__ == "__main__":
    main()
