"""
ACE-Step V1.5 Pipeline
Handler wrapper connecting model and UI
"""
import os
import sys

# Load environment variables from .env file in project root
# This allows configuration without hardcoding values
# Falls back to .env.example if .env is not found
try:
    from dotenv import load_dotenv
    # Get project root directory
    _current_file = os.path.abspath(__file__)
    _project_root = os.path.dirname(os.path.dirname(_current_file))
    _env_path = os.path.join(_project_root, '.env')
    _env_example_path = os.path.join(_project_root, '.env.example')
    
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        print(f"Loaded configuration from {_env_path}")
    elif os.path.exists(_env_example_path):
        load_dotenv(_env_example_path)
        print(f"Loaded configuration from {_env_example_path} (fallback)")
except ImportError:
    # python-dotenv not installed, skip loading .env
    pass

# Clear proxy settings that may affect Gradio
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

try:
    # When executed as a module: `python -m acestep.acestep_v15_pipeline`
    from .handler import AceStepHandler
    from .llm_inference import LLMHandler
    from .dataset_handler import DatasetHandler
    from .gradio_ui import create_gradio_interface
except ImportError:
    # When executed as a script: `python acestep/acestep_v15_pipeline.py`
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.dataset_handler import DatasetHandler
    from acestep.gradio_ui import create_gradio_interface


def create_demo(init_params=None, language='en'):
    """
    Create Gradio demo interface
    
    Args:
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
                    Keys: 'pre_initialized' (bool), 'checkpoint', 'config_path', 'device',
                          'init_llm', 'lm_model_path', 'backend', 'use_flash_attention',
                          'offload_to_cpu', 'offload_dit_to_cpu', 'init_status',
                          'dit_handler', 'llm_handler' (initialized handlers if pre-initialized),
                          'language' (UI language code)
        language: UI language code ('en', 'zh', 'ja', default: 'en')
    
    Returns:
        Gradio Blocks instance
    """
    # Use pre-initialized handlers if available, otherwise create new ones
    if init_params and init_params.get('pre_initialized') and 'dit_handler' in init_params:
        dit_handler = init_params['dit_handler']
        llm_handler = init_params['llm_handler']
    else:
        dit_handler = AceStepHandler()  # DiT handler
        llm_handler = LLMHandler()      # LM handler
    
    dataset_handler = DatasetHandler()  # Dataset handler
    
    # Create Gradio interface with all handlers and initialization parameters
    demo = create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=init_params, language=language)
    
    return demo


def get_gpu_memory_gb():
    """
    Get GPU memory in GB. Returns 0 if no GPU is available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Get total memory of the first GPU in GB
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)  # Convert bytes to GB
            return memory_gb
        else:
            return 0
    except Exception as e:
        print(f"Warning: Failed to detect GPU memory: {e}", file=sys.stderr)
        return 0


def main():
    """Main entry function"""
    import argparse
    
    # Detect GPU memory to auto-configure offload settings
    gpu_memory_gb = get_gpu_memory_gb()
    auto_offload = gpu_memory_gb > 0 and gpu_memory_gb < 16
    
    if auto_offload:
        print(f"Detected GPU memory: {gpu_memory_gb:.2f} GB (< 16GB)")
        print("Auto-enabling CPU offload to reduce GPU memory usage")
    elif gpu_memory_gb > 0:
        print(f"Detected GPU memory: {gpu_memory_gb:.2f} GB (>= 16GB)")
        print("CPU offload disabled by default")
    else:
        print("No GPU detected, running on CPU")
    
    parser = argparse.ArgumentParser(description="Gradio Demo for ACE-Step V1.5")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the gradio server on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name (default: 127.0.0.1, use 0.0.0.0 for all interfaces)")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh", "ja"], help="UI language: en (English), zh (中文), ja (日本語)")
    
    # Service mode argument
    parser.add_argument("--service_mode", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, 
                       help="Enable service mode (default: False). When enabled, uses preset models and restricts UI options.")
    
    # Service initialization arguments
    parser.add_argument("--init_service", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Initialize service on startup (default: False)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path (optional, for display purposes)")
    parser.add_argument("--config_path", type=str, default=None, help="Main model path (e.g., 'acestep-v15-turbo')")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Processing device (default: auto)")
    parser.add_argument("--init_llm", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help="Initialize 5Hz LM (default: True)")
    parser.add_argument("--lm_model_path", type=str, default=None, help="5Hz LM model path (e.g., 'acestep-5Hz-lm-0.6B')")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "pt"], help="5Hz LM backend (default: vllm)")
    parser.add_argument("--use_flash_attention", type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, help="Use flash attention (default: auto-detect)")
    parser.add_argument("--offload_to_cpu", type=lambda x: x.lower() in ['true', '1', 'yes'], default=auto_offload, help=f"Offload models to CPU (default: {'True' if auto_offload else 'False'}, auto-detected based on GPU VRAM)")
    parser.add_argument("--offload_dit_to_cpu", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Offload DiT to CPU (default: False)")

    # API mode argument
    parser.add_argument("--enable-api", action="store_true", help="Enable API endpoints (default: False)")

    # Authentication arguments
    parser.add_argument("--auth-username", type=str, default=None, help="Username for Gradio authentication")
    parser.add_argument("--auth-password", type=str, default=None, help="Password for Gradio authentication")
    parser.add_argument("--api-key", type=str, default=None, help="API key for API endpoints authentication")

    args = parser.parse_args()
    
    # Service mode defaults (can be configured via .env file)
    if args.service_mode:
        print("Service mode enabled - applying preset configurations...")
        # Force init_service in service mode
        args.init_service = True
        # Default DiT model for service mode (from env or fallback)
        if args.config_path is None:
            args.config_path = os.environ.get(
                "SERVICE_MODE_DIT_MODEL",
                "acestep-v15-turbo-fix-inst-shift-dynamic"
            )
        # Default LM model for service mode (from env or fallback)
        if args.lm_model_path is None:
            args.lm_model_path = os.environ.get(
                "SERVICE_MODE_LM_MODEL",
                "acestep-5Hz-lm-1.7B-v4-fix"
            )
        # Backend for service mode (from env or fallback to vllm)
        args.backend = os.environ.get("SERVICE_MODE_BACKEND", "vllm")
        print(f"  DiT model: {args.config_path}")
        print(f"  LM model: {args.lm_model_path}")
        print(f"  Backend: {args.backend}")
    
    try:
        init_params = None
        
        # If init_service is True, perform initialization before creating UI
        if args.init_service:
            print("Initializing service from command line...")
            
            # Create handler instances for initialization
            dit_handler = AceStepHandler()
            llm_handler = LLMHandler()
            
            # Auto-select config_path if not provided
            if args.config_path is None:
                available_models = dit_handler.get_available_acestep_v15_models()
                if available_models:
                    args.config_path = "acestep-v15-turbo" if "acestep-v15-turbo" in available_models else available_models[0]
                    print(f"Auto-selected config_path: {args.config_path}")
                else:
                    print("Error: No available models found. Please specify --config_path", file=sys.stderr)
                    sys.exit(1)
            
            # Get project root (same logic as in handler)
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            
            # Determine flash attention setting
            use_flash_attention = args.use_flash_attention
            if use_flash_attention is None:
                use_flash_attention = dit_handler.is_flash_attention_available()
            
            # Initialize DiT handler
            print(f"Initializing DiT model: {args.config_path} on {args.device}...")
            init_status, enable_generate = dit_handler.initialize_service(
                project_root=project_root,
                config_path=args.config_path,
                device=args.device,
                use_flash_attention=use_flash_attention,
                compile_model=False,
                offload_to_cpu=args.offload_to_cpu,
                offload_dit_to_cpu=args.offload_dit_to_cpu
            )
            
            if not enable_generate:
                print(f"Error initializing DiT model: {init_status}", file=sys.stderr)
                sys.exit(1)
            
            print(f"DiT model initialized successfully")
            
            # Initialize LM handler if requested
            lm_status = ""
            if args.init_llm:
                if args.lm_model_path is None:
                    # Try to get default LM model
                    available_lm_models = llm_handler.get_available_5hz_lm_models()
                    if available_lm_models:
                        args.lm_model_path = available_lm_models[0]
                        print(f"Using default LM model: {args.lm_model_path}")
                    else:
                        print("Warning: No LM models available, skipping LM initialization", file=sys.stderr)
                        args.init_llm = False
                
                if args.init_llm and args.lm_model_path:
                    checkpoint_dir = os.path.join(project_root, "checkpoints")
                    print(f"Initializing 5Hz LM: {args.lm_model_path} on {args.device}...")
                    lm_status, lm_success = llm_handler.initialize(
                        checkpoint_dir=checkpoint_dir,
                        lm_model_path=args.lm_model_path,
                        backend=args.backend,
                        device=args.device,
                        offload_to_cpu=args.offload_to_cpu,
                        dtype=dit_handler.dtype
                    )
                    
                    if lm_success:
                        print(f"5Hz LM initialized successfully")
                        init_status += f"\n{lm_status}"
                    else:
                        print(f"Warning: 5Hz LM initialization failed: {lm_status}", file=sys.stderr)
                        init_status += f"\n{lm_status}"
            
            # Prepare initialization parameters for UI
            init_params = {
                'pre_initialized': True,
                'service_mode': args.service_mode,
                'checkpoint': args.checkpoint,
                'config_path': args.config_path,
                'device': args.device,
                'init_llm': args.init_llm,
                'lm_model_path': args.lm_model_path,
                'backend': args.backend,
                'use_flash_attention': use_flash_attention,
                'offload_to_cpu': args.offload_to_cpu,
                'offload_dit_to_cpu': args.offload_dit_to_cpu,
                'init_status': init_status,
                'enable_generate': enable_generate,
                'dit_handler': dit_handler,
                'llm_handler': llm_handler,
                'language': args.language
            }
            
            print("Service initialization completed successfully!")
        
        # Create and launch demo
        print(f"Creating Gradio interface with language: {args.language}...")
        demo = create_demo(init_params=init_params, language=args.language)
        
        # Enable queue for multi-user support
        # This ensures proper request queuing and prevents concurrent generation conflicts
        print("Enabling queue for multi-user support...")
        demo.queue(
            max_size=20,  # Maximum queue size (adjust based on your needs)
            status_update_rate="auto",  # Update rate for queue status
        )

        # Enable API endpoints if requested
        if args.enable_api:
            print("Enabling API endpoints...")
            from acestep.gradio_ui.api_routes import setup_api_routes
            setup_api_routes(demo, dit_handler, llm_handler, api_key=args.api_key)
            if args.api_key:
                print("API authentication enabled")
            print("API endpoints enabled: /health, /v1/models, /release_task, /query_result, /create_random_sample, /format_lyrics")

        print(f"Launching server on {args.server_name}:{args.port}...")

        # Setup authentication if provided
        auth = None
        if args.auth_username and args.auth_password:
            auth = (args.auth_username, args.auth_password)
            print("Authentication enabled")

        demo.launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True,
            prevent_thread_lock=False,
            inbrowser=False,
            auth=auth,
        )
    except Exception as e:
        print(f"Error launching Gradio: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
