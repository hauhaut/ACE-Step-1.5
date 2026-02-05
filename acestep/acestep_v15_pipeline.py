"""
ACE-Step V1.5 Pipeline
Handler wrapper connecting model and UI
"""
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

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
        logger.info(f"Loaded configuration from {_env_path}")
    elif os.path.exists(_env_example_path):
        load_dotenv(_env_example_path)
        logger.info(f"Loaded configuration from {_env_example_path} (fallback)")
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
    from .gpu_config import get_gpu_config, get_gpu_memory_gb, print_gpu_config_info, set_global_gpu_config
except ImportError:
    # When executed as a script: `python acestep/acestep_v15_pipeline.py`
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.dataset_handler import DatasetHandler
    from acestep.gradio_ui import create_gradio_interface
    from acestep.gpu_config import get_gpu_config, get_gpu_memory_gb, print_gpu_config_info, set_global_gpu_config


# Module-level singleton handlers - persist across page refreshes
_dit_handler_instance = None
_llm_handler_instance = None
_dit_handler_lock = threading.Lock()
_llm_handler_lock = threading.Lock()


def get_dit_handler():
    """Get or create singleton DiT handler instance"""
    global _dit_handler_instance
    if _dit_handler_instance is None:
        with _dit_handler_lock:
            if _dit_handler_instance is None:
                _dit_handler_instance = AceStepHandler()
    return _dit_handler_instance


def get_llm_handler():
    """Get or create singleton LLM handler instance"""
    global _llm_handler_instance
    if _llm_handler_instance is None:
        with _llm_handler_lock:
            if _llm_handler_instance is None:
                _llm_handler_instance = LLMHandler()
    return _llm_handler_instance


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
    # Use pre-initialized handlers if available, otherwise use singletons
    if init_params and init_params.get('pre_initialized') and 'dit_handler' in init_params:
        dit_handler = init_params['dit_handler']
        llm_handler = init_params['llm_handler']
    else:
        # Use singleton handlers - persists model state across page refreshes
        dit_handler = get_dit_handler()
        llm_handler = get_llm_handler()
    
    dataset_handler = DatasetHandler()  # Dataset handler
    
    # Create Gradio interface with all handlers and initialization parameters
    demo = create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=init_params, language=language)
    
    return demo


def main():
    """Main entry function"""
    import argparse

    def str_to_bool(value: str) -> bool:
        """Convert string to boolean with proper validation."""
        if value.lower() in ('true', '1', 'yes'):
            return True
        if value.lower() in ('false', '0', 'no'):
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}. Use true/false, yes/no, or 1/0.")

    # Detect GPU memory and get configuration
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)  # Set global config for use across modules
    
    gpu_memory_gb = gpu_config.gpu_memory_gb
    auto_offload = gpu_memory_gb > 0 and gpu_memory_gb < 16
    
    # Log GPU configuration info
    logger.info(f"\n{'='*60}")
    logger.info("GPU Configuration Detected:")
    logger.info(f"{'='*60}")
    logger.info(f"  GPU Memory: {gpu_memory_gb:.2f} GB")
    logger.info(f"  Configuration Tier: {gpu_config.tier}")
    logger.info(f"  Max Duration (with LM): {gpu_config.max_duration_with_lm}s ({gpu_config.max_duration_with_lm // 60} min)")
    logger.info(f"  Max Duration (without LM): {gpu_config.max_duration_without_lm}s ({gpu_config.max_duration_without_lm // 60} min)")
    logger.info(f"  Max Batch Size (with LM): {gpu_config.max_batch_size_with_lm}")
    logger.info(f"  Max Batch Size (without LM): {gpu_config.max_batch_size_without_lm}")
    logger.info(f"  Default LM Init: {gpu_config.init_lm_default}")
    logger.info(f"  Available LM Models: {gpu_config.available_lm_models or 'None'}")
    logger.info(f"{'='*60}\n")

    if auto_offload:
        logger.info(f"Auto-enabling CPU offload (GPU < 16GB)")
    elif gpu_memory_gb > 0:
        logger.info(f"CPU offload disabled by default (GPU >= 16GB)")
    else:
        logger.info("No GPU detected, running on CPU")
    
    parser = argparse.ArgumentParser(description="Gradio Demo for ACE-Step V1.5")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the gradio server on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name (default: 127.0.0.1, use 0.0.0.0 for all interfaces)")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh", "ja"], help="UI language: en (English), zh (中文), ja (日本語)")
    
    # Service mode argument
    parser.add_argument("--service_mode", type=str_to_bool, default=False, 
                       help="Enable service mode (default: False). When enabled, uses preset models and restricts UI options.")
    
    # Service initialization arguments
    parser.add_argument("--init_service", type=str_to_bool, default=False, help="Initialize service on startup (default: False)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path (optional, for display purposes)")
    parser.add_argument("--config_path", type=str, default=None, help="Main model path (e.g., 'acestep-v15-turbo')")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "xpu"], help="Processing device (default: auto)")
    parser.add_argument("--init_llm", type=str_to_bool, default=None, help="Initialize 5Hz LM (default: auto based on GPU memory)")
    parser.add_argument("--lm_model_path", type=str, default=None, help="5Hz LM model path (e.g., 'acestep-5Hz-lm-0.6B')")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "pt"], help="5Hz LM backend (default: vllm)")
    parser.add_argument("--use_flash_attention", type=str_to_bool, default=None, help="Use flash attention (default: auto-detect)")
    parser.add_argument("--offload_to_cpu", type=str_to_bool, default=auto_offload, help=f"Offload models to CPU (default: {'True' if auto_offload else 'False'}, auto-detected based on GPU VRAM)")
    parser.add_argument("--offload_dit_to_cpu", type=str_to_bool, default=False, help="Offload DiT to CPU (default: False)")
    parser.add_argument("--download-source", type=str, default=None, choices=["huggingface", "modelscope", "auto"], help="Preferred model download source (default: auto-detect based on network)")

    # API mode argument
    parser.add_argument("--enable-api", action="store_true", help="Enable API endpoints (default: False)")

    # Authentication arguments
    parser.add_argument("--auth-username", type=str, default=None, help="Username for Gradio authentication")
    parser.add_argument("--auth-password", type=str, default=None, help="Password for Gradio authentication")
    parser.add_argument("--api-key", type=str, default=None, help="API key for API endpoints authentication")

    args = parser.parse_args()

    # Enable API requires init_service
    if args.enable_api:
        args.init_service = True
        # Load config from .env if not specified
        if args.config_path is None:
            args.config_path = os.environ.get("ACESTEP_CONFIG_PATH")
        if args.lm_model_path is None:
            args.lm_model_path = os.environ.get("ACESTEP_LM_MODEL_PATH")
        if os.environ.get("ACESTEP_LM_BACKEND"):
            args.backend = os.environ.get("ACESTEP_LM_BACKEND")

    # Service mode defaults (can be configured via .env file)
    if args.service_mode:
        logger.info("Service mode enabled - applying preset configurations...")
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
        logger.info(f"  DiT model: {args.config_path}")
        logger.info(f"  LM model: {args.lm_model_path}")
        logger.info(f"  Backend: {args.backend}")
    
    try:
        init_params = None
        dit_handler = None
        llm_handler = None

        # If init_service is True, perform initialization before creating UI
        if args.init_service:
            logger.info("Initializing service from command line...")

            # Use singleton handler instances for initialization
            dit_handler = get_dit_handler()
            llm_handler = get_llm_handler()
            
            # Auto-select config_path if not provided
            if args.config_path is None:
                available_models = dit_handler.get_available_acestep_v15_models()
                if available_models:
                    args.config_path = "acestep-v15-turbo" if "acestep-v15-turbo" in available_models else available_models[0]
                    logger.info(f"Auto-selected config_path: {args.config_path}")
                else:
                    logger.error("Error: No available models found. Please specify --config_path")
                    sys.exit(1)
            
            # Get project root (same logic as in handler)
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            
            # Determine flash attention setting
            use_flash_attention = args.use_flash_attention
            if use_flash_attention is None:
                use_flash_attention = dit_handler.is_flash_attention_available()

            # Determine download source preference
            prefer_source = None
            if args.download_source and args.download_source != "auto":
                prefer_source = args.download_source
                logger.info(f"Using preferred download source: {prefer_source}")

            # Initialize DiT handler
            logger.info(f"Initializing DiT model: {args.config_path} on {args.device}...")
            init_status, enable_generate = dit_handler.initialize_service(
                project_root=project_root,
                config_path=args.config_path,
                device=args.device,
                use_flash_attention=use_flash_attention,
                compile_model=False,
                offload_to_cpu=args.offload_to_cpu,
                offload_dit_to_cpu=args.offload_dit_to_cpu,
                prefer_source=prefer_source
            )

            if not enable_generate:
                logger.error(f"Error initializing DiT model: {init_status}")
                sys.exit(1)

            logger.info(f"DiT model initialized successfully")
            
            # Initialize LM handler if requested
            # Auto-determine init_llm based on GPU config if not explicitly set
            if args.init_llm is None:
                args.init_llm = gpu_config.init_lm_default
                logger.info(f"Auto-setting init_llm to {args.init_llm} based on GPU configuration")
            
            lm_status = ""
            if args.init_llm:
                if args.lm_model_path is None:
                    # Try to get default LM model
                    available_lm_models = llm_handler.get_available_5hz_lm_models()
                    if available_lm_models:
                        args.lm_model_path = available_lm_models[0]
                        logger.info(f"Using default LM model: {args.lm_model_path}")
                    else:
                        logger.warning("No LM models available, skipping LM initialization")
                        args.init_llm = False

                if args.init_llm and args.lm_model_path:
                    checkpoint_dir = os.path.join(project_root, "checkpoints")
                    logger.info(f"Initializing 5Hz LM: {args.lm_model_path} on {args.device}...")
                    lm_status, lm_success = llm_handler.initialize(
                        checkpoint_dir=checkpoint_dir,
                        lm_model_path=args.lm_model_path,
                        backend=args.backend,
                        device=args.device,
                        offload_to_cpu=args.offload_to_cpu,
                        dtype=dit_handler.dtype
                    )

                    if lm_success:
                        logger.info(f"5Hz LM initialized successfully")
                        init_status += f"\n{lm_status}"
                    else:
                        logger.warning(f"5Hz LM initialization failed: {lm_status}")
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
                'language': args.language,
                'gpu_config': gpu_config,  # Pass GPU config to UI
            }
            
            logger.info("Service initialization completed successfully!")

        # Create and launch demo
        logger.info(f"Creating Gradio interface with language: {args.language}...")
        
        # If not using init_service, still pass gpu_config to init_params
        if init_params is None:
            init_params = {
                'gpu_config': gpu_config,
                'language': args.language,
            }
        
        demo = create_demo(init_params=init_params, language=args.language)
        
        # Enable queue for multi-user support
        # This ensures proper request queuing and prevents concurrent generation conflicts
        logger.info("Enabling queue for multi-user support...")
        demo.queue(
            max_size=20,  # Maximum queue size (adjust based on your needs)
            status_update_rate="auto",  # Update rate for queue status
        )

        logger.info(f"Launching server on {args.server_name}:{args.port}...")

        # Setup authentication if provided
        auth = None
        if args.auth_username and args.auth_password:
            auth = (args.auth_username, args.auth_password)
            logger.info("Authentication enabled")

        # Enable API endpoints if requested
        if args.enable_api:
            logger.info("Enabling API endpoints...")
            from acestep.gradio_ui.api_routes import setup_api_routes

            # Launch Gradio first with prevent_thread_lock=True
            demo.launch(
                server_name=args.server_name,
                server_port=args.port,
                share=args.share,
                debug=args.debug,
                show_error=True,
                prevent_thread_lock=True,  # Don't block, so we can add routes
                inbrowser=False,
                auth=auth,
            )

            # Now add API routes to Gradio's FastAPI app (app is available after launch)
            setup_api_routes(demo, dit_handler, llm_handler, api_key=args.api_key)

            if args.api_key:
                logger.info("API authentication enabled")
            logger.info("API endpoints enabled: /health, /v1/models, /release_task, /query_result, /create_random_sample, /format_lyrics")

            # Keep the main thread alive
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
        else:
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
        logger.error(f"Error launching Gradio: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
