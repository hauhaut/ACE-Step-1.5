"""
ACE-Step V1.5 Pipeline
Handler wrapper connecting model and UI
"""
import os
import sys

# Clear proxy settings that may affect Gradio
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

from .handler import AceStepHandler
from .llm_inference import LLMHandler
from .dataset_handler import DatasetHandler
from .gradio_ui import create_gradio_interface


def create_demo(init_params=None):
    """
    Create Gradio demo interface
    
    Args:
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
                    Keys: 'pre_initialized' (bool), 'checkpoint', 'config_path', 'device',
                          'init_llm', 'lm_model_path', 'backend', 'use_flash_attention',
                          'offload_to_cpu', 'offload_dit_to_cpu', 'init_status',
                          'dit_handler', 'llm_handler' (initialized handlers if pre-initialized)
    
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
    demo = create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=init_params)
    
    return demo


def main():
    """Main entry function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gradio Demo for ACE-Step V1.5")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the gradio server on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name (default: 127.0.0.1, use 0.0.0.0 for all interfaces)")
    
    # Service initialization arguments
    parser.add_argument("--init_service", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Initialize service on startup (default: False)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path (optional, for display purposes)")
    parser.add_argument("--config_path", type=str, default=None, help="Main model path (e.g., 'acestep-v15-turbo')")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Processing device (default: auto)")
    parser.add_argument("--init_llm", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help="Initialize 5Hz LM (default: True)")
    parser.add_argument("--lm_model_path", type=str, default=None, help="5Hz LM model path (e.g., 'acestep-5Hz-lm-0.6B')")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "pt"], help="5Hz LM backend (default: vllm)")
    parser.add_argument("--use_flash_attention", type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, help="Use flash attention (default: auto-detect)")
    parser.add_argument("--offload_to_cpu", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Offload models to CPU (default: False)")
    parser.add_argument("--offload_dit_to_cpu", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Offload DiT to CPU (default: False)")
    
    args = parser.parse_args()
    
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
                'llm_handler': llm_handler
            }
            
            print("Service initialization completed successfully!")
        
        # Create and launch demo
        print("Creating Gradio interface...")
        demo = create_demo(init_params=init_params)
        print(f"Launching server on {args.server_name}:{args.port}...")
        demo.launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True,
            prevent_thread_lock=False,  # Keep thread locked to maintain server running
            inbrowser=False,  # Don't auto-open browser
        )
    except Exception as e:
        print(f"Error launching Gradio: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
