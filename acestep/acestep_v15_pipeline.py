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


def create_demo():
    """
    Create Gradio demo interface
    
    Returns:
        Gradio Blocks instance
    """
    # Create independent handler instances
    dit_handler = AceStepHandler()  # DiT handler
    llm_handler = LLMHandler()      # LM handler
    dataset_handler = DatasetHandler()  # Dataset handler
    
    # Create Gradio interface with all handlers
    demo = create_gradio_interface(dit_handler, llm_handler, dataset_handler)
    
    return demo


def main():
    """Main entry function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gradio Demo for ACE-Step V1.5")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the gradio server on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name (default: 127.0.0.1, use 0.0.0.0 for all interfaces)")
    args = parser.parse_args()
    
    try:
        # Create and launch demo
        print("Creating Gradio interface...")
        demo = create_demo()
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
