"""Test client for OpenRouter-compatible ACE-Step API.

Usage:
    python -m openrouter.client_test --base-url http://127.0.0.1:8002
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime

import requests


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("Testing: GET /health")
    print("=" * 60)
    
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_list_models(base_url: str, api_key: str = None) -> bool:
    """Test models list endpoint."""
    print("\n" + "=" * 60)
    print("Testing: GET /api/v1/models")
    print("=" * 60)
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        resp = requests.get(f"{base_url}/api/v1/models", headers=headers, timeout=10)
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_chat_completions(
    base_url: str,
    prompt: str,
    lyrics: str = "",
    api_key: str = None,
    save_audio: bool = True,
) -> bool:
    """Test chat completions endpoint."""
    print("\n" + "=" * 60)
    print("Testing: POST /v1/chat/completions")
    print("=" * 60)
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Build request
    request_body = {
        "model": "acestep/music-v1.5",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "modalities": ["audio"],
        "temperature": 0.85,
        "lyrics": lyrics,
        "vocal_language": "en",
    }
    
    print(f"Request: {json.dumps(request_body, indent=2)}")
    print("\nGenerating music... (this may take a while)")
    
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=request_body,
            timeout=300,  # 5 minutes timeout for generation
        )
        
        print(f"\nStatus: {resp.status_code}")
        
        if resp.status_code != 200:
            print(f"Error Response: {resp.text}")
            return False
        
        data = resp.json()
        
        # Print response without audio data (too large)
        display_data = data.copy()
        if display_data.get("choices"):
            for choice in display_data["choices"]:
                if choice.get("message", {}).get("audio", {}).get("data"):
                    audio_data = choice["message"]["audio"]["data"]
                    choice["message"]["audio"]["data"] = f"<base64 audio, {len(audio_data)} chars>"
        
        print(f"Response: {json.dumps(display_data, indent=2)}")
        
        # Save audio if requested
        if save_audio and data.get("choices"):
            audio_info = data["choices"][0].get("message", {}).get("audio", {})
            audio_base64 = audio_info.get("data", "")
            
            if audio_base64:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"test_output_{timestamp}.mp3"
                
                audio_bytes = base64.b64decode(audio_base64)
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
                
                print(f"\nAudio saved to: {output_file}")
                print(f"Audio size: {len(audio_bytes)} bytes")
        
        return True
        
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test OpenRouter API client")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8002",
        help="API base URL (default: http://127.0.0.1:8002)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="API key for authentication",
    )
    parser.add_argument(
        "--prompt",
        default="A cheerful pop song with upbeat rhythm and catchy melody",
        help="Music generation prompt",
    )
    parser.add_argument(
        "--lyrics",
        default="",
        help="Optional lyrics for the song",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip music generation test (only test health and models)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save generated audio to file",
    )
    args = parser.parse_args()
    
    print(f"Testing ACE-Step OpenRouter API at: {args.base_url}")
    
    results = []
    
    # Test health
    results.append(("Health Check", test_health(args.base_url)))
    
    # Test models list
    results.append(("List Models", test_list_models(args.base_url, args.api_key)))
    
    # Test chat completions (music generation)
    if not args.skip_generation:
        results.append((
            "Chat Completions",
            test_chat_completions(
                args.base_url,
                args.prompt,
                args.lyrics,
                args.api_key,
                save_audio=not args.no_save,
            ),
        ))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
