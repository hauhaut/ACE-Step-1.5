#!/usr/bin/env python3
"""
Test client for ACE-Step OpenRouter API.

Usage:
    python test_client.py [--url URL] [--prompt PROMPT]
"""

import argparse
import base64
import json
import sys
import time

import httpx

DEFAULT_URL = "http://localhost:8000"
DEFAULT_PROMPT = "Create an upbeat electronic dance music track"


def test_models(url: str) -> bool:
    """Test GET /api/v1/models"""
    print("\n=== Testing GET /api/v1/models ===")
    try:
        r = httpx.get(f"{url}/api/v1/models", timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"✅ Found {len(data['data'])} model(s)")
        for m in data["data"]:
            print(f"   - {m['id']}: {m['name']}")
            print(f"     Description: {m['description']}")
            print(f"     Modalities: Input {m['input_modalities']}, Output {m['output_modalities']}")
            print(f"     Pricing: ${m['pricing']['prompt']}/token (prompt), ${m['pricing']['completion']}/token (completion), ${m['pricing']['request']}/request")
            if m.get('supported_sampling_parameters'):
                print(f"     Sampling Parameters: {', '.join(m['supported_sampling_parameters'])}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_completion(url: str, prompt: str, output_file: str = "output.mp3") -> bool:
    """Test POST /v1/chat/completions"""
    print(f"\n=== Testing POST /v1/chat/completions ===")
    print(f"Prompt: {prompt[:60]}...")
    
    payload = {
        "model": "acestep/music-gen-v1",
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["audio"],  # 指定输出模态为音频
        "duration": 30,  # Short for testing
    }
    
    try:
        start = time.time()
        r = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=300)
        elapsed = time.time() - start
        
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"Status: {r.status_code}")
        
        if r.status_code != 200:
            print(f"❌ Error: {r.text}")
            return False
        
        data = r.json()
        print(f"✅ ID: {data['id']}")
        
        audio = data["choices"][0]["message"].get("audio")
        if audio:
            audio_bytes = base64.b64decode(audio["data"])
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            print(f"💾 Saved: {output_file} ({len(audio_bytes)/1024:.1f} KB)")
        
        return True
    except httpx.TimeoutException:
        print("❌ Timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_health(url: str) -> bool:
    """Test GET /health"""
    print("\n=== Testing GET /health ===")
    try:
        r = httpx.get(f"{url}/health", timeout=5)
        print(f"✅ {r.json()}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", default="output.mp3")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation test")
    args = parser.parse_args()
    
    print(f"Server: {args.url}")
    
    results = {
        "health": test_health(args.url),
        "models": test_models(args.url),
    }
    
    if not args.skip_gen:
        results["completion"] = test_completion(args.url, args.prompt, args.output)
    
    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'✅' if ok else '❌'}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
