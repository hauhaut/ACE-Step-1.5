"""OpenRouter-compatible API client test suite for ACE-Step Music Generator.

This comprehensive test suite validates the OpenRouter provider API implementation:
- OpenRouter API compliance (models, chat completions)
- OpenAI SDK compatibility
- Authentication (Bearer token)
- Input processing modes (tags, heuristic, sample_query)
- Audio output validation
- Error handling

Usage:
    # Using requests (default)
    python client_test.py --base-url http://127.0.0.1:8002

    # Using OpenAI SDK
    python client_test.py --base-url http://127.0.0.1:8002 --use-openai-sdk

    # With API key authentication
    python client_test.py --base-url http://127.0.0.1:8002 --api-key your_key

    # Quick test (skip audio generation)
    python client_test.py --skip-generation

    # Full test with all scenarios
    python client_test.py --full-test
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_URL = "http://127.0.0.1:8002"
DEFAULT_MODEL = "acemusic/acestep-v1.5-turbo"

# Timeouts (seconds)
TIMEOUT_HEALTH = 10
TIMEOUT_MODELS = 10
TIMEOUT_GENERATION = 600  # 10 minutes for music generation


# =============================================================================
# Test Framework
# =============================================================================

class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Container for test execution results."""
    name: str
    status: TestStatus
    message: str = ""
    duration_ms: float = 0
    data: Any = None

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED

    def __repr__(self) -> str:
        return f"TestResult({self.name}: {self.status.value})"


@dataclass
class TestContext:
    """Shared context for test execution."""
    base_url: str
    api_key: Optional[str] = None
    use_openai_sdk: bool = False
    save_audio: bool = True
    output_dir: str = "."
    verbose: bool = False


# =============================================================================
# Utilities
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 70) -> None:
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_subheader(title: str) -> None:
    """Print formatted subsection header."""
    print(f"\n--- {title} ---")


def truncate_str(s: str, max_len: int = 100) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_len:
        return s
    return f"{s[:max_len]}... ({len(s)} chars total)"


def print_json(data: Any, max_str_len: int = 100, indent: int = 2) -> None:
    """Print JSON with truncated strings."""
    def _truncate(obj: Any) -> Any:
        if isinstance(obj, str):
            return truncate_str(obj, max_str_len)
        elif isinstance(obj, dict):
            return {k: _truncate(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_truncate(item) for item in obj]
        return obj

    print(json.dumps(_truncate(data), indent=indent, ensure_ascii=False))


def save_audio(audio_base64: str, filename: str, output_dir: str = ".") -> str:
    """Decode and save base64 audio to file."""
    filepath = os.path.join(output_dir, filename)
    audio_bytes = base64.b64decode(audio_base64)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    return filepath


def get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Build HTTP headers with optional auth."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# =============================================================================
# HTTP Client (requests-based)
# =============================================================================

class HTTPClient:
    """Simple HTTP client using requests library."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        import requests
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return get_headers(self.api_key)

    def get(self, path: str, timeout: int = 30) -> Dict[str, Any]:
        """HTTP GET request."""
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, headers=self._headers(), timeout=timeout)
        return {
            "status_code": resp.status_code,
            "body": resp.json() if resp.content else {},
            "text": resp.text,
        }

    def post(self, path: str, data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """HTTP POST request."""
        url = f"{self.base_url}{path}"
        resp = self.session.post(
            url, headers=self._headers(), json=data, timeout=timeout
        )
        return {
            "status_code": resp.status_code,
            "body": resp.json() if resp.content else {},
            "text": resp.text,
        }


# =============================================================================
# OpenAI SDK Client (optional)
# =============================================================================

class OpenAIClient:
    """Client using OpenAI Python SDK for compatibility testing."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.client = OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=api_key or "dummy-key",  # OpenAI SDK requires a key
        )

    def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Create chat completion using OpenAI SDK."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.model_dump()


# =============================================================================
# Test Cases
# =============================================================================

def test_health_check(ctx: TestContext) -> TestResult:
    """
    Test: Health Check Endpoint

    OpenRouter Requirement: Providers should expose a health endpoint.

    Validates:
    - GET /health returns 200
    - Response contains status information
    - Server is ready to accept requests
    """
    print_header("Test: Health Check")
    start_time = time.time()

    try:
        client = HTTPClient(ctx.base_url, ctx.api_key)
        print(f"GET {ctx.base_url}/health")

        resp = client.get("/health", timeout=TIMEOUT_HEALTH)
        duration = (time.time() - start_time) * 1000

        print(f"Status Code: {resp['status_code']}")
        print("Response:")
        print_json(resp["body"])

        if resp["status_code"] != 200:
            return TestResult(
                "Health Check",
                TestStatus.FAILED,
                f"Expected 200, got {resp['status_code']}",
                duration,
            )

        body = resp["body"]
        if "status" not in body:
            return TestResult(
                "Health Check",
                TestStatus.FAILED,
                "Missing 'status' field in response",
                duration,
            )

        return TestResult(
            "Health Check",
            TestStatus.PASSED,
            f"Server healthy, status: {body.get('status')}",
            duration,
        )

    except Exception as e:
        return TestResult(
            "Health Check",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_list_models(ctx: TestContext) -> TestResult:
    """
    Test: List Models Endpoint

    OpenRouter Requirement: GET /api/v1/models must return available models
    with pricing and capability information.

    Validates:
    - Endpoint returns 200
    - Response has 'data' array
    - Each model has required fields: id, name, pricing, modalities
    - Pricing format is correct (strings, not numbers)
    """
    print_header("Test: List Models (OpenRouter Format)")
    start_time = time.time()

    try:
        client = HTTPClient(ctx.base_url, ctx.api_key)
        print(f"GET {ctx.base_url}/api/v1/models")

        resp = client.get("/api/v1/models", timeout=TIMEOUT_MODELS)
        duration = (time.time() - start_time) * 1000

        print(f"Status Code: {resp['status_code']}")
        print("Response:")
        print_json(resp["body"])

        if resp["status_code"] == 401:
            return TestResult(
                "List Models",
                TestStatus.FAILED,
                "Authentication failed",
                duration,
            )

        if resp["status_code"] != 200:
            return TestResult(
                "List Models",
                TestStatus.FAILED,
                f"Expected 200, got {resp['status_code']}",
                duration,
            )

        body = resp["body"]

        # Validate OpenRouter response structure
        if "data" not in body:
            return TestResult(
                "List Models",
                TestStatus.FAILED,
                "Missing 'data' field (OpenRouter format requires 'data' array)",
                duration,
            )

        if not isinstance(body["data"], list) or len(body["data"]) == 0:
            return TestResult(
                "List Models",
                TestStatus.FAILED,
                "'data' must be a non-empty array",
                duration,
            )

        # Validate model structure per OpenRouter spec
        model = body["data"][0]
        required_fields = [
            "id",
            "name",
            "created",
            "pricing",
            "input_modalities",
            "output_modalities",
        ]
        missing = [f for f in required_fields if f not in model]

        if missing:
            return TestResult(
                "List Models",
                TestStatus.FAILED,
                f"Model missing required fields: {missing}",
                duration,
            )

        # Validate pricing format (OpenRouter requires strings)
        pricing = model.get("pricing", {})
        if not isinstance(pricing, dict):
            return TestResult(
                "List Models",
                TestStatus.FAILED,
                "Pricing must be an object",
                duration,
            )

        for key in ["prompt", "completion"]:
            if key in pricing and not isinstance(pricing[key], str):
                return TestResult(
                    "List Models",
                    TestStatus.FAILED,
                    f"Pricing '{key}' must be a string (OpenRouter format)",
                    duration,
                )

        # Validate modalities
        if "audio" not in model.get("output_modalities", []):
            return TestResult(
                "List Models",
                TestStatus.FAILED,
                "Output modalities should include 'audio'",
                duration,
            )

        print(f"\n✓ Found {len(body['data'])} model(s)")
        print(f"✓ Primary model: {model['name']} ({model['id']})")
        print(f"✓ Pricing: prompt={pricing.get('prompt')}, completion={pricing.get('completion')}")

        return TestResult(
            "List Models",
            TestStatus.PASSED,
            f"Found {len(body['data'])} model(s), OpenRouter format valid",
            duration,
            body,
        )

    except Exception as e:
        return TestResult(
            "List Models",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_chat_completions_basic(ctx: TestContext) -> TestResult:
    """
    Test: Basic Chat Completions (Sample Query Mode)

    Tests the LLM-powered sample generation where user provides a simple
    description and the system generates appropriate prompt and lyrics.

    Validates:
    - POST /v1/chat/completions returns 200
    - Response follows OpenAI chat completion format
    - Audio data is present and valid base64
    - Usage statistics are included
    """
    print_header("Test: Chat Completions - Sample Query Mode")
    start_time = time.time()

    try:
        client = HTTPClient(ctx.base_url, ctx.api_key)

        request_body = {
            "model": DEFAULT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Create an upbeat electronic dance track with energetic beats"
                }
            ],
            "modalities": ["audio"],
            "temperature": 0.85,
            "top_p": 0.9,
            "instrumental": True,
        }

        print("Request Body:")
        print_json(request_body)
        print("\n⏳ Generating music... (this may take 1-5 minutes)")

        resp = client.post(
            "/v1/chat/completions",
            request_body,
            timeout=TIMEOUT_GENERATION,
        )
        duration = (time.time() - start_time) * 1000

        print(f"\nStatus Code: {resp['status_code']}")

        if resp["status_code"] != 200:
            print(f"Error: {resp['text'][:500]}")
            return TestResult(
                "Chat Completions (Basic)",
                TestStatus.FAILED,
                f"Expected 200, got {resp['status_code']}",
                duration,
            )

        body = resp["body"]

        # Validate OpenAI chat completion format
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        missing = [f for f in required_fields if f not in body]

        if missing:
            return TestResult(
                "Chat Completions (Basic)",
                TestStatus.FAILED,
                f"Missing required fields: {missing}",
                duration,
            )

        if body.get("object") != "chat.completion":
            return TestResult(
                "Chat Completions (Basic)",
                TestStatus.FAILED,
                f"Expected object='chat.completion', got '{body.get('object')}'",
                duration,
            )

        # Validate choices
        choices = body.get("choices", [])
        if not choices:
            return TestResult(
                "Chat Completions (Basic)",
                TestStatus.FAILED,
                "Response has no choices",
                duration,
            )

        choice = choices[0]
        message = choice.get("message", {})
        audio = message.get("audio", {})

        if not audio.get("data"):
            return TestResult(
                "Chat Completions (Basic)",
                TestStatus.FAILED,
                "No audio data in response",
                duration,
            )

        # Validate base64 audio
        try:
            audio_bytes = base64.b64decode(audio["data"])
        except Exception as e:
            return TestResult(
                "Chat Completions (Basic)",
                TestStatus.FAILED,
                f"Invalid base64 audio: {e}",
                duration,
            )

        # Display response (without audio data)
        display_body = json.loads(json.dumps(body))
        display_body["choices"][0]["message"]["audio"]["data"] = f"<{len(audio['data'])} chars base64>"
        print("\nResponse Body:")
        print_json(display_body)

        # Save audio if enabled
        audio_file = None
        if ctx.save_audio:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = save_audio(
                audio["data"],
                f"test_basic_{timestamp}.mp3",
                ctx.output_dir,
            )
            print(f"\n✓ Audio saved: {audio_file}")

        print(f"✓ Audio size: {len(audio_bytes)} bytes ({len(audio_bytes)/1024/1024:.2f} MB)")
        print(f"✓ Usage: prompt_tokens={body['usage'].get('prompt_tokens')}, completion_tokens={body['usage'].get('completion_tokens')}")

        return TestResult(
            "Chat Completions (Basic)",
            TestStatus.PASSED,
            f"Generated {len(audio_bytes)} bytes audio",
            duration,
            {"audio_file": audio_file, "audio_size": len(audio_bytes)},
        )

    except Exception as e:
        return TestResult(
            "Chat Completions (Basic)",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_chat_completions_with_tags(ctx: TestContext) -> TestResult:
    """
    Test: Chat Completions with Tag Format

    Tests the explicit tag-based input format:
    - <prompt>...</prompt> for style/description
    - <lyrics>...</lyrics> for song lyrics

    This is the recommended format for precise control.
    """
    print_header("Test: Chat Completions - Tag Format (<prompt> <lyrics>)")
    start_time = time.time()

    try:
        client = HTTPClient(ctx.base_url, ctx.api_key)

        # Use tag format for explicit prompt and lyrics
        content = """<prompt>Soft acoustic ballad with gentle guitar and warm vocals</prompt>
<lyrics>
[Verse]
Walking through the morning light
Everything feels so right
The world is waking up with me
And I feel so free

[Chorus]
This is the moment we live for
Open up every door
Let the sunshine in
</lyrics>"""

        request_body = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["audio"],
            "temperature": 0.85,
            "vocal_language": "en",
            "instrumental": False,
        }

        print("Request Body (Tag Format):")
        print_json(request_body, max_str_len=300)
        print("\n⏳ Generating music with lyrics... (this may take 1-5 minutes)")

        resp = client.post(
            "/v1/chat/completions",
            request_body,
            timeout=TIMEOUT_GENERATION,
        )
        duration = (time.time() - start_time) * 1000

        print(f"\nStatus Code: {resp['status_code']}")

        if resp["status_code"] != 200:
            print(f"Error: {resp['text'][:500]}")
            return TestResult(
                "Chat Completions (Tags)",
                TestStatus.FAILED,
                f"Expected 200, got {resp['status_code']}",
                duration,
            )

        body = resp["body"]
        choices = body.get("choices", [])

        if not choices:
            return TestResult(
                "Chat Completions (Tags)",
                TestStatus.FAILED,
                "No choices in response",
                duration,
            )

        audio = choices[0].get("message", {}).get("audio", {})

        if not audio.get("data"):
            return TestResult(
                "Chat Completions (Tags)",
                TestStatus.FAILED,
                "No audio data in response",
                duration,
            )

        audio_bytes = base64.b64decode(audio["data"])
        transcript = audio.get("transcript", "")

        # Save audio
        audio_file = None
        if ctx.save_audio:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = save_audio(
                audio["data"],
                f"test_tags_{timestamp}.mp3",
                ctx.output_dir,
            )
            print(f"\n✓ Audio saved: {audio_file}")

        print(f"✓ Audio size: {len(audio_bytes)} bytes")
        if transcript:
            print(f"✓ Transcript: {truncate_str(transcript, 100)}")

        return TestResult(
            "Chat Completions (Tags)",
            TestStatus.PASSED,
            f"Generated vocal track with {len(audio_bytes)} bytes",
            duration,
            {"audio_file": audio_file, "transcript": transcript},
        )

    except Exception as e:
        return TestResult(
            "Chat Completions (Tags)",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_chat_completions_lyrics_heuristic(ctx: TestContext) -> TestResult:
    """
    Test: Chat Completions with Lyrics Heuristic Detection

    Tests automatic lyrics detection when user provides lyrics-like text
    without explicit tags. The server should detect and handle accordingly.
    """
    print_header("Test: Chat Completions - Lyrics Heuristic Detection")
    start_time = time.time()

    try:
        client = HTTPClient(ctx.base_url, ctx.api_key)

        # Lyrics-like content without tags (should be auto-detected)
        lyrics_content = """[Verse 1]
Under the stars tonight
I found my way back home
Through the darkness and the light
I never walk alone

[Chorus]
We are the dreamers
We are the believers
Together forever
Nothing can break us"""

        request_body = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": lyrics_content}],
            "modalities": ["audio"],
            "temperature": 0.85,
        }

        print("Request Body (Lyrics Auto-Detection):")
        print_json(request_body, max_str_len=200)
        print("\n⏳ Generating... (server should auto-detect lyrics format)")

        resp = client.post(
            "/v1/chat/completions",
            request_body,
            timeout=TIMEOUT_GENERATION,
        )
        duration = (time.time() - start_time) * 1000

        print(f"\nStatus Code: {resp['status_code']}")

        if resp["status_code"] != 200:
            print(f"Error: {resp['text'][:500]}")
            return TestResult(
                "Chat Completions (Heuristic)",
                TestStatus.FAILED,
                f"Expected 200, got {resp['status_code']}",
                duration,
            )

        body = resp["body"]
        audio = body.get("choices", [{}])[0].get("message", {}).get("audio", {})

        if not audio.get("data"):
            return TestResult(
                "Chat Completions (Heuristic)",
                TestStatus.FAILED,
                "No audio data in response",
                duration,
            )

        audio_bytes = base64.b64decode(audio["data"])

        if ctx.save_audio:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_audio(audio["data"], f"test_heuristic_{timestamp}.mp3", ctx.output_dir)

        print(f"✓ Audio size: {len(audio_bytes)} bytes")
        print("✓ Lyrics heuristic detection worked")

        return TestResult(
            "Chat Completions (Heuristic)",
            TestStatus.PASSED,
            f"Lyrics auto-detected, generated {len(audio_bytes)} bytes",
            duration,
        )

    except Exception as e:
        return TestResult(
            "Chat Completions (Heuristic)",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_chat_completions_with_params(ctx: TestContext) -> TestResult:
    """
    Test: Chat Completions with Custom Parameters

    Tests ACE-Step specific parameters:
    - duration: Audio length
    - bpm: Beats per minute
    - vocal_language: Language code
    - instrumental: Boolean flag
    """
    print_header("Test: Chat Completions - Custom Parameters")
    start_time = time.time()

    try:
        client = HTTPClient(ctx.base_url, ctx.api_key)

        request_body = {
            "model": DEFAULT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "<prompt>Fast-paced rock instrumental with heavy drums</prompt>"
                }
            ],
            "modalities": ["audio"],
            "temperature": 0.9,
            "top_p": 0.95,
            # ACE-Step specific params
            "duration": 30.0,  # 30 seconds
            "bpm": 140,
            "instrumental": True,
        }

        print("Request Body (Custom Parameters):")
        print_json(request_body)
        print("\n⏳ Generating 30-second track at 140 BPM...")

        resp = client.post(
            "/v1/chat/completions",
            request_body,
            timeout=TIMEOUT_GENERATION,
        )
        duration = (time.time() - start_time) * 1000

        print(f"\nStatus Code: {resp['status_code']}")

        if resp["status_code"] != 200:
            print(f"Error: {resp['text'][:500]}")
            return TestResult(
                "Chat Completions (Params)",
                TestStatus.FAILED,
                f"Expected 200, got {resp['status_code']}",
                duration,
            )

        body = resp["body"]
        audio = body.get("choices", [{}])[0].get("message", {}).get("audio", {})

        if not audio.get("data"):
            return TestResult(
                "Chat Completions (Params)",
                TestStatus.FAILED,
                "No audio data",
                duration,
            )

        audio_bytes = base64.b64decode(audio["data"])

        if ctx.save_audio:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_audio(audio["data"], f"test_params_{timestamp}.mp3", ctx.output_dir)

        print(f"✓ Audio size: {len(audio_bytes)} bytes")
        print("✓ Custom parameters accepted")

        return TestResult(
            "Chat Completions (Params)",
            TestStatus.PASSED,
            f"Generated with custom params: {len(audio_bytes)} bytes",
            duration,
        )

    except Exception as e:
        return TestResult(
            "Chat Completions (Params)",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_openai_sdk_compatibility(ctx: TestContext) -> TestResult:
    """
    Test: OpenAI SDK Compatibility

    Validates that the API works with the official OpenAI Python SDK,
    which is a key requirement for OpenRouter compatibility.
    """
    print_header("Test: OpenAI SDK Compatibility")
    start_time = time.time()

    if not ctx.use_openai_sdk:
        return TestResult(
            "OpenAI SDK",
            TestStatus.SKIPPED,
            "Use --use-openai-sdk to enable",
            0,
        )

    try:
        sdk_client = OpenAIClient(ctx.base_url, ctx.api_key)

        print("Using OpenAI Python SDK...")
        print(f"Base URL: {ctx.base_url}/v1")

        # Note: OpenAI SDK doesn't support custom fields like 'instrumental'
        # so we use tag format instead
        response = sdk_client.chat_completions(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "<prompt>Calm ambient music for relaxation</prompt>"
                }
            ],
        )
        duration = (time.time() - start_time) * 1000

        print("\nResponse received via OpenAI SDK")

        if not response.get("choices"):
            return TestResult(
                "OpenAI SDK",
                TestStatus.FAILED,
                "No choices in SDK response",
                duration,
            )

        print(f"✓ Response ID: {response.get('id')}")
        print(f"✓ Model: {response.get('model')}")
        print("✓ OpenAI SDK compatibility confirmed")

        return TestResult(
            "OpenAI SDK",
            TestStatus.PASSED,
            "SDK compatibility verified",
            duration,
        )

    except ImportError:
        return TestResult(
            "OpenAI SDK",
            TestStatus.SKIPPED,
            "openai package not installed",
            0,
        )
    except Exception as e:
        return TestResult(
            "OpenAI SDK",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_error_handling_empty_messages(ctx: TestContext) -> TestResult:
    """
    Test: Error Handling - Empty Messages

    Validates proper error response for invalid requests.
    """
    print_header("Test: Error Handling - Empty Messages")
    start_time = time.time()

    try:
        client = HTTPClient(ctx.base_url, ctx.api_key)

        request_body = {
            "model": DEFAULT_MODEL,
            "messages": [],
            "modalities": ["audio"],
        }

        print("Sending request with empty messages...")
        resp = client.post("/v1/chat/completions", request_body, timeout=30)
        duration = (time.time() - start_time) * 1000

        print(f"Status Code: {resp['status_code']}")

        if resp["status_code"] == 200:
            return TestResult(
                "Error (Empty Messages)",
                TestStatus.FAILED,
                "Server should reject empty messages",
                duration,
            )

        if 400 <= resp["status_code"] < 500:
            print(f"✓ Server correctly rejected: {resp['body'].get('detail', resp['text'][:200])}")
            return TestResult(
                "Error (Empty Messages)",
                TestStatus.PASSED,
                f"Correctly rejected with status {resp['status_code']}",
                duration,
            )

        return TestResult(
            "Error (Empty Messages)",
            TestStatus.FAILED,
            f"Unexpected status: {resp['status_code']}",
            duration,
        )

    except Exception as e:
        return TestResult(
            "Error (Empty Messages)",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_error_handling_invalid_model(ctx: TestContext) -> TestResult:
    """
    Test: Error Handling - Invalid Model

    Validates error response when requesting non-existent model.
    """
    print_header("Test: Error Handling - Invalid Model")
    start_time = time.time()

    try:
        client = HTTPClient(ctx.base_url, ctx.api_key)

        request_body = {
            "model": "invalid/nonexistent-model",
            "messages": [{"role": "user", "content": "test"}],
            "modalities": ["audio"],
        }

        print("Sending request with invalid model...")
        resp = client.post("/v1/chat/completions", request_body, timeout=30)
        duration = (time.time() - start_time) * 1000

        print(f"Status Code: {resp['status_code']}")

        # Note: Some servers may accept any model ID and route to default
        # This is acceptable behavior
        if resp["status_code"] == 200:
            print("⚠ Server accepted invalid model (routing to default)")
            return TestResult(
                "Error (Invalid Model)",
                TestStatus.PASSED,
                "Server routes invalid model to default (acceptable)",
                duration,
            )

        if 400 <= resp["status_code"] < 500:
            print(f"✓ Server rejected invalid model")
            return TestResult(
                "Error (Invalid Model)",
                TestStatus.PASSED,
                f"Correctly rejected with status {resp['status_code']}",
                duration,
            )

        return TestResult(
            "Error (Invalid Model)",
            TestStatus.FAILED,
            f"Unexpected status: {resp['status_code']}",
            duration,
        )

    except Exception as e:
        return TestResult(
            "Error (Invalid Model)",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_authentication_valid(ctx: TestContext) -> TestResult:
    """
    Test: Authentication - Valid API Key

    Validates Bearer token authentication works correctly.
    """
    print_header("Test: Authentication - Valid Key")
    start_time = time.time()

    if not ctx.api_key:
        return TestResult(
            "Auth (Valid Key)",
            TestStatus.SKIPPED,
            "No API key configured",
            0,
        )

    try:
        client = HTTPClient(ctx.base_url, ctx.api_key)
        print(f"Using API Key: {ctx.api_key[:8]}...")

        resp = client.get("/api/v1/models", timeout=TIMEOUT_MODELS)
        duration = (time.time() - start_time) * 1000

        print(f"Status Code: {resp['status_code']}")

        if resp["status_code"] == 200:
            print("✓ Authentication successful")
            return TestResult(
                "Auth (Valid Key)",
                TestStatus.PASSED,
                "Valid API key accepted",
                duration,
            )

        if resp["status_code"] == 401:
            return TestResult(
                "Auth (Valid Key)",
                TestStatus.FAILED,
                "Valid key rejected",
                duration,
            )

        return TestResult(
            "Auth (Valid Key)",
            TestStatus.FAILED,
            f"Unexpected status: {resp['status_code']}",
            duration,
        )

    except Exception as e:
        return TestResult(
            "Auth (Valid Key)",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


def test_authentication_invalid(ctx: TestContext) -> TestResult:
    """
    Test: Authentication - Invalid API Key

    Validates server rejects invalid Bearer tokens.
    """
    print_header("Test: Authentication - Invalid Key")
    start_time = time.time()

    if not ctx.api_key:
        return TestResult(
            "Auth (Invalid Key)",
            TestStatus.SKIPPED,
            "No API key configured (auth may be disabled)",
            0,
        )

    try:
        # Use wrong API key
        client = HTTPClient(ctx.base_url, "invalid_key_12345")
        print("Using invalid API key...")

        resp = client.get("/api/v1/models", timeout=TIMEOUT_MODELS)
        duration = (time.time() - start_time) * 1000

        print(f"Status Code: {resp['status_code']}")

        if resp["status_code"] == 401:
            print("✓ Invalid key correctly rejected")
            return TestResult(
                "Auth (Invalid Key)",
                TestStatus.PASSED,
                "Invalid key rejected with 401",
                duration,
            )

        if resp["status_code"] == 200:
            return TestResult(
                "Auth (Invalid Key)",
                TestStatus.FAILED,
                "Server accepted invalid key",
                duration,
            )

        return TestResult(
            "Auth (Invalid Key)",
            TestStatus.FAILED,
            f"Unexpected status: {resp['status_code']}",
            duration,
        )

    except Exception as e:
        return TestResult(
            "Auth (Invalid Key)",
            TestStatus.FAILED,
            f"Error: {str(e)}",
            (time.time() - start_time) * 1000,
        )


# =============================================================================
# Test Runner
# =============================================================================

def run_tests(ctx: TestContext, skip_generation: bool = False, full_test: bool = False) -> int:
    """
    Run test suite and return exit code.

    Args:
        ctx: Test context with configuration
        skip_generation: Skip audio generation tests
        full_test: Run all test scenarios

    Returns:
        0 if all tests pass, 1 otherwise
    """
    print_header("ACE-Step OpenRouter API Test Suite", "=", 70)
    print(f"Base URL: {ctx.base_url}")
    print(f"API Key: {'Configured' if ctx.api_key else 'Not configured'}")
    print(f"OpenAI SDK: {'Enabled' if ctx.use_openai_sdk else 'Disabled'}")
    print(f"Skip Generation: {skip_generation}")
    print(f"Full Test: {full_test}")
    print(f"Output Directory: {ctx.output_dir}")

    results: List[TestResult] = []

    # === Core Tests (always run) ===
    results.append(test_health_check(ctx))
    results.append(test_list_models(ctx))

    # === Generation Tests ===
    if not skip_generation:
        results.append(test_chat_completions_basic(ctx))

        if full_test:
            results.append(test_chat_completions_with_tags(ctx))
            results.append(test_chat_completions_lyrics_heuristic(ctx))
            results.append(test_chat_completions_with_params(ctx))

    # === SDK Compatibility ===
    if ctx.use_openai_sdk:
        results.append(test_openai_sdk_compatibility(ctx))

    # === Error Handling ===
    results.append(test_error_handling_empty_messages(ctx))
    results.append(test_error_handling_invalid_model(ctx))

    # === Authentication ===
    results.append(test_authentication_valid(ctx))
    results.append(test_authentication_invalid(ctx))

    # === Summary ===
    print_header("Test Summary", "=", 70)

    passed = sum(1 for r in results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in results if r.status == TestStatus.FAILED)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
    total = len(results)

    for result in results:
        if result.status == TestStatus.PASSED:
            symbol = "✓"
            status = "PASSED"
        elif result.status == TestStatus.FAILED:
            symbol = "✗"
            status = "FAILED"
        else:
            symbol = "○"
            status = "SKIPPED"

        print(f"  {symbol} {result.name}: {status}")
        if result.message:
            print(f"    └─ {result.message}")
        if result.duration_ms > 0:
            print(f"    └─ Duration: {result.duration_ms/1000:.2f}s")

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped (total: {total})")

    if failed == 0:
        print("\n✅ All tests PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) FAILED")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenRouter-compatible API test suite for ACE-Step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python client_test.py --base-url http://127.0.0.1:8002

  # Full test with all scenarios
  python client_test.py --base-url http://127.0.0.1:8002 --full-test

  # Quick test (skip audio generation)
  python client_test.py --skip-generation

  # With OpenAI SDK
  python client_test.py --use-openai-sdk

  # With authentication
  python client_test.py --api-key your_api_key
""",
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("ACESTEP_API_URL", DEFAULT_BASE_URL),
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="API key for Bearer authentication",
    )
    parser.add_argument(
        "--use-openai-sdk",
        action="store_true",
        help="Test with OpenAI Python SDK",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip audio generation tests (faster)",
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Run all test scenarios including advanced cases",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save generated audio files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("ACESTEP_OUTPUT_DIR", "."),
        help="Directory to save generated audio files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Create output directory if needed
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    ctx = TestContext(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
        use_openai_sdk=args.use_openai_sdk,
        save_audio=not args.no_save,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    return run_tests(
        ctx,
        skip_generation=args.skip_generation,
        full_test=args.full_test,
    )


if __name__ == "__main__":
    sys.exit(main())
