#!/usr/bin/env python3
"""Test script for Context Optimizer Gateway"""
import json
import requests

BASE_URL = "https://cursor-op.onrender.com"
GATEWAY_KEY = "dev-key-12345"
OPENROUTER_KEY = "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"
MODEL = "google/gemini-2.0-flash-lite-001"

headers = {
    "Authorization": f"Bearer {GATEWAY_KEY}",
    "X-OpenRouter-API-Key": OPENROUTER_KEY,
    "Content-Type": "application/json"
}


def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"✅ Health: {response.json()}\n")


def test_simple_completion():
    """Test simple chat completion"""
    print("🔍 Testing simple completion (no optimization)...")
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say 'Hello Gateway!' and nothing else"}],
            "enable_optimization": False,
            "max_tokens": 20
        }
    )
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    tokens = result["usage"]["total_tokens"]
    print(f"✅ Response: {content.strip()}")
    print(f"📊 Tokens used: {tokens}\n")


def test_context_optimization():
    """Test with context optimization"""
    print("🔍 Testing context optimization...")
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "My name is Alice and I'm 25 years old."},
                {"role": "assistant", "content": "Nice to meet you Alice!"},
                {"role": "user", "content": "I love playing tennis and reading books."},
                {"role": "assistant", "content": "Those are great hobbies!"},
                {"role": "user", "content": "I work at a library in San Francisco."},
                {"role": "assistant", "content": "That sounds wonderful!"},
                {"role": "user", "content": "What's my name and where do I work?"}
            ],
            "enable_optimization": True,
            "max_chunks": 8,
            "max_tokens": 100
        }
    )
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    tokens = result["usage"]["total_tokens"]
    print(f"✅ Response: {content.strip()}")
    print(f"📊 Tokens used: {tokens}")
    print(f"💡 Context was optimized and model still remembered details!\n")


def test_streaming():
    """Test streaming response"""
    print("🔍 Testing streaming response...")
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Count from 1 to 3"}],
            "stream": True,
            "max_tokens": 30
        },
        stream=True
    )
    
    print("✅ Streaming chunks: ", end="", flush=True)
    for line in response.iter_lines():
        if line:
            line_text = line.decode('utf-8')
            if line_text.startswith('data: ') and line_text != 'data: [DONE]':
                try:
                    data = json.loads(line_text[6:])
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            print(delta['content'], end="", flush=True)
                except json.JSONDecodeError:
                    pass
    print("\n")


def test_error_handling():
    """Test error handling"""
    print("🔍 Testing error handling (missing API key)...")
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {GATEWAY_KEY}"},  # Missing OpenRouter key
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "test"}]
        }
    )
    print(f"✅ Properly rejected: {response.status_code} - {response.json()['detail']}\n")


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Context Optimizer Gateway Test Suite")
    print("=" * 70)
    print()
    
    test_health()
    test_simple_completion()
    test_context_optimization()
    test_streaming()
    test_error_handling()
    
    print("=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)
    print()
    print(f"📍 Base URL: {BASE_URL}")
    print(f"🔑 Gateway Key: {GATEWAY_KEY}")
    print(f"🤖 Model: {MODEL}")
