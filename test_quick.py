#!/usr/bin/env python3
"""
Quick test to verify the gateway works with OpenRouter API
Run this to make sure your setup is correct
"""

import os
import sys
from openai import OpenAI

def test_basic():
    """Test basic completion"""
    print("Testing basic completion...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("   Set it with: export OPENROUTER_API_KEY='sk-or-v1-...'")
        return False
    
    try:
        client = OpenAI(
            base_url="https://cursor-op.onrender.com/v1",
            api_key=api_key
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say 'Hello World' and nothing else."}
            ],
            max_tokens=10
        )
        
        content = response.choices[0].message.content
        print(f"✅ Response: {content}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_optimization():
    """Test context optimization"""
    print("\nTesting context optimization...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return False
    
    try:
        client = OpenAI(
            base_url="https://cursor-op.onrender.com/v1",
            api_key=api_key
        )
        
        # Create a conversation with 5+ messages
        messages = [
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "That's nice!"},
            {"role": "user", "content": "I love Python programming."},
            {"role": "assistant", "content": "Great choice!"},
            {"role": "user", "content": "What was my favorite color?"}
        ]
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            extra_body={
                "enable_optimization": True,
                "max_chunks": 10
            }
        )
        
        content = response.choices[0].message.content
        print(f"✅ Response with optimization: {content}")
        
        # Check if it remembered the color
        if "blue" in content.lower():
            print("✅ Context optimization working (remembered 'blue')")
        else:
            print("⚠️  Context might not be fully preserved")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_streaming():
    """Test streaming"""
    print("\nTesting streaming...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return False
    
    try:
        client = OpenAI(
            base_url="https://cursor-op.onrender.com/v1",
            api_key=api_key
        )
        
        stream = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Count from 1 to 3"}
            ],
            stream=True,
            max_tokens=20
        )
        
        print("✅ Streaming response: ", end="")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_json_mode():
    """Test JSON mode"""
    print("\nTesting JSON mode...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return False
    
    try:
        client = OpenAI(
            base_url="https://cursor-op.onrender.com/v1",
            api_key=api_key
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": "Return a JSON object with a 'status' field set to 'ok'"}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        print(f"✅ JSON response: {content}")
        
        # Try to parse JSON
        import json
        json.loads(content)
        print("✅ Valid JSON")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Context Optimizer Gateway - Quick Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Basic Completion", test_basic),
        ("Context Optimization", test_optimization),
        ("Streaming", test_streaming),
        ("JSON Mode", test_json_mode),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {e}")
            results.append(False)
    
    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Gateway is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check your API key and connection.")
        sys.exit(1)
