#!/usr/bin/env python3
"""
Test script to simulate what Kilocode/Cursor would send.
This helps us verify that we can receive and manipulate context.
"""

import os
import sys
import httpx
from dotenv import load_dotenv

load_dotenv()

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if not API_KEY:
    print("❌ Set OPENROUTER_API_KEY in .env")
    sys.exit(1)


def test_simple_request():
    """Test a simple 2-message request."""
    print("\n" + "="*60)
    print("TEST 1: Simple Request (2 messages)")
    print("="*60)
    
    response = httpx.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json={
            "model": "google/gemini-2.0-flash-001",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in exactly 3 words."}
            ],
            "max_tokens": 50
        },
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        timeout=60.0
    )
    
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"Response: {content}")
    else:
        print(f"Error: {response.text}")


def test_long_conversation():
    """Test a longer conversation (what Kilocode would send)."""
    print("\n" + "="*60)
    print("TEST 2: Long Conversation (simulating Kilocode)")
    print("="*60)
    
    # Simulate what an AI IDE would send - lots of context
    messages = [
        {"role": "system", "content": """You are an expert software engineer helping with code.
The user is working on a Python project with FastAPI.
Always provide clear, correct code."""},
        
        {"role": "user", "content": "I'm building an API gateway. Here's my current code:\n\n```python\nfrom fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\ndef root():\n    return {'hello': 'world'}\n```"},
        
        {"role": "assistant", "content": "That's a good starting point! I can see you have a basic FastAPI application. What would you like to add to it?"},
        
        {"role": "user", "content": "I want to add a POST endpoint that accepts JSON data and validates it with Pydantic."},
        
        {"role": "assistant", "content": """Here's how to add a validated POST endpoint:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    quantity: int = 1

@app.post('/items')
def create_item(item: Item):
    return {"item": item, "total": item.price * item.quantity}
```

This creates an `/items` endpoint that validates the incoming JSON."""},
        
        {"role": "user", "content": "Perfect! Now I also need to add authentication. Can you show me how to add API key auth?"},
        
        {"role": "assistant", "content": """Sure! Here's how to add API key authentication:

```python
from fastapi import FastAPI, Header, HTTPException

@app.post('/items')
def create_item(
    item: Item,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    if x_api_key != "secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"item": item, "total": item.price * item.quantity}
```"""},
        
        {"role": "user", "content": "Great! One more thing - how do I add logging?"}
    ]
    
    response = httpx.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json={
            "model": "google/gemini-2.0-flash-001",
            "messages": messages,
            "max_tokens": 300
        },
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        timeout=60.0
    )
    
    print(f"Status: {response.status_code}")
    
    # Check our custom headers
    for key, value in response.headers.items():
        if key.startswith("x-context"):
            print(f"📊 {key}: {value}")
    
    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"\n📝 Response preview:\n{content[:500]}...")
    else:
        print(f"Error: {response.text}")


def test_with_tool_calls():
    """Test a request that includes tool call context."""
    print("\n" + "="*60)
    print("TEST 3: Request with Tool/Function context")
    print("="*60)
    
    messages = [
        {"role": "system", "content": "You are an assistant. Use the provided tools when needed."},
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Tokyo"}'
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"temp": 22, "condition": "sunny"}'
        },
        {"role": "assistant", "content": "The weather in Tokyo is 22°C and sunny."},
        {"role": "user", "content": "And what about tomorrow?"}
    ]
    
    response = httpx.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json={
            "model": "google/gemini-2.0-flash-001",
            "messages": messages,
            "max_tokens": 100
        },
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        timeout=60.0
    )
    
    print(f"Status: {response.status_code}")
    
    for key, value in response.headers.items():
        if key.startswith("x-context"):
            print(f"📊 {key}: {value}")


if __name__ == "__main__":
    print("🧪 Context Optimizer Gateway - Test Suite")
    print("="*60)
    print(f"Gateway: {GATEWAY_URL}")
    print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    
    test_simple_request()
    test_long_conversation()
    test_with_tool_calls()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
