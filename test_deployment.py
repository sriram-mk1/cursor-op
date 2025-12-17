import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://cursor-op.up.railway.app/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-oss-20b:free"  # Free model for testing

if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env")
    exit(1)

def test_deployment():
    print(f"Testing deployment at: {API_URL}")
    print("-" * 50)

    # 1. Create a fake conversation history with REPETITIVE content to force retrieval
    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Here is some context about the project. The project is called CursorOp. It is a RAG system."},
        {"role": "assistant", "content": "I understand. CursorOp is a RAG system."},
        {"role": "user", "content": "The main file is main.py. It uses FastAPI."},
        {"role": "assistant", "content": "Okay, main.py uses FastAPI."},
        {"role": "user", "content": "Tell me about CursorOp and main.py."} # This should match the previous chunks
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "enable_optimization": True
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cursor-op.up.railway.app",
        "X-Title": "Deployment Test",
        "X-Session-Id": "test-session-123" # Fixed session ID
    }

    print(f"Sending request with {len(messages)} messages...")
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        duration = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Duration: {duration:.2f}s")
        
        print("\nAll Response Headers:")
        for k, v in response.headers.items():
            print(f"{k}: {v}")

        if response.status_code == 200:
            data = response.json()
            if 'choices' in data:
                content = data['choices'][0]['message']['content']
                print("\nResponse Content:")
                print("-" * 20)
                print(content[:200] + "..." if len(content) > 200 else content)
                print("-" * 20)
            else:
                print(f"Unexpected response format: {data}")
                    
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_deployment()
