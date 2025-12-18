import httpx
import os
from dotenv import load_dotenv

load_dotenv()

GATEWAY_URL = "http://localhost:8000"
API_KEY = os.getenv("OPENROUTER_API_KEY")

def test_adaptive():
    # Create a large history (> 10,000 chars)
    large_text = "This is some filler text to increase the character count. " * 200
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Let's talk about Python optimization."},
        {"role": "assistant", "content": "Optimization is key. " + large_text},
        {"role": "user", "content": "What about FastAPI performance?"},
        {"role": "assistant", "content": "FastAPI is very fast. " + large_text},
        {"role": "user", "content": "How do I use Pydantic V2?"}
    ]
    
    print(f"🚀 Sending request with {sum(len(m['content']) for m in messages):,} chars...")
    response = httpx.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json={
            "model": "google/gemini-2.0-flash-001",
            "messages": messages,
            "enable_optimization": True,
            "debug": True
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60.0
    )
    
    if response.status_code == 200:
        data = response.json()
        debug = data.get("_debug", {})
        print(f"✅ Success! {debug.get('original_count')} -> {debug.get('optimized_count')} msgs")
        print(f"Optimized content preview: {debug.get('reconstructed_messages')[1]['content'][:200]}...")
    else:
        print(f"❌ Failed: {response.text}")

if __name__ == "__main__":
    test_adaptive()
