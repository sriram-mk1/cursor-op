import httpx
import asyncio
import time
import random

async def send_test_request(session_id: str):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "x-v1-key": "v1-ae887ef2485a47e9a2425da7fbca33852ad50c1149814",
        "Content-Type": "application/json"
    }
    
    topics = ['quantum computing', 'ancient rome', 'machine learning', 'baking', 'space travel']
    topic = random.choice(topics)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Let's talk about {topic}. What are the basics?"}
    ]
    
    # Simulate a growing conversation
    history_len = random.randint(2, 15)
    for i in range(history_len):
        messages.insert(1, {"role": "user", "content": f"Detail {i} about {topic}: " + " ".join([topic] * random.randint(5, 20))})
        messages.insert(2, {"role": "assistant", "content": f"That's interesting! Here is more info about {topic} part {i}."})

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "stream": False
    }
    
    # Occasionally omit session_id to test fallback
    if random.random() > 0.3:
        payload["session_id"] = session_id
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            start = time.time()
            # We use a dummy OR key if needed, but the gateway will handle it
            resp = await client.post(url, json=payload, headers=headers)
            print(f"[{session_id}] Sent {len(messages)} msgs | Status: {resp.status_code} | {time.time()-start:.2f}s")
        except Exception as e:
            print(f"Error: {e}")

async def main():
    sessions = ["session-alpha", "session-beta", "session-gamma"]
    while True:
        session = random.choice(sessions)
        await send_test_request(session)
        await asyncio.sleep(random.uniform(2, 5))

if __name__ == "__main__":
    asyncio.run(main())
