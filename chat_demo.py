import os
import time
import asyncio
import logging
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv
import httpx
import tiktoken
from context_optimizer.engine import ContextOptimizer

# Load environment variables
load_dotenv()

# Configure logging to show optimization details
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("chat-demo")

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"  # Fast and cheap for demo
SESSION_ID = "demo-session-1"

if not OPENROUTER_API_KEY:
    print("❌ Error: OPENROUTER_API_KEY not found in .env")
    exit(1)

def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        return len(text) // 4

async def ingest_codebase(optimizer: ContextOptimizer):
    """Ingest the current project files into the optimizer."""
    print(f"\n{'='*60}")
    print("📂 INGESTING CODEBASE...")
    print(f"{'='*60}")
    
    files_to_ingest = [
        "main.py",
        "context_optimizer/engine.py",
        "README.md",
        "requirements.txt"
    ]
    
    events = []
    total_tokens = 0
    
    for file_path in files_to_ingest:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                tokens = count_tokens(content)
                total_tokens += tokens
                
                events.append({
                    "role": "tool",
                    "source": "file_read",
                    "content": f"File: {file_path}\n\n{content}",
                    "metadata": {"path": file_path}
                })
                print(f"  ✓ Loaded {file_path} ({tokens} tokens)")
    
    start_time = time.time()
    optimizer.ingest(SESSION_ID, events)
    duration = time.time() - start_time
    
    print(f"\n✅ Ingestion complete in {duration:.3f}s")
    print(f"   Total Context Size: {total_tokens:,} tokens")
    print(f"{'='*60}\n")

async def chat_loop(optimizer: ContextOptimizer):
    print(f"💬 Starting Chat with RAG (Model: {MODEL})")
    print("   Ask questions about the codebase! (Type 'quit' to exit)\n")
    
    conversation_history = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break
                if not user_input:
                    continue
                
                # 1. Optimize Context
                print(f"\n{'─'*60}")
                print("🧠 OPTIMIZING CONTEXT...")
                
                start_opt = time.time()
                
                # We optimize based on the user's query against the ingested codebase
                optimization_result = optimizer.optimize(
                    session_id=SESSION_ID,
                    query_text=user_input,
                    target_token_budget=2000 # Keep context under 2k tokens
                )
                
                opt_duration = time.time() - start_opt
                
                # Display Optimization Stats
                orig_tokens = optimization_result['original_tokens']
                opt_tokens = optimization_result['optimized_tokens']
                savings = optimization_result['percent_saved']
                
                print(f"   ⏱️  Time: {opt_duration:.3f}s")
                print(f"   📉 Reduction: {orig_tokens} → {opt_tokens} tokens ({savings:.1f}% saved)")
                print(f"   📑 Selected Chunks: {len(optimization_result['optimized_context'])}")
                
                # Show what chunks were selected (briefly)
                print("   🔍 Top Relevant Content:")
                for i, chunk in enumerate(optimization_result['optimized_context'][:3]):
                    preview = chunk['content'].replace('\n', ' ')[:80]
                    print(f"      {i+1}. {preview}...")
                print(f"{'─'*60}\n")

                # 2. Construct Prompt
                # System prompt gets the optimized context
                system_content = "You are a helpful AI assistant answering questions about a codebase.\n"
                system_content += "Use the following context to answer the user's question:\n\n"
                
                for chunk in optimization_result['optimized_context']:
                    system_content += f"---\n{chunk['content']}\n"
                
                messages = [
                    {"role": "system", "content": system_content},
                ]
                
                # Add recent conversation history (last 2 turns) for continuity
                messages.extend(conversation_history[-4:]) 
                messages.append({"role": "user", "content": user_input})

                # 3. Call LLM
                print("🤖 Assistant: ", end="", flush=True)
                
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "Context Optimizer Demo"
                    },
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "stream": False # Simple non-streaming for now to keep code clean
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data['choices'][0]['message']['content']
                    print(answer)
                    
                    # Update history
                    conversation_history.append({"role": "user", "content": user_input})
                    conversation_history.append({"role": "assistant", "content": answer})
                else:
                    print(f"❌ API Error: {response.status_code} - {response.text}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Initialize Optimizer
    optimizer = ContextOptimizer()
    
    # Run Async Loop
    asyncio.run(ingest_codebase(optimizer))
    asyncio.run(chat_loop(optimizer))
