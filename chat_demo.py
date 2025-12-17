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
MODEL = "openai/gpt-oss-20b:free"  # Fast and cheap for demo
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
                
                # --- 1. OPTIMIZED REQUEST ---
                print(f"\n{'─'*60}")
                print("🚀 RUN 1: OPTIMIZED REQUEST")
                
                start_opt = time.time()
                
                optimization_result = optimizer.optimize(
                    session_id=SESSION_ID,
                    query_text=user_input
                )
                
                opt_duration = time.time() - start_opt
                
                # Stats
                orig_tokens = optimization_result['original_tokens']
                opt_tokens = optimization_result['optimized_tokens']
                savings = optimization_result['percent_saved']
                
                print(f"   ⏱️  Optimization Time: {opt_duration:.3f}s")
                print(f"   📉 Context Reduction: {orig_tokens} → {opt_tokens} tokens ({savings:.1f}% saved)")
                print(f"   📑 Selected Chunks: {len(optimization_result['optimized_context'])}")
                
                # Build Optimized Prompt
                system_content_opt = "You are a helpful AI assistant answering questions about a codebase.\n"
                system_content_opt += "Use the following context to answer the user's question:\n\n"
                for chunk in optimization_result['optimized_context']:
                    system_content_opt += f"---\n{chunk['content']}\n"
                
                messages_opt = [{"role": "system", "content": system_content_opt}]
                messages_opt.extend(conversation_history[-4:]) 
                messages_opt.append({"role": "user", "content": user_input})

                # Call LLM (Optimized)
                print("   🤖 Calling LLM (Optimized)... ", end="", flush=True)
                start_llm_opt = time.time()
                response_opt = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "https://cursor-op.up.railway.app",
                        "X-Title": "Context Optimizer Demo"
                    },
                    json={"model": MODEL, "messages": messages_opt, "stream": False}
                )
                llm_duration_opt = time.time() - start_llm_opt
                
                if response_opt.status_code == 200:
                    answer_opt = response_opt.json()['choices'][0]['message']['content']
                    print(f"Done ({llm_duration_opt:.2f}s)")
                else:
                    print(f"Error: {response_opt.status_code}")
                    answer_opt = "Error"

                # --- 2. UNOPTIMIZED REQUEST (BASELINE) ---
                print(f"\n{'─'*60}")
                print("🐢 RUN 2: UNOPTIMIZED REQUEST (Full Context)")
                
                # Build Unoptimized Prompt (All chunks)
                all_chunks = optimizer.sessions.get(SESSION_ID, [])
                system_content_full = "You are a helpful AI assistant answering questions about a codebase.\n"
                system_content_full += "Use the following context to answer the user's question:\n\n"
                for chunk in all_chunks:
                    system_content_full += f"---\n{chunk['content']}\n"
                
                messages_full = [{"role": "system", "content": system_content_full}]
                messages_full.extend(conversation_history[-4:])
                messages_full.append({"role": "user", "content": user_input})
                
                full_tokens = count_tokens(system_content_full)
                print(f"   📦 Full Context Size: {full_tokens:,} tokens")

                # Call LLM (Unoptimized)
                print("   🤖 Calling LLM (Full)... ", end="", flush=True)
                start_llm_full = time.time()
                response_full = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "https://cursor-op.up.railway.app",
                        "X-Title": "Context Optimizer Demo"
                    },
                    json={"model": MODEL, "messages": messages_full, "stream": False}
                )
                llm_duration_full = time.time() - start_llm_full
                
                if response_full.status_code == 200:
                    answer_full = response_full.json()['choices'][0]['message']['content']
                    print(f"Done ({llm_duration_full:.2f}s)")
                else:
                    print(f"Error: {response_full.status_code}")
                    answer_full = "Error"

                # --- COMPARISON SUMMARY ---
                print(f"\n{'='*60}")
                print("📊 COMPARISON SUMMARY")
                print(f"{'='*60}")
                print(f"Metric          | Optimized      | Unoptimized")
                print(f"----------------|----------------|----------------")
                print(f"Input Tokens    | {opt_tokens:<14,} | {full_tokens:<14,}")
                print(f"LLM Latency     | {llm_duration_opt:<13.2f}s | {llm_duration_full:<13.2f}s")
                print(f"Total Time      | {opt_duration + llm_duration_opt:<13.2f}s | {llm_duration_full:<13.2f}s")
                print(f"{'='*60}\n")
                
                print(f"💡 Optimized Answer:\n{answer_opt}\n")
                
                # Update history with optimized answer
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": answer_opt})

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
