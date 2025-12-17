#!/usr/bin/env python3
"""
Interactive demo to test context optimization with real conversation and code.
"""

import logging
from context_optimizer import ContextOptimizer

# Setup logging to see everything
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def main():
    print("\n" + "="*80)
    print(" "*20 + "CONTEXT OPTIMIZER DEMO")
    print("="*80 + "\n")
    
    # Initialize optimizer
    optimizer = ContextOptimizer()
    session_id = "demo-session"
    
    # Create mock conversation with our actual code
    events = [
        {
            "role": "user",
            "source": "chat",
            "content": "I need to build a context optimizer that reduces token usage for LLM conversations."
        },
        {
            "role": "assistant",
            "source": "chat",
            "content": "I'll help you build a context optimizer. We'll use semantic embeddings to retrieve relevant chunks and aggressively shrink the context to fit within token budgets."
        },
        {
            "role": "tool",
            "source": "file_read",
            "content": """
# Context Optimizer Engine Code

import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class ContextOptimizer:
    def __init__(self):
        self.model = SentenceTransformer("minishlab/potion-base-2M", device="cpu", truncate_dim=128)
        self.sessions = {}
    
    def ingest(self, session_id: str, events: List[Dict]):
        # Chunk events into smaller pieces
        chunks = []
        for event in events:
            content = event.get('content', '')
            # Split into 200-token chunks
            chunks.extend(self._chunk_text(content, 200))
        self.sessions[session_id] = chunks
    
    def optimize(self, session_id: str, query: str, max_chunks: int = 12):
        # Retrieve relevant chunks using embeddings
        chunks = self.sessions.get(session_id, [])
        query_emb = self.model.encode(query)
        chunk_embs = self.model.encode(chunks)
        
        # Compute similarity and rank
        similarities = self.model.similarity(query_emb, chunk_embs)
        top_chunks = sorted(zip(similarities[0], chunks), reverse=True)[:max_chunks]
        
        # Shrink each chunk to 100 tokens max
        optimized = []
        for score, chunk in top_chunks:
            shrunk = self._shrink(chunk, max_tokens=100)
            optimized.append(shrunk)
        
        return optimized
"""
        },
        {
            "role": "user",
            "source": "chat",
            "content": "How does the chunking work? I want to make sure it splits content into smaller pieces."
        },
        {
            "role": "tool",
            "source": "file_read",
            "content": """
def _chunk_text(self, text: str, max_tokens: int = 200) -> List[str]:
    '''
    Split text into smaller chunks of max_tokens size.
    This ensures we have granular pieces for better retrieval.
    '''
    tokens = self.encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = self.encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks
"""
        },
        {
            "role": "assistant",
            "source": "chat",
            "content": "The chunking works by encoding the text into tokens using tiktoken, then splitting it into fixed-size chunks of 200 tokens each. This creates smaller, more granular pieces that can be individually ranked by relevance."
        },
        {
            "role": "user",
            "source": "chat",
            "content": "What about the shrinking step? How do we reduce token count?"
        },
        {
            "role": "tool",
            "source": "file_read",
            "content": """
# Aggressive shrinking logic
def _shrink(self, content: str, max_tokens: int = 100) -> str:
    tokens = self.encoding.encode(content)
    if len(tokens) <= max_tokens:
        return content
    
    # Take first max_tokens and add ellipsis
    shrunk_tokens = tokens[:max_tokens]
    return self.encoding.decode(shrunk_tokens) + "..."
"""
        },
        {
            "role": "assistant",
            "source": "chat",
            "content": "The shrinking step takes each selected chunk and truncates it to a maximum of 100 tokens. If a chunk is longer, we keep only the first 100 tokens and add an ellipsis. This aggressively reduces the total token count while preserving the most important information at the start of each chunk."
        },
        {
            "role": "user",
            "source": "chat",
            "content": "Can you show me the FastAPI integration code?"
        },
        {
            "role": "tool",
            "source": "file_read",
            "content": """
# FastAPI Integration (main.py)

from fastapi import FastAPI, Header
from context_optimizer import ContextOptimizer

app = FastAPI()
optimizer = ContextOptimizer()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, authorization: str = Header(None)):
    messages = request.messages
    
    # Extract history and current query
    history = messages[:-1]
    current_query = messages[-1]['content']
    
    # Optimize context
    session_id = request.user or "default"
    optimizer.ingest(session_id, history)
    result = optimizer.optimize(session_id, current_query, max_chunks=12)
    
    # Inject optimized context into system message
    optimized_messages = [
        {"role": "system", "content": f"Previous context:\\n{result['optimized_context']}"},
        messages[-1]
    ]
    
    # Forward to OpenRouter
    response = await forward_to_openrouter(optimized_messages)
    return response
"""
        },
        {
            "role": "assistant",
            "source": "chat",
            "content": "The FastAPI integration extracts the conversation history, ingests it into the optimizer, retrieves and shrinks the most relevant chunks, then injects the optimized context into a system message before forwarding to OpenRouter. This reduces the token count by 40-90% while preserving relevance."
        }
    ]
    
    # Ingest the conversation
    print("\n" + "▶"*40)
    print("INGESTING CONVERSATION...")
    print("▶"*40 + "\n")
    optimizer.ingest(session_id, events)
    
    # Interactive query loop
    print("\n" + "="*80)
    print("INTERACTIVE DEMO - Enter queries to test optimization")
    print("Type 'quit' to exit")
    print("="*80 + "\n")
    
    test_queries = [
        "How does the chunking work?",
        "Show me the FastAPI integration",
        "Explain the shrinking logic"
    ]
    
    print("Suggested queries:")
    for i, q in enumerate(test_queries, 1):
        print(f"  {i}. {q}")
    print()
    
    while True:
        query = input("\n🔍 Enter query (or number 1-3, or 'quit'): ").strip()
        
        if query.lower() == 'quit':
            print("\n👋 Goodbye!\n")
            break
        
        # Handle numbered selection
        if query.isdigit() and 1 <= int(query) <= len(test_queries):
            query = test_queries[int(query) - 1]
            print(f"   Using: '{query}'")
        
        if not query:
            continue
        
        # Run optimization
        print("\n" + "▶"*40)
        print(f"OPTIMIZING FOR QUERY: '{query}'")
        print("▶"*40 + "\n")
        
        result = optimizer.optimize(
            session_id=session_id,
            query_text=query,
            max_chunks=8,
            target_token_budget=500
        )
        
        # Display results
        if result.get('optimized_context'):
            print("\n" + "─"*80)
            print("OPTIMIZED CONTEXT (what would be sent to LLM):")
            print("─"*80)
            
            for i, chunk in enumerate(result['optimized_context'], 1):
                print(f"\n[Chunk {i}] Role: {chunk['role']}, Source: {chunk['source']}")
                print(f"Tokens: {chunk['original_tokens']} → {chunk['shrunk_tokens']} ({chunk['reduction']:.1f}% reduction)")
                print(f"Content:\n{chunk['content']}\n")
            
            print("─"*80)
            print(f"📊 SUMMARY:")
            print(f"   Original:  {result['original_tokens']:,} tokens")
            print(f"   Optimized: {result['optimized_tokens']:,} tokens")
            print(f"   Saved:     {result['original_tokens'] - result['optimized_tokens']:,} tokens ({result['percent_saved']:.1f}%)")
            print("─"*80)

if __name__ == "__main__":
    main()
