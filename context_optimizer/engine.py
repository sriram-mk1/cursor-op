import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
from rank_bm25 import BM25Okapi

# Configure logging
logger = logging.getLogger("context-optimizer")
logger.setLevel(logging.INFO)

class ContextOptimizer:
    """
    A lightweight, in-memory RAG system using BM25 for relevance scoring.
    
    Features:
    - Zero heavy dependencies (no torch/transformers).
    - Extremely fast (< 10ms retrieval).
    - Session-based in-memory storage.
    - Token-aware context shrinking.
    """

    def __init__(self):
        """
        Initialize the ContextOptimizer.
        """
        logger.info("Initializing ContextOptimizer (BM25 Engine)...")
        
        # In-memory storage: session_id -> List[Dict]
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize Tokenizer
        # Using cl100k_base (GPT-4) as a standard reference.
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback if cl100k_base not found (rare)
            self.tokenizer = tiktoken.get_encoding("p50k_base")
            
        logger.info("ContextOptimizer initialized successfully.")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer for BM25. 
        Splits on whitespace and removes non-alphanumeric chars.
        """
        text = text.lower()
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string using tiktoken."""
        return len(self.tokenizer.encode(text))

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            
            if end == len(tokens):
                break
            
            start += chunk_size - overlap
            
        return chunks

    def ingest(self, session_id: str, events: List[Dict[str, Any]]):
        """
        Ingest conversation events into memory.
        
        Args:
            session_id: Unique identifier for the conversation session.
            events: List of message dictionaries (role, content, etc.).
        """
        # Initialize session if not exists
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        # We'll rebuild the session data for simplicity in this "all in session" model.
        # In a real app, you might append, but here we want to ensure we have the latest state.
        # If the user passes the full history every time, we should clear and re-ingest 
        # OR handle deduplication.
        # For this implementation, we'll assume 'events' contains new or full history.
        # Let's just append new ones? 
        # The prompt implies "reset the project... except main.py".
        # The main.py calls ingest with "history except last message".
        # So we should probably clear and re-ingest to be safe and stateless-ish, 
        # or check for duplicates.
        # Let's clear for now to guarantee "fresh" context for the request.
        self.sessions[session_id] = []
        
        for i, event in enumerate(events):
            content = event.get("content", "")
            role = event.get("role", "user")
            timestamp = event.get("ts", time.time())
            
            if not content:
                continue
                
            chunks = self._chunk_text(content)
            
            for chunk in chunks:
                self.sessions[session_id].append({
                    "content": chunk,
                    "role": role,
                    "timestamp": timestamp,
                    "original_index": i,
                    "token_count": self._count_tokens(chunk),
                    "tokenized_text": self._tokenize(chunk) # Pre-compute for BM25
                })

    def optimize(self, 
                 session_id: str, 
                 query_text: str, 
                 max_chunks: int = 10, 
                 target_token_budget: Optional[int] = None, 
                 cache_ttl_sec: int = 300) -> Dict[str, Any]:
        """
        Retrieve and optimize context using BM25.
        """
        start_time = time.time()
        
        if session_id not in self.sessions or not self.sessions[session_id]:
            return {
                "optimized_context": [],
                "raw_token_est": 0,
                "optimized_token_est": 0,
                "percent_saved_est": 0.0
            }
            
        session_chunks = self.sessions[session_id]
        
        # 1. Calculate Raw Tokens (Total history size)
        raw_tokens = sum(chunk["token_count"] for chunk in session_chunks)
        
        # 2. Prepare BM25
        tokenized_corpus = [chunk["tokenized_text"] for chunk in session_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 3. Query BM25
        tokenized_query = self._tokenize(query_text)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # 4. Rank Chunks
        # Zip scores with chunks
        scored_chunks = []
        for i, score in enumerate(doc_scores):
            # We can add a small boost for recency?
            # Let's keep it simple: Pure relevance first.
            scored_chunks.append({
                "chunk": session_chunks[i],
                "score": score,
                "index": i
            })
            
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # 5. Select Top Chunks within Budget
        selected_chunks = []
        current_tokens = 0
        
        # If no budget provided, use a sensible default or just max_chunks
        budget = target_token_budget if target_token_budget else 2000
        
        for item in scored_chunks:
            chunk = item["chunk"]
            
            # Filter out zero-score items (irrelevant) unless we are desperate?
            # Usually BM25 returns 0 for no keyword overlap.
            # If we have very few chunks, we might want to include recent ones even if score is 0.
            # Let's include if score > 0 OR it's very recent (last 3 chunks).
            is_recent = (len(session_chunks) - item["index"]) <= 3
            
            if item["score"] > 0 or is_recent:
                if current_tokens + chunk["token_count"] <= budget:
                    selected_chunks.append(chunk)
                    current_tokens += chunk["token_count"]
                    
                if len(selected_chunks) >= max_chunks:
                    break
        
        # 6. Re-sort by original index (Chronological)
        # This is crucial for chat context to make sense.
        selected_chunks.sort(key=lambda x: x["original_index"])
        
        # 7. Format Output
        optimized_context = []
        for chunk in selected_chunks:
            optimized_context.append({
                "summary": f"[{chunk['role']}] {chunk['content']}",
                "score": 0 # Placeholder
            })
            
        percent_saved = 0.0
        if raw_tokens > 0:
            percent_saved = ((raw_tokens - current_tokens) / raw_tokens) * 100
            
        elapsed = time.time() - start_time
        
        return {
            "optimized_context": optimized_context,
            "raw_token_est": raw_tokens,
            "optimized_token_est": current_tokens,
            "percent_saved_est": percent_saved
        }
