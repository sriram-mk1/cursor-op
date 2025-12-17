import logging
import time
from typing import List, Dict, Any, Optional
import tiktoken
from collections import defaultdict

logger = logging.getLogger("context-optimizer")

class ContextOptimizer:
    def __init__(self):
        self.sessions = defaultdict(list)
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers import SentenceTransformer
            # Use a smaller, quantized model to fit within 512MB RAM
            # all-MiniLM-L6-v2 is ~80MB and very efficient
            # Or use the static model with truncation if available
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            self.model.max_seq_length = 256  # Limit sequence length to save memory
            logger.info("Loaded lightweight embedding model: sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed. Falling back to keyword matching.")
        except Exception as e:
            logger.warning(f"Failed to load static embedding model: {e}. Falling back to keyword matching.")

    def ingest(self, session_id: str, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Ingest events into the session history."""
        new_count = 0
        for event in events:
            # Simple deduplication: check if content already exists in session
            # In a real system, use SimHash or vector similarity
            if not any(e.get('content') == event.get('content') for e in self.sessions[session_id]):
                self.sessions[session_id].append(event)
                new_count += 1
        
        return {"ingested": new_count, "deduped": len(events) - new_count}

    def get_stats(self, session_id: str) -> Dict[str, Any]:
        chunks = self.sessions.get(session_id, [])
        return {
            "chunks": len(chunks),
            "index_terms": 0, 
            "dedup_rate": 0.0 
        }

    def optimize(self, session_id: str, query_text: str, max_chunks: int = 12, target_token_budget: Optional[int] = None, cache_ttl_sec: int = 300) -> Dict[str, Any]:
        """Optimize context by selecting relevant chunks and shrinking them."""
        
        all_chunks = self.sessions.get(session_id, [])
        if not all_chunks:
            return {}

        # Log BEFORE context
        logger.info(f"--- OPTIMIZATION START for Session {session_id} ---")
        logger.info(f"Query: {query_text}")
        logger.info(f"Total available chunks: {len(all_chunks)}")
        
        # 1. Retrieval / Ranking
        scored_chunks = []
        if self.model:
            try:
                query_embedding = self.model.encode(query_text)
                # Encode all chunks 
                chunk_texts = [chunk.get('content', '') for chunk in all_chunks]
                chunk_embeddings = self.model.encode(chunk_texts)
                
                similarities = self.model.similarity(query_embedding, chunk_embeddings)
                
                for i, chunk in enumerate(all_chunks):
                    score = similarities[0][i].item()
                    scored_chunks.append((score, chunk))
                
                logger.info("Used static embeddings for retrieval.")
            except Exception as e:
                logger.error(f"Embedding retrieval failed: {e}")
                # Fallback to recent chunks
                for i, chunk in enumerate(all_chunks):
                    scored_chunks.append((i, chunk)) 
        else:
            # Fallback: just use recency
            for i, chunk in enumerate(all_chunks):
                scored_chunks.append((i, chunk))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Select top chunks
        selected_chunks = [chunk for _, chunk in scored_chunks[:max_chunks]]
        
        # Sort selected chunks by time (original order) to maintain flow
        selected_chunks.sort(key=lambda x: all_chunks.index(x))

        # 2. Shrinking / Token Budget
        optimized_context = []
        total_raw_tokens = 0
        total_opt_tokens = 0
        total_raw_chars = 0
        total_opt_chars = 0

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except:
            encoding = tiktoken.get_encoding("p50k_base")

        for chunk in selected_chunks:
            content = chunk.get('content', '')
            raw_tokens = len(encoding.encode(content))
            raw_chars = len(content)
            
            optimized_content = content
            # Shrinking logic: truncate long logs or file reads
            if chunk.get('source') in ['file_read', 'logs'] and len(content) > 1000:
                optimized_content = content[:500] + "\n... [truncated] ...\n" + content[-500:]
            
            opt_tokens = len(encoding.encode(optimized_content))
            opt_chars = len(optimized_content)

            optimized_context.append({
                "type": chunk.get("source", "unknown"),
                "role": chunk.get("role", "unknown"),
                "source": chunk.get("source", "unknown"),
                "raw_tokens": raw_tokens,
                "raw_chars": raw_chars,
                "optimized_tokens": opt_tokens,
                "optimized_chars": opt_chars,
                "summary": f"[{chunk.get('role')}]: {optimized_content}" 
            })
            
            total_raw_tokens += raw_tokens
            total_opt_tokens += opt_tokens
            total_raw_chars += raw_chars
            total_opt_chars += opt_chars

        # Enforce token budget if specified
        if target_token_budget and total_opt_tokens > target_token_budget:
            logger.info(f"Token budget exceeded ({total_opt_tokens} > {target_token_budget}). Truncating...")
            while total_opt_tokens > target_token_budget and optimized_context:
                # Remove least relevant? Or just oldest?
                # Since we sorted by time, removing from start removes oldest.
                # But maybe we should remove lowest score?
                # For now, remove from start (oldest)
                removed = optimized_context.pop(0)
                total_opt_tokens -= removed['optimized_tokens']
                total_opt_chars -= removed['optimized_chars']

        percent_saved = 0
        if total_raw_tokens > 0:
            percent_saved = ((total_raw_tokens - total_opt_tokens) / total_raw_tokens) * 100

        # Log AFTER context
        logger.info(f"Optimized: {len(optimized_context)} chunks, ~{total_opt_tokens} tokens")
        logger.info(f"Savings: {percent_saved:.1f}%")
        logger.info(f"--- OPTIMIZATION END ---")

        return {
            "optimized_context": optimized_context,
            "raw_token_est": total_raw_tokens,
            "optimized_token_est": total_opt_tokens,
            "percent_saved_est": percent_saved,
            "raw_chars": total_raw_chars,
            "optimized_chars": total_opt_chars
        }
