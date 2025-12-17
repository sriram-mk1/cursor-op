"""
Context Optimizer - Smart Chunking + RAG Pipeline
=================================================
Fast, simple, effective context compression using:
- Smart chunking (respects code blocks, logs, etc.)
- Hybrid retrieval: BM25 + Semantic (FastEmbed - ONNX, no PyTorch!)
- Query-based context selection
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

log = logging.getLogger("gateway")

# Global model instance - initialized once at startup
_embedder = None

def init_embedder():
    """Initialize the FastEmbed model globally."""
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            log.info("🔄 Initializing FastEmbed (BAAI/bge-small-en-v1.5)...")
            _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            log.info("✅ FastEmbed initialized")
        except Exception as e:
            log.error(f"❌ Failed to initialize FastEmbed: {e}")

# Simple LRU-style cache for embeddings to save CPU/Latency
_EMBEDDING_CACHE = {}
MAX_CACHE_SIZE = 1000

def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Get embeddings with simple caching."""
    global _embedder, _EMBEDDING_CACHE
    
    if _embedder is None:
        init_embedder()
    
    results = []
    to_embed = []
    to_embed_indices = []
    
    for i, text in enumerate(texts):
        if text in _EMBEDDING_CACHE:
            results.append(_EMBEDDING_CACHE[text])
        else:
            results.append(None)
            to_embed.append(text)
            to_embed_indices.append(i)
    
    if to_embed:
        embeddings = list(_embedder.embed(to_embed))
        for i, emb in enumerate(embeddings):
            # Cache it
            if len(_EMBEDDING_CACHE) >= MAX_CACHE_SIZE:
                # Simple clear if full (could be smarter but this is fast)
                _EMBEDDING_CACHE.clear()
            
            _EMBEDDING_CACHE[to_embed[i]] = emb
            results[to_embed_indices[i]] = emb
            
    return results


@dataclass
class Chunk:
    """A chunk of content from a message."""
    content: str
    role: str
    msg_index: int
    chunk_index: int
    is_code: bool = False
    is_recent: bool = False
    score: float = 0.0
    
    def __hash__(self):
        return hash((self.content, self.msg_index, self.chunk_index))


class SmartChunker:
    """
    Smart chunking that respects content structure.
    """
    
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    
    def __init__(self, target_chunk_size: int = 600): # Increased for better context
        self.target_size = target_chunk_size
    
    def chunk_message(self, content: str, role: str, msg_index: int, is_recent: bool = False) -> List[Chunk]:
        """Chunk a message content smartly."""
        if not content or not content.strip():
            return []
        
        chunks = []
        chunk_idx = 0
        
        # Extract code blocks
        code_blocks = list(self.CODE_BLOCK_PATTERN.finditer(content))
        
        last_end = 0
        for match in code_blocks:
            before = content[last_end:match.start()].strip()
            if before:
                for text_chunk in self._split_text(before):
                    chunks.append(Chunk(content=text_chunk, role=role, msg_index=msg_index, chunk_index=chunk_idx, is_recent=is_recent))
                    chunk_idx += 1
            
            chunks.append(Chunk(content=match.group(), role=role, msg_index=msg_index, chunk_index=chunk_idx, is_code=True, is_recent=is_recent))
            chunk_idx += 1
            last_end = match.end()
        
        remaining = content[last_end:].strip()
        if remaining:
            for text_chunk in self._split_text(remaining):
                chunks.append(Chunk(content=text_chunk, role=role, msg_index=msg_index, chunk_index=chunk_idx, is_recent=is_recent))
                chunk_idx += 1
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.target_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) <= self.target_size:
                current += ("\n\n" if current else "") + para
            else:
                if current: chunks.append(current)
                current = para
        
        if current: chunks.append(current)
        return chunks


class HybridRetriever:
    """
    Hybrid retrieval: BM25 + FastEmbed semantic similarity.
    """
    
    def __init__(self, bm25_weight: float = 0.4, semantic_weight: float = 0.6, min_score: float = 0.3):
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.min_score = min_score
    
    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 15) -> List[Chunk]:
        if not chunks: return []
        
        bm25_scores = self._bm25_scores(query, chunks)
        semantic_scores = self._semantic_scores(query, chunks)
        
        # 1. Score all chunks
        for i, chunk in enumerate(chunks):
            bm25 = bm25_scores[i] if bm25_scores else 0
            semantic = semantic_scores[i] if semantic_scores else 0
            
            # Recency is critical for flow
            recency_boost = 1.5 if chunk.is_recent else 1.0
            # Code is high signal
            code_boost = 1.2 if chunk.is_code else 1.0
            
            chunk.score = (self.bm25_weight * bm25 + self.semantic_weight * semantic) * recency_boost * code_boost
        
        # 2. Select top chunks
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        selected = []
        selected_indices = set()
        
        for chunk in sorted_chunks:
            if len(selected) >= top_k: break
            if chunk.score >= self.min_score or chunk.is_recent:
                selected.append(chunk)
                selected_indices.add((chunk.msg_index, chunk.chunk_index))
        
        # 3. CONTEXT EXPANSION: If we have space, add neighbors of selected chunks
        # This prevents "choppy" context by keeping related lines together
        expanded = list(selected)
        if len(expanded) < top_k + 5:
            for chunk in selected:
                # Check neighbors in the same message
                for offset in [-1, 1]:
                    neighbor_idx = chunk.chunk_index + offset
                    if neighbor_idx < 0: continue
                    
                    # Find neighbor in all_chunks
                    for candidate in chunks:
                        if candidate.msg_index == chunk.msg_index and candidate.chunk_index == neighbor_idx:
                            if (candidate.msg_index, candidate.chunk_index) not in selected_indices:
                                candidate.score *= 0.8 # Slightly lower priority for neighbors
                                expanded.append(candidate)
                                selected_indices.add((candidate.msg_index, candidate.chunk_index))
                                break
                if len(expanded) >= top_k + 5: break
                
        return expanded[:top_k + 5]
    
    def _bm25_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        try:
            from rank_bm25 import BM25Okapi
            corpus = [chunk.content.lower().split() for chunk in chunks]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query.lower().split())
            max_s = max(scores) if max(scores) > 0 else 1
            return [s / max_s for s in scores]
        except: return [0.0] * len(chunks)
    
    def _semantic_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        try:
            query_emb = get_embeddings([query])[0]
            chunk_embs = get_embeddings([c.content for c in chunks])
            query_norm = query_emb / np.linalg.norm(query_emb)
            similarities = []
            for emb in chunk_embs:
                sim = np.dot(query_norm, emb / np.linalg.norm(emb))
                similarities.append(float(sim))
            return similarities
        except: return [0.0] * len(chunks)


class ContextOptimizer:
    """
    Main context optimization pipeline.
    """
    
    def __init__(self, max_context_chunks: int = 20):
        self.chunker = SmartChunker(target_chunk_size=800) # Larger chunks for better coherence
        self.retriever = HybridRetriever(bm25_weight=0.4, semantic_weight=0.6, min_score=0.25)
        self.max_chunks = max_context_chunks
        self.request_count = 0
    
    def optimize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize the message context."""
        self.request_count += 1
        
        if len(messages) <= 3:
            return messages
        
        system_msg = None
        other_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                if system_msg is None: system_msg = msg
            else:
                other_messages.append(msg)
        
        if not other_messages: return messages
        
        query_msg = other_messages[-1]
        query_text = self._extract_text(query_msg.get("content", ""))
        history = other_messages[:-1]
        
        if len(history) <= 2: return messages
        
        # Chunk history
        all_chunks = []
        for idx, msg in enumerate(history):
            content = self._extract_text(msg.get("content", ""))
            role = msg.get("role", "unknown")
            # Last 3 messages are "recent"
            is_recent = idx >= len(history) - 3
            all_chunks.extend(self.chunker.chunk_message(content, role, idx, is_recent))
        
        # Retrieve
        relevant = self.retriever.retrieve(query_text, all_chunks, top_k=self.max_chunks)
        relevant.sort(key=lambda c: (c.msg_index, c.chunk_index))
        
        # Reconstruct
        optimized = []
        if system_msg: optimized.append(system_msg)
        
        if relevant:
            context_parts = []
            current_msg_idx = -1
            
            for chunk in relevant:
                if chunk.msg_index != current_msg_idx:
                    if context_parts: context_parts.append("-" * 10)
                    context_parts.append(f"[{chunk.role.upper()}]:")
                    current_msg_idx = chunk.msg_index
                context_parts.append(chunk.content)
            
            optimized.append({
                "role": "user",
                "content": f"### RELEVANT CONVERSATION HISTORY ###\n\n" + "\n".join(context_parts)
            })
            optimized.append({
                "role": "assistant", 
                "content": "Understood. I have reviewed the relevant history. How can I help?"
            })
        
        optimized.append(query_msg)
        
        # One-line summary log
        log.info(f"✨ Optimized: {len(other_messages)}->{len(optimized)} msgs | {len(all_chunks)}->{len(relevant)} chunks")
        
        return optimized
    
    def _extract_text(self, content) -> str:
        if isinstance(content, str): return content
        if isinstance(content, list):
            return " ".join([p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"])
        return str(content) if content else ""
