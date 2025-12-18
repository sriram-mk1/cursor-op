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
    """Streamlined Hybrid Retrieval (BM25 + Semantic)."""
    
    def __init__(self, bm25_weight: float = 0.4, semantic_weight: float = 0.6, min_score: float = 0.25):
        self.bm25_weight, self.semantic_weight, self.min_score = bm25_weight, semantic_weight, min_score
    
    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 15) -> List[Chunk]:
        if not chunks: return []
        
        bm25_scores = self._bm25_scores(query, chunks)
        semantic_scores = self._semantic_scores(query, chunks)
        
        # Score and filter in one pass
        scored = []
        msg_counts = {}
        
        for i, chunk in enumerate(chunks):
            # Calculate base score
            s = (self.bm25_weight * (bm25_scores[i] if bm25_scores else 0) + 
                 self.semantic_weight * (semantic_scores[i] if semantic_scores else 0))
            
            # Apply boosts
            s *= (1.6 if chunk.is_recent else 1.0) * (1.25 if chunk.is_code else 1.0)
            
            # Diversity penalty
            if msg_counts.get(chunk.msg_index, 0) >= 3: s *= 0.6
            
            chunk.score = s
            if s >= self.min_score or chunk.is_recent:
                scored.append(chunk)
                msg_counts[chunk.msg_index] = msg_counts.get(chunk.msg_index, 0) + 1
        
        # Sort and take top_k
        scored.sort(key=lambda x: x.score, reverse=True)
        selected = scored[:top_k]
        
        # Quick neighbor expansion for cohesion
        indices = {(c.msg_index, c.chunk_index) for c in selected}
        expanded = list(selected)
        
        for c in selected:
            if len(expanded) >= top_k + 5: break
            for offset in [-1, 1]:
                nid = c.chunk_index + offset
                if nid < 0: continue
                # Find neighbor
                for cand in chunks:
                    if cand.msg_index == c.msg_index and cand.chunk_index == nid:
                        if (cand.msg_index, cand.chunk_index) not in indices:
                            expanded.append(cand)
                            indices.add((cand.msg_index, cand.chunk_index))
                            break
        
        return expanded[:top_k + 5]
    
    def _bm25_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        try:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi([c.content.lower().split() for c in chunks])
            scores = bm25.get_scores(query.lower().split())
            m = max(scores) if any(scores) else 1
            return [s / m for s in scores]
        except: return [0.0] * len(chunks)
    
    def _semantic_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        try:
            q_emb = get_embeddings([query])[0]
            c_embs = get_embeddings([c.content for c in chunks])
            q_norm = q_emb / np.linalg.norm(q_emb)
            return [float(np.dot(q_norm, e / np.linalg.norm(e))) for e in c_embs]
        except: return [0.0] * len(chunks)


class ContextOptimizer:
    """Simplified Context Optimization Pipeline."""
    
    def __init__(self, max_context_chunks: int = 20):
        self.chunker = SmartChunker(target_chunk_size=800)
        self.retriever = HybridRetriever()
        self.max_chunks = max_context_chunks
        self.THRESHOLD = 10000 # Chars
    
    def optimize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= 3: return messages
        
        # Adaptive Threshold
        total_chars = sum(len(self._extract_text(m.get("content", ""))) for m in messages)
        if total_chars < self.THRESHOLD: return messages
            
        # Split system/history/query
        system = next((m for m in messages if m.get("role") == "system"), None)
        history = [m for m in messages if m.get("role") != "system"]
        query_msg = history.pop()
        
        # Chunking
        all_chunks = []
        for i, m in enumerate(history):
            all_chunks.extend(self.chunker.chunk_message(
                self._extract_text(m.get("content", "")), 
                m.get("role", "user"), i, i >= len(history) - 3
            ))
        
        # Exponential Selectivity
        k = self.max_chunks
        if len(all_chunks) > 200: k = 8
        elif len(all_chunks) > 100: k = 12
            
        # Retrieval & Reconstruction
        relevant = sorted(self.retriever.retrieve(self._extract_text(query_msg.get("content", "")), all_chunks, k), 
                         key=lambda c: (c.msg_index, c.chunk_index))
        
        if not relevant: return messages
        
        ctx_body = []
        last_idx = -1
        for c in relevant:
            if c.msg_index != last_idx:
                ctx_body.append(f"\n{'='*20}\n### {c.role.upper()} ###")
                last_idx = c.msg_index
            ctx_body.append(c.content)
            
        optimized = []
        if system: optimized.append(system)
        optimized.extend([
            {"role": "user", "content": f"Summarized history for context:\n" + "\n".join(ctx_body)},
            {"role": "assistant", "content": "Context received. How can I help?"},
            query_msg
        ])
        
        log.info(f"✨ {total_chars:,}ch | {len(history)+1}->{len(optimized)}ms | {len(all_chunks)}->{len(relevant)}ck")
        return optimized
    
    def _extract_text(self, content) -> str:
        if isinstance(content, str): return content
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content) if content else ""


