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

# Lazy load embedder
_embedder = None


def get_embedder():
    """Lazy load the FastEmbed embedding model."""
    global _embedder
    
    if _embedder is not None:
        return _embedder
    
    try:
        from fastembed import TextEmbedding
        log.info("🔄 Loading FastEmbed model...")
        # all-MiniLM-L6-v2 equivalent, but ONNX (no PyTorch!)
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        log.info("✅ FastEmbed model loaded")
        return _embedder
    except Exception as e:
        log.error(f"❌ Failed to load embedding model: {e}")
        return None


@dataclass
class Chunk:
    """A chunk of content from a message."""
    content: str
    role: str
    msg_index: int
    chunk_index: int
    is_code: bool = False
    is_recent: bool = False
    
    def __hash__(self):
        return hash((self.content, self.msg_index, self.chunk_index))


class SmartChunker:
    """
    Smart chunking that respects content structure.
    - Code blocks stay intact
    - Stack traces stay together
    - Natural paragraph breaks
    """
    
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    
    def __init__(self, target_chunk_size: int = 300):
        self.target_size = target_chunk_size
    
    def chunk_message(self, content: str, role: str, msg_index: int, is_recent: bool = False) -> List[Chunk]:
        """Chunk a message content smartly."""
        if not content or not content.strip():
            return []
        
        chunks = []
        chunk_idx = 0
        
        # Extract code blocks first (preserve them intact)
        code_blocks = list(self.CODE_BLOCK_PATTERN.finditer(content))
        
        last_end = 0
        for match in code_blocks:
            # Text before code block
            before = content[last_end:match.start()].strip()
            if before:
                for text_chunk in self._split_text(before):
                    chunks.append(Chunk(
                        content=text_chunk,
                        role=role,
                        msg_index=msg_index,
                        chunk_index=chunk_idx,
                        is_code=False,
                        is_recent=is_recent
                    ))
                    chunk_idx += 1
            
            # Code block as single chunk
            chunks.append(Chunk(
                content=match.group(),
                role=role,
                msg_index=msg_index,
                chunk_index=chunk_idx,
                is_code=True,
                is_recent=is_recent
            ))
            chunk_idx += 1
            last_end = match.end()
        
        # Remaining text
        remaining = content[last_end:].strip()
        if remaining:
            for text_chunk in self._split_text(remaining):
                chunks.append(Chunk(
                    content=text_chunk,
                    role=role,
                    msg_index=msg_index,
                    chunk_index=chunk_idx,
                    is_code=False,
                    is_recent=is_recent
                ))
                chunk_idx += 1
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split plain text into reasonable chunks."""
        if len(text) <= self.target_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) <= self.target_size:
                current += ("\n\n" if current else "") + para
            else:
                if current:
                    chunks.append(current)
                current = para
        
        if current:
            chunks.append(current)
        
        return chunks


class HybridRetriever:
    """
    Hybrid retrieval: BM25 + FastEmbed semantic similarity.
    Lightweight and fast (ONNX, no PyTorch).
    """
    
    def __init__(self, bm25_weight: float = 0.3, semantic_weight: float = 0.7):
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
    
    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 10) -> List[Chunk]:
        """Retrieve most relevant chunks."""
        if not chunks:
            return []
        
        if len(chunks) <= top_k:
            return chunks
        
        bm25_scores = self._bm25_scores(query, chunks)
        semantic_scores = self._semantic_scores(query, chunks)
        
        final_scores = []
        for i, chunk in enumerate(chunks):
            bm25 = bm25_scores[i] if bm25_scores else 0
            semantic = semantic_scores[i] if semantic_scores else 0
            
            # Boosts
            recency_boost = 1.5 if chunk.is_recent else 1.0
            code_boost = 1.2 if chunk.is_code else 1.0
            
            score = (self.bm25_weight * bm25 + self.semantic_weight * semantic) * recency_boost * code_boost
            final_scores.append((score, chunk))
        
        final_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in final_scores[:top_k]]
    
    def _bm25_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        """BM25 keyword matching scores."""
        try:
            from rank_bm25 import BM25Okapi
            
            corpus = [chunk.content.lower().split() for chunk in chunks]
            query_tokens = query.lower().split()
            
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query_tokens)
            
            max_score = max(scores) if max(scores) > 0 else 1
            return [s / max_score for s in scores]
        except Exception as e:
            log.warning(f"BM25 failed: {e}")
            return [0.0] * len(chunks)
    
    def _semantic_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        """Semantic similarity using FastEmbed."""
        embedder = get_embedder()
        if embedder is None:
            return [0.0] * len(chunks)
        
        try:
            # FastEmbed returns generators, convert to lists
            query_emb = list(embedder.embed([query]))[0]
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_embs = list(embedder.embed(chunk_texts))
            
            # Cosine similarity
            query_norm = query_emb / np.linalg.norm(query_emb)
            similarities = []
            for emb in chunk_embs:
                emb_norm = emb / np.linalg.norm(emb)
                sim = np.dot(query_norm, emb_norm)
                similarities.append(float(sim))
            
            return similarities
        except Exception as e:
            log.warning(f"Semantic retrieval failed: {e}")
            return [0.0] * len(chunks)


class ContextOptimizer:
    """
    Main context optimization pipeline.
    
    Flow:
    1. Receive messages (ignore system prompt)
    2. Chunk non-system messages smartly
    3. Retrieve relevant chunks for the query
    4. Reconstruct minimal context
    """
    
    def __init__(self, max_context_chunks: int = 15):
        self.chunker = SmartChunker(target_chunk_size=400)
        self.retriever = HybridRetriever(bm25_weight=0.3, semantic_weight=0.7)
        self.max_chunks = max_context_chunks
        self.request_count = 0
    
    def optimize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize the message context."""
        self.request_count += 1
        
        if len(messages) <= 3:
            log.info(f"📨 Pass-through: {len(messages)} messages (too few)")
            return messages
        
        # Separate system message (don't touch it)
        system_msg = None
        other_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                if system_msg is None:
                    system_msg = msg
            else:
                other_messages.append(msg)
        
        if not other_messages:
            return messages
        
        # Query = last user message
        query_msg = other_messages[-1]
        query_text = self._extract_text(query_msg.get("content", ""))
        
        # History = everything except last
        history = other_messages[:-1]
        
        if len(history) <= 2:
            log.info(f"📨 Pass-through: short history ({len(history)} msgs)")
            return messages
        
        log.info(f"{'='*60}")
        log.info(f"🔄 OPTIMIZING #{self.request_count}")
        log.info(f"{'='*60}")
        log.info(f"📥 Input: {len(other_messages)} msgs (excl. system)")
        
        # Chunk history
        all_chunks = []
        for idx, msg in enumerate(history):
            content = self._extract_text(msg.get("content", ""))
            role = msg.get("role", "unknown")
            is_recent = idx >= len(history) - 2
            
            chunks = self.chunker.chunk_message(content, role, idx, is_recent)
            all_chunks.extend(chunks)
        
        log.info(f"📦 {len(all_chunks)} chunks")
        
        # Retrieve relevant
        relevant = self.retriever.retrieve(query_text, all_chunks, top_k=self.max_chunks)
        log.info(f"🎯 Selected {len(relevant)}/{len(all_chunks)} chunks")
        
        # Log what we're keeping vs dropping for visibility
        relevant_set = set(relevant)
        for chunk in all_chunks:
            status = "✅ KEEP" if chunk in relevant_set else "❌ DROP"
            preview = chunk.content[:50].replace("\n", " ")
            log.debug(f"  {status} | {chunk.role}: {preview}...")
        
        # Sort by original order
        relevant.sort(key=lambda c: (c.msg_index, c.chunk_index))
        
        # Reconstruct
        optimized = []
        
        if system_msg:
            optimized.append(system_msg)
        
        if relevant:
            context_parts = []
            current_role = None
            
            for chunk in relevant:
                if chunk.role != current_role:
                    if current_role:
                        context_parts.append("")
                    context_parts.append(f"--- {chunk.role.upper()} ---")
                    current_role = chunk.role
                context_parts.append(chunk.content)
            
            context_content = "\n".join(context_parts)
            
            optimized.append({
                "role": "user",
                "content": f"[RELEVANT CONTEXT FROM CONVERSATION]:\n{context_content}"
            })
            optimized.append({
                "role": "assistant", 
                "content": "I've analyzed the relevant parts of our previous conversation. I'm ready to help with your request."
            })
        
        optimized.append(query_msg)
        
        log.info(f"📤 Output: {len(optimized)} msgs")
        log.info(f"{'='*60}")
        
        return optimized
    
    def _extract_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            return " ".join(texts)
        return str(content) if content else ""
