"""
Context Optimizer - Smart Chunking + RAG Pipeline
=================================================
Fast, simple, effective context compression using:
- Smart chunking (respects code blocks, logs, etc.)
- Hybrid retrieval: BM25 + Semantic (MiniLM-L6-v2)
- Query-based context selection
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

log = logging.getLogger("gateway")

# Lazy load heavy models
_embedder = None
_embedder_loading = False


def get_embedder():
    """Lazy load the embedding model."""
    global _embedder, _embedder_loading
    
    if _embedder is not None:
        return _embedder
    
    if _embedder_loading:
        return None
    
    _embedder_loading = True
    try:
        from sentence_transformers import SentenceTransformer
        log.info("🔄 Loading embedding model: all-MiniLM-L6-v2...")
        _embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        log.info("✅ Embedding model loaded")
        return _embedder
    except Exception as e:
        log.error(f"❌ Failed to load embedding model: {e}")
        _embedder_loading = False
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
    
    # Patterns for content that should stay together
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')
    STACK_TRACE_PATTERN = re.compile(r'(Traceback.*?(?=\n\n|\Z))', re.DOTALL)
    
    def __init__(self, target_chunk_size: int = 300):
        """
        Args:
            target_chunk_size: Target size in characters (not tokens, for speed)
        """
        self.target_size = target_chunk_size
    
    def chunk_message(self, content: str, role: str, msg_index: int, is_recent: bool = False) -> List[Chunk]:
        """Chunk a message content smartly."""
        if not content or not content.strip():
            return []
        
        chunks = []
        chunk_idx = 0
        
        # Extract code blocks first (preserve them intact)
        code_blocks = []
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            code_blocks.append((match.start(), match.end(), match.group()))
        
        # Process content around code blocks
        last_end = 0
        for start, end, code in code_blocks:
            # Text before code block
            before = content[last_end:start].strip()
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
            
            # Code block as single chunk (even if large)
            chunks.append(Chunk(
                content=code,
                role=role,
                msg_index=msg_index,
                chunk_index=chunk_idx,
                is_code=True,
                is_recent=is_recent
            ))
            chunk_idx += 1
            last_end = end
        
        # Remaining text after last code block
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
        
        # Try to split on paragraph breaks first
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
    Hybrid retrieval using BM25 + Semantic similarity.
    Fast and effective.
    """
    
    def __init__(self, bm25_weight: float = 0.3, semantic_weight: float = 0.7):
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
    
    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 10) -> List[Chunk]:
        """
        Retrieve most relevant chunks using hybrid approach.
        """
        if not chunks:
            return []
        
        if len(chunks) <= top_k:
            return chunks
        
        # Get scores from both methods
        bm25_scores = self._bm25_scores(query, chunks)
        semantic_scores = self._semantic_scores(query, chunks)
        
        # Combine scores
        final_scores = []
        for i, chunk in enumerate(chunks):
            bm25 = bm25_scores[i] if bm25_scores else 0
            semantic = semantic_scores[i] if semantic_scores else 0
            
            # Boost recent chunks
            recency_boost = 1.5 if chunk.is_recent else 1.0
            # Slight boost for code (often important)
            code_boost = 1.2 if chunk.is_code else 1.0
            
            score = (self.bm25_weight * bm25 + self.semantic_weight * semantic) * recency_boost * code_boost
            final_scores.append((score, chunk))
        
        # Sort by score and return top_k
        final_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in final_scores[:top_k]]
    
    def _bm25_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        """Get BM25 scores for chunks."""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize (simple whitespace)
            corpus = [chunk.content.lower().split() for chunk in chunks]
            query_tokens = query.lower().split()
            
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query_tokens)
            
            # Normalize to 0-1
            max_score = max(scores) if max(scores) > 0 else 1
            return [s / max_score for s in scores]
        except Exception as e:
            log.warning(f"BM25 failed: {e}")
            return [0.0] * len(chunks)
    
    def _semantic_scores(self, query: str, chunks: List[Chunk]) -> List[float]:
        """Get semantic similarity scores using embeddings."""
        embedder = get_embedder()
        if embedder is None:
            return [0.0] * len(chunks)
        
        try:
            # Encode query and chunks
            query_emb = embedder.encode(query, convert_to_numpy=True)
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_embs = embedder.encode(chunk_texts, convert_to_numpy=True)
            
            # Cosine similarity
            query_norm = query_emb / np.linalg.norm(query_emb)
            chunk_norms = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True)
            similarities = np.dot(chunk_norms, query_norm)
            
            # Already 0-1 range for cosine similarity
            return similarities.tolist()
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
        """
        Optimize the message context.
        
        Returns optimized messages list.
        """
        self.request_count += 1
        
        if len(messages) <= 3:
            # Too few messages to optimize
            log.info(f"📨 Pass-through: {len(messages)} messages (too few to optimize)")
            return messages
        
        # Separate system message (we don't touch it)
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
        
        # Get the query (last user message)
        query_msg = other_messages[-1]
        query_text = self._extract_text(query_msg.get("content", ""))
        
        # History = everything except the last message
        history = other_messages[:-1]
        
        if len(history) <= 2:
            # Very short history, just pass through
            log.info(f"📨 Pass-through: short history ({len(history)} msgs)")
            return messages
        
        log.info(f"{'='*60}")
        log.info(f"🔄 OPTIMIZING REQUEST #{self.request_count}")
        log.info(f"{'='*60}")
        log.info(f"📥 Input: {len(other_messages)} messages (excl. system)")
        
        # Chunk the history
        all_chunks = []
        for idx, msg in enumerate(history):
            content = self._extract_text(msg.get("content", ""))
            role = msg.get("role", "unknown")
            is_recent = idx >= len(history) - 2  # Last 2 messages are "recent"
            
            chunks = self.chunker.chunk_message(content, role, idx, is_recent)
            all_chunks.extend(chunks)
        
        log.info(f"📦 Chunked into {len(all_chunks)} chunks")
        
        # Retrieve relevant chunks
        relevant = self.retriever.retrieve(query_text, all_chunks, top_k=self.max_chunks)
        
        log.info(f"🎯 Retrieved {len(relevant)} relevant chunks")
        
        # Sort by original order (maintain conversation flow)
        relevant.sort(key=lambda c: (c.msg_index, c.chunk_index))
        
        # Reconstruct context
        optimized_messages = []
        
        # 1. System message first (untouched)
        if system_msg:
            optimized_messages.append(system_msg)
        
        # 2. Add optimized context as a single context message
        if relevant:
            context_parts = []
            current_role = None
            
            for chunk in relevant:
                if chunk.role != current_role:
                    if current_role:
                        context_parts.append("")  # Separator
                    context_parts.append(f"[{chunk.role.upper()}]:")
                    current_role = chunk.role
                context_parts.append(chunk.content)
            
            context_content = "\n".join(context_parts)
            
            optimized_messages.append({
                "role": "user",
                "content": f"[PREVIOUS CONTEXT - Summarized]\n{context_content}"
            })
            
            optimized_messages.append({
                "role": "assistant",
                "content": "I've reviewed the previous context. How can I help you now?"
            })
        
        # 3. Add the current query
        optimized_messages.append(query_msg)
        
        log.info(f"📤 Output: {len(optimized_messages)} messages")
        log.info(f"{'='*60}")
        
        return optimized_messages
    
    def _extract_text(self, content) -> str:
        """Extract text from content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            return " ".join(texts)
        return str(content) if content else ""
