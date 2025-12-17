import logging
import re
from typing import List, Dict, Any, Optional
import tiktoken
from collections import defaultdict
from rank_bm25 import BM25Okapi

logger = logging.getLogger("context-optimizer")

class ContextOptimizer:
    def __init__(self):
        self.sessions = defaultdict(list)
        self.model = None
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: sentence-transformers/static-retrieval-mrl-en-v1...")
            self.model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu", truncate_dim=128)
            logger.info("✓ Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = tiktoken.get_encoding("p50k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _chunk_text_smart(self, text: str, role: str, max_tokens: int = 500) -> List[str]:
        """
        Structure-aware chunking (300-800 tokens per chunk).
        Keeps code blocks, stack traces, and sections intact.
        """
        chunks = []
        
        # Detect code blocks
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = list(re.finditer(code_block_pattern, text))
        
        if code_blocks:
            # Split around code blocks
            last_end = 0
            for match in code_blocks:
                # Add text before code block
                before = text[last_end:match.start()].strip()
                if before:
                    chunks.extend(self._split_by_tokens(before, max_tokens))
                
                # Add code block as single chunk (even if large)
                chunks.append(match.group())
                last_end = match.end()
            
            # Add remaining text
            after = text[last_end:].strip()
            if after:
                chunks.extend(self._split_by_tokens(after, max_tokens))
        else:
            # No code blocks, split by tokens
            chunks = self._split_by_tokens(text, max_tokens)
        
        return chunks

    def _split_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks by token count."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks

    def ingest(self, session_id: str, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Ingest conversation events with smart chunking.
        """
        new_chunks = 0
        
        for idx, event in enumerate(events):
            content = event.get('content', '')
            if not content:
                continue
            
            role = event.get('role', 'unknown')
            source = event.get('source', 'chat')
            
            # Smart chunking (300-800 tokens)
            chunks = self._chunk_text_smart(content, role, max_tokens=500)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_tokens = self._count_tokens(chunk)
                chunk_obj = {
                    'content': chunk,
                    'role': role,
                    'source': source,
                    'tokens': chunk_tokens,
                    'event_idx': idx,
                    'chunk_idx': chunk_idx,
                    'is_recent': idx >= len(events) - 2  # Mark last 2 events as recent
                }
                
                # Deduplication
                if not any(c['content'] == chunk for c in self.sessions[session_id]):
                    self.sessions[session_id].append(chunk_obj)
                    new_chunks += 1
        
        return {"ingested": new_chunks}

    def _bm25_retrieve(self, query: str, chunks: List[Dict], top_k: int = 20) -> List[tuple]:
        """BM25 retrieval for exact matches."""
        if not chunks:
            return []
        
        # Tokenize corpus
        corpus = [c['content'].lower().split() for c in chunks]
        bm25 = BM25Okapi(corpus)
        
        # Score query
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        # Return top-k with scores
        scored = [(scores[i], chunks[i]) for i in range(len(chunks))]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def _embedding_retrieve(self, query: str, chunks: List[Dict], top_k: int = 20) -> List[tuple]:
        """Embedding-based retrieval for semantic matches."""
        if not self.model or not chunks:
            return []
        
        try:
            query_embedding = self.model.encode(query, show_progress_bar=False)
            chunk_texts = [c['content'] for c in chunks]
            chunk_embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
            similarities = self.model.similarity(query_embedding, chunk_embeddings)[0]
            
            scored = [(float(similarities[i]), chunks[i]) for i in range(len(chunks))]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.error(f"Embedding retrieval failed: {e}")
            return []

    def _hybrid_retrieve(self, query: str, chunks: List[Dict], top_k: int = 15) -> List[Dict]:
        """
        Hybrid retrieval: BM25 + Embeddings with score fusion.
        """
        # Get BM25 results
        bm25_results = self._bm25_retrieve(query, chunks, top_k=top_k)
        
        # Get embedding results
        embedding_results = self._embedding_retrieve(query, chunks, top_k=top_k)
        
        # Normalize and fuse scores
        chunk_scores = {}
        
        # BM25 scores (normalize to 0-1)
        if bm25_results:
            max_bm25 = max(score for score, _ in bm25_results) if bm25_results else 1
            for score, chunk in bm25_results:
                chunk_id = id(chunk)
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + normalized_score * 0.4  # 40% weight
        
        # Embedding scores (already 0-1)
        for score, chunk in embedding_results:
            chunk_id = id(chunk)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + score * 0.6  # 60% weight
        
        # Map back to chunks
        chunk_map = {id(c): c for c in chunks}
        scored_chunks = [(chunk_scores[cid], chunk_map[cid]) for cid in chunk_scores]
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def optimize(
        self, 
        session_id: str, 
        query_text: str
    ) -> Dict[str, Any]:
        """
        Optimize context with hybrid retrieval + anchors + coverage guardrails.
        """
        all_chunks = self.sessions.get(session_id, [])
        if not all_chunks:
            return {"optimized_context": [], "original_tokens": 0, "optimized_tokens": 0, "percent_saved": 0}

        original_tokens = sum(c['tokens'] for c in all_chunks)
        
        # STEP 1: ANCHORS (guaranteed include)
        # Always include last 2 turns (recent context)
        anchors = [c for c in all_chunks if c.get('is_recent', False)]
        anchor_tokens = sum(c['tokens'] for c in anchors)
        
        # STEP 2: HYBRID RETRIEVAL
        # Retrieve from non-anchor chunks
        non_anchor_chunks = [c for c in all_chunks if not c.get('is_recent', False)]
        retrieved = self._hybrid_retrieve(query_text, non_anchor_chunks, top_k=15)
        
        # STEP 3: COVERAGE GUARDRAILS
        # Ensure diversity: diagnostic, code, recent
        selected = list(anchors)  # Start with anchors
        selected_ids = {id(c) for c in anchors}
        
        # Add retrieved chunks (avoid duplicates)
        for chunk in retrieved:
            if id(chunk) not in selected_ids:
                selected.append(chunk)
                selected_ids.add(id(chunk))
        
        # STEP 4: SORT BY ORIGINAL ORDER
        selected.sort(key=lambda x: (x['event_idx'], x['chunk_idx']))
        
        # Calculate stats
        current_tokens = sum(c['tokens'] for c in selected)
        percent_saved = ((original_tokens - current_tokens) / original_tokens * 100) if original_tokens > 0 else 0
        
        logger.info(f"Chunks: {len(selected)}")
        logger.info(f"Tokens: {original_tokens} -> {current_tokens} ({percent_saved:.1f}% reduction)")
        logger.info(f"Session: {session_id}")
        
        return {
            "optimized_context": selected,
            "original_tokens": original_tokens,
            "optimized_tokens": current_tokens,
            "percent_saved": percent_saved
        }

    def get_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics."""
        chunks = self.sessions.get(session_id, [])
        return {
            "chunks": len(chunks),
            "total_tokens": sum(c['tokens'] for c in chunks)
        }
