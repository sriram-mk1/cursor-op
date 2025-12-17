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
        
        # Initialize embedding model with truncation
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: minishlab/potion-base-2M with 128-dim truncation...")
            self.model = SentenceTransformer("minishlab/potion-base-2M", device="cpu", truncate_dim=128)
            logger.info(f"✓ Model loaded successfully. Embedding dimension: 128")
        except ImportError:
            logger.error("sentence-transformers not installed!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.info("✓ Tokenizer initialized (cl100k_base)")
        except:
            self.encoding = tiktoken.get_encoding("p50k_base")
            logger.info("✓ Tokenizer initialized (p50k_base)")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _chunk_text(self, text: str, max_tokens: int = 200) -> List[str]:
        """
        Split text into smaller chunks of max_tokens size.
        This ensures we have granular pieces for better retrieval.
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks

    def ingest(self, session_id: str, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Ingest conversation events and chunk them into smaller pieces.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"INGESTION START - Session: {session_id}")
        logger.info(f"{'='*80}")
        
        new_chunks = 0
        total_chars = 0
        total_tokens = 0
        
        for idx, event in enumerate(events):
            content = event.get('content', '')
            if not content:
                continue
            
            role = event.get('role', 'unknown')
            source = event.get('source', 'chat')
            
            # Chunk the content into smaller pieces (200 tokens each)
            chunks = self._chunk_text(content, max_tokens=200)
            
            logger.info(f"\nEvent {idx + 1}/{len(events)} - Role: {role}, Source: {source}")
            logger.info(f"  Original length: {len(content)} chars, {self._count_tokens(content)} tokens")
            logger.info(f"  Split into {len(chunks)} chunks")
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_tokens = self._count_tokens(chunk)
                chunk_obj = {
                    'content': chunk,
                    'role': role,
                    'source': source,
                    'tokens': chunk_tokens,
                    'event_idx': idx,
                    'chunk_idx': chunk_idx
                }
                
                # Simple deduplication: check if exact content exists
                if not any(c['content'] == chunk for c in self.sessions[session_id]):
                    self.sessions[session_id].append(chunk_obj)
                    new_chunks += 1
                    total_chars += len(chunk)
                    total_tokens += chunk_tokens
                    logger.info(f"    Chunk {chunk_idx + 1}: {chunk_tokens} tokens - '{chunk[:60]}...'")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"INGESTION COMPLETE")
        logger.info(f"  New chunks added: {new_chunks}")
        logger.info(f"  Total chunks in session: {len(self.sessions[session_id])}")
        logger.info(f"  Total characters: {total_chars:,}")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"{'='*80}\n")
        
        return {"ingested": new_chunks, "deduped": len(events) - new_chunks}

    def optimize(
        self, 
        session_id: str, 
        query_text: str, 
        max_chunks: int = 12, 
        target_token_budget: Optional[int] = None,
        cache_ttl_sec: int = 300
    ) -> Dict[str, Any]:
        """
        Optimize context by:
        1. Retrieving most relevant chunks using embeddings
        2. Selecting top chunks within budget
        3. Shrinking content aggressively
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZATION START")
        logger.info(f"{'='*80}")
        logger.info(f"Session: {session_id}")
        logger.info(f"Query: '{query_text}'")
        logger.info(f"Max chunks: {max_chunks}")
        logger.info(f"Token budget: {target_token_budget or 'unlimited'}")
        
        all_chunks = self.sessions.get(session_id, [])
        if not all_chunks:
            logger.warning("No chunks found in session!")
            return {}

        logger.info(f"\nTotal available chunks: {len(all_chunks)}")
        
        # Calculate original token count
        original_tokens = sum(c['tokens'] for c in all_chunks)
        logger.info(f"Original total tokens: {original_tokens:,}")
        
        # STEP 1: RETRIEVAL - Use embeddings to find relevant chunks
        logger.info(f"\n{'─'*80}")
        logger.info("STEP 1: RETRIEVAL (Semantic Similarity)")
        logger.info(f"{'─'*80}")
        
        scored_chunks = []
        
        if self.model:
            try:
                logger.info("Encoding query...")
                query_embedding = self.model.encode(query_text, show_progress_bar=False)
                
                logger.info("Encoding chunks...")
                chunk_texts = [c['content'] for c in all_chunks]
                chunk_embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
                
                logger.info("Computing similarities...")
                similarities = self.model.similarity(query_embedding, chunk_embeddings)
                
                for i, chunk in enumerate(all_chunks):
                    score = float(similarities[0][i])
                    scored_chunks.append((score, chunk))
                    logger.info(f"  Chunk {i+1}: similarity={score:.4f}, tokens={chunk['tokens']}, preview='{chunk['content'][:50]}...'")
                
                logger.info("✓ Used semantic embeddings for retrieval")
            except Exception as e:
                logger.error(f"Embedding failed: {e}, falling back to recency")
                for i, chunk in enumerate(all_chunks):
                    scored_chunks.append((i, chunk))
        else:
            logger.warning("No model available, using recency")
            for i, chunk in enumerate(all_chunks):
                scored_chunks.append((i, chunk))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # STEP 2: SELECTION - Pick top chunks
        logger.info(f"\n{'─'*80}")
        logger.info("STEP 2: SELECTION (Top-K Ranking)")
        logger.info(f"{'─'*80}")
        
        selected_chunks = [chunk for _, chunk in scored_chunks[:max_chunks]]
        logger.info(f"Selected top {len(selected_chunks)} chunks")
        
        # Sort by original order to maintain conversation flow
        selected_chunks.sort(key=lambda x: (x['event_idx'], x['chunk_idx']))
        
        # STEP 3: SHRINKING - Aggressively reduce content
        logger.info(f"\n{'─'*80}")
        logger.info("STEP 3: SHRINKING (Aggressive Reduction)")
        logger.info(f"{'─'*80}")
        
        optimized_context = []
        total_optimized_tokens = 0
        
        for i, chunk in enumerate(selected_chunks):
            original_content = chunk['content']
            original_tokens = chunk['tokens']
            
            # AGGRESSIVE SHRINKING: Take only first 100 tokens
            shrunk_content = original_content
            if original_tokens > 100:
                tokens = self.encoding.encode(original_content)
                shrunk_tokens = tokens[:100]
                shrunk_content = self.encoding.decode(shrunk_tokens) + "..."
            
            shrunk_tokens = self._count_tokens(shrunk_content)
            
            # Check budget
            if target_token_budget and (total_optimized_tokens + shrunk_tokens) > target_token_budget:
                logger.info(f"  Chunk {i+1}: SKIPPED (would exceed budget)")
                continue
            
            optimized_context.append({
                "role": chunk['role'],
                "source": chunk['source'],
                "content": shrunk_content,
                "original_tokens": original_tokens,
                "shrunk_tokens": shrunk_tokens,
                "reduction": ((original_tokens - shrunk_tokens) / original_tokens * 100) if original_tokens > 0 else 0
            })
            
            total_optimized_tokens += shrunk_tokens
            
            logger.info(f"  Chunk {i+1}: {original_tokens} → {shrunk_tokens} tokens ({((original_tokens - shrunk_tokens) / original_tokens * 100):.1f}% reduction)")
            logger.info(f"    Preview: '{shrunk_content[:60]}...'")
        
        # Calculate final stats
        percent_saved = ((original_tokens - total_optimized_tokens) / original_tokens * 100) if original_tokens > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Original tokens:   {original_tokens:,}")
        logger.info(f"Optimized tokens:  {total_optimized_tokens:,}")
        logger.info(f"Tokens saved:      {original_tokens - total_optimized_tokens:,}")
        logger.info(f"Reduction:         {percent_saved:.1f}%")
        logger.info(f"Chunks selected:   {len(optimized_context)}")
        logger.info(f"{'='*80}\n")
        
        return {
            "optimized_context": optimized_context,
            "original_tokens": original_tokens,
            "optimized_tokens": total_optimized_tokens,
            "percent_saved": percent_saved,
            "chunks_selected": len(optimized_context)
        }

    def get_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics."""
        chunks = self.sessions.get(session_id, [])
        return {
            "chunks": len(chunks),
            "total_tokens": sum(c['tokens'] for c in chunks)
        }
