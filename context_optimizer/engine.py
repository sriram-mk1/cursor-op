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
            logger.info("Loading embedding model: sentence-transformers/static-retrieval-mrl-en-v1 with 128-dim truncation...")
            self.model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu", truncate_dim=128)
            logger.info(f"✓ Model loaded successfully. Embedding dimension: 128")
        except ImportError as e:
            logger.error(f"sentence-transformers import failed: {e}")
            logger.error("Please install it with: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load model: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
        target_token_budget: int = 2000,
        min_similarity: float = 0.35
    ) -> Dict[str, Any]:
        """
        Optimize context dynamically:
        1. Retrieve chunks sorted by relevance
        2. Keep ALL chunks above min_similarity threshold
        3. Fill up to target_token_budget
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZATION START")
        logger.info(f"Session: {session_id} | Query: '{query_text}'")
        logger.info(f"Budget: {target_token_budget} tokens | Min Similarity: {min_similarity}")
        
        all_chunks = self.sessions.get(session_id, [])
        if not all_chunks:
            return {"optimized_context": [], "original_tokens": 0, "optimized_tokens": 0, "percent_saved": 0}

        original_tokens = sum(c['tokens'] for c in all_chunks)
        
        # STEP 1: RETRIEVAL
        scored_chunks = []
        if self.model:
            try:
                query_embedding = self.model.encode(query_text, show_progress_bar=False)
                chunk_texts = [c['content'] for c in all_chunks]
                chunk_embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
                similarities = self.model.similarity(query_embedding, chunk_embeddings)[0]
                
                for i, chunk in enumerate(all_chunks):
                    scored_chunks.append((float(similarities[i]), chunk))
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                # Fallback: keep recent chunks
                for i, chunk in enumerate(all_chunks):
                    scored_chunks.append((float(i)/len(all_chunks), chunk))
        else:
             for i, chunk in enumerate(all_chunks):
                scored_chunks.append((float(i)/len(all_chunks), chunk))
        
        # Sort by relevance (highest first)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # STEP 2: DYNAMIC SELECTION
        selected_chunks = []
        current_tokens = 0
        
        for score, chunk in scored_chunks:
            # 1. Similarity Check - Primary filter for relevance
            if score < min_similarity:
                continue
                
            # 2. Budget Check - Soft limit (only skip if WAY over budget, e.g. > 2x)
            if target_token_budget and current_tokens > target_token_budget * 2:
                logger.info(f"  Stopping selection: exceeded 2x budget ({current_tokens} > {target_token_budget*2})")
                break
                
            selected_chunks.append(chunk)
            current_tokens += chunk['tokens']
            
        # Sort by original order to maintain flow
        selected_chunks.sort(key=lambda x: (x['event_idx'], x['chunk_idx']))
        
        # Calculate stats
        percent_saved = ((original_tokens - current_tokens) / original_tokens * 100) if original_tokens > 0 else 0
        
        logger.info(f"Selected {len(selected_chunks)}/{len(all_chunks)} chunks")
        logger.info(f"Tokens: {original_tokens} → {current_tokens} ({percent_saved:.1f}% saved)")
        logger.info(f"{'='*80}\n")
        
        return {
            "optimized_context": selected_chunks,
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
