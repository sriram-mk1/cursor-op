import re
import sqlite3
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
from simhash import Simhash
import tiktoken

log = logging.getLogger("gateway")

# Global model instance
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            log.info("🔄 Loading Snowflake/snowflake-arctic-embed-xs...")
            _embedder = TextEmbedding(model_name="Snowflake/snowflake-arctic-embed-xs")
            log.info("✅ Embedder loaded")
        except Exception as e:
            log.error(f"❌ Failed to load embedder: {e}")
    return _embedder

@dataclass
class Chunk:
    id: str
    content: str
    role: str
    msg_index: int
    chunk_index: int
    type: str = "text" # text, code, log, heading
    source_file: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0
    score: float = 0.0
    embedding: Optional[np.ndarray] = None
    simhash: Optional[int] = None

class SmartChunker:
    """Structure-aware chunking."""
    
    CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')
    STACK_TRACE_RE = re.compile(r'(\w+\.py", line \d+, in [\s\S]+?\n\s+[\s\S]+?\n)')
    HEADING_RE = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
    
    def __init__(self, target_tokens: int = 500):
        self.target_tokens = target_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_message(self, content: str, role: str, msg_idx: int) -> List[Chunk]:
        if not content: return []
        
        chunks = []
        
        # 1. Extract structural elements
        # We'll use a simple strategy: find all blocks, then fill in the gaps
        blocks = []
        for match in self.CODE_BLOCK_RE.finditer(content):
            blocks.append((match.start(), match.end(), "code"))
        for match in self.STACK_TRACE_RE.finditer(content):
            blocks.append((match.start(), match.end(), "log"))
        for match in self.HEADING_RE.finditer(content):
            blocks.append((match.start(), match.end(), "heading"))
            
        blocks.sort()
        
        # Merge overlapping blocks (if any)
        merged = []
        if blocks:
            curr_start, curr_end, curr_type = blocks[0]
            for next_start, next_end, next_type in blocks[1:]:
                if next_start < curr_end:
                    curr_end = max(curr_end, next_end)
                else:
                    merged.append((curr_start, curr_end, curr_type))
                    curr_start, curr_end, curr_type = next_start, next_end, next_type
            merged.append((curr_start, curr_end, curr_type))
            
        # 2. Process blocks and gaps
        last_pos = 0
        chunk_idx = 0
        
        def add_chunk(text: str, ctype: str):
            nonlocal chunk_idx
            text = text.strip()
            if not text: return
            
            tokens = len(self.encoder.encode(text))
            # If text is too large, split it further (simple split for now)
            if tokens > self.target_tokens * 1.5:
                parts = [text[i:i+2000] for i in range(0, len(text), 2000)]
                for p in parts:
                    cid = hashlib.md5(f"{msg_idx}-{chunk_idx}-{p[:50]}".encode()).hexdigest()
                    chunks.append(Chunk(id=cid, content=p, role=role, msg_index=msg_idx, chunk_index=chunk_idx, type=ctype, tokens=len(self.encoder.encode(p))))
                    chunk_idx += 1
            else:
                cid = hashlib.md5(f"{msg_idx}-{chunk_idx}-{text[:50]}".encode()).hexdigest()
                chunks.append(Chunk(id=cid, content=text, role=role, msg_index=msg_idx, chunk_index=chunk_idx, type=ctype, tokens=tokens))
                chunk_idx += 1

        for start, end, ctype in merged:
            # Gap before block
            if start > last_pos:
                add_chunk(content[last_pos:start], "text")
            # The block itself
            add_chunk(content[start:end], ctype)
            last_pos = end
            
        # Final gap
        if last_pos < len(content):
            add_chunk(content[last_pos:], "text")
            
        return chunks

class HybridRetriever:
    """SQLite FTS5 + FastEmbed Hybrid Retrieval."""
    
    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, content, role UNINDEXED)")
        self.chunk_map = {}

    def index_chunks(self, chunks: List[Chunk]):
        embedder = get_embedder()
        texts = [c.content for c in chunks if not c.embedding]
        
        if texts and embedder:
            embeddings = list(embedder.embed(texts))
            text_idx = 0
            for c in chunks:
                if not c.embedding:
                    c.embedding = embeddings[text_idx]
                    text_idx += 1
        
        for c in chunks:
            if c.id not in self.chunk_map:
                self.db.execute("INSERT INTO chunks_fts(id, content, role) VALUES (?, ?, ?)", (c.id, c.content, c.role))
                self.chunk_map[c.id] = c
                # Compute SimHash for dedup
                c.simhash = Simhash(c.content).value

    def retrieve(self, query: str, top_k: int = 20) -> List[Chunk]:
        # 1. Lexical Search (BM25)
        bm25_results = []
        try:
            # Clean query for FTS
            clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
            if clean_query:
                cursor = self.db.execute(
                    "SELECT id, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT 50",
                    (clean_query,)
                )
                for row in cursor:
                    cid, score = row
                    if cid in self.chunk_map:
                        chunk = self.chunk_map[cid]
                        # BM25 in SQLite is lower = better, we normalize later
                        bm25_results.append((chunk, -score))
        except Exception as e:
            log.warning(f"BM25 search failed: {e}")

        # 2. Semantic Search
        semantic_results = []
        embedder = get_embedder()
        if embedder:
            query_emb = list(embedder.embed([query]))[0]
            query_norm = query_emb / np.linalg.norm(query_emb)
            
            for chunk in self.chunk_map.values():
                if chunk.embedding is not None:
                    sim = np.dot(query_norm, chunk.embedding / np.linalg.norm(chunk.embedding))
                    semantic_results.append((chunk, float(sim)))
            
            semantic_results.sort(key=lambda x: x[1], reverse=True)
            semantic_results = semantic_results[:50]

        # 3. RRF (Reciprocal Rank Fusion) or Simple Union + Normalize
        # We'll use a simple union with normalized scores
        combined = {}
        
        def normalize(results):
            if not results: return {}
            min_s = min(r[1] for r in results)
            max_s = max(r[1] for r in results)
            if max_s == min_s: return {r[0].id: 1.0 for r in results}
            return {r[0].id: (r[1] - min_s) / (max_s - min_s) for r in results}

        bm25_norm = normalize(bm25_results)
        semantic_norm = normalize(semantic_results)
        
        all_ids = set(bm25_norm.keys()) | set(semantic_norm.keys())
        for cid in all_ids:
            score = 0.4 * bm25_norm.get(cid, 0) + 0.6 * semantic_norm.get(cid, 0)
            chunk = self.chunk_map[cid]
            chunk.score = score
            combined[cid] = chunk
            
        return sorted(combined.values(), key=lambda x: x.score, reverse=True)[:top_k]

class DedupManager:
    """SimHash based deduplication."""
    
    def dedup(self, chunks: List[Chunk], threshold: int = 3) -> List[Chunk]:
        if not chunks: return []
        
        unique = []
        hashes = []
        
        for c in chunks:
            is_dup = False
            c_hash = Simhash(c.content)
            for h in hashes:
                if c_hash.distance(h) <= threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
                hashes.append(c_hash)
        return unique

class PromptPacker:
    """Budget-aware packing with coverage guardrails."""
    
    def __init__(self, total_budget: int = 4000):
        self.total_budget = total_budget

    def pack(self, query_chunk: Chunk, recent_chunks: List[Chunk], relevant_chunks: List[Chunk]) -> List[Chunk]:
        packed = []
        current_tokens = 0
        
        # 1. Anchors (Query + Last 2 turns)
        anchors = [query_chunk] + recent_chunks[:4] # Roughly last 2 turns
        for c in anchors:
            if current_tokens + c.tokens <= self.total_budget:
                packed.append(c)
                current_tokens += c.tokens
        
        # 2. Coverage Guardrails
        # Ensure at least one of each type if available and not already in packed
        selected_ids = {c.id for c in packed}
        
        types_needed = {"code", "log"}
        for ctype in types_needed:
            for c in relevant_chunks:
                if c.type == ctype and c.id not in selected_ids:
                    if current_tokens + c.tokens <= self.total_budget:
                        packed.append(c)
                        current_tokens += c.tokens
                        selected_ids.add(c.id)
                        break
        
        # 3. Fill remaining budget by score
        for c in relevant_chunks:
            if c.id not in selected_ids:
                if current_tokens + c.tokens <= self.total_budget:
                    packed.append(c)
                    current_tokens += c.tokens
                    selected_ids.add(c.id)
                    
        # Sort by original flow for LLM coherence
        packed.sort(key=lambda x: (x.msg_index, x.chunk_index))
        return packed

class ContextOptimizer:
    """The New Architecture Orchestrator."""
    
    def __init__(self):
        self.chunker = SmartChunker()
        self.retriever = HybridRetriever()
        self.deduper = DedupManager()
        self.packer = PromptPacker(total_budget=6000)
        self.all_chunks = []
        self.query_cache = {}

    def optimize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= 2: return messages
        
        # 1. Extract and Index
        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        history = [m for m in messages if m.get("role") != "system"]
        query_msg = history.pop()
        
        # Incremental indexing
        new_chunks = []
        # We only re-chunk the last few messages to be fast, but for simplicity here we chunk all
        # In a real session, we'd cache chunks per message ID
        all_history_chunks = []
        for i, m in enumerate(history):
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            all_history_chunks.extend(self.chunker.chunk_message(content, m.get("role", "user"), i))
            
        # Dedup history
        clean_history = self.deduper.dedup(all_history_chunks)
        self.retriever.index_chunks(clean_history)
        
        # 2. Retrieve
        query_text = query_msg.get("content", "")
        if isinstance(query_text, list):
            query_text = " ".join(p.get("text", "") for p in query_text if p.get("type") == "text")
            
        # Cache check
        if query_text in self.query_cache and time.time() - self.query_cache[query_text][1] < 60:
            relevant = self.query_cache[query_text][0]
        else:
            relevant = self.retriever.retrieve(query_text, top_k=30)
            self.query_cache[query_text] = (relevant, time.time())
            
        # 3. Pack
        query_chunk = Chunk(id="query", content=query_text, role="user", msg_index=999, chunk_index=0, tokens=len(tiktoken.get_encoding("cl100k_base").encode(query_text)))
        recent = clean_history[-10:] # Last bits of history
        
        packed = self.packer.pack(query_chunk, recent, relevant)
        
        # 4. Reconstruct
        ctx_parts = []
        last_msg_idx = -1
        for c in packed:
            if c.id == "query": continue
            if c.msg_index != last_msg_idx:
                ctx_parts.append(f"\n--- {c.role.upper()} ---")
                last_msg_idx = c.msg_index
            ctx_parts.append(c.content)
            
        optimized = []
        if system_msg: optimized.append(system_msg)
        
        optimized.append({
            "role": "user",
            "content": f"Context for current request:\n" + "\n".join(ctx_parts)
        })
        optimized.append({
            "role": "assistant",
            "content": "I have the context. Ready for your request."
        })
        optimized.append(query_msg)
        
        log.info(f"✨ Hybrid RAG: {len(history)+1}->{len(optimized)} msgs | {len(packed)} chunks")
        return optimized
