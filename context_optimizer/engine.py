import re
import sqlite3
import logging
import hashlib
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import numpy as np
from simhash import Simhash
import tiktoken

log = logging.getLogger("gateway")

# Global model and caches
_embedder = None
_embedding_cache = {} # text_hash -> vector
_retrieval_cache = {} # (session_id, query_hash) -> optimized_msgs

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            log.info("🔄 Loading Snowflake/snowflake-arctic-embed-xs (dim=128)...")
            _embedder = TextEmbedding(model_name="Snowflake/snowflake-arctic-embed-xs")
            log.info("✅ Embedder loaded")
        except Exception as e:
            log.error(f"❌ Failed to load embedder: {e}")
    return _embedder

@dataclass
class Chunk:
    id: str
    session_id: str
    source: str
    type: str
    text: str
    created_at: float = field(default_factory=time.time)
    token_est: int = 0
    lex_terms: str = ""
    embed_vec: Optional[List[float]] = None
    score: float = 0.0

class SessionManager:
    """High-performance multi-session storage."""
    def __init__(self, db_path: str = ":memory:"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_db()
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _setup_db(self):
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, session_id UNINDEXED, text, lex_terms)")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS chunks_meta (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                source TEXT,
                type TEXT,
                text TEXT,
                created_at REAL,
                token_est INTEGER,
                embed_vec BLOB
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_session ON chunks_meta(session_id)")
        self.db.commit()

    def ingest_chunk(self, chunk: Chunk):
        lex = re.sub(r'[^\w\s]', ' ', chunk.text).lower()
        chunk.lex_terms = lex
        chunk.token_est = len(self.encoder.encode(chunk.text))
        
        self.db.execute(
            "INSERT OR IGNORE INTO chunks_fts(id, session_id, text, lex_terms) VALUES (?, ?, ?, ?)",
            (chunk.id, chunk.session_id, chunk.text, chunk.lex_terms)
        )
        
        vec_blob = np.array(chunk.embed_vec, dtype=np.float32).tobytes() if chunk.embed_vec else None
        self.db.execute(
            "INSERT OR IGNORE INTO chunks_meta(id, session_id, source, type, text, created_at, token_est, embed_vec) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chunk.id, chunk.session_id, chunk.source, chunk.type, chunk.text, chunk.created_at, chunk.token_est, vec_blob)
        )
        self.db.commit()

    def get_session_chunks(self, session_id: str, limit: int = 200) -> List[Chunk]:
        cursor = self.db.execute(
            "SELECT id, session_id, source, type, text, created_at, token_est, embed_vec FROM chunks_meta WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit)
        )
        chunks = []
        for row in cursor:
            vec = np.frombuffer(row[7], dtype=np.float32).tolist() if row[7] else None
            chunks.append(Chunk(
                id=row[0], session_id=row[1], source=row[2], type=row[3],
                text=row[4], created_at=row[5], token_est=row[6], embed_vec=vec
            ))
        return chunks

    def hybrid_search(self, session_id: str, query_text: str, query_vec: Optional[np.ndarray], top_k: int = 45) -> List[Chunk]:
        lex_ids = {}
        try:
            q = re.sub(r'[^\w\s]', ' ', query_text).strip()
            if q:
                cursor = self.db.execute(
                    "SELECT id, bm25(chunks_fts) FROM chunks_fts WHERE session_id = ? AND chunks_fts MATCH ? LIMIT 80",
                    (session_id, q)
                )
                for row in cursor:
                    lex_ids[row[0]] = -row[1]
        except: pass

        all_session_chunks = self.get_session_chunks(session_id, limit=600)
        semantic_scores = {}
        if query_vec is not None:
            q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
            for c in all_session_chunks:
                if c.embed_vec:
                    c_vec = np.array(c.embed_vec)
                    sim = np.dot(q_norm, c_vec / (np.linalg.norm(c_vec) + 1e-9))
                    semantic_scores[c.id] = float(sim)

        def norm(d):
            if not d: return {}
            vs = list(d.values())
            mi, ma = min(vs), max(vs)
            if mi == ma: return {k: 1.0 for k in d}
            return {k: (v - mi) / (ma - mi + 1e-9) for k, v in d.items()}

        lex_norm = norm(lex_ids)
        sem_norm = norm(semantic_scores)
        
        combined = []
        now = time.time()
        for c in all_session_chunks:
            # Hybrid Score
            base_score = 0.3 * lex_norm.get(c.id, 0) + 0.7 * sem_norm.get(c.id, 0)
            
            # Recency Boost (up to 20% boost for very recent chunks)
            age_hours = (now - c.created_at) / 3600
            recency_boost = 1.2 / (1.0 + age_hours) 
            
            score = base_score * recency_boost
            
            if score > 0.03: # More lenient threshold
                c.score = score
                combined.append(c)
        
        return sorted(combined, key=lambda x: x.score, reverse=True)[:top_k]

class SmartChunker:
    """Fast, reliable structure-aware chunking."""
    def __init__(self, target_tokens: int = 800): # Larger chunks for more context
        self.target_tokens = target_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_and_classify(self, text: str, source: str) -> List[Tuple[str, str]]:
        if not text: return []
        parts = re.split(r'(\n#{1,6}\s+.*|```[\s\S]*?```|\n\n)', text)
        chunks = []
        for p in parts:
            p = p.strip()
            if not p: continue
            ctype = "exploratory"
            if p.startswith("```") or p.startswith("#"): ctype = "authoritative"
            elif "error" in p.lower() or "traceback" in p.lower(): ctype = "diagnostic"
            elif source == "chat" or source == "user" or source == "assistant": ctype = "historical"
            chunks.append((p, ctype))
        return chunks

class ContextOptimizer:
    """Turbo-charged V1 Pipeline."""
    def __init__(self,):
        self.sessions = SessionManager()
        self.chunker = SmartChunker()
        self.seen_messages = {} # session_id -> set of msg hashes

    def ingest_event(self, session_id: str, content: Any, source: str):
        if not content: return
        
        # Normalize content to string
        if isinstance(content, list):
            text = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        else:
            text = str(content)
        
        # 🛡️ ANTI-INCEPTION: Never index reconstructed context or TOON blocks
        if "RELEVANT SESSION CONTEXT" in text or "url: /context/session.toon" in text:
            return
            
        if not text.strip(): return
        
        msg_hash = hashlib.md5(text.encode()).hexdigest()
        if session_id not in self.seen_messages: self.seen_messages[session_id] = set()
        if msg_hash in self.seen_messages[session_id]: return
        self.seen_messages[session_id].add(msg_hash)
        
        # 2. Chunk & Embed (with Cache)
        raw_chunks = self.chunker.chunk_and_classify(text, source)
        embedder = get_embedder()
        
        for content_part, ctype in raw_chunks:
            # Skip chunks that look like metadata or headers
            if content_part.startswith("---") or content_part.startswith("# Session Context"):
                continue
                
            part_hash = hashlib.md5(content_part.encode()).hexdigest()
            
            # Global Embedding Cache
            if part_hash in _embedding_cache:
                vec = _embedding_cache[part_hash]
            elif embedder:
                vec = list(embedder.embed([content_part]))[0][:128].tolist()
                _embedding_cache[part_hash] = vec
            else:
                vec = None
            
            chunk_id = f"{session_id}-{part_hash[:16]}-{ctype[:3]}"
            self.sessions.ingest_chunk(Chunk(
                id=chunk_id, session_id=session_id, source=source,
                type=ctype, text=content_part, embed_vec=vec
            ))

    def optimize(self, session_id: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start_time = time.time()
        
        # 1. Fast Retrieval Cache
        query_text = str(messages[-1].get("content", ""))
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        cache_key = (session_id, query_hash)
        
        if cache_key in _retrieval_cache:
            cached_msgs, ts = _retrieval_cache[cache_key]
            if time.time() - ts < 30: # 30s cache
                log.info(f"⚡️ Cache Hit [{session_id}] | 🔍 \"{query_text[:40]}...\"")
                return cached_msgs

        # 2. Incremental Ingest (Only new messages)
        ingest_start = time.time()
        for m in messages:
            self.ingest_event(session_id, m.get("content", ""), m.get("role", "user"))
        ingest_time = (time.time() - ingest_start) * 1000

        # 3. Hybrid Search
        search_start = time.time()
        embedder = get_embedder()
        query_vec = list(embedder.embed([query_text]))[0][:128] if embedder else None
        relevant = self.sessions.hybrid_search(session_id, query_text, query_vec, top_k=45)
        search_time = (time.time() - search_start) * 1000

        # 4. TOON Reconstruction
        pack_start = time.time()
        toon_context = self._to_toon(relevant)
        
        optimized = []
        system = next((m for m in messages if m.get("role") == "system"), None)
        if system: optimized.append(system)
        
        if toon_context:
            optimized.append({
                "role": "system", 
                "content": f"--- SESSION CONTEXT (TOON) ---\n{toon_context}"
            })
        
        optimized.append(messages[-1])
        pack_time = (time.time() - pack_start) * 1000
        total_time = (time.time() - start_time) * 1000

        # 5. Detailed Compact Logging
        chunk_summary = ", ".join([f"{c.source}/{c.type[:3]}" for c in relevant[:5]])
        if len(relevant) > 5: chunk_summary += "..."
        
        # Estimate tokens
        total_tokens = sum(len(self.sessions.encoder.encode(json.dumps(m))) for m in optimized)
        
        log.info(f"✨ [{session_id}] | 🔍 \"{query_text[:30]}...\" | 📦 {len(relevant)} chunks ({chunk_summary}) | 🎟️ {total_tokens:,} tkn | ⚡️ {total_time:.0f}ms (I:{ingest_time:.0f} S:{search_time:.0f} P:{pack_time:.0f})")
        
        _retrieval_cache[cache_key] = (optimized, time.time())
        return optimized

    def _to_toon(self, chunks: List[Chunk]) -> str:
        if not chunks: return ""
        
        # Group by type then source
        tree = {}
        for c in chunks:
            if c.type not in tree: tree[c.type] = {}
            if c.source not in tree[c.type]: tree[c.type][c.source] = []
            tree[c.type][c.source].append(c)
            
        lines = ["# Session Summary", ""]
        for ctype, sources in tree.items():
            lines.append(f"{ctype}[{sum(len(cs) for cs in sources.values())}]:")
            for source, cs in sources.items():
                lines.append(f"  {source}[{len(cs)}]:")
                for c in cs:
                    txt = c.text.replace('"', '\\"').replace('\n', ' ')
                    lines.append(f"    - \"{txt}\"")
        return "\n".join(lines)
