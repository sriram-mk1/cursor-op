import re
import sqlite3
import logging
import hashlib
import time
import json
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import tiktoken

log = logging.getLogger("gateway")

# --- Pre-compiled Globals ---
RE_CLEAN = re.compile(r'[^\w\s]')
RE_BLOCKS = re.compile(r'(\n#{1,6}\s+.*|```[\s\S]*?```|\n\n)')
ENCODER = tiktoken.get_encoding("cl100k_base")

# --- Global Model & Caches ---
_embedder = None
_embedder_ready = threading.Event()
_embedding_cache = {}
_retrieval_cache = {}

def _load_model_bg():
    global _embedder
    try:
        from fastembed import TextEmbedding
        log.info("🔄 Loading Snowflake/snowflake-arctic-embed-xs in background...")
        _embedder = TextEmbedding(model_name="Snowflake/snowflake-arctic-embed-xs")
        _embedder_ready.set()
        log.info("✅ Embedder ready")
    except Exception as e:
        log.error(f"❌ Failed to load embedder: {e}")

# Start loading immediately
threading.Thread(target=_load_model_bg, daemon=True).start()

@dataclass
class Chunk:
    id: str
    session_id: str
    source: str
    type: str
    text: str
    created_at: float = field(default_factory=time.time)
    token_est: int = 0
    embed_vec: Optional[List[float]] = None
    score: float = 0.0

class SessionManager:
    """Ultra-fast in-memory session storage."""
    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self._setup_db()

    def _setup_db(self):
        self.db.execute("PRAGMA journal_mode=OFF") # Max speed for in-memory
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

    def ingest_batch(self, chunks: List[Chunk]):
        for c in chunks:
            lex = RE_CLEAN.sub(' ', c.text).lower()
            c.token_est = len(ENCODER.encode(c.text))
            
            self.db.execute(
                "INSERT OR IGNORE INTO chunks_fts(id, session_id, text, lex_terms) VALUES (?, ?, ?, ?)",
                (c.id, c.session_id, c.text, lex)
            )
            
            vec_blob = np.array(c.embed_vec, dtype=np.float32).tobytes() if c.embed_vec else None
            self.db.execute(
                "INSERT OR IGNORE INTO chunks_meta(id, session_id, source, type, text, created_at, token_est, embed_vec) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (c.id, c.session_id, c.source, c.type, c.text, c.created_at, c.token_est, vec_blob)
            )
        self.db.commit()

    def hybrid_search(self, session_id: str, query_text: str, query_vec: Optional[np.ndarray], top_k: int = 45) -> List[Chunk]:
        lex_ids = {}
        try:
            q = RE_CLEAN.sub(' ', query_text).strip()
            if q:
                cursor = self.db.execute(
                    "SELECT id, bm25(chunks_fts) FROM chunks_fts WHERE session_id = ? AND chunks_fts MATCH ? LIMIT 80",
                    (session_id, q)
                )
                for row in cursor:
                    lex_ids[row[0]] = -row[1]
        except: pass

        # Get recent chunks for this session
        cursor = self.db.execute(
            "SELECT id, session_id, source, type, text, created_at, token_est, embed_vec FROM chunks_meta WHERE session_id = ? ORDER BY created_at DESC LIMIT 500",
            (session_id,)
        )
        
        all_chunks = []
        semantic_scores = {}
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9) if query_vec is not None else None
        
        for row in cursor:
            vec_blob = row[7]
            c = Chunk(id=row[0], session_id=row[1], source=row[2], type=row[3], text=row[4], created_at=row[5], token_est=row[6])
            
            if q_norm is not None and vec_blob:
                c_vec = np.frombuffer(vec_blob, dtype=np.float32)
                sim = np.dot(q_norm, c_vec / (np.linalg.norm(c_vec) + 1e-9))
                semantic_scores[c.id] = float(sim)
            
            all_chunks.append(c)

        def norm(d):
            if not d: return {}
            vs = list(d.values())
            mi, ma = min(vs), max(vs)
            if mi == ma: return {k: 1.0 for k in d}
            return {k: (v - mi) / (ma - mi + 1e-9) for k, v in d.items()}

        lex_norm = norm(lex_ids)
        sem_norm = norm(semantic_scores)
        
        now = time.time()
        for c in all_chunks:
            base = 0.3 * lex_norm.get(c.id, 0) + 0.7 * sem_norm.get(c.id, 0)
            age_boost = 1.2 / (1.0 + (now - c.created_at) / 3600)
            c.score = base * age_boost
            
        return sorted([c for c in all_chunks if c.score > 0.02], key=lambda x: x.score, reverse=True)[:top_k]

    def terminate_session(self, session_id: str):
        self.db.execute("DELETE FROM chunks_fts WHERE session_id = ?", (session_id,))
        self.db.execute("DELETE FROM chunks_meta WHERE session_id = ?", (session_id,))
        self.db.commit()

class ContextOptimizer:
    """Instant-start, Turbo-charged V1 Pipeline."""
    def __init__(self):
        self.sessions = SessionManager()
        self.seen_messages = {} # session_id -> set of msg hashes
        self.last_access = {} # session_id -> timestamp

    def _cleanup(self):
        now = time.time()
        to_kill = [sid for sid, ts in self.last_access.items() if now - ts > 1800] # 30m
        for sid in to_kill:
            log.info(f"🌱 Germinating session {sid}")
            self.sessions.terminate_session(sid)
            self.seen_messages.pop(sid, None)
            self.last_access.pop(sid, None)

    def ingest_event(self, session_id: str, content: Any, source: str):
        if not content: return
        self.last_access[session_id] = time.time()
        
        if isinstance(content, list):
            text = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        else:
            text = str(content)
        
        if "RELEVANT SESSION CONTEXT" in text or "url: /context/session.toon" in text: return
        if not text.strip(): return
        
        msg_hash = hashlib.md5(text.encode()).hexdigest()
        if session_id not in self.seen_messages: self.seen_messages[session_id] = set()
        if msg_hash in self.seen_messages[session_id]: return
        self.seen_messages[session_id].add(msg_hash)
        
        raw_chunks = RE_BLOCKS.split(text)
        to_ingest = []
        to_embed = []
        
        for p in raw_chunks:
            p = p.strip()
            if not p or p.startswith("---") or p.startswith("# Session Context"): continue
            
            ctype = "exploratory"
            if p.startswith("```") or p.startswith("#"): ctype = "authoritative"
            elif "error" in p.lower() or "traceback" in p.lower(): ctype = "diagnostic"
            elif source in ("chat", "user", "assistant"): ctype = "historical"
            
            p_hash = hashlib.md5(p.encode()).hexdigest()
            chunk = Chunk(id=f"{session_id}-{p_hash[:16]}-{ctype[:3]}", session_id=session_id, source=source, type=ctype, text=p)
            
            if p_hash in _embedding_cache:
                chunk.embed_vec = _embedding_cache[p_hash]
            elif _embedder_ready.is_set():
                to_embed.append((p, p_hash, chunk))
            
            to_ingest.append(chunk)

        # Batch Embedding
        if to_embed:
            texts = [x[0] for x in to_embed]
            vecs = list(_embedder.embed(texts))
            for i, (p, p_hash, chunk) in enumerate(to_embed):
                v = vecs[i][:128].tolist()
                chunk.embed_vec = v
                _embedding_cache[p_hash] = v
        
        self.sessions.ingest_batch(to_ingest)

    def optimize(self, session_id: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start_time = time.time()
        self._cleanup()
        
        query_text = str(messages[-1].get("content", ""))
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        cache_key = (session_id, query_hash)
        
        if cache_key in _retrieval_cache:
            cached_msgs, ts = _retrieval_cache[cache_key]
            if time.time() - ts < 30:
                log.info(f"⚡️ Cache Hit [{session_id}]")
                return cached_msgs

        # Ingest
        ingest_start = time.time()
        for m in messages:
            self.ingest_event(session_id, m.get("content", ""), m.get("role", "user"))
        ingest_time = (time.time() - ingest_start) * 1000

        # Search
        search_start = time.time()
        query_vec = None
        if _embedder_ready.is_set():
            query_vec = list(_embedder.embed([query_text]))[0][:128]
        
        relevant = self.sessions.hybrid_search(session_id, query_text, query_vec, top_k=45)
        search_time = (time.time() - search_start) * 1000

        # Pack
        pack_start = time.time()
        toon_context = self._to_toon(relevant)
        
        optimized = []
        system = next((m for m in messages if m.get("role") == "system"), None)
        if system: optimized.append(system)
        if toon_context:
            optimized.append({"role": "system", "content": f"--- SESSION CONTEXT (TOON) ---\n{toon_context}"})
        optimized.append(messages[-1])
        
        pack_time = (time.time() - pack_start) * 1000
        total_time = (time.time() - start_time) * 1000

        log.info(f"✨ [{session_id}] | 📦 {len(relevant)} chunks | ⚡️ {total_time:.0f}ms (I:{ingest_time:.0f} S:{search_time:.0f} P:{pack_time:.0f})")
        
        _retrieval_cache[cache_key] = (optimized, time.time())
        return optimized

    def _to_toon(self, chunks: List[Chunk]) -> str:
        if not chunks: return ""
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
