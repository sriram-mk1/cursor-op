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
        lex_scores = {}
        try:
            q = RE_CLEAN.sub(' ', query_text).strip()
            if q:
                cursor = self.db.execute(
                    "SELECT id, bm25(chunks_fts) FROM chunks_fts WHERE session_id = ? AND chunks_fts MATCH ? LIMIT 100",
                    (session_id, q)
                )
                for row in cursor:
                    lex_scores[row[0]] = -row[1]
        except: pass

        # Vectorized Semantic Search
        cursor = self.db.execute(
            "SELECT id, embed_vec, created_at FROM chunks_meta WHERE session_id = ? ORDER BY created_at DESC LIMIT 500",
            (session_id,)
        )
        rows = cursor.fetchall()
        if not rows: return []
        
        ids = [r[0] for r in rows]
        vec_blobs = [r[1] for r in rows]
        times = np.array([r[2] for r in rows])
        
        sem_scores = {}
        if query_vec is not None and any(vec_blobs):
            # Fast matrix construction
            valid_idx = [i for i, v in enumerate(vec_blobs) if v]
            if valid_idx:
                matrix = np.frombuffer(b''.join([vec_blobs[i] for i in valid_idx]), dtype=np.float32).reshape(len(valid_idx), -1)
                q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
                m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
                sims = np.dot(matrix / m_norms, q_norm)
                for i, idx in enumerate(valid_idx):
                    sem_scores[ids[idx]] = float(sims[i])

        def norm_dict(d):
            if not d: return {}
            vs = list(d.values())
            mi, ma = min(vs), max(vs)
            if mi == ma: return {k: 1.0 for k in d}
            diff = ma - mi + 1e-9
            return {k: (v - mi) / diff for k, v in d.items()}

        l_norm = norm_dict(lex_scores)
        s_norm = norm_dict(sem_scores)
        
        now = time.time()
        scored_ids = []
        for i, cid in enumerate(ids):
            base = 0.3 * l_norm.get(cid, 0) + 0.7 * s_norm.get(cid, 0)
            age_boost = 1.2 / (1.0 + (now - times[i]) / 3600)
            score = base * age_boost
            if score > 0.01:
                scored_ids.append((cid, score))
        
        top_ids = sorted(scored_ids, key=lambda x: x[1], reverse=True)[:top_k]
        if not top_ids: return []
        
        # Final fetch of full objects for top_k only
        placeholders = ','.join(['?'] * len(top_ids))
        id_map = {cid: score for cid, score in top_ids}
        cursor = self.db.execute(
            f"SELECT id, session_id, source, type, text, created_at, token_est FROM chunks_meta WHERE id IN ({placeholders})",
            [x[0] for x in top_ids]
        )
        
        final = []
        for r in cursor:
            c = Chunk(id=r[0], session_id=r[1], source=r[2], type=r[3], text=r[4], created_at=r[5], token_est=r[6])
            c.score = id_map.get(c.id, 0)
            final.append(c)
            
        return sorted(final, key=lambda x: x.score, reverse=True)

    def terminate_session(self, session_id: str):
        self.db.execute("DELETE FROM chunks_fts WHERE session_id = ?", (session_id,))
        self.db.execute("DELETE FROM chunks_meta WHERE session_id = ?", (session_id,))
        self.db.commit()

class ContextOptimizer:
    """Extreme Performance V1 Pipeline."""
    def __init__(self):
        self.sessions = SessionManager()
        self.seen_messages = {} # session_id -> set of msg hashes
        self.last_access = {} # session_id -> timestamp

    def _cleanup(self):
        now = time.time()
        to_kill = [sid for sid, ts in self.last_access.items() if now - ts > 600] # 10m cleanup
        for sid in to_kill:
            self.sessions.terminate_session(sid)
            self.seen_messages.pop(sid, None)
            self.last_access.pop(sid, None)

    def ingest_event(self, session_id: str, content: Any, source: str):
        if not content: return
        self.last_access[session_id] = time.time()
        
        # Fast normalization
        if isinstance(content, list):
            text = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        else:
            text = str(content)
        
        if not text or "RELEVANT SESSION CONTEXT" in text[:100]: return
        
        msg_hash = hashlib.md5(text.encode()).hexdigest()
        if session_id not in self.seen_messages: self.seen_messages[session_id] = set()
        if msg_hash in self.seen_messages[session_id]: return
        self.seen_messages[session_id].add(msg_hash)
        
        # Fast chunking
        raw_chunks = [p.strip() for p in RE_BLOCKS.split(text) if p.strip()]
        to_ingest = []
        to_embed = []
        
        for p in raw_chunks:
            if p.startswith("---") or p.startswith("# Session Context"): continue
            
            p_hash = hashlib.md5(p.encode()).hexdigest()
            ctype = "exploratory"
            if p[0] in ('`', '#'): ctype = "authoritative"
            elif "error" in p.lower() or "traceback" in p.lower(): ctype = "diagnostic"
            elif source in ("chat", "user", "assistant"): ctype = "historical"
            
            chunk = Chunk(id=f"{session_id}-{p_hash[:16]}-{ctype[:3]}", session_id=session_id, source=source, type=ctype, text=p)
            
            if p_hash in _embedding_cache:
                chunk.embed_vec = _embedding_cache[p_hash]
            elif _embedder_ready.is_set():
                to_embed.append((p, p_hash, chunk))
            
            to_ingest.append(chunk)

        if to_embed:
            vecs = list(_embedder.embed([x[0] for x in to_embed]))
            for i, (p, p_hash, chunk) in enumerate(to_embed):
                v = vecs[i][:128].tolist()
                chunk.embed_vec = v
                _embedding_cache[p_hash] = v
        
        self.sessions.ingest_batch(to_ingest)

    def optimize(self, session_id: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start_time = time.time()
        self._cleanup()
        
        last_msg = messages[-1]
        query_text = str(last_msg.get("content", ""))
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        cache_key = (session_id, query_hash)
        
        if cache_key in _retrieval_cache:
            cached_msgs, ts = _retrieval_cache[cache_key]
            if time.time() - ts < 15: return cached_msgs

        # Fast Ingest: Only process the last message if session is active
        ingest_start = time.time()
        if session_id in self.last_access and len(messages) > 1:
            self.ingest_event(session_id, query_text, last_msg.get("role", "user"))
        else:
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
        optimized.append(last_msg)
        
        pack_time = (time.time() - pack_start) * 1000
        total_time = (time.time() - start_time) * 1000

        log.info(f"⚡️ [{session_id}] | {len(relevant)} chunks | {total_time:.0f}ms (I:{ingest_time:.0f} S:{search_time:.0f} P:{pack_time:.0f})")
        
        _retrieval_cache[cache_key] = (optimized, time.time())
        return optimized

    def _to_toon(self, chunks: List[Chunk]) -> str:
        if not chunks: return ""
        tree = {}
        for c in chunks:
            if c.type not in tree: tree[c.type] = {}
            if c.source not in tree[c.type]: tree[c.type][c.source] = []
            tree[c.type][c.source].append(c.text)
            
        lines = ["# Session Summary", ""]
        for ctype, sources in tree.items():
            lines.append(f"{ctype}:")
            for source, texts in sources.items():
                lines.append(f"  {source}:")
                for txt in texts:
                    lines.append(f"    - \"{txt.replace(chr(10), ' ')}\"")
        return "\n".join(lines)
