import re
import sqlite3
import logging
import hashlib
import time
import json
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import tiktoken

log = logging.getLogger("gateway")

# --- Pre-compiled Globals ---
RE_CLEAN = re.compile(r'[^\w\s]')
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
        log.info("🔄 Loading Snowflake/snowflake-arctic-embed-xs...")
        _embedder = TextEmbedding(model_name="Snowflake/snowflake-arctic-embed-xs")
        _embedder_ready.set()
        log.info("✅ Embedder ready")
    except Exception as e:
        log.error(f"❌ Embedder error: {e}")

threading.Thread(target=_load_model_bg, daemon=True).start()

@dataclass
class Chunk:
    id: str
    session_id: str
    source: str
    type: str
    text: str
    created_at: float = field(default_factory=time.time)
    score: float = 0.0

class SessionManager:
    """Fast in-memory storage with vectorized search."""
    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=OFF")
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
                embed_vec BLOB
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_session ON chunks_meta(session_id)")
        self.db.commit()

    def ingest_batch(self, chunks: List[Dict[str, Any]]):
        for c in chunks:
            lex = RE_CLEAN.sub(' ', c['text']).lower()
            self.db.execute("INSERT OR IGNORE INTO chunks_fts VALUES (?, ?, ?, ?)", (c['id'], c['session_id'], c['text'], lex))
            vec_blob = np.array(c['embed_vec'], dtype=np.float32).tobytes() if c.get('embed_vec') else None
            self.db.execute("INSERT OR IGNORE INTO chunks_meta VALUES (?, ?, ?, ?, ?, ?, ?)", 
                           (c['id'], c['session_id'], c['source'], c['type'], c['text'], c['created_at'], vec_blob))
        self.db.commit()

    def search(self, session_id: str, query_text: str, query_vec: Optional[np.ndarray], top_k: int = 45) -> List[Chunk]:
        lex_scores = {}
        try:
            q = RE_CLEAN.sub(' ', query_text).strip()
            if q:
                cursor = self.db.execute("SELECT id, bm25(chunks_fts) FROM chunks_fts WHERE session_id = ? AND chunks_fts MATCH ? LIMIT 100", (session_id, q))
                for row in cursor: lex_scores[row[0]] = -row[1]
        except: pass

        cursor = self.db.execute("SELECT id, embed_vec, created_at FROM chunks_meta WHERE session_id = ? ORDER BY created_at DESC LIMIT 500", (session_id,))
        rows = cursor.fetchall()
        if not rows: return []
        
        ids, vec_blobs, times = [r[0] for r in rows], [r[1] for r in rows], np.array([r[2] for r in rows])
        sem_scores = {}
        if query_vec is not None:
            valid = [i for i, v in enumerate(vec_blobs) if v]
            if valid:
                matrix = np.frombuffer(b''.join([vec_blobs[i] for i in valid]), dtype=np.float32).reshape(len(valid), 128)
                q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
                sims = np.dot(matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9), q_norm)
                for i, idx in enumerate(valid): sem_scores[ids[idx]] = float(sims[i])

        def norm(d):
            if not d: return {}
            vs = list(d.values())
            mi, ma = min(vs), max(vs)
            return {k: (v - mi) / (ma - mi + 1e-9) for k, v in d.items()}

        l_n, s_n = norm(lex_scores), norm(sem_scores)
        now = time.time()
        results = []
        for i, cid in enumerate(ids):
            score = (0.3 * l_n.get(cid, 0) + 0.7 * s_n.get(cid, 0)) * (1.2 / (1.0 + (now - times[i]) / 3600))
            if score > 0.01: results.append((cid, score))
        
        top = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        if not top: return []
        
        id_map = {cid: s for cid, s in top}
        placeholders = ','.join(['?'] * len(top))
        cursor = self.db.execute(f"SELECT id, session_id, source, type, text, created_at FROM chunks_meta WHERE id IN ({placeholders})", [x[0] for x in top])
        
        final = []
        for r in cursor:
            c = Chunk(id=r[0], session_id=r[1], source=r[2], type=r[3], text=r[4], created_at=r[5])
            c.score = id_map.get(c.id, 0)
            final.append(c)
        return sorted(final, key=lambda x: x.score, reverse=True)

class ContextOptimizer:
    """Simplified, High-Performance RAG."""
    def __init__(self):
        self.sessions = SessionManager()
        self.seen = {} # session_id -> set
        self.last = {} # session_id -> ts

    def ingest(self, session_id: str, content: Any, role: str):
        if not content or role == "system": return
        text = " ".join(p.get("text", "") for p in content if isinstance(p, dict)) if isinstance(content, list) else str(content)
        if not text.strip() or "RELEVANT SESSION CONTEXT" in text[:100]: return
        
        self.last[session_id] = time.time()
        m_hash = hashlib.md5(text.encode()).hexdigest()
        if session_id not in self.seen: self.seen[session_id] = set()
        if m_hash in self.seen[session_id]: return
        self.seen[session_id].add(m_hash)
        
        # Simple robust chunking
        parts = [p.strip() for p in re.split(r'(\n\n|```[\s\S]*?```)', text) if p.strip()]
        to_ingest = []
        to_embed = []
        
        for p in parts:
            p_hash = hashlib.md5(p.encode()).hexdigest()
            ctype = "authoritative" if p[0] in ('`', '#') else "diagnostic" if "error" in p.lower() else "historical"
            chunk = {"id": f"{session_id}-{p_hash[:16]}", "session_id": session_id, "source": role, "type": ctype, "text": p, "created_at": time.time()}
            
            if p_hash in _embedding_cache:
                chunk["embed_vec"] = _embedding_cache[p_hash]
            elif _embedder_ready.is_set():
                to_embed.append((p, p_hash, chunk))
            to_ingest.append(chunk)

        if to_embed:
            vecs = list(_embedder.embed([x[0] for x in to_embed]))
            for i, (p, p_hash, chunk) in enumerate(to_embed):
                v = vecs[i][:128].tolist()
                chunk["embed_vec"] = v
                _embedding_cache[p_hash] = v
        
        self.sessions.ingest_batch(to_ingest)

    def optimize(self, session_id: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start = time.time()
        # Fast-path ingest
        if session_id in self.last:
            self.ingest(session_id, messages[-1].get("content"), messages[-1].get("role"))
        else:
            for m in messages: self.ingest(session_id, m.get("content"), m.get("role"))
        
        query = str(messages[-1].get("content", ""))
        q_vec = list(_embedder.embed([query]))[0][:128] if _embedder_ready.is_set() else None
        relevant = self.sessions.search(session_id, query, q_vec)
        
        # Build TOON
        tree = {}
        for c in relevant:
            if c.type not in tree: tree[c.type] = {}
            if c.source not in tree[c.type]: tree[c.type][c.source] = []
            tree[c.type][c.source].append(c.text.replace('\n', ' '))
            
        toon = ["# Session Summary"]
        for t, srcs in tree.items():
            toon.append(f"{t}:")
            for s, txts in srcs.items():
                toon.append(f"  {s}:")
                for txt in txts: toon.append(f"    - \"{txt}\"")
        
        optimized = []
        sys = next((m for m in messages if m.get("role") == "system"), None)
        if sys: optimized.append(sys)
        if relevant: optimized.append({"role": "system", "content": f"--- SESSION CONTEXT ---\n" + "\n".join(toon)})
        optimized.append(messages[-1])
        
        log.info(f"⚡️ [{session_id}] | {len(relevant)} chunks | {int((time.time()-start)*1000)}ms")
        return optimized
