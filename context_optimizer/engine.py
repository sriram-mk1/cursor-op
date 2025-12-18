import re
import sqlite3
import logging
import hashlib
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
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
    source: str  # tool name / filename / chat
    type: str    # authoritative | diagnostic | exploratory | historical
    text: str
    created_at: float = field(default_factory=time.time)
    token_est: int = 0
    lex_terms: str = ""
    embed_vec: Optional[List[float]] = None

    def to_dict(self):
        return asdict(self)

class SessionManager:
    """Manages multi-session chunk storage and retrieval using SQLite."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_db()
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _setup_db(self):
        # FTS5 for lexical search
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, session_id UNINDEXED, text, lex_terms)")
        # Metadata table
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
        # Normalize text for lexical terms
        lex = re.sub(r'[^\w\s]', ' ', chunk.text).lower()
        chunk.lex_terms = lex
        chunk.token_est = len(self.encoder.encode(chunk.text))
        
        # Store in FTS (Ignore if exists)
        self.db.execute(
            "INSERT OR IGNORE INTO chunks_fts(id, session_id, text, lex_terms) VALUES (?, ?, ?, ?)",
            (chunk.id, chunk.session_id, chunk.text, chunk.lex_terms)
        )
        
        # Store in Meta (Ignore if exists)
        vec_blob = np.array(chunk.embed_vec, dtype=np.float32).tobytes() if chunk.embed_vec else None
        self.db.execute(
            "INSERT OR IGNORE INTO chunks_meta(id, session_id, source, type, text, created_at, token_est, embed_vec) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chunk.id, chunk.session_id, chunk.source, chunk.type, chunk.text, chunk.created_at, chunk.token_est, vec_blob)
        )
        self.db.commit()

    def get_session_chunks(self, session_id: str, limit: int = 100) -> List[Chunk]:
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

    def hybrid_search(self, session_id: str, query_text: str, query_vec: Optional[np.ndarray], top_k: int = 20) -> List[Chunk]:
        # 1. Lexical Search (BM25)
        lex_ids = {}
        try:
            q = re.sub(r'[^\w\s]', ' ', query_text).strip()
            if q:
                cursor = self.db.execute(
                    "SELECT id, bm25(chunks_fts) FROM chunks_fts WHERE session_id = ? AND chunks_fts MATCH ? LIMIT 50",
                    (session_id, q)
                )
                for row in cursor:
                    lex_ids[row[0]] = -row[1] # Lower is better
        except: pass

        # 2. Semantic Search (if vec provided)
        all_session_chunks = self.get_session_chunks(session_id, limit=500)
        semantic_scores = {}
        if query_vec is not None:
            q_norm = query_vec / np.linalg.norm(query_vec)
            for c in all_session_chunks:
                if c.embed_vec:
                    c_vec = np.array(c.embed_vec)
                    sim = np.dot(q_norm, c_vec / np.linalg.norm(c_vec))
                    semantic_scores[c.id] = float(sim)

        # 3. Fusion
        results = []
        # Normalize scores
        def norm(d):
            if not d: return {}
            vs = list(d.values())
            mi, ma = min(vs), max(vs)
            if mi == ma: return {k: 1.0 for k in d}
            return {k: (v - mi) / (ma - mi) for k, v in d.items()}

        lex_norm = norm(lex_ids)
        sem_norm = norm(semantic_scores)
        
        combined = {}
        for c in all_session_chunks:
            score = 0.4 * lex_norm.get(c.id, 0) + 0.6 * sem_norm.get(c.id, 0)
            if score > 0:
                c.score = score
                combined[c.id] = c
        
        return sorted(combined.values(), key=lambda x: x.score, reverse=True)[:top_k]

class SmartChunker:
    """Simple, reliable structure-aware chunking."""
    
    def __init__(self, target_tokens: int = 500):
        self.target_tokens = target_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_and_classify(self, text: str, source: str) -> List[Tuple[str, str]]:
        if not text: return []
        
        # Split by logical blocks: code, headings, or double newlines
        parts = re.split(r'(\n#{1,6}\s+.*|```[\s\S]*?```|\n\n)', text)
        chunks = []
        
        for p in parts:
            p = p.strip()
            if not p: continue
            
            ctype = "exploratory"
            if p.startswith("```"): ctype = "authoritative"
            elif p.startswith("#"): ctype = "authoritative"
            elif "error" in p.lower() or "traceback" in p.lower(): ctype = "diagnostic"
            elif source == "chat": ctype = "historical"
            
            chunks.append((p, ctype))
            
        return chunks

class ContextOptimizer:
    """V1 Pipeline Orchestrator."""
    
    def __init__(self):
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
            
        if not text.strip(): return
        
        msg_hash = hashlib.md5(text.encode()).hexdigest()
        if session_id not in self.seen_messages:
            self.seen_messages[session_id] = set()
        if msg_hash in self.seen_messages[session_id]:
            return
        self.seen_messages[session_id].add(msg_hash)
        
        raw_chunks = self.chunker.chunk_and_classify(text, source)
        embedder = get_embedder()
        for content_part, ctype in raw_chunks:
            vec = None
            if embedder:
                vec = list(embedder.embed([content_part]))[0][:128].tolist()
            
            # Use a more robust ID: session + content hash + type
            c_hash = hashlib.md5(content_part.encode()).hexdigest()
            chunk_id = f"{session_id}-{c_hash[:16]}-{ctype[:3]}"
            
            chunk = Chunk(
                id=chunk_id,
                session_id=session_id,
                source=source,
                type=ctype,
                text=content_part,
                embed_vec=vec
            )
            self.sessions.ingest_chunk(chunk)

    def build_query(self, messages: List[Dict[str, Any]]) -> Tuple[str, Optional[np.ndarray]]:
        # Just use the last message for the query to keep it clean
        last_msg = messages[-1]
        query_text = last_msg.get("content", "")
        if isinstance(query_text, list):
            query_text = " ".join(p.get("text", "") for p in query_text if p.get("type") == "text")
            
        embedder = get_embedder()
        query_vec = None
        if embedder and query_text:
            query_vec = list(embedder.embed([query_text]))[0][:128]
            
        return query_text, query_vec

    def optimize(self, session_id: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Ingest ALL messages in the current request to ensure they are indexed for this session
        for m in messages:
            self.ingest_event(session_id, m.get("content", ""), m.get("role", "user"))
        
        # Build query
        query_text, query_vec = self.build_query(messages)
        
        # Retrieve
        relevant = self.sessions.hybrid_search(session_id, query_text, query_vec, top_k=25)
        
        # TOON Reconstruction
        toon_context = self._to_toon(relevant)
        
        optimized = []
        system = next((m for m in messages if m.get("role") == "system"), None)
        if system: optimized.append(system)
        
        if toon_context:
            optimized.append({
                "role": "system", 
                "content": (
                    "CRITICAL: You are an AI with access to a session history. "
                    "The following context is retrieved from the current session. "
                    "Use it to answer the user's request accurately. "
                    "If the user asks about something in the history, it IS in the context below.\n\n"
                    + toon_context
                )
            })
        
        optimized.append(messages[-1]) # Current query
        
        log.info(f"✨ V1 RAG [{session_id}]: {len(relevant)} chunks retrieved")
        return optimized

    def _to_toon(self, chunks: List[Chunk]) -> str:
        if not chunks: return ""
        lines = ["---", "url: /context/session.toon", "---", "# Session Context", ""]
        
        # Group by type
        by_type = {}
        for c in chunks:
            if c.type not in by_type: by_type[c.type] = []
            by_type[c.type].append(c)
            
        for ctype, cs in by_type.items():
            lines.append(f"{ctype}[{len(cs)}]:")
            for c in cs:
                # TOON format: source: "text"
                text = c.text.replace('"', '\\"').replace('\n', ' ')
                lines.append(f"  - {c.source}: \"{text}\"")
        return "\n".join(lines)
