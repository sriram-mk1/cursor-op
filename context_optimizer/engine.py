import os
import json
import redis
import re
import logging
import hashlib
import time
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import numpy as np
import tiktoken

log = logging.getLogger("gateway")

# --- Pre-compiled Globals ---
RE_CLEAN = re.compile(r'[^\w\s]')
ENCODER = tiktoken.get_encoding("cl100k_base")

@dataclass
class Atom:
    line_index: int
    msg_index: int
    source: str
    text: str
    tokens: int
    timestamp: float
    terms: Set[str] = field(default_factory=set)
    provides: Set[str] = field(default_factory=set) # Track function/class definitions
    score: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d['terms'] = list(self.terms)
        d['provides'] = list(self.provides)
        return d
    
    @classmethod
    def from_dict(cls, d):
        d['terms'] = set(d.get('terms', []))
        d['provides'] = set(d.get('provides', []))
        return cls(**d)

class AtomizedScorer:
    """V3: BM25+ Scorer with Structural Intelligence & Distillation."""
    def __init__(self):
        self.stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "while", "in", "is", "it", "of", "to"}
        self.k1 = 1.2
        self.b = 0.75
        self.delta = 1.0
        
        # Phase 2: Definition Detectors
        self.re_defs = [
            re.compile(r'(?:def|class|async def)\s+([a-zA-Z_]\w*)'), # Python
            re.compile(r'(?:function|class|const|let|var)\s+([a-zA-Z_]\w*)'), # JS/TS
            re.compile(r'interface\s+([a-zA-Z_]\w*)'), # TS Interface
            re.compile(r'type\s+([a-zA-Z_]\w*)\s*=') # TS Type
        ]

    def _to_string(self, content: Any) -> str:
        if isinstance(content, str): return content
        if isinstance(content, list):
            return " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        return str(content)

    def distill(self, text: str) -> str:
        """Phase 3: Semantic Distillation (Lossless-ish Compression)."""
        if len(text) < 10: return text
        
        # 1. Remove obvious comments (careful with strings)
        distilled = re.sub(r'#.*$', '', text) # Python comments
        distilled = re.sub(r'//.*$', '', distilled) # JS comments
        
        # 2. Compress whitespace
        distilled = " ".join(distilled.split())
        
        # 3. Truncate extreme lines (Boilerplate avoidance)
        if len(distilled) > 250:
            distilled = distilled[:120] + " ... " + distilled[-120:]
            
        return distilled.strip()

    def get_terms(self, text: Any) -> Set[str]:
        text_str = self._to_string(text)
        terms = set(RE_CLEAN.sub(' ', text_str).lower().split())
        return terms - self.stop_words

    def extract_definitions(self, text: str) -> Set[str]:
        defs = set()
        for r in self.re_defs:
            matches = r.findall(text)
            for m in matches: defs.add(m)
        return defs

    def score_atoms(self, atoms: List[Atom], query: str) -> List[Atom]:
        if not atoms or not query: return atoms
        
        query_terms = self.get_terms(query)
        if not query_terms: return atoms
        
        num_docs = len(atoms)
        avgdl = sum(len(a.terms) for a in atoms) / num_docs if num_docs > 0 else 1
        
        idf = {}
        for term in query_terms:
            doc_freq = sum(1 for a in atoms if term in a.terms)
            idf[term] = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        
        for atom in atoms:
            if not atom.terms:
                atom.score = 0.0
                continue
                
            bm25_score = 0.0
            doc_len = len(atom.terms)
            
            for term in query_terms:
                if term in atom.terms:
                    tf = 1 
                    numerator = idf[term] * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
                    bm25_score += (numerator / denominator) + self.delta
            
            recency_boost = 1.0 + (atom.msg_index / max(1, len(atoms)))
            atom.score = bm25_score * recency_boost
            
            # Phase 2: Definition Boost
            if atom.provides:
                atom.score *= 1.5
                
            if any(c in atom.text for c in ('{', '}', '(', ')', '=', ':', '`')):
                atom.score *= 1.2
                
        return atoms

class ContextOptimizer:
    """V3: Full Structural-Temporal Distillation Engine."""
    def __init__(self):
        self.scorer = AtomizedScorer()
        self.base_budget = 1800
        self.session_ttl = 86400 # 24 Hours
        
        # Redis Setup
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                self.redis = redis.from_url(redis_url, decode_responses=True)
                log.info("📡 [V3 RAG] Connected to Redis KV Cache (Railway)")
            except Exception as e:
                log.error(f"❌ Redis connection failed: {e}")
                self.redis = None
        else:
            self.redis = None
            log.warning("🏠 [V3 RAG] No REDIS_URL found, using local RAM storage")
            
        self.local_sessions: Dict[str, Dict[str, Any]] = {}

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        if self.redis:
            data = self.redis.get(f"v3_sess:{session_id}")
            if data:
                raw = json.loads(data)
                return {
                    "history": raw.get("history", []),
                    "atoms": [Atom.from_dict(a) for a in raw.get("atoms", [])]
                }
        return self.local_sessions.get(session_id, {"history": [], "atoms": []})

    def _save_session(self, session_id: str, history: List, atoms: List):
        data = {
            "history": history,
            "atoms": [a.to_dict() for a in atoms]
        }
        if self.redis:
            self.redis.setex(f"v3_sess:{session_id}", self.session_ttl, json.dumps(data))
        else:
            self.local_sessions[session_id] = {"history": history, "atoms": atoms}

    def ingest(self, session_id: str, messages: List[Dict[str, Any]]):
        session = self._get_session(session_id)
        history = session["history"]
        atoms = session["atoms"]
        
        changed = False
        new_atoms = []
        for msg in messages:
            content = msg.get("content", "")
            if not content or msg.get("role") == "system": continue
            
            text = self.scorer._to_string(content)
            if "SESSION CONTEXT" in text[:100] or "# Session Summary" in text[:100]:
                continue
                
            m_hash = hashlib.md5(text.encode()).hexdigest()
            if any(m.get("hash") == m_hash for m in history):
                continue
            
            msg_idx = len(history)
            msg_ts = time.time()
            history.append({
                "role": msg.get("role"),
                "content": text,
                "hash": m_hash,
                "timestamp": msg_ts
            })
            
            lines = text.split('\n')
            for line in lines:
                raw_text = line.strip()
                if not raw_text: continue
                
                # Phase 3: Immediate Distillation on Ingestion
                distilled_text = self.scorer.distill(raw_text)
                if not distilled_text: continue
                
                # Phase 2: Definition Extraction
                defs = self.scorer.extract_definitions(raw_text)
                
                new_atoms.append(Atom(
                    line_index=len(atoms) + len(new_atoms),
                    msg_index=msg_idx,
                    source=msg.get("role"),
                    text=distilled_text,
                    tokens=len(ENCODER.encode(distilled_text)),
                    timestamp=msg_ts,
                    terms=self.scorer.get_terms(distilled_text),
                    provides=defs
                ))
            changed = True
        
        if changed:
            atoms.extend(new_atoms)
            self._save_session(session_id, history, atoms)

    def optimize(self, session_id: str, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        
        self.ingest(session_id, messages)
        session = self._get_session(session_id)
        atoms = session["atoms"]
        if not atoms: return messages, {}
        
        query = self.scorer._to_string(messages[-1].get("content", ""))
        query_terms = self.scorer.get_terms(query)
        
        specificity = len(query_terms)
        if specificity <= 2:
            max_tokens = 3000 
        elif specificity >= 6:
            max_tokens = 1200 
        else:
            max_tokens = self.base_budget

        scored_atoms = self.scorer.score_atoms(atoms, query)
        
        top_atoms = sorted([a for a in scored_atoms if a.score > 0], key=lambda x: x.score, reverse=True)[:60]
        if not top_atoms:
            return messages[-5:] if len(messages) > 5 else messages, {"status": "no_matches"}

        selected_indices: Set[int] = set()
        
        # Phase 2: Definition Map for Linking
        def_map = {} # name -> line_index
        for a in atoms:
            for d in a.provides: def_map[d] = a.line_index

        for atom in top_atoms:
            # Expand neighborhood
            for offset in range(-2, 3):
                idx = atom.line_index + offset
                if 0 <= idx < len(atoms):
                    selected_indices.add(idx)
                    
            # Structural Link: If this atom calls something we have a definition for, add it
            for d_name, d_idx in def_map.items():
                if d_name in atom.text:
                    # Found reference! Pull in the definition cluster
                    for offset in range(-1, 5): # Definitions are usually small, grab a bit more
                        idx = d_idx + offset
                        if 0 <= idx < len(atoms):
                            selected_indices.add(idx)
        
        final_atoms = sorted([atoms[idx] for idx in selected_indices], key=lambda x: x.line_index)
        
        packed_lines = []
        current_tokens = 0
        last_msg_idx = -1
        
        for atom in final_atoms:
            header = ""
            if atom.msg_index != last_msg_idx:
                header = f"\n[{atom.source.upper()}]:\n"
                last_msg_idx = atom.msg_index
            
            line_text = f"{header}  {atom.text}"
            line_tokens = atom.tokens + (len(ENCODER.encode(header)) if header else 0)
            
            if current_tokens + line_tokens > max_tokens:
                break
                
            packed_lines.append(line_text)
            current_tokens += line_tokens
            
        optimized = []
        sys_prompt = next((m for m in messages if m.get("role") == "system"), None)
        if sys_prompt: optimized.append(sys_prompt)
        
        if packed_lines:
            context_text = "--- RELEVANT SESSION HISTORY (DISTILLED V3) ---\n" + "\n".join(packed_lines)
            optimized.append({"role": "system", "content": context_text})
            
        optimized.append(messages[-1])
        
        overhead = (time.time() - start_time) * 1000
        log.info(f"⚡️ [V3 RAG] {session_id} | Spec: {specificity} | Bud: {max_tokens} | Structural Links | {overhead:.1f}ms")
        
        return optimized, {
            "version": "3.1.0",
            "specificity": specificity,
            "budget": max_tokens,
            "selected_lines": len(packed_lines),
            "overhead_ms": overhead
        }
