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
import tiktoken

import numpy as np

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
    """V3.2: NumPy-Accelerated Scorer with Gaussian Density & Temporal Decay."""
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
        
        # 1. Smarter comment removal (avoiding common false positives like strings)
        distilled = re.sub(r'(?m)^\s*#.*$', '', text) 
        distilled = re.sub(r'(?m)^\s*//.*$', '', distilled) 
        distilled = re.sub(r'\s+#(?![^{}]*\}).*$', '', distilled) 
        distilled = re.sub(r'\s+//(?![^{}]*\}).*$', '', distilled)
        
        # 2. Compress whitespace
        distilled = " ".join(distilled.split())
        
        # 3. Truncate extreme lines
        if len(distilled) > 300:
            distilled = distilled[:140] + " ... " + distilled[-140:]
            
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
        
        # Vectorized BM25 Core
        raw_scores = np.zeros(num_docs)
        
        # Pre-calculate IDFs
        idf = {}
        for term in query_terms:
            doc_freq = sum(1 for a in atoms if term in a.terms)
            idf[term] = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        
        for i, atom in enumerate(atoms):
            if not atom.terms: continue
            
            doc_score = 0.0
            doc_len = len(atom.terms)
            for term in query_terms:
                if term in atom.terms:
                    # BM25 Component
                    tf = 1 
                    numerator = idf[term] * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
                    doc_score += (numerator / denominator) + self.delta
            
            # Application of Structural Logic
            if atom.provides: doc_score *= 1.8
            if any(c in atom.text for c in ('{', '}', '(', ')', '=', ':', '`')):
                doc_score *= 1.2
                
            raw_scores[i] = doc_score

        # ---------------------------------------------------------
        # V3.2 Power Step: Gaussian Density Smoothing
        # ---------------------------------------------------------
        # This "spreads" the score of a hit to its neighbors, 
        # naturally highlighting functional blocks.
        kernel_size = 5
        sigma = 1.0
        x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        kernel = np.exp(-0.5 * (x / sigma)**2)
        kernel /= kernel.sum()
        
        # Apply smoothing
        smoothed_scores = np.convolve(raw_scores, kernel, mode='same')
        
        # ---------------------------------------------------------
        # V3.2 Power Step: Exponential Temporal Decay
        # ---------------------------------------------------------
        # We want a smooth decay from 1.0 (newest) down to 0.4 (oldest)
        msg_indices = np.array([a.msg_index for a in atoms])
        max_msg = max(1, msg_indices[-1])
        # Modern Exponential Decay Policy
        temporal_weights = 0.4 + 0.6 * np.exp(2.0 * (msg_indices / max_msg - 1))
        
        # Final Blend
        final_scores = (raw_scores * 0.7 + smoothed_scores * 0.3) * temporal_weights
        
        for i, atom in enumerate(atoms):
            atom.score = float(final_scores[i])
                
        return atoms

class ContextOptimizer:
    """V3: Full Structural-Temporal Distillation Engine."""
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        
        self.scorer = AtomizedScorer()
        self.base_budget = 1800
        self.session_ttl = 86400 # 24 Hours
        
        # Redis Setup: Smart detection with strict timeouts
        redis_url = os.getenv("REDIS_URL") or os.getenv("REDISURL") or os.getenv("REDISHOST")
        if redis_url and not redis_url.startswith("redis"):
            redis_url = f"redis://{redis_url}"
            
        if redis_url:
            try:
                # Add strict timeouts to prevent hanging on cold starts or network issues
                self.redis = redis.from_url(
                    redis_url, 
                    decode_responses=True,
                    socket_timeout=3.0,
                    socket_connect_timeout=3.0,
                    retry_on_timeout=False
                )
                # Verify connection immediately
                self.redis.ping()
                log.info(f"📡 [V3 RAG] Connected to Redis KV Cache")
            except Exception as e:
                log.error(f"❌ Redis connection failed (falling back to RAM): {e}")
                self.redis = None
        else:
            self.redis = None
            log.warning("🏠 [V3 RAG] No REDIS_URL found, using local RAM storage")
            
        self.local_sessions: Dict[str, Dict[str, Any]] = {}

    def _get_session(self, user_key: str, session_id: str) -> Dict[str, Any]:
        """Strictly isolated session retrieval."""
        storage_key = f"v3:{hashlib.md5(user_key.encode()).hexdigest()[:10]}:{session_id}"
        if self.redis:
            data = self.redis.get(storage_key)
            if data:
                raw = json.loads(data)
                return {
                    "history": raw.get("history", []),
                    "atoms": [Atom.from_dict(a) for a in raw.get("atoms", [])]
                }
        return self.local_sessions.get(storage_key, {"history": [], "atoms": []})

    def _save_session(self, user_key: str, session_id: str, history: List, atoms: List):
        """Strictly isolated session persistence."""
        storage_key = f"v3:{hashlib.md5(user_key.encode()).hexdigest()[:10]}:{session_id}"
        data = {
            "history": history,
            "atoms": [a.to_dict() for a in atoms]
        }
        if self.redis:
            self.redis.setex(storage_key, self.session_ttl, json.dumps(data))
        else:
            self.local_sessions[storage_key] = {"history": history, "atoms": atoms}

    def ingest(self, user_key: str, session_id: str, messages: List[Dict[str, Any]]):
        session = self._get_session(user_key, session_id)
        history = session["history"]
        atoms = session["atoms"]
        
        # Phase 4: Boilerplate Noise Filters (The "Blackhole" Filter)
        NOISE_PATTERNS = [
            r"<(task|environment_details|slug|name|model|tool_format|todos|update_todo_list)>",
            r"</(task|environment_details|slug|name|model|tool_format|todos|update_todo_list)>",
            r"# (VSCode Visible Files|VSCode Open Tabs|Current Time|Current Cost|Current Mode|Current Workspace Directory-)",
            r"Current time in ISO 8601",
            r"User time zone:",
            r"No files found\.",
            r"^\$?\d+\.\d{2}$", # Catches $0.00
            r"You have not created a todo list yet",
            r"REMINDERS",
            r"\| # \| Content \| Status \|",
            r"\[(ask_followup_question|update_todo_list)\] Result:"
        ]
        noise_re = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

        changed = False
        new_atoms = []
        for msg in messages:
            content = msg.get("content", "")
            if not content or msg.get("role") == "system": continue
            
            raw_text = self.scorer._to_string(content)
            m_hash = hashlib.md5(raw_text.encode()).hexdigest()
            if any(m.get("hash") == m_hash for m in history):
                continue
            
            msg_idx = len(history)
            msg_ts = time.time()
            history.append({
                "role": msg.get("role"),
                "content": raw_text,
                "hash": m_hash,
                "timestamp": msg_ts
            })
            
            lines = raw_text.split('\n')
            for line in lines:
                clean_line = line.strip()
                if not clean_line or len(clean_line) < 2: continue
                
                # Apply Phase 4 Noise Filter: Environmental garbage detection
                if any(r.search(clean_line) for r in noise_re): 
                    continue
                
                # We store the RAW line for the LLM to see, 
                # but we can still use distillation internally for term extraction
                new_atoms.append(Atom(
                    line_index=len(atoms) + len(new_atoms),
                    msg_index=msg_idx,
                    source=msg.get("role"),
                    text=clean_line, # <--- PRESERVE ORIGINAL CONTEXT
                    tokens=len(ENCODER.encode(clean_line)),
                    timestamp=msg_ts,
                    terms=self.scorer.get_terms(clean_line), # Terms are already distilled internally
                    provides=self.scorer.extract_definitions(clean_line)
                ))
            changed = True
        
        if changed:
            atoms.extend(new_atoms)
            self._save_session(user_key, session_id, history, atoms)

    def optimize(self, user_key: str, session_id: str, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        
        self.ingest(user_key, session_id, messages)
        session = self._get_session(user_key, session_id)
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
        
        # Higher density selection for the Top-K
        top_atoms = sorted([a for a in scored_atoms if a.score > 0.1], key=lambda x: x.score, reverse=True)[:50]
        if not top_atoms:
            return messages[-5:] if len(messages) > 5 else messages, {"status": "no_matches"}

        selected_indices: Set[int] = set()
        def_map = {} # name -> line_index
        for a in atoms:
            for d in a.provides: def_map[d] = a.line_index

        for atom in top_atoms:
            # Expand neighborhood for context flow
            for offset in range(-1, 2):
                idx = atom.line_index + offset
                if 0 <= idx < len(atoms): selected_indices.add(idx)
                    
            # Structural Link
            for d_name, d_idx in def_map.items():
                if d_name in atom.text:
                    for offset in range(-1, 4):
                        idx = d_idx + offset
                        if 0 <= idx < len(atoms): selected_indices.add(idx)
        
        final_atoms = sorted([atoms[idx] for idx in selected_indices], key=lambda x: x.line_index)
        
        packed_lines = []
        current_tokens = 0
        last_msg_idx = -1
        
        for atom in final_atoms:
            header = ""
            if atom.msg_index != last_msg_idx:
                prefix = "User" if atom.source == "user" else "AI"
                header = f"\n### {prefix} (T-{int(time.time() - atom.timestamp)}s ago)\n"
                last_msg_idx = atom.msg_index
            
            line_text = f"{header}  {atom.text}"
            line_tokens = atom.tokens + (len(ENCODER.encode(header)) if header else 0)
            
            if current_tokens + line_tokens > max_tokens: break
                
            packed_lines.append(line_text)
            current_tokens += line_tokens
            
        optimized = []
        sys_prompt = next((m for m in messages if m.get("role") == "system"), None)
        if sys_prompt: optimized.append(sys_prompt)
        
        if packed_lines:
            # V3.3 Hyper-Clean Markdown Wrap
            context_text = (
                "## CONTEXT OPTIMIZER (V3.3 ACTIVE)\n"
                "The following fragments have been distilled for maximum relevance:\n"
                + "\n".join(packed_lines)
                + "\n\n---\n*End of distilled context*"
            )
            optimized.append({"role": "system", "content": context_text})
            
        optimized.append(messages[-1])
        overhead = (time.time() - start_time) * 1000
        
        return optimized, {
            "version": "3.3.0",
            "specificity": specificity,
            "budget": max_tokens,
            "total_lines": len(atoms),
            "selected_lines": len(packed_lines),
            "overhead_ms": overhead,
            "sequence": [
                {
                    "source": a.source,
                    "text": a.text,
                    "score": round(a.score, 3),
                    "line_index": a.line_index
                }
                for a in final_atoms[:30]
            ]
        }
