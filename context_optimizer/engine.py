import re
import logging
import hashlib
import time
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
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
    score: float = 0.0

class AtomizedScorer:
    """V3: BM25+ Scorer with Temporal Decay."""
    def __init__(self):
        self.stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "while", "in", "is", "it", "of", "to"}
        # BM25+ Constants
        self.k1 = 1.2
        self.b = 0.75
        self.delta = 1.0
        # Temporal Decay Constant (decay half-life approx 12 hours)
        self.lambda_decay = 0.05 

    def _to_string(self, content: Any) -> str:
        if isinstance(content, str): return content
        if isinstance(content, list):
            return " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        return str(content)

    def get_terms(self, text: Any) -> Set[str]:
        text_str = self._to_string(text)
        terms = set(RE_CLEAN.sub(' ', text_str).lower().split())
        return terms - self.stop_words

    def score_atoms(self, atoms: List[Atom], query: str) -> List[Atom]:
        if not atoms or not query: return atoms
        
        query_terms = self.get_terms(query)
        if not query_terms: return atoms
        
        # Calculate Corpus Stats for BM25
        num_docs = len(atoms)
        avgdl = sum(len(a.terms) for a in atoms) / num_docs if num_docs > 0 else 1
        
        # Calculate IDF for each query term
        idf = {}
        for term in query_terms:
            doc_freq = sum(1 for a in atoms if term in a.terms)
            idf[term] = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        now = time.time()
        
        for atom in atoms:
            if not atom.terms:
                atom.score = 0.0
                continue
                
            bm25_score = 0.0
            doc_len = len(atom.terms)
            
            for term in query_terms:
                if term in atom.terms:
                    tf = 1 # We treat line-level as binary or low-frequency
                    numerator = idf[term] * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
                    bm25_score += (numerator / denominator) + self.delta
            
            # Application of Temporal Decay
            # age_hours = max(0, (now - atom.timestamp) / 3600)
            # decay = math.exp(-self.lambda_decay * age_hours)
            # atom.score = bm25_score * decay
            
            # Simpler Recency Boost: linear decay within session
            recency_boost = 1.0 + (atom.msg_index / max(1, len(atoms)))
            atom.score = bm25_score * recency_boost
            
            # Structural Boost (V2 legacy)
            if any(c in atom.text for c in ('{', '}', '(', ')', '=', ':', '`')):
                atom.score *= 1.2
                
        return atoms

class ContextOptimizer:
    """V3: Structural-Temporal Distillation Engine."""
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.session_atoms: Dict[str, List[Atom]] = {}
        self.scorer = AtomizedScorer()
        self.base_budget = 1800

    def ingest(self, session_id: str, messages: List[Dict[str, Any]]):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            self.session_atoms[session_id] = []
            
        new_atoms = []
        for msg in messages:
            content = msg.get("content", "")
            if not content or msg.get("role") == "system": continue
            
            text = self.scorer._to_string(content)
            if "SESSION CONTEXT" in text[:100] or "# Session Summary" in text[:100]:
                continue
                
            m_hash = hashlib.md5(text.encode()).hexdigest()
            if any(m.get("hash") == m_hash for m in self.sessions[session_id]):
                continue
            
            msg_idx = len(self.sessions[session_id])
            msg_ts = time.time()
            self.sessions[session_id].append({
                "role": msg.get("role"),
                "content": text,
                "hash": m_hash,
                "timestamp": msg_ts
            })
            
            # Speculative Atomization (Pre-process lines)
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                new_atoms.append(Atom(
                    line_index=len(self.session_atoms[session_id]) + len(new_atoms),
                    msg_index=msg_idx,
                    source=msg.get("role"),
                    text=line,
                    tokens=len(ENCODER.encode(line)),
                    timestamp=msg_ts,
                    terms=self.scorer.get_terms(line)
                ))
        
        self.session_atoms[session_id].extend(new_atoms)

    def optimize(self, session_id: str, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        
        # 1. Speculative Ingest
        self.ingest(session_id, messages)
        atoms = self.session_atoms.get(session_id, [])
        if not atoms: return messages, {}
        
        # 2. Dynamic Budgeting
        query = self.scorer._to_string(messages[-1].get("content", ""))
        query_terms = self.scorer.get_terms(query)
        
        # Ambiguity detection: brief queries get more context
        specificity = len(query_terms)
        if specificity <= 2:
            max_tokens = 3000 # broad search for ambiguous query
        elif specificity >= 6:
            max_tokens = 1200 # narrow, precise context only
        else:
            max_tokens = self.base_budget

        # 3. V3 Scoring: BM25+ with Temporal Weighting
        scored_atoms = self.scorer.score_atoms(atoms, query)
        
        # 4. Cluster expansion
        top_atoms = sorted([a for a in scored_atoms if a.score > 0], key=lambda x: x.score, reverse=True)[:60]
        if not top_atoms:
            return messages[-5:] if len(messages) > 5 else messages, {"status": "no_matches"}

        selected_indices: Set[int] = set()
        for atom in top_atoms:
            # Expand logic context
            for offset in range(-2, 3):
                idx = atom.line_index + offset
                if 0 <= idx < len(atoms):
                    selected_indices.add(idx)
        
        final_atoms = sorted([atoms[idx] for idx in selected_indices], key=lambda x: x.line_index)
        
        # 5. Packing
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
            
        # 6. Rebuild
        optimized = []
        sys_prompt = next((m for m in messages if m.get("role") == "system"), None)
        if sys_prompt: optimized.append(sys_prompt)
        
        if packed_lines:
            context_text = "--- RELEVANT SESSION HISTORY (ATOMIZED V3) ---\n" + "\n".join(packed_lines)
            optimized.append({"role": "system", "content": context_text})
            
        optimized.append(messages[-1])
        
        overhead = (time.time() - start_time) * 1000
        log.info(f"⚡️ [V3 RAG] {session_id} | Spec: {specificity} | Bud: {max_tokens} | Score: BM25+ | {overhead:.1f}ms")
        
        return optimized, {
            "version": "3.0.0",
            "specificity": specificity,
            "budget": max_tokens,
            "selected_lines": len(packed_lines),
            "overhead_ms": overhead
        }
