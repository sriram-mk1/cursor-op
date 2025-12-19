import re
import logging
import hashlib
import time
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
    score: float = 0.0

class AtomizedScorer:
    """Ultra-fast vectorized lexical matcher."""
    def __init__(self):
        self.stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "while", "in", "is", "it", "of", "to"}

    def get_terms(self, text: str) -> Set[str]:
        terms = set(RE_CLEAN.sub(' ', text).lower().split())
        return terms - self.stop_words

    def score_atoms(self, atoms: List[Atom], query: str) -> List[Atom]:
        if not atoms or not query: return atoms
        
        query_terms = list(self.get_terms(query))
        if not query_terms: return atoms
        
        # Vectorized scoring: One pass over all atoms
        # We use a simple but effective term-frequency / line-length heuristic
        for atom in atoms:
            atom_terms = self.get_terms(atom.text)
            if not atom_terms: continue
            
            matches = sum(1 for t in query_terms if t in atom_terms)
            if matches == 0: continue
            
            # Score = (matches / sqrt(query_len)) * (matches / sqrt(atom_len))
            # This rewards density and exact matches
            atom.score = (matches / (len(query_terms)**0.5)) * (matches / (len(atom_terms)**0.5 + 1))
            
            # Boost for code-like lines
            if any(c in atom.text for c in ('{', '}', '(', ')', '=', ':', '`')):
                atom.score *= 1.2
                
        return atoms

class ContextOptimizer:
    """V2 Atomized Sequential RAG Engine."""
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.scorer = AtomizedScorer()
        self.max_context_tokens = 1800 # Aggressive token budget

    def ingest(self, session_id: str, messages: List[Dict[str, Any]]):
        """Store only original messages, avoiding inception."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        for msg in messages:
            content = msg.get("content", "")
            if not content or msg.get("role") == "system": continue
            
            # Anti-Inception: Skip if it looks like our own RAG context
            text = str(content)
            if "SESSION CONTEXT" in text[:100] or "# Session Summary" in text[:100]:
                continue
                
            # Deduplicate by hash
            m_hash = hashlib.md5(text.encode()).hexdigest()
            if any(m.get("hash") == m_hash for m in self.sessions[session_id]):
                continue
                
            self.sessions[session_id].append({
                "role": msg.get("role"),
                "content": text,
                "hash": m_hash,
                "timestamp": time.time()
            })

    def optimize(self, session_id: str, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        
        # 1. Update session with new OG messages
        self.ingest(session_id, messages)
        history = self.sessions.get(session_id, [])
        if not history: return messages
        
        # 2. Atomize: Split everything into lines
        atoms: List[Atom] = []
        line_counter = 0
        for i, msg in enumerate(history):
            lines = msg["content"].split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                atoms.append(Atom(
                    line_index=line_counter,
                    msg_index=i,
                    source=msg["role"],
                    text=line,
                    tokens=len(ENCODER.encode(line))
                ))
                line_counter += 1
        
        if not atoms: return messages
        
        # 3. Score: Vectorized lexical match
        query = messages[-1].get("content", "")
        scored_atoms = self.scorer.score_atoms(atoms, query)
        
        # 4. Cluster Expansion: Pick top atoms and pull in neighbors
        top_atoms = sorted([a for a in scored_atoms if a.score > 0], key=lambda x: x.score, reverse=True)[:50]
        if not top_atoms:
            # Fallback: Just take the most recent messages if no matches
            return messages[-5:] if len(messages) > 5 else messages

        selected_indices: Set[int] = set()
        for atom in top_atoms:
            # Expand: Grab 2 lines before and 2 lines after for context
            for offset in range(-2, 3):
                idx = atom.line_index + offset
                if 0 <= idx < len(atoms):
                    selected_indices.add(idx)
        
        # 5. Chronological Reassembly
        final_atoms = sorted([atoms[idx] for idx in selected_indices], key=lambda x: x.line_index)
        
        # 6. Telegraphic Packing (within token budget)
        packed_lines = []
        current_tokens = 0
        last_msg_idx = -1
        
        for atom in final_atoms:
            # Add source header if message changed
            header = ""
            if atom.msg_index != last_msg_idx:
                header = f"\n[{atom.source.upper()}]:\n"
                last_msg_idx = atom.msg_index
            
            line_text = f"{header}  {atom.text}"
            line_tokens = len(ENCODER.encode(line_text))
            
            if current_tokens + line_tokens > self.max_context_tokens:
                break
                
            packed_lines.append(line_text)
            current_tokens += line_tokens
            
        # 7. Rebuild Message List
        optimized = []
        # Keep original system prompt if exists
        sys_prompt = next((m for m in messages if m.get("role") == "system"), None)
        if sys_prompt: optimized.append(sys_prompt)
        
        # Add the "Swiss Cheese" context
        if packed_lines:
            context_text = "--- RELEVANT SESSION HISTORY (ATOMIZED) ---\n" + "\n".join(packed_lines)
            optimized.append({"role": "system", "content": context_text})
            
        # Always add the last user message
        optimized.append(messages[-1])
        
        overhead = (time.time() - start_time) * 1000
        log.info(f"⚡️ [V2 RAG] {session_id} | {len(packed_lines)} lines | {current_tokens} tokens | {overhead:.1f}ms")
        
        details = {
            "total_lines": len(atoms),
            "selected_lines": len(packed_lines),
            "overhead_ms": overhead,
            "sequence": [
                {
                    "source": a.source,
                    "text": a.text[:50] + "..." if len(a.text) > 50 else a.text,
                    "score": a.score,
                    "line_index": a.line_index
                }
                for a in final_atoms
            ]
        }
        
        return optimized, details
