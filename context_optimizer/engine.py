import os
import json
import logging
import hashlib
import time
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import tiktoken
import numpy as np

log = logging.getLogger("gateway")
ENCODER = tiktoken.get_encoding("cl100k_base")

@dataclass
class Atom:
    line_index: int
    msg_index: int
    source: str
    text: str
    tokens: int
    timestamp: float
    score: float = 0.0

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class SimpleReconstructor:
    """V5: Simple, high-fidelity context reconstruction engine."""
    def __init__(self):
        self.stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "while", "in", "is", "it", "of", "to"}

    def split_into_atoms(self, messages: List[Dict[str, Any]]) -> List[Atom]:
        atoms = []
        for m_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content: continue
            
            # Handle multi-modal or structured content
            text = content if isinstance(content, str) else json.dumps(content)
            lines = text.split('\n')
            ts = msg.get("timestamp", time.time())
            
            for line in lines:
                if not line.strip(): continue
                atoms.append(Atom(
                    line_index=len(atoms),
                    msg_index=m_idx,
                    source=msg.get("role", "user"),
                    text=line,
                    tokens=len(ENCODER.encode(line)),
                    timestamp=ts
                ))
        return atoms

    def score_atoms(self, atoms: List[Atom], query: str) -> List[Atom]:
        if not atoms or not query: return atoms
        
        query_terms = set(query.lower().split()) - self.stop_words
        if not query_terms: return atoms
        
        for atom in atoms:
            atom_text = atom.text.lower()
            matches = sum(1 for term in query_terms if term in atom_text)
            # Simple scoring: matches + temporal boost
            atom.score = (matches / len(query_terms)) if query_terms else 0
            # Boost recent signals
            temporal_boost = math.exp(atom.msg_index - len(set(a.msg_index for a in atoms)))
            atom.score *= (1.0 + temporal_boost)
            
        return atoms

class ContextOptimizer:
    """V5: Simplified Context Reconstruction Engine."""
    def __init__(self):
        self.reconstructor = SimpleReconstructor()
        from database import Database
        self.db = Database()

    def optimize(self, user_key: str, session_id: str, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        
        if not messages or len(messages) <= 1:
            return messages, {}

        # 1. Isolate the current query and history
        # We only do RAG over the history (exclude the very last message)
        history = messages[:-1]
        active_query_msg = messages[-1]
        
        # 2. Split history into atoms
        atoms = self.reconstructor.split_into_atoms(history)
        if not atoms: 
            return messages, {}
        
        # 3. Score based on active query
        query = active_query_msg.get("content", "")
        if not isinstance(query, str): query = json.dumps(query)
        
        scored_atoms = self.reconstructor.score_atoms(atoms, query)
        
        # 4. Select top atoms from history
        top_atoms = sorted(scored_atoms, key=lambda x: x.score, reverse=True)[:20]
        selected_indices = set()
        for atom in top_atoms:
            # Keep neighbors for flow
            for i in range(atom.line_index - 1, atom.line_index + 2):
                if 0 <= i < len(atoms):
                    selected_indices.add(i)
        
        reconstructed_atoms = sorted([atoms[i] for i in selected_indices], key=lambda x: x.line_index)
        
        # 5. Rebuild the context block
        packed_lines = [a.text for a in reconstructed_atoms]
        
        optimized = []
        # Preserve original system prompt if it exists in history
        sys_prompt = next((m for m in history if m.get("role") == "system"), None)
        if sys_prompt: optimized.append(sys_prompt)
        
        if packed_lines:
            context_text = (
                "--- HISTORICAL CONTEXT RECONSTRUCTION ---\n"
                + "\n".join(packed_lines)
                + "\n--- END RECONSTRUCTION ---"
            )
            optimized.append({"role": "system", "content": context_text})
            
        # Always append the active query at the end
        optimized.append(active_query_msg)
        
        return optimized, {
            "total_lines": len(atoms),
            "selected_lines": len(packed_lines),
            "total_history_tokens": sum(a.tokens for a in atoms),
            "overhead_ms": (time.time() - start_time) * 1000,
            "sequence": [
                {
                    "id": f"atom-{a.line_index}",
                    "source": a.source,
                    "text": a.text,
                    "score": round(a.score, 3),
                    "line_index": a.line_index,
                    "selected": a.line_index in selected_indices
                }
                for a in scored_atoms[:100]  # Return top 100 for visualization
            ]
        }
