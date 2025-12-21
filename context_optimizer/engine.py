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
from fastembed import TextEmbedding

log = logging.getLogger("gateway")
ENCODER = tiktoken.get_encoding("cl100k_base")

# Singleton for Embedding Model to save RAM
class EmbeddingManager:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            # Using nomic-embed-text-v1.5 (Small, MRL support, high performance)
            # Fits in ~150MB RAM
            log.info("Loading Embedding Model...")
            self._model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        return self._model

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return np.array(list(self.model.embed(queries)))

    def embed_chunks(self, texts: List[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts)))

@dataclass
class Atom:
    msg_index: int
    text: str
    tokens: int
    timestamp: float
    score: float = 0.0
    embedding: Optional[np.ndarray] = None
    atom_index: int = 0
    role: str = "user"

    def to_dict(self):
        d = asdict(self)
        if d["embedding"] is not None:
            d.pop("embedding") # Don't send embeddings to frontend
        return d

class StructuralReconstructor:
    """V6: Structural Code-Block Context Reconstruction Engine."""
    def __init__(self):
        self.stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "while", "in", "is", "it", "of", "to"}

    def split_into_atoms(self, messages: List[Dict[str, Any]]) -> List[Atom]:
        atoms = []
        for m_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content: continue
            
            role = msg.get("role", "user")
            text = content if isinstance(content, str) else json.dumps(content)
            lines = text.split('\n')
            ts = msg.get("timestamp", time.time())
            
            current_block = []
            
            for line in lines:
                stripped = line.strip()
                # Detection for block starters
                is_block_start = any(stripped.startswith(k) for k in [
                    "def ", "class ", "async def ", "interface ", "function ", "export ", "module.exports",
                    "import ", "from ", "const ", "let ", "var "
                ])
                
                # If we hit a block start and already have content, flush the previous block
                if is_block_start and current_block and len("\n".join(current_block)) > 100:
                    atoms.append(self._create_atom(current_block, m_idx, len(atoms), role, ts))
                    current_block = []
                
                current_block.append(line)
                
                # Hard limit on block size to keep granularity
                if len(current_block) > 30:
                    atoms.append(self._create_atom(current_block, m_idx, len(atoms), role, ts))
                    current_block = []
            
            if current_block:
                atoms.append(self._create_atom(current_block, m_idx, len(atoms), role, ts))
                
        return atoms

    def _create_atom(self, lines: List[str], msg_idx: int, atom_idx: int, role: str, ts: float) -> Atom:
        text = "\n".join(lines)
        return Atom(
            msg_index=msg_idx,
            atom_index=atom_idx,
            role=role,
            text=text,
            tokens=len(ENCODER.encode(text)),
            timestamp=ts
        )

    def score_atoms(self, atoms: List[Atom], query_text: str, query_emb: np.ndarray) -> List[Atom]:
        if not atoms: return atoms
        
        # 1. Keyword scoring (BM25-lite)
        query_terms = set(query_text.lower().split()) - self.stop_words
        
        # 2. Semantic scoring (Vector)
        atom_texts = [a.text for a in atoms]
        embeddings = EmbeddingManager().embed_chunks(atom_texts)
        
        # We only use first 128 dims for MRL efficiency if we want, 
        # but let's just do standard cosine for now.
        for i, atom in enumerate(atoms):
            atom_emb = embeddings[i]
            
            # Cosine Similarity
            semantic_score = np.dot(query_emb, atom_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(atom_emb) + 1e-9)
            
            # Keyword score
            matches = sum(1 for term in query_terms if term in atom.text.lower())
            keyword_score = (matches / len(query_terms)) if query_terms else 0
            
            # Recency Boost (Temporal)
            # More recent messages get a boost
            total_turns = max(1, len(set(a.msg_index for a in atoms)))
            temporal_boost = math.exp((atom.msg_index - total_turns) / 2.0)
            
            # Hybrid Formula
            # 0.6 Semantic + 0.3 Keyword + 0.1 Temporal adjustment
            atom.score = (0.6 * semantic_score) + (0.3 * keyword_score)
            atom.score *= (1.0 + 0.1 * temporal_boost)
            
        return atoms

class ContextOptimizer:
    """V6: Structural & Semantic Context Reconstruction Engine."""
    def __init__(self):
        self.reconstructor = StructuralReconstructor()
        from database import Database
        self.db = Database()
        self.embedder = EmbeddingManager()

    def optimize(self, user_key: str, session_id: str, current_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        
        if not current_messages:
            return current_messages, {}

        active_query_msg = current_messages[-1]
        query_text = active_query_msg.get("content", "")
        if not isinstance(query_text, str): query_text = json.dumps(query_text)
        
        # 1. Retrieve Historical Context from DB (Original/Unoptimized)
        # This prevents context loss over multi-turn conversations
        hashed_key = hashlib.sha256(user_key.encode()).hexdigest() if user_key != "anon" else "anon"
        history = self.db.get_session_history(session_id, hashed_key)
        
        # If no DB history, fallback to current payload history (minus active query)
        if not history and len(current_messages) > 1:
            history = current_messages[:-1]
        
        if not history:
            return current_messages, {"mode": "no_history"}

        # 2. Structural Chunking
        atoms = self.reconstructor.split_into_atoms(history)
        if not atoms:
            return current_messages, {"mode": "no_atoms"}

        # 3. Hybrid Scoring
        query_emb = self.embedder.embed_queries([query_text])[0]
        scored_atoms = self.reconstructor.score_atoms(atoms, query_text, query_emb)
        
        # 4. Filter by Semantic Gatekeeper
        # If no atom is relevant enough (> 0.3 similarity), we pass very little context
        max_score = max(a.score for a in scored_atoms) if scored_atoms else 0
        
        # 5. Selection with Coherence Preservation
        # Choose top blocks
        top_atoms = sorted(scored_atoms, key=lambda x: x.score, reverse=True)[:15]
        
        # If similarity is too low, strictly limit context
        REL_THRESHOLD = 0.25 
        if max_score < REL_THRESHOLD:
            # Maybe just keep the last 2 messages for basic flow
            selected_indices = {a.atom_index for a in atoms if a.msg_index >= len(history) - 2}
            gate_triggered = True
        else:
            selected_indices = set()
            for atom in top_atoms:
                if atom.score > REL_THRESHOLD * 0.8:
                    selected_indices.add(atom.atom_index)
                    # Include one neighboring block for flow if it's from the same message
                    for neighbor_idx in [atom.atom_index - 1, atom.atom_index + 1]:
                        if 0 <= neighbor_idx < len(atoms):
                            if atoms[neighbor_idx].msg_index == atom.msg_index:
                                selected_indices.add(neighbor_idx)
            gate_triggered = False

        reconstructed_atoms = sorted([atoms[i] for i in selected_indices], key=lambda x: x.atom_index)
        
        # 6. Rebuild context block
        packed_content = []
        last_msg_idx = -1
        
        for a in reconstructed_atoms:
            if a.msg_index != last_msg_idx:
                # Add a marker between different historical turns
                packed_content.append(f"\n--- [Historical {a.role.upper()} @ turn {a.msg_index}] ---")
                last_msg_idx = a.msg_index
            packed_content.append(a.text)
        
        optimized = []
        # Keep original system prompt if it exists in history
        sys_prompt = next((m for m in history if m.get("role") == "system"), None)
        if sys_prompt: optimized.append(sys_prompt)
        
        if packed_content:
            context_text = (
                "Relevant Historical Context (Reconstructed):\n"
                + "\n".join(packed_content)
                + "\n--- End Context ---"
            )
            optimized.append({"role": "system", "content": context_text})
            
        optimized.append(active_query_msg)
        
        # 7. Prepare Sequence Metadata for Frontend
        sequence_data = []
        for a in sorted(scored_atoms, key=lambda x: x.score, reverse=True)[:50]:
            d = a.to_dict()
            d["id"] = f"atom-{a.atom_index}"
            d["selected"] = a.atom_index in selected_indices
            sequence_data.append(d)

        return optimized, {
            "total_blocks": len(atoms),
            "selected_blocks": len(reconstructed_atoms),
            "max_relevance": round(max_score, 3),
            "gate_triggered": gate_triggered,
            "overhead_ms": (time.time() - start_time) * 1000,
            "sequence": sequence_data
        }
