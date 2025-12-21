import modal
import os
import time
import json
import math
import hashlib
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

load_dotenv()

# 1. Define the compute environment
image = modal.Image.debian_slim().pip_install(
    "fastembed", 
    "numpy", 
    "tiktoken", 
    "supabase",
    "python-dotenv",
    "fastapi"
)

app = modal.App("v1-context-optimizer")

# Reuse the Atom definition
@dataclass
class Atom:
    msg_index: int
    text: str
    tokens: int
    timestamp: float
    score: float = 0.0
    atom_index: int = 0
    role: str = "user"
    symbols: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

# 2. Structural & Semantic Logic (Moved to Modal)
@app.cls(
    image=image,
    container_idle_timeout=300, # Keep alive for 5 mins
    keep_warm=1,                
    secrets=[modal.Secret.from_name("supabase-keys")] 
)
class RemoteOptimizer:
    @modal.enter()
    def setup(self):
        """Load expensive models and shared state."""
        from fastembed import TextEmbedding
        import tiktoken
        from supabase import create_client

        print("🚀 Initializing Ultra-Fast Context Engine v6.2...")
        self.model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "while", "in", "is", "it", "of", "to"}
        
        # High-Speed In-Memory Cache
        self.embedding_cache = {} 
        self.history_cache = {}

        url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            self.supabase = create_client(url, key)
        else:
            self.supabase = None

    def _extract_symbols(self, text: str) -> List[str]:
        """Fast regex extraction of code symbols (classes, functions)."""
        # Look for 'class Name', 'def name(', 'async def name('
        found = re.findall(r'(?:class|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)', text)
        return list(set(found))

    def _split_into_atoms(self, messages: List[Dict[str, Any]]) -> List[Atom]:
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
                    "import ", "from ", "const ", "let ", "var ", "@"
                ])
                if is_block_start and current_block and len("\n".join(current_block)) > 120:
                    atoms.append(self._create_atom(current_block, m_idx, len(atoms), role, ts))
                    current_block = []
                current_block.append(line)
                if len(current_block) > 40:
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
            tokens=len(self.encoder.encode(text)),
            timestamp=ts,
            symbols=self._extract_symbols(text)
        )

    @modal.method()
    def optimize(self, user_key: str, session_id: str, current_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        return self._optimize_internal(user_key, session_id, current_messages)

    def _optimize_internal(self, user_key: str, session_id: str, current_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        
        if not current_messages:
            return current_messages, {}

        active_query_msg = current_messages[-1]
        query_text = active_query_msg.get("content", "")
        if not isinstance(query_text, str): query_text = json.dumps(query_text)
        
        # 1. Fetch History with local caching
        history = []
        hist_cache_key = f"{session_id}_{user_key}"
        
        if hist_cache_key in self.history_cache and (time.time() - self.history_cache[hist_cache_key]['ts'] < 20):
            history = self.history_cache[hist_cache_key]['data']
        elif self.supabase:
            try:
                hashed_key = hashlib.sha256(user_key.encode()).hexdigest() if user_key != "anon" else "anon"
                response = self.supabase.table("analytics")\
                    .select("raw_messages")\
                    .eq("session_id", session_id)\
                    .eq("hashed_key", hashed_key)\
                    .order("timestamp", desc=True)\
                    .limit(1)\
                    .execute()
                if response.data:
                    history = response.data[0].get("raw_messages", [])
                    self.history_cache[hist_cache_key] = {'data': history, 'ts': time.time()}
            except: pass
        
        if not history and len(current_messages) > 1:
            history = current_messages[:-1]
        
        if not history:
            return current_messages, {"mode": "no_history", "engine": "modal-v6.2-anchor"}

        # 2. Chunking & Symbol Pre-processing
        atoms = self._split_into_atoms(history)
        if not atoms:
            return current_messages, {"mode": "no_atoms", "engine": "modal-v6.2-anchor"}

        # 3. Path & Symbol Anchoring (Fast Path)
        # Identify what the user is actually talking about
        query_symbols = set(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)', query_text))
        
        # 4. Shadow Contextualized Batch Embedding (Sub-50ms)
        # We embed the "Shadow" prompt (Context + Atom) for scoring accuracy
        # but the shadow is NEVER used in the final output.
        query_emb = list(self.model.embed([f"Represent this query for retrieving relevant code: {query_text}"]))[0]
        
        atom_embeddings = []
        uncached_indices = []
        uncached_shadows = []
        
        for i, a in enumerate(atoms):
            # The Shadow Prompt: Purely for vector positioning
            shadow_text = f"Context: Historical {a.role} message at turn {a.msg_index}. Content: {a.text}"
            txt_hash = hashlib.md5(shadow_text.encode()).hexdigest()
            
            if txt_hash in self.embedding_cache:
                atom_embeddings.append((i, self.embedding_cache[txt_hash]))
            else:
                uncached_shadows.append(shadow_text)
                uncached_indices.append(i)
                atom_embeddings.append((i, None))
        
        if uncached_shadows:
            new_embs = list(self.model.embed(uncached_shadows))
            for idx, emb in zip(uncached_indices, new_embs):
                shadow_text = f"Context: Historical {atoms[idx].role} message at turn {atoms[idx].msg_index}. Content: {atoms[idx].text}"
                txt_hash = hashlib.md5(shadow_text.encode()).hexdigest()
                self.embedding_cache[txt_hash] = emb
                atom_embeddings[idx] = (idx, emb)
        
        # 5. Hybrid Scoring v6.2 (Symbol Anchoring)
        anchor_boost_count = 0
        for i, atom in enumerate(atoms):
            emb = atom_embeddings[i][1]
            
            # Semantic Alignment
            semantic_score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9)
            
            # Symbol Anchoring: Massive boost if query mentions a symbol defined in this atom
            matching_symbols = query_symbols.intersection(set(atom.symbols))
            anchor_boost = 1.3 if matching_symbols else 1.0
            if matching_symbols: anchor_boost_count += 1
            
            # Recency bias (decay over turns)
            total_turns = max(1, atoms[-1].msg_index + 1)
            turn_bias = 1.0 + (atom.msg_index / total_turns) * 0.1
            
            atom.score = semantic_score * anchor_boost * turn_bias

        # 6. Optimized Selection (Zero Noise)
        max_score = float(max(a.score for a in atoms))
        REL_THRESHOLD = 0.22
        
        selected_indices = set()
        gate_triggered = False
        
        if max_score < REL_THRESHOLD:
            # High-fidelity fallback (Last 3 messages to maintain thread flow)
            selected_indices = {a.atom_index for a in atoms if a.msg_index >= max(0, atoms[-1].msg_index - 2)}
            gate_triggered = True
        else:
            # Powerful selection + strict sibling inclusion
            top_atoms = sorted(atoms, key=lambda x: x.score, reverse=True)[:15]
            for atom in top_atoms:
                if atom.score > REL_THRESHOLD * 0.6:
                    selected_indices.add(atom.atom_index)
                    # Sibling flow (maintain block continuity)
                    for side in [-1, 1]:
                        neighbor = atom.atom_index + side
                        if 0 <= neighbor < len(atoms) and atoms[neighbor].msg_index == atom.msg_index:
                            selected_indices.add(neighbor)

        # 7. Pure Reconstruction (STRICT: NO ADDED FIELDS)
        reconstructed_atoms = sorted([atoms[i] for i in selected_indices], key=lambda x: x.atom_index)
        
        packed_content = []
        last_msg_idx = -1
        # Markers are only added between historical turns for clarity
        for a in reconstructed_atoms:
            if a.msg_index != last_msg_idx:
                packed_content.append(f"\n--- [TURN {a.msg_index}] ---")
                last_msg_idx = a.msg_index
            packed_content.append(a.text)
            
        optimized = []
        sys_prompt = next((m for m in history if m.get("role") == "system"), None)
        if sys_prompt: optimized.append(sys_prompt)
        
        if packed_content:
            context_text = "ORIGINAL_CONTEXT_SNAPSHOT:\n" + "\n".join(packed_content)
            optimized.append({"role": "system", "content": context_text})
            
        optimized.append(active_query_msg)
        
        return optimized, {
            "total_blocks": len(atoms),
            "selected_blocks": len(reconstructed_atoms),
            "max_relevance": round(max_score, 3),
            "anchor_hits": anchor_boost_count,
            "gate_triggered": gate_triggered,
            "overhead_ms": (time.time() - start_time) * 1000,
            "sequence": [a.to_dict() for a in sorted(atoms, key=lambda x: x.score, reverse=True)[:30]],
            "engine": "modal-v6.2-anchor"
        }

    @modal.web_endpoint(method="POST")
    def optimize_web(self, payload: Dict[str, Any]):
        """Speed-optimized endpoint."""
        return self._optimize_internal(
            payload["user_key"], 
            payload["session_id"], 
            payload["current_messages"]
        )
