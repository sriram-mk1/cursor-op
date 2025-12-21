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

        print("🚀 Initializing Remote Optimizer v6.1 (High Speed)...")
        self.model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "while", "in", "is", "it", "of", "to"}
        
        # In-Memory Cache for Embeddings (Container Life)
        self.embedding_cache = {} 
        self.history_cache = {}

        url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            self.supabase = create_client(url, key)
        else:
            self.supabase = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Cache-aware embedding."""
        txt_hash = hashlib.md5(text.encode()).hexdigest()
        if txt_hash in self.embedding_cache:
            return self.embedding_cache[txt_hash]
        
        emb = list(self.model.embed([text]))[0]
        self.embedding_cache[txt_hash] = emb
        return emb

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
                # Code-aware markers
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
            timestamp=ts
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
        
        # Optional: Skip Supabase if we have a very recent cache hit in the container
        if hist_cache_key in self.history_cache and (time.time() - self.history_cache[hist_cache_key]['ts'] < 30):
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
            except Exception as e:
                print(f"Fetch Error: {e}")
        
        if not history and len(current_messages) > 1:
            history = current_messages[:-1]
        
        if not history:
            return current_messages, {"mode": "no_history", "engine": "modal-remote-v6.1"}

        # 2. Chunking
        atoms = self._split_into_atoms(history)
        if not atoms:
            return current_messages, {"mode": "no_atoms", "engine": "modal-remote-v6.1"}

        # 3. Batch Embeddings (Optimized with Cache)
        query_emb = list(self.model.embed([query_text]))[0]
        
        # Optimized embedding logic to only embed what changed
        atom_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, a in enumerate(atoms):
            txt_hash = hashlib.md5(a.text.encode()).hexdigest()
            if txt_hash in self.embedding_cache:
                atom_embeddings.append((i, self.embedding_cache[txt_hash]))
            else:
                uncached_texts.append(a.text)
                uncached_indices.append(i)
                atom_embeddings.append((i, None)) # Placeholder
        
        if uncached_texts:
            new_embs = list(self.model.embed(uncached_texts))
            for idx, emb in zip(uncached_indices, new_embs):
                txt_hash = hashlib.md5(atoms[idx].text.encode()).hexdigest()
                self.embedding_cache[txt_hash] = emb
                atom_embeddings[idx] = (idx, emb)
        
        atom_embeddings = [e[1] for e in atom_embeddings]
        
        # 4. Hybrid Scoring v6.1
        query_terms = set(query_text.lower().split()) - self.stop_words
        
        # Path Boosting: Identify filenames or classes mentioned in the query
        path_matches = re.findall(r'(\w+\.(?:py|js|ts|tsx|css|html|json))', query_text)
        
        for i, atom in enumerate(atoms):
            emb = atom_embeddings[i]
            # Semantic (Cosine)
            semantic_score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9)
            
            # Keyword
            matches = sum(1 for term in query_terms if term in atom.text.lower())
            keyword_score = (matches / len(query_terms)) if query_terms else 0
            
            # Recency
            total_turns = max(1, len(set(a.msg_index for a in atoms)))
            temporal_boost = math.exp((atom.msg_index - total_turns) / 4.0)
            
            # Path Boosting (The "Powerful" part)
            path_boost = 1.0
            if any(p.lower() in atom.text.lower() for p in path_matches):
                path_boost = 1.5
            
            # Hybrid Formula
            atom.score = (0.7 * semantic_score) + (0.3 * keyword_score)
            atom.score *= (1.0 + 0.1 * temporal_boost)
            atom.score *= path_boost

        # 5. Selection
        max_score = float(max(a.score for a in atoms)) if atoms else 0
        top_atoms = sorted(atoms, key=lambda x: x.score, reverse=True)[:20]
        
        REL_THRESHOLD = 0.22 # Lowered slightly for better recall in complex code
        selected_indices = set()
        gate_triggered = False
        
        if max_score < REL_THRESHOLD:
            # Fallback to last 2 turns
            last_msg_idx = max(0, atoms[-1].msg_index - 1)
            selected_indices = {a.atom_index for a in atoms if a.msg_index >= last_msg_idx}
            gate_triggered = True
        else:
            for atom in top_atoms:
                if atom.score > REL_THRESHOLD * 0.7:
                    selected_indices.add(atom.atom_index)
                    # Include 2 siblings for better code flow
                    for offset in [-1, 1, -2, 2]:
                        n_idx = atom.atom_index + offset
                        if 0 <= n_idx < len(atoms) and atoms[n_idx].msg_index == atom.msg_index:
                            selected_indices.add(n_idx)

        reconstructed_atoms = sorted([atoms[i] for i in selected_indices], key=lambda x: x.atom_index)
        
        # 6. Build Context Block
        packed_content = []
        last_msg_idx = -1
        for a in reconstructed_atoms:
            if a.msg_index != last_msg_idx:
                packed_content.append(f"\n--- [{a.role.upper()} @ Turn {a.msg_index}] ---")
                last_msg_idx = a.msg_index
            packed_content.append(a.text)
            
        optimized = []
        sys_prompt = next((m for m in history if m.get("role") == "system"), None)
        if sys_prompt: optimized.append(sys_prompt)
        
        if packed_content:
            context_text = "POWERFUL CONTEXT RECONSTRUCTION v6.1 (Active Scan):\n" + "\n".join(packed_content) + "\n--- End Scan ---"
            optimized.append({"role": "system", "content": context_text})
            
        optimized.append(active_query_msg)
        
        # 7. Metadata (limit to 40 for UI clarity)
        sequence_data = []
        for a in sorted(atoms, key=lambda x: x.score, reverse=True)[:40]:
            d = a.to_dict()
            d["id"] = f"atom-{a.atom_index}"
            d["selected"] = a.atom_index in selected_indices
            sequence_data.append(d)

        return optimized, {
            "total_blocks": len(atoms),
            "selected_blocks": len(reconstructed_atoms),
            "max_relevance": round(float(max_score), 3),
            "gate_triggered": gate_triggered,
            "overhead_ms": (time.time() - start_time) * 1000,
            "sequence": sequence_data,
            "engine": "modal-remote-v6.1"
        }

    @modal.web_endpoint(method="POST")
    def optimize_web(self, payload: Dict[str, Any]):
        """Speed-optimized endpoint."""
        return self._optimize_internal(
            payload["user_key"], 
            payload["session_id"], 
            payload["current_messages"]
        )
