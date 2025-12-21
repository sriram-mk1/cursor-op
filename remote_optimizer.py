import modal
import os
import time
import json
import math
import hashlib
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
    container_idle_timeout=300, # Keep alive for 5 mins after last call
    keep_warm=1,                # Keep 1 instance ready at all times
    secrets=[modal.Secret.from_name("supabase-keys")] 
)
class RemoteOptimizer:
    @modal.enter()
    def setup(self):
        """Load expensive models once at startup."""
        from fastembed import TextEmbedding
        import tiktoken
        import numpy as np
        from supabase import create_client

        print("🚀 Initializing Remote Optimizer...")
        self.model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.np = np
        self.stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "while", "in", "is", "it", "of", "to"}
        
        # Initialize Supabase inside Modal for direct history fetching
        url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            self.supabase = create_client(url, key)
            print("✅ Supabase Connected (Remote)")
        else:
            self.supabase = None
            print("⚠️ Supabase Credentials Missing in Modal Secrets")

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
                is_block_start = any(stripped.startswith(k) for k in [
                    "def ", "class ", "async def ", "interface ", "function ", "export ", "module.exports",
                    "import ", "from ", "const ", "let ", "var "
                ])
                if is_block_start and current_block and len("\n".join(current_block)) > 100:
                    atoms.append(self._create_atom(current_block, m_idx, len(atoms), role, ts))
                    current_block = []
                current_block.append(line)
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
        
        # 1. Fetch History directly from Supabase (Remote)
        history = []
        if self.supabase:
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
            except Exception as e:
                print(f"Fetch Error: {e}")
        
        if not history and len(current_messages) > 1:
            history = current_messages[:-1]
        
        if not history:
            return current_messages, {"mode": "no_history"}

        # 2. Chunking
        atoms = self._split_into_atoms(history)
        if not atoms:
            return current_messages, {"mode": "no_atoms"}

        # 3. Batch Embeddings (Fast Path)
        query_emb = list(self.model.embed([query_text]))[0]
        atom_texts = [a.text for a in atoms]
        atom_embeddings = list(self.model.embed(atom_texts))
        
        query_terms = set(query_text.lower().split()) - self.stop_words
        
        # 4. Scoring
        for i, atom in enumerate(atoms):
            emb = atom_embeddings[i]
            # Semantic
            semantic_score = self.np.dot(query_emb, emb) / (self.np.linalg.norm(query_emb) * self.np.linalg.norm(emb) + 1e-9)
            # Keyword
            matches = sum(1 for term in query_terms if term in atom.text.lower())
            keyword_score = (matches / len(query_terms)) if query_terms else 0
            # Temporal
            total_turns = max(1, len(set(a.msg_index for a in atoms)))
            temporal_boost = math.exp((atom.msg_index - total_turns) / 2.0)
            
            atom.score = (0.6 * semantic_score) + (0.3 * keyword_score)
            atom.score *= (1.0 + 0.1 * temporal_boost)

        # 5. Selection
        max_score = max(a.score for a in atoms)
        top_atoms = sorted(atoms, key=lambda x: x.score, reverse=True)[:15]
        
        REL_THRESHOLD = 0.25
        selected_indices = set()
        gate_triggered = False
        
        if max_score < REL_THRESHOLD:
            # Fallback to last 2 turns
            selected_indices = {a.atom_index for a in atoms if a.msg_index >= max(0, atoms[-1].msg_index - 1)}
            gate_triggered = True
        else:
            for atom in top_atoms:
                if atom.score > REL_THRESHOLD * 0.8:
                    selected_indices.add(atom.atom_index)
                    # Include siblings in same message
                    for n_idx in [atom.atom_index - 1, atom.atom_index + 1]:
                        if 0 <= n_idx < len(atoms) and atoms[n_idx].msg_index == atom.msg_index:
                            selected_indices.add(n_idx)

        reconstructed_atoms = sorted([atoms[i] for i in selected_indices], key=lambda x: x.atom_index)
        
        # 6. Rebuild
        packed_content = []
        last_msg_idx = -1
        for a in reconstructed_atoms:
            if a.msg_index != last_msg_idx:
                packed_content.append(f"\n--- [Historical {a.role.upper()} @ turn {a.msg_index}] ---")
                last_msg_idx = a.msg_index
            packed_content.append(a.text)
            
        optimized = []
        sys_prompt = next((m for m in history if m.get("role") == "system"), None)
        if sys_prompt: optimized.append(sys_prompt)
        
        if packed_content:
            context_text = "Relevant Historical Context (Reconstructed):\n" + "\n".join(packed_content) + "\n--- End Context ---"
            optimized.append({"role": "system", "content": context_text})
            
        optimized.append(active_query_msg)
        
        # 7. Metadata
        sequence_data = []
        for a in sorted(atoms, key=lambda x: x.score, reverse=True)[:50]:
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
            "engine": "modal-remote-v6"
        }

    @modal.web_endpoint(method="POST")
    def optimize_web(self, payload: Dict[str, Any]):
        """Simple HTTP endpoint for Railway to call."""
        return self._optimize_internal(
            payload["user_key"], 
            payload["session_id"], 
            payload["current_messages"]
        )
