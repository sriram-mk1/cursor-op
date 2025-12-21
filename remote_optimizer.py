import modal
import os
import time
import json
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
    "python-dotenv",
    "fastapi"
)

app = modal.App("v1-context-optimizer")

@dataclass
class Chunk:
    msg_index: int
    chunk_index: int
    text: str
    tokens: int
    role: str
    symbols: List[str] = field(default_factory=list)
    score: float = 0.0
    selected: bool = False

# 2. Ultra-Simplified & Isolated Optimizer
@app.cls(
    image=image,
    container_idle_timeout=300,
    keep_warm=1
)
class RemoteOptimizer:
    @modal.enter()
    def setup(self):
        from fastembed import TextEmbedding
        import tiktoken
        print("🚀 v10: Initializing Stateless Context Engine...")
        self.model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _extract_symbols(self, text: str) -> List[str]:
        return list(set(re.findall(r'(?:class|def|function|const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)', text)))

    def _chunk_messages(self, messages: List[Dict[str, Any]]) -> List[Chunk]:
        chunks = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not isinstance(content, str): content = json.dumps(content)
            
            parts = re.split(r'(\n\n|```)', content)
            current_text = ""
            for p in parts:
                if not p: continue
                if len(current_text) + len(p) > 500:
                    if current_text:
                        chunks.append(self._create_chunk(current_text, i, len(chunks), role))
                        current_text = ""
                current_text += p
            if current_text:
                chunks.append(self._create_chunk(current_text, i, len(chunks), role))
        return chunks

    def _create_chunk(self, text: str, msg_idx: int, chunk_idx: int, role: str) -> Chunk:
        return Chunk(
            msg_index=msg_idx,
            chunk_index=chunk_idx,
            role=role,
            text=text,
            tokens=len(self.encoder.encode(text)),
            symbols=self._extract_symbols(text)
        )

    def _optimize_internal(self, user_key: str, session_id: str, current_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        if len(current_messages) <= 1:
            return current_messages, {"mode": "passthrough", "overhead_ms": 0}

        system_msgs = [m for m in current_messages[:-1] if m.get("role") == "system"]
        # History is everything between system prompts and last message
        history_msgs = [m for m in current_messages[:-1] if m.get("role") != "system"]
        query_msg = current_messages[-1]
        query_content = query_msg.get("content", "")
        query_text = query_content if isinstance(query_content, str) else json.dumps(query_content)
        
        if not history_msgs:
            return current_messages, {"mode": "no_history", "overhead_ms": 0}

        # 2. Chunking (Only history)
        chunks = self._chunk_messages(history_msgs)
        if not chunks:
            return current_messages, {"mode": "no_chunks", "overhead_ms": 0}

        # 3. Embedding (Stateless)
        query_emb = list(self.model.embed([f"search_query: {query_text}"]))[0]
        chunk_texts = [f"search_document: {c.text}" for c in chunks]
        chunk_embs = list(self.model.embed(chunk_texts))

        # 4. Hybrid Scoring
        query_symbols = set(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)', query_text))
        for i, (chunk, emb) in enumerate(zip(chunks, chunk_embs)):
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9)
            matching = query_symbols.intersection(set(chunk.symbols))
            boost = 1.3 if matching else 1.0
            recency = 1.0 + (chunk.msg_index / len(history_msgs)) * 0.1
            chunk.score = float(sim * boost * recency)

        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        selected_indices = set()
        total_tokens = 0
        for i, c in enumerate(sorted_chunks):
            if c.score < 0.25 and total_tokens > 1000: break 
            if total_tokens + c.tokens > 6000: break
            selected_indices.add((c.msg_index, c.chunk_index))
            c.selected = True
            total_tokens += c.tokens

        optimized_history = []
        for msg_idx, msg in enumerate(history_msgs):
            msg_chunks = [c for c in chunks if c.msg_index == msg_idx and c.selected]
            if msg_chunks:
                msg_chunks.sort(key=lambda x: x.chunk_index)
                reconstructed_content = "\n".join([c.text for c in msg_chunks])
                new_msg = msg.copy()
                new_msg["content"] = reconstructed_content
                optimized_history.append(new_msg)

        optimized = system_msgs + optimized_history + [query_msg]
        snapshot_text = "\n\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in optimized_history])

        return optimized, {
            "snapshot": snapshot_text,
            "log": {
                "sequence": [asdict(c) for c in chunks],
                "total_chunks": len(chunks),
                "selected_chunks": len(selected_indices),
                "top_score": round(max(c.score for c in chunks), 3) if chunks else 0,
                "overhead_ms": (time.time() - start_time) * 1000
            },
            "engine": "v10-stateless-fixed"
        }

    @modal.method()
    def optimize(self, user_key: str, session_id: str, current_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        return self._optimize_internal(user_key, session_id, current_messages)

    @modal.web_endpoint(method="POST")
    def optimize_web(self, payload: Dict[str, Any]):
        return self._optimize_internal(
            payload["user_key"], 
            payload["session_id"], 
            payload.get("current_messages", [])
        )
