import os
import json

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
        
        # Persistence: Use Database class directly
        from database import Database
        self.db = Database()
        self.local_sessions: Dict[str, Dict[str, Any]] = {}

    def _get_session(self, user_key: str, session_id: str) -> Dict[str, Any]:
        """Strictly isolated session retrieval from Supabase."""
        # storage_key no longer needed for internal dict lookup if we use DB, 
        # but useful for local fallback
        storage_key = f"v3:{hashlib.md5(user_key.encode()).hexdigest()[:10]}:{session_id}"
        
        # Try DB first
        if self.db and self.db.supabase:
            data = self.db.get_session_state(session_id)
            if data:
                return {
                    "history": data.get("history", []),
                    "atoms": [Atom.from_dict(a) for a in data.get("atoms", [])]
                }
        
        return self.local_sessions.get(storage_key, {"history": [], "atoms": []})

    def _save_session(self, user_key: str, session_id: str, history: List, atoms: List):
        """Strictly isolated session persistence to Supabase."""
        storage_key = f"v3:{hashlib.md5(user_key.encode()).hexdigest()[:10]}:{session_id}"
        data = {
            "history": history,
            "atoms": [a.to_dict() for a in atoms]
        }
        
        # Save to DB
        if self.db and self.db.supabase:
            self.db.save_session_state(session_id, data)
        else:
            self.local_sessions[storage_key] = {"history": history, "atoms": atoms}

    def ingest(self, user_key: str, session_id: str, messages: List[Dict[str, Any]]):
        session = self._get_session(user_key, session_id)
        history = session["history"]
        atoms = session["atoms"]
        
        # V3.7 Ultra-Clean "Purge" Filter 
        NOISE_PATTERNS = [
            r"<(task|environment_details|slug|name|model|tool_format|todos|update_todo_list|ask_followup_question|attempt_completion|result|feedback)>",
            r"</(task|environment_details|slug|name|model|tool_format|todos|update_todo_list|ask_followup_question|attempt_completion|result|feedback)>",
            r"# (VSCode Visible Files|VSCode Open Tabs|Current Time|Current Cost|Current Mode|Current Workspace Directory-|Reminder: Instructions for Tool Use|Next Steps)",
            r"Current time in ISO 8601",
            r"User time zone:",
            r"No files found\.",
            r"^\$?\d+\.\d{2}$",
            r"You have not created a todo list yet",
            r"REMINDERS",
            r"\| # \| Content \| Status \|",
            r"\[(ask_followup_question|update_todo_list|attempt_completion)\] Result:",
            r"\[ERROR\] You did not use a tool",
            r"Tool uses are formatted using XML-style tags",
            r"(Always use the actual tool name|If you have completed|If you require additional)",
            r"\(This is an automated message, so do not respond to it conversationally\.\)"
        ]
        noise_re = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]
        xml_strip_re = re.compile(r'<[^>]+>') # Universal XML strip

        changed = False
        new_atoms = []
        for msg in messages:
            content = msg.get("content", "")
            if not content or msg.get("role") == "system": continue
            
            raw_text = self.scorer._to_string(content)
            
            # V3.7 surgical history cleaning: Strip XML from history too
            clean_history_text = xml_strip_re.sub('', raw_text).strip()
            # Also strip the noise patterns from the history strings
            for r in noise_re:
                clean_history_text = r.sub('', clean_history_text).strip()
            
            if not clean_history_text: continue

            m_hash = hashlib.md5(clean_history_text.encode()).hexdigest()
            if any(m.get("hash") == m_hash for m in history):
                continue
            
            msg_idx = len(history)
            msg_ts = time.time()
            history.append({
                "role": msg.get("role"),
                "content": clean_history_text, # Hyper-clean history
                "hash": m_hash,
                "timestamp": msg_ts
            })
            
            lines = raw_text.split('\n') # Still use raw lines for atom splitting to find defs
            for line in lines:
                clean_line = line.strip()
                if not clean_line or len(clean_line) < 2: continue
                
                # 1. Skip if it's purely a noise pattern
                if any(r.search(clean_line) for r in noise_re): 
                    continue
                
                # 2. Universal XML Stripping for stored atoms
                # This kills all <tags> while keeping the content
                stripped_content = xml_strip_re.sub('', clean_line).strip()
                if not stripped_content: continue

                new_atoms.append(Atom(
                    line_index=len(atoms) + len(new_atoms),
                    msg_index=msg_idx,
                    source=msg.get("role"),
                    text=stripped_content, # <--- ULTRA CLEAN PRESERVATION
                    tokens=len(ENCODER.encode(stripped_content)),
                    timestamp=msg_ts,
                    terms=self.scorer.get_terms(stripped_content),
                    provides=self.scorer.extract_definitions(clean_line) # Keep def detection on raw line
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
        
        # Phase 5: High-Fidelity Ceiling (Remove strict budgeting)
        # We allow up to 30k tokens now for massive context awareness
        max_tokens = 30000 
        specificity = len(query_terms)

        scored_atoms = self.scorer.score_atoms(atoms, query)
        
        # Density selection: Be more generous with what we include
        top_atoms = sorted([a for a in scored_atoms if a.score > 0.05], key=lambda x: x.score, reverse=True)[:100]
        if not top_atoms:
            return messages[-10:] if len(messages) > 10 else messages, {"status": "no_matches"}

        selected_indices: Set[int] = set()
        def_map = {} 
        for a in atoms:
            for d in a.provides: def_map[d] = a.line_index

        for atom in top_atoms:
            # Expand neighborhood for "Deep Flow" context
            # Grab 5 lines before and after for every hit to prevent fragmentation
            for offset in range(-5, 6):
                idx = atom.line_index + offset
                if 0 <= idx < len(atoms): selected_indices.add(idx)
                    
            # Enhanced Structural Linking
            for d_name, d_idx in def_map.items():
                if d_name in atom.text:
                    # If we find a reference, grab the WHOLE block (15 lines)
                    for offset in range(-2, 13):
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
                header = f"\n### [{prefix} @ T-{int(time.time() - atom.timestamp)}s ago]\n"
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
            # V3.9 High-Fidelity Reconstruction
            context_text = (
                "# CONTEXT OPTIMIZER (V3.9 HIGH-FIDELITY ACTIVE)\n"
                "The following fragments have been intelligently retrieved and linked from the conversation history.\n"
                "Structural linking has been applied to restore code integrity.\n"
                "--- START RETRIEVED CONTEXT ---\n"
                + "\n".join(packed_lines)
                + "\n--- END RETRIEVED CONTEXT ---\n"
            )
            optimized.append({"role": "system", "content": context_text})
            
        optimized.append(messages[-1])
        overhead = (time.time() - start_time) * 1000
        
        # Phase 6: Accurate Savings Tracking
        # Calculate what the total context size would have been WITHOUT optimization
        total_history_tokens = sum(len(ENCODER.encode(str(m.get("content", "")))) for m in messages)
        
        return optimized, {
            "version": "3.9.0",
            "specificity": specificity,
            "budget": max_tokens,
            "total_lines": len(atoms),
            "selected_lines": len(packed_lines),
            "total_history_tokens": total_history_tokens,
            "optimized_tokens": current_tokens,
            "overhead_ms": overhead,
            "sequence": [
                {
                    "source": a.source,
                    "text": a.text,
                    "score": round(a.score, 3),
                    "line_index": a.line_index
                }
                for a in final_atoms[:50]
            ]
        }
