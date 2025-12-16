import math
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cache import QueryCache
from .shrinker import shrink_chunk
from .simhash import bucket_key, compute_simhash, hamming_distance
from .tokenizer import chunk_content, estimate_tokens, normalize_text
from .types import ChunkType


MAX_CHUNKS_PER_SESSION = 2000
SIMHASH_BUCKET_BITS = 16
SIMHASH_HAMMING_THRESHOLD = 3
BM25_K1 = 1.5
BM25_B = 0.75
MAX_CANDIDATES = 400


@dataclass
class Chunk:
    id: str
    session_id: str
    role: str
    source: str
    content: str
    chunk_type: ChunkType
    tokens: List[str]
    simhash: int
    created_ts: float = field(default_factory=time.monotonic)

    @property
    def char_length(self) -> int:
        return len(self.content)

    @property
    def token_length(self) -> int:
        return len(self.tokens)


@dataclass
class SessionState:
    chunks: Dict[str, Chunk] = field(default_factory=dict)
    index: Dict[str, Dict[str, int]] = field(default_factory=dict)
    doc_lengths: Dict[str, int] = field(default_factory=dict)
    chunk_queue: deque = field(default_factory=deque)
    pinned_ids: set = field(default_factory=set)
    simhash_buckets: Dict[int, set] = field(default_factory=dict)
    cache: QueryCache = field(default_factory=QueryCache)
    last_optimize: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=lambda: {"ingested": 0, "deduped": 0, "cache_hits": 0, "cache_misses": 0})

    @property
    def avg_doc_len(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)


class ContextOptimizer:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def ingest(self, session_id: str, events: List[Dict]) -> Dict[str, int]:
        state = self.sessions.setdefault(session_id, SessionState())
        ingested = 0
        for event in events:
            content = event.get("content", "").strip()
            if not content:
                continue
            for snippet in chunk_content(content):
                normalized_tokens = normalize_text(snippet)
                if not normalized_tokens:
                    continue
                chunk_type = self._classify_chunk(event, snippet)
                chunk_id = uuid.uuid4().hex
                simhash_value = compute_simhash(normalized_tokens)
                chunk = Chunk(
                    id=chunk_id,
                    session_id=session_id,
                    role=event.get("role", "tool"),
                    source=event.get("source", "unknown"),
                    content=snippet,
                    chunk_type=chunk_type,
                    tokens=normalized_tokens,
                    simhash=simhash_value,
                )
                if self._is_duplicate(state, chunk):
                    state.stats["deduped"] += 1
                    continue
                self._persist_chunk(state, chunk)
                if self._should_pin(event, snippet):
                    state.pinned_ids.add(chunk_id)
                ingested += 1
        state.stats["ingested"] += ingested
        return {"ingested": ingested, "deduped": state.stats["deduped"]}

    def optimize(
        self,
        session_id: str,
        query_text: str,
        max_chunks: int = 12,
        target_token_budget: Optional[int] = None,
        cache_ttl_sec: int = 60,
    ) -> Dict:
        state = self.sessions.setdefault(session_id, SessionState())
        max_chunks = max(1, max_chunks)
        query_terms = normalize_text(query_text)
        cache_key = (session_id, tuple(query_terms), max_chunks, target_token_budget)
        cached = state.cache.get(cache_key, ttl=cache_ttl_sec)
        if cached:
            state.stats["cache_hits"] += 1
            state.last_optimize = {**cached, "query_text": query_text, "timestamp": time.time()}
            return cached
        state.stats["cache_misses"] += 1

        candidates = self._collect_candidates(state, query_terms, max_chunks)
        scored = self._score_candidates(state, candidates, query_terms)
        deduped_chunks = self._dedupe_chunks(state, scored, max_chunks)
        optimized_items = self._shrink_and_trim(deduped_chunks, query_terms, target_token_budget)
        result = self._build_result(optimized_items)
        state.cache.set(cache_key, result)
        state.last_optimize = {**result, "query_text": query_text, "timestamp": time.time()}
        return result

    def get_stats(self, session_id: str) -> Dict[str, any]:
        state = self.sessions.get(session_id)
        if not state:
            return {"chunks": 0, "index_terms": 0, "dedup_rate": 0.0, "last_optimize": {}, "cache_hit_rate": 0.0}
        total = state.stats["cache_hits"] + state.stats["cache_misses"]
        return {
            "chunks": len(state.chunks),
            "index_terms": len(state.index),
            "dedup_rate": (state.stats["deduped"] / (state.stats["ingested"] or 1)),
            "index_size": sum(len(postings) for postings in state.index.values()),
            "last_optimize": state.last_optimize,
            "cache_hit_rate": (state.stats["cache_hits"] / total) if total else 0.0,
        }

    def _collect_candidates(self, state: SessionState, query_terms: List[str], max_chunks: int) -> List[str]:
        candidates = set(state.pinned_ids)
        for term in query_terms:
            postings = state.index.get(term)
            if not postings:
                continue
            for chunk_id in postings:
                candidates.add(chunk_id)
                if len(candidates) >= MAX_CANDIDATES:
                    break
        if not candidates:
            candidates = list(state.chunk_queue)[-max_chunks:]
        return list(candidates)

    def _score_candidates(self, state: SessionState, candidates: List[str], query_terms: List[str]) -> List[Tuple[str, float]]:
        scored = []
        N = max(len(state.chunks), 1)
        avgdl = state.avg_doc_len or 1
        for chunk_id in candidates:
            chunk = state.chunks.get(chunk_id)
            if not chunk:
                continue
            score = 0.0
            if not query_terms:
                age = time.monotonic() - chunk.created_ts
                score = 1 / (1 + age)
            else:
                for term in query_terms:
                    postings = state.index.get(term)
                    if not postings:
                        continue
                    tf = postings.get(chunk_id, 0)
                    if tf == 0:
                        continue
                    df = max(1, len(postings))
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    doc_len = state.doc_lengths.get(chunk_id, avgdl)
                    denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avgdl)
                    score += idf * ((tf * (BM25_K1 + 1)) / denom)
            scored.append((chunk_id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def _dedupe_chunks(
        self, state: SessionState, scored: List[Tuple[str, float]], max_chunks: int
    ) -> List[Chunk]:
        seen = []
        selected = []
        for chunk_id, _score in scored:
            chunk = state.chunks.get(chunk_id)
            if not chunk:
                continue
            if any(hamming_distance(chunk.simhash, other) <= SIMHASH_HAMMING_THRESHOLD for other in seen):
                continue
            seen.append(chunk.simhash)
            selected.append(chunk)
            if len(selected) >= max_chunks:
                break
        return selected

    def _shrink_and_trim(
        self, chunks: List[Chunk], query_terms: List[str], token_budget: Optional[int]
    ) -> List[Dict]:
        trimmed = []
        total_optimized_tokens = 0
        for chunk in chunks:
            summary = shrink_chunk(chunk.chunk_type, chunk.content, query_terms)
            optimized_tokens = estimate_tokens(summary)
            total_optimized_tokens += optimized_tokens
            trimmed.append(
                {
                    "chunk_id": chunk.id,
                    "role": chunk.role,
                    "source": chunk.source,
                    "type": chunk.chunk_type.value,
                    "raw_chars": chunk.char_length,
                    "raw_tokens": chunk.token_length,
                    "summary": summary,
                    "optimized_tokens": optimized_tokens,
                    "optimized_chars": len(summary),
                }
            )
        if token_budget and total_optimized_tokens > token_budget:
            while trimmed and total_optimized_tokens > token_budget:
                removed = trimmed.pop()
                total_optimized_tokens -= removed["optimized_tokens"]
        return trimmed

    def _build_result(self, optimized_items: List[Dict]) -> Dict:
        raw_chars = sum(item["raw_chars"] for item in optimized_items)
        optimized_chars = sum(item["optimized_chars"] for item in optimized_items)
        raw_tokens = sum(item["raw_tokens"] for item in optimized_items)
        optimized_tokens = sum(item["optimized_tokens"] for item in optimized_items)
        percent_saved = (
            (1 - optimized_tokens / raw_tokens) * 100 if raw_tokens else 0.0
        )
        return {
            "optimized_context": optimized_items,
            "raw_chars": raw_chars,
            "optimized_chars": optimized_chars,
            "raw_token_est": raw_tokens,
            "optimized_token_est": optimized_tokens,
            "percent_saved_est": round(percent_saved, 2),
            "selected_chunk_ids": [item["chunk_id"] for item in optimized_items],
        }

    def _persist_chunk(self, state: SessionState, chunk: Chunk) -> None:
        state.chunks[chunk.id] = chunk
        state.doc_lengths[chunk.id] = chunk.token_length or 1
        state.chunk_queue.append(chunk.id)
        bucket = bucket_key(chunk.simhash, SIMHASH_BUCKET_BITS)
        state.simhash_buckets.setdefault(bucket, set()).add(chunk.id)
        frequencies = Counter(chunk.tokens)
        for term, freq in frequencies.items():
            postings = state.index.setdefault(term, {})
            postings[chunk.id] = freq
        self._shrink_rolling_window(state)

    def _shrink_rolling_window(self, state: SessionState) -> None:
        while len(state.chunk_queue) > MAX_CHUNKS_PER_SESSION:
            candidate = state.chunk_queue.popleft()
            if candidate in state.pinned_ids:
                state.chunk_queue.append(candidate)
                if len(state.chunk_queue) > MAX_CHUNKS_PER_SESSION * 2:
                    break
                continue
            self._remove_chunk(state, candidate)

    def _remove_chunk(self, state: SessionState, chunk_id: str) -> None:
        chunk = state.chunks.pop(chunk_id, None)
        if not chunk:
            return
        state.doc_lengths.pop(chunk_id, None)
        bucket = bucket_key(chunk.simhash, SIMHASH_BUCKET_BITS)
        bucket_set = state.simhash_buckets.get(bucket)
        if bucket_set:
            bucket_set.discard(chunk_id)
            if not bucket_set:
                state.simhash_buckets.pop(bucket, None)
        for postings in state.index.values():
            postings.pop(chunk_id, None)

    def _is_duplicate(self, state: SessionState, chunk: Chunk) -> bool:
        bucket = bucket_key(chunk.simhash, SIMHASH_BUCKET_BITS)
        candidates = state.simhash_buckets.get(bucket, set())
        return any(
            hamming_distance(chunk.simhash, state.chunks[cand].simhash) <= SIMHASH_HAMMING_THRESHOLD
            for cand in candidates
            if cand in state.chunks
        )

    def _classify_chunk(self, event: Dict, snippet: str) -> ChunkType:
        content = snippet.lower()
        source = event.get("source", "").lower()
        role = event.get("role", "")
        if "log" in source or role == "tool" or "error" in content or "exception" in content:
            return ChunkType.DIAGNOSTIC
        if any(token in content for token in ("goal", "objective", "mission", "priority")):
            return ChunkType.AUTHORITATIVE
        if "history" in source or "document" in source or "note" in source:
            return ChunkType.HISTORICAL
        return ChunkType.EXPLORATORY

    def _should_pin(self, event: Dict, snippet: str) -> bool:
        source = event.get("source", "").lower()
        if "system" in source or "prompt" in source:
            return True
        if event.get("role") == "user" and any(keyword in snippet.lower() for keyword in ("goal", "objective")):
            return True
        return False
