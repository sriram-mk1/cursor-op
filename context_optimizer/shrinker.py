import re
from itertools import groupby
from typing import Iterable, List

from .types import ChunkType


def _clean_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _joined_snippet(lines: Iterable[str]) -> str:
    return " ".join(lines)


def shrink_authoritative(content: str, query_terms: Iterable[str]) -> str:
    # remove inline comment markers
    content = re.sub(r"//.*?$|#.*?$", "", content, flags=re.MULTILINE)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    lines = _clean_lines(content)
    matches = [
        idx
        for idx, line in enumerate(lines)
        if any(term in line.lower() for term in query_terms)
    ]
    if matches:
        window_start = max(0, min(matches) - 1)
        window_end = min(len(lines), max(matches) + 2)
        return _joined_snippet(line for line in lines[window_start:window_end])
    return _joined_snippet(lines[:3]) or content.strip()


def shrink_diagnostic(content: str, _: Iterable[str]) -> str:
    text = re.sub(r"\d{4}-\d{2}-\d{2}T?\d{2}:\d{2}:\d{2}\S*", "", content)
    lines = _clean_lines(text)
    deduped = [next(group) for _, group in groupby(lines)]
    first_error = next((line for line in deduped if "error" in line.lower()), deduped[:1][0] if deduped else "")
    stack_frames = [line for line in deduped if "at " in line.lower()]
    snippet = [first_error] + stack_frames[:3]
    return " ".join(line for line in snippet if line)


def shrink_exploratory(content: str, _: Iterable[str]) -> str:
    lines = _clean_lines(content)
    if lines:
        return lines[0]
    return content[:200].strip()


def shrink_historical(content: str, _: Iterable[str]) -> str:
    lines = _clean_lines(content)
    if not lines:
        return ""
    bullet_summary = [f"• {lines[idx]}" for idx in range(min(3, len(lines) - 2))]
    tail = lines[-2:]
    return " ".join(bullet_summary + tail) if bullet_summary else " ".join(tail)


SHRINKER_MAP = {
    ChunkType.AUTHORITATIVE: shrink_authoritative,
    ChunkType.DIAGNOSTIC: shrink_diagnostic,
    ChunkType.EXPLORATORY: shrink_exploratory,
    ChunkType.HISTORICAL: shrink_historical,
}


def shrink_chunk(chunk_type: ChunkType, content: str, query_terms: Iterable[str]) -> str:
    shrinker = SHRINKER_MAP.get(chunk_type, shrink_exploratory)
    return shrinker(content, query_terms)
