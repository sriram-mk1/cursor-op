import re
from typing import Iterable, List

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+(?:'[a-z]+)?")
STOPWORDS = {
    "the",
    "and",
    "is",
    "at",
    "which",
    "on",
    "for",
    "with",
    "a",
    "an",
    "of",
    "to",
    "in",
    "that",
    "this",
    "it",
}


def tokenize_text(text: str) -> List[str]:
    """Very light tokenizer for English-like tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def normalize_text(text: str) -> List[str]:
    """Lowercase tokens with stopword dropping."""
    tokens = tokenize_text(text)
    return [tok for tok in tokens if tok not in STOPWORDS]


def estimate_tokens(text: str) -> int:
    """Rough token estimate using tokenizer."""
    return len(tokenize_text(text))


def chunk_content(text: str, target_tokens: int = 600, max_chars: int = 4000) -> List[str]:
    """Break a text payload into chunks roughly target_tokens length."""
    tokens = tokenize_text(text)
    if not tokens:
        snippet = text.strip()
        return [snippet] if snippet else []

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + target_tokens)
        while end > start:
            snippet_tokens = tokens[start:end]
            chunk_text = " ".join(snippet_tokens)
            if len(chunk_text) <= max_chars or end - start <= 1:
                break
            end -= 1
        chunks.append(" ".join(tokens[start:end]))
        start = end
    return chunks
