import hashlib
from typing import Iterable


def _hash_token(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def compute_simhash(tokens: Iterable[str], bits: int = 64) -> int:
    """Compute a 64-bit SimHash signature for given tokens."""
    if not tokens:
        return 0
    vector = [0] * bits
    for token in tokens:
        token_hash = _hash_token(token)
        for i in range(bits):
            bit_mask = 1 << i
            vector[i] += 1 if token_hash & bit_mask else -1
    signature = 0
    for i, weight in enumerate(vector):
        if weight > 0:
            signature |= 1 << i
    return signature


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two SimHash signatures."""
    xor = a ^ b
    return xor.bit_count()


def bucket_key(signature: int, prefix_bits: int = 16) -> int:
    """Return top prefix bits for quick bucketization."""
    return signature >> (64 - prefix_bits)
