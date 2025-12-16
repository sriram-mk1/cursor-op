from enum import Enum


class ChunkType(str, Enum):
    AUTHORITATIVE = "authoritative"
    DIAGNOSTIC = "diagnostic"
    EXPLORATORY = "exploratory"
    HISTORICAL = "historical"
