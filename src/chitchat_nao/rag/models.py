from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source_path: str
    section: str
    content_hash: str


@dataclass(frozen=True)
class RetrievedContext:
    id: str
    text: str
    source_path: str
    section: str
    score: float
    rank: int
