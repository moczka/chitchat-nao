from dataclasses import dataclass
from enum import Enum


class ResponseMode(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REDIRECT = "redirect"
    ERROR = "error"


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


@dataclass(frozen=True)
class AskResult:
    mode: ResponseMode
    spoken_text: str
    evidence: tuple[RetrievedContext, ...]
    provenance_verified: bool
    diagnostics: tuple[str, ...]
    clarification_attempts: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
