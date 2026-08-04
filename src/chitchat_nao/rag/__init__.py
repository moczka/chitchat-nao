"""Small, local retrieval components."""

from typing import Any

from .models import AskResult, DocumentChunk, ResponseMode, RetrievedContext


def ask(*args: Any, **kwargs: Any) -> AskResult:
    """Lazily dispatch to the RAG assistant."""
    from .assistant import ask as _ask

    return _ask(*args, **kwargs)


__all__ = [
    "AskResult",
    "DocumentChunk",
    "ResponseMode",
    "RetrievedContext",
    "ask",
]
