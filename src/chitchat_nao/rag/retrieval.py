from .embedding import EmbeddingProvider, normalize_vectors
from .models import DocumentChunk, RetrievedContext


class Retriever:
    def __init__(
        self, chunks: list[DocumentChunk], provider: EmbeddingProvider
    ) -> None:
        self._chunks = chunks
        self._provider = provider
        self._vectors = normalize_vectors(
            provider.embed([chunk.text for chunk in chunks])
        )

    def search(self, query: str, top_k: int = 2) -> list[RetrievedContext]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = normalize_vectors(self._provider.embed([query]))[0]
        scores = self._vectors @ query_vector
        ordered = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), self._chunks[index].id),
        )
        results: list[RetrievedContext] = []
        for rank, index in enumerate(ordered[:top_k], start=1):
            chunk = self._chunks[index]
            results.append(
                RetrievedContext(
                    chunk.id,
                    chunk.text,
                    chunk.source_path,
                    chunk.section,
                    float(scores[index]),
                    rank,
                )
            )
        return results
