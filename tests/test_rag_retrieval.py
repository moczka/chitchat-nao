import unittest

import numpy as np

from chitchat_nao.rag.models import DocumentChunk
from chitchat_nao.rag.retrieval import Retriever


class FixedProvider:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [self.vectors[text] for text in texts], dtype=np.float32
        )


class RetrievalTests(unittest.TestCase):
    def test_search_returns_normalized_cosine_order_and_metadata(self) -> None:
        chunks = [
            DocumentChunk("a", "alpha", "a.md", "A", "a"),
            DocumentChunk("b", "beta", "b.md", "B", "b"),
        ]
        provider = FixedProvider(
            {"alpha": [3, 0], "beta": [0, 2], "query": [1, 0]}
        )

        results = Retriever(chunks, provider).search("query", 2)

        self.assertEqual([result.id for result in results], ["a", "b"])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(results[0].source_path, "a.md")
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_equal_scores_are_ordered_by_chunk_id(self) -> None:
        chunks = [
            DocumentChunk("b", "beta", "b.md", "B", "b"),
            DocumentChunk("a", "alpha", "a.md", "A", "a"),
        ]
        provider = FixedProvider(
            {"alpha": [1, 0], "beta": [1, 0], "query": [1, 0]}
        )

        results = Retriever(chunks, provider).search("query", 2)

        self.assertEqual([result.id for result in results], ["a", "b"])
