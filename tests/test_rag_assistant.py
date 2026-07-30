import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from chitchat_nao.rag.assistant import main, validate_generated_answer
from chitchat_nao.rag.generation import GenerationRequest
from chitchat_nao.rag.models import RetrievedContext


class FakeEmbeddingProvider:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class AssistantTests(unittest.TestCase):
    def _run_with_generated_answer(
        self, generated_answer: str, results: list[RetrievedContext]
    ) -> str:
        class FakeRetriever:
            def __init__(self, chunks: object, provider: object) -> None:
                pass

            def search(
                self, question: str, top_k: int
            ) -> list[RetrievedContext]:
                return results

        class FakeGenerator:
            def __init__(self, model_path: Path) -> None:
                pass

            def generate(self, request: GenerationRequest) -> str:
                return generated_answer

        output = StringIO()
        original_argv = sys.argv
        try:
            sys.argv = ["assistant", "--question", "question"]
            with redirect_stdout(output):
                main(
                    embedding_provider_factory=FakeEmbeddingProvider,
                    generator_factory=FakeGenerator,
                    retriever_factory=FakeRetriever,
                )
        finally:
            sys.argv = original_argv
        return output.getvalue()

    def test_allowed_citations_print_generated_answer_unchanged(self) -> None:
        first = "chunk-" + "a" * 64
        second = "chunk-" + "b" * 64
        results = [
            RetrievedContext(first, "one", "one.md", "One", 0.9, 1),
            RetrievedContext(second, "two", "two.md", "Two", 0.8, 2),
        ]
        answer = "One [S1] and two [S2]."

        rendered = self._run_with_generated_answer(answer, results)

        self.assertTrue(rendered.startswith(answer + "\n"))

    def test_rejected_citations_print_only_stable_fallback(self) -> None:
        valid = "chunk-" + "a" * 64
        result = RetrievedContext(valid, "one", "one.md", "One", 0.9, 1)
        cases = (
            "answer without citation",
            "answer [S]",
            "answer [S2]",
            "answer [S1] and [S2]",
            f"answer [{valid}]",
        )

        for answer in cases:
            with self.subTest(answer=answer):
                rendered = self._run_with_generated_answer(answer, [result])
                self.assertNotIn(answer, rendered)
                self.assertEqual(
                    rendered.splitlines()[0],
                    "[answer withheld: citation structural validation failed]",
                )

    def test_case_variant_unknown_and_malformed_labels_are_rejected(
        self,
    ) -> None:
        valid = "chunk-" + "a" * 64
        result = RetrievedContext(valid, "one", "one.md", "One", 0.9, 1)
        cases = (
            "answer [s1]",
            "answer [S01]",
            "answer [S1x]",
        )

        for answer in cases:
            with self.subTest(answer=answer):
                rendered = self._run_with_generated_answer(answer, [result])
                self.assertNotIn(answer, rendered)
                self.assertTrue(
                    rendered.startswith(
                        "[answer withheld: citation structural "
                        "validation failed]\n"
                    )
                )

    def test_canonical_chunk_ids_are_not_accepted_as_citations(self) -> None:
        canonical_id = "chunk-" + "a" * 64
        result = RetrievedContext(canonical_id, "one", "one.md", "One", 0.9, 1)

        self.assertFalse(
            validate_generated_answer(f"answer [{canonical_id}]", [result])
        )

    def test_empty_results_reject_generated_answer(self) -> None:
        answer = "answer [" + "chunk-" + "a" * 64 + "]"

        rendered = self._run_with_generated_answer(answer, [])

        self.assertNotIn(answer, rendered)
        self.assertIn(
            "[answer withheld: citation structural validation failed]",
            rendered,
        )

    def test_empty_results_do_not_invoke_generator(self) -> None:
        invocations = 0

        class FakeRetriever:
            def __init__(self, chunks: object, provider: object) -> None:
                pass

            def search(
                self, question: str, top_k: int
            ) -> list[RetrievedContext]:
                return []

        class FakeGenerator:
            def __init__(self, model_path: Path) -> None:
                nonlocal invocations
                invocations += 1

            def generate(self, request: GenerationRequest) -> str:
                nonlocal invocations
                invocations += 1
                return "answer [S1]"

        output = StringIO()
        original_argv = sys.argv
        try:
            sys.argv = ["assistant", "--question", "question"]
            with redirect_stdout(output):
                main(
                    embedding_provider_factory=FakeEmbeddingProvider,
                    generator_factory=FakeGenerator,
                    retriever_factory=FakeRetriever,
                )
        finally:
            sys.argv = original_argv

        self.assertEqual(invocations, 0)
        self.assertEqual(
            output.getvalue().splitlines()[0],
            "[answer withheld: citation structural validation failed]",
        )

    def test_validation_uses_only_current_results(self) -> None:
        current = RetrievedContext(
            "chunk-" + "a" * 64, "one", "one.md", "One", 0.9, 1
        )
        answer = "answer [S2]"

        self.assertFalse(validate_generated_answer(answer, [current]))
        self.assertTrue(validate_generated_answer("answer [S1]", [current]))

    def test_model_and_top_k_are_passed_to_their_factories(self) -> None:
        captured: dict[str, object] = {}

        class FakeRetriever:
            def __init__(self, chunks: object, provider: object) -> None:
                pass

            def search(
                self, question: str, top_k: int
            ) -> list[RetrievedContext]:
                captured["question"] = question
                captured["top_k"] = top_k
                return [
                    RetrievedContext(
                        "chunk-" + "a" * 64,
                        "one",
                        "one.md",
                        "One",
                        0.9,
                        1,
                    )
                ]

        class FakeGenerator:
            def __init__(self, model_path: Path) -> None:
                captured["model_path"] = model_path

            def generate(self, request: GenerationRequest) -> str:
                return "answer [S1]"

        original_argv = sys.argv
        try:
            sys.argv = [
                "assistant",
                "--question",
                "question",
                "--model",
                "/tmp/test-model.gguf",
                "--top-k",
                "1",
            ]
            main(
                embedding_provider_factory=FakeEmbeddingProvider,
                generator_factory=FakeGenerator,
                retriever_factory=FakeRetriever,
            )
        finally:
            sys.argv = original_argv

        self.assertEqual(captured["model_path"], Path("/tmp/test-model.gguf"))
        self.assertEqual(captured["top_k"], 1)

    def test_one_question_prints_answer_and_ranked_diagnostics(self) -> None:
        class FakeGenerator:
            def __init__(self, model_path: object) -> None:
                self.model_path = model_path

            def generate(self, request: GenerationRequest) -> str:
                self.request = request
                return "The supplied answer. [S1]"

        output = StringIO()
        original_argv = sys.argv
        try:
            sys.argv = ["assistant", "--question", "Who can join?"]
            with redirect_stdout(output):
                main(
                    embedding_provider_factory=FakeEmbeddingProvider,
                    generator_factory=FakeGenerator,
                )
        finally:
            sys.argv = original_argv

        rendered = output.getvalue()
        self.assertIn("The supplied answer.", rendered)
        self.assertIn("rank=1", rendered)
        self.assertIn("score=", rendered)
        self.assertIn("source=", rendered)
        self.assertIn("chunk-", rendered)
