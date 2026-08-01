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

    def test_allowed_citations_print_cleaned_answer_with_valid_diagnostic(
        self,
    ) -> None:
        first = "chunk-" + "a" * 64
        second = "chunk-" + "b" * 64
        results = [
            RetrievedContext(first, "one", "one.md", "One", 0.9, 1),
            RetrievedContext(second, "two", "two.md", "Two", 0.8, 2),
        ]
        answer = "One [S1]."

        rendered = self._run_with_generated_answer(answer, results)

        self.assertEqual(rendered.splitlines()[0], "One.")
        self.assertNotRegex(rendered.splitlines()[0], r"\[S\d+\]")
        self.assertIn(
            "Citation formatting was structurally valid",
            rendered,
        )

    def test_invalid_or_missing_citations_do_not_withhold_answer(self) -> None:
        valid = "chunk-" + "a" * 64
        result = RetrievedContext(valid, "one", "one.md", "One", 0.9, 1)
        cases = (
            ("answer without citation", "answer without citation"),
            ("answer [S]", "answer [S]"),
            ("answer [S2]", "answer"),
            ("answer [S1] and [S2]", "answer and"),
            (f"answer [{valid}]", f"answer [{valid}]"),
        )

        for answer, expected_spoken_text in cases:
            with self.subTest(answer=answer):
                rendered = self._run_with_generated_answer(answer, [result])
                self.assertEqual(
                    rendered.splitlines()[0], expected_spoken_text
                )
                self.assertNotRegex(rendered.splitlines()[0], r"\[S\d+\]")
                self.assertIn("rank=1", rendered)
                self.assertIn("source=one.md", rendered)
                self.assertIn(f"id={valid}", rendered)
                self.assertIn(
                    "Citation formatting was missing or invalid", rendered
                )
                self.assertNotIn(
                    "[answer withheld: citation structural validation failed]",
                    rendered,
                )

    def test_case_variant_unknown_and_malformed_labels_are_returned(
        self,
    ) -> None:
        valid = "chunk-" + "a" * 64
        result = RetrievedContext(valid, "one", "one.md", "One", 0.9, 1)
        cases = (
            ("answer [s1]", "answer [s1]"),
            ("answer [S01]", "answer"),
            ("answer [S1x]", "answer [S1x]"),
        )

        for answer, expected_spoken_text in cases:
            with self.subTest(answer=answer):
                rendered = self._run_with_generated_answer(answer, [result])
                self.assertEqual(
                    rendered.splitlines()[0], expected_spoken_text
                )
                self.assertNotRegex(rendered.splitlines()[0], r"\[S\d+\]")
                self.assertIn("rank=1", rendered)
                self.assertIn("source=one.md", rendered)
                self.assertIn(
                    "Citation formatting was missing or invalid", rendered
                )
                self.assertNotIn(
                    "[answer withheld: citation structural validation failed]",
                    rendered,
                )

    def test_cli_reports_unverified_provenance_after_spoken_text(self) -> None:
        result = RetrievedContext(
            "chunk-" + "a" * 64, "one", "one.md", "One", 0.9, 1
        )
        answer = "answer without citation"

        rendered = self._run_with_generated_answer(answer, [result])
        lines = rendered.splitlines()

        self.assertEqual(lines[0], answer)
        self.assertIn("provenance_verified=False", rendered)
        self.assertIn("diagnostic=", rendered)
        self.assertIn("citation", rendered.lower())
        self.assertTrue(
            all(
                line.startswith(
                    ("provenance_verified=", "diagnostic=", "rank=")
                )
                for line in lines[1:]
            )
        )

    def test_cli_reports_generator_error_after_speech_fallback(self) -> None:
        result = RetrievedContext(
            "chunk-" + "a" * 64, "one", "one.md", "One", 0.9, 1
        )
        invoked = False

        class FakeRetriever:
            def __init__(self, chunks: object, provider: object) -> None:
                pass

            def search(
                self, question: str, top_k: int
            ) -> list[RetrievedContext]:
                return [result]

        class FailingGenerator:
            def __init__(self, model_path: Path) -> None:
                pass

            def generate(self, request: GenerationRequest) -> str:
                nonlocal invoked
                invoked = True
                raise RuntimeError("synthetic generator failure")

        output = StringIO()
        original_argv = sys.argv
        try:
            sys.argv = ["assistant", "--question", "question"]
            with redirect_stdout(output):
                main(
                    embedding_provider_factory=FakeEmbeddingProvider,
                    generator_factory=FailingGenerator,
                    retriever_factory=FakeRetriever,
                )
        finally:
            sys.argv = original_argv

        rendered = output.getvalue()
        lines = rendered.splitlines()
        self.assertTrue(invoked)
        self.assertEqual(lines[0], "I couldn't generate a response right now.")
        self.assertIn("provenance_verified=False", rendered)
        self.assertIn(
            "diagnostic=Generator error: synthetic generator failure", rendered
        )
        self.assertTrue(
            all(
                line.startswith(
                    ("provenance_verified=", "diagnostic=", "rank=")
                )
                for line in lines[1:]
            )
        )

    def test_canonical_chunk_ids_are_not_accepted_as_citations(self) -> None:
        canonical_id = "chunk-" + "a" * 64
        result = RetrievedContext(canonical_id, "one", "one.md", "One", 0.9, 1)

        self.assertFalse(
            validate_generated_answer(f"answer [{canonical_id}]", [result])
        )

    def test_empty_results_print_natural_redirect(self) -> None:
        answer = "answer [" + "chunk-" + "a" * 64 + "]"

        rendered = self._run_with_generated_answer(answer, [])

        self.assertNotIn(answer, rendered)
        self.assertIn("I can help with Computer Club questions", rendered)
        self.assertNotIn(
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
        self.assertIn(
            "I can help with Computer Club questions",
            output.getvalue(),
        )
        self.assertNotIn(
            "[answer withheld: citation structural validation failed]",
            output.getvalue(),
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
        class FakeRetriever:
            def __init__(self, chunks: object, provider: object) -> None:
                pass

            def search(
                self, question: str, top_k: int
            ) -> list[RetrievedContext]:
                return [
                    RetrievedContext(
                        "chunk-" + "a" * 64,
                        "Question 1: Who can join the Computer Club? "
                        "Answer 1: Anyone! Regardless of major or year, any "
                        "Quincy College student and alumni may join.",
                        "faq.md",
                        "FAQ",
                        0.9,
                        1,
                    )
                ]

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
                    retriever_factory=FakeRetriever,
                )
        finally:
            sys.argv = original_argv

        rendered = output.getvalue()
        self.assertEqual(rendered.splitlines()[0], "The supplied answer.")
        self.assertIn("The supplied answer.", rendered)
        self.assertIn("provenance_verified=False", rendered)
        self.assertIn("diagnostic=", rendered)
        self.assertIn("rank=1", rendered)
        self.assertIn("score=", rendered)
        self.assertIn("source=", rendered)
        self.assertIn("chunk-", rendered)
