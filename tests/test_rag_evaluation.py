import json
import math
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from chitchat_nao.rag.evaluator import (
    EvalCaseResult,
    EvaluationReport,
    evaluate_corpus,
    format_inspection,
    load_knowledge_base,
    format_report,
    load_eval_cases,
)
from chitchat_nao.rag.ingest import ingest_markdown
from chitchat_nao.rag.models import DocumentChunk, RetrievedContext
from chitchat_nao.rag.retrieval import Retriever


class WordOverlapProvider:
    def __init__(self, corpus: list[str]) -> None:
        vocabulary = {
            word
            for text in corpus
            for word in re.findall(r"[a-z0-9]+", text.lower())
        }
        self.vocabulary = sorted(vocabulary)
        document_frequency = {
            word: sum(
                word in re.findall(r"[a-z0-9]+", text.lower())
                for text in corpus
            )
            for word in self.vocabulary
        }
        self.inverse_document_frequency = {
            word: math.log((1 + len(corpus)) / (1 + frequency)) + 1
            for word, frequency in document_frequency.items()
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [
                    re.findall(r"[a-z0-9]+", text.lower()).count(word)
                    * self.inverse_document_frequency[word]
                    for word in self.vocabulary
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


class EvaluationTests(unittest.TestCase):
    def test_isolated_synthetic_corpus_passes_answerable_retrieval_at_top_two(
        self,
    ) -> None:
        fixture_root = (
            Path(__file__).parent / "fixtures" / "synthetic_retrieval_corpus"
        )
        eval_path = fixture_root / "synthetic_retrieval_eval.json"

        self.assertTrue(fixture_root.is_dir())
        self.assertNotIn("knowledge_base", fixture_root.resolve().parts)
        self.assertNotEqual(
            fixture_root.resolve(), Path("knowledge_base").resolve()
        )
        markdown_paths = sorted(fixture_root.glob("*.md"))
        self.assertGreaterEqual(len(markdown_paths), 5)

        chunks = load_knowledge_base(fixture_root)
        ingested_chunks = [
            chunk
            for path in markdown_paths
            for chunk in ingest_markdown(
                path, path.relative_to(fixture_root).as_posix()
            )
        ]
        self.assertEqual(chunks, ingested_chunks)
        self.assertTrue(all(chunk.id.startswith("chunk-") for chunk in chunks))
        self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))

        raw_cases = json.loads(eval_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(raw_cases), 24)
        self.assertLessEqual(len(raw_cases), 30)
        scenario_categories = {case["scenario_category"] for case in raw_cases}
        self.assertTrue(
            {
                "direct_fact",
                "paraphrase",
                "ambiguous_competing",
                "weak_relevant",
                "unrelated",
                "adversarial_trick_unanswered",
                "nonce_fact_contrast",
            }.issubset(scenario_categories)
        )
        self.assertGreaterEqual(
            sum(
                case["scenario_category"] == "nonce_fact_contrast"
                for case in raw_cases
            ),
            2,
        )

        cases = load_eval_cases(eval_path)
        self.assertEqual(len(cases), len(raw_cases))
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        raw_cases_by_question = {case["question"]: case for case in raw_cases}
        for case in cases:
            if case.category == "answerable":
                self.assertTrue(case.relevant_chunk_ids)
                for chunk_id in case.relevant_chunk_ids:
                    self.assertIn(chunk_id, chunks_by_id)
                    self.assertNotIn(
                        "knowledge_base", chunks_by_id[chunk_id].source_path
                    )
                gold_text = " ".join(
                    chunks_by_id[chunk_id].text
                    for chunk_id in case.relevant_chunk_ids
                ).lower()
                self.assertTrue(
                    all(
                        phrase.lower() in gold_text
                        for phrase in case.expected_answer_contains
                    )
                )
                self.assertEqual(
                    raw_cases_by_question[case.question]["category"],
                    "answerable",
                )

        provider = WordOverlapProvider(
            [chunk.text for chunk in chunks]
            + [case.question for case in cases]
        )
        retriever = Retriever(chunks, provider)
        report = evaluate_corpus(
            eval_path,
            lambda question: retriever.search(question, top_k=2),
        )

        answerable = [
            case for case in report.cases if case.category == "answerable"
        ]
        self.assertTrue(answerable)
        self.assertTrue(
            all(
                len(case.retrieved_ids) == 2
                and case.has_gold_hit
                and case.recall_at_2 == 1.0
                for case in answerable
            )
        )
        self.assertEqual(report.recall_at_2, 1.0)
        self.assertTrue(
            all(
                case.category != "answerable" and case.recall_at_2 is None
                for case in report.cases
                if case.category != "answerable"
            )
        )

    def test_knowledge_base_ids_are_independent_of_cwd_and_path_form(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_base = root / "knowledge_base"
            knowledge_base.mkdir()
            (knowledge_base / "faq.md").write_text(
                "# FAQ\n\nStable answer.\n", encoding="utf-8"
            )
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                relative_ids = [
                    chunk.id
                    for chunk in load_knowledge_base(Path("knowledge_base"))
                ]
                os.chdir(root / "knowledge_base")
                absolute_ids = [
                    chunk.id
                    for chunk in load_knowledge_base(knowledge_base.resolve())
                ]
            finally:
                os.chdir(original_cwd)

        self.assertEqual(relative_ids, absolute_ids)

    def test_eval_gold_ids_exist_and_point_to_answer_chunks(self) -> None:
        knowledge_base = load_knowledge_base(Path("knowledge_base"))
        chunks_by_id = {chunk.id: chunk for chunk in knowledge_base}
        cases = load_eval_cases(Path("src/chitchat_nao/rag/eval_corpus.json"))

        answerable = [case for case in cases if case.category == "answerable"]
        self.assertTrue(all(case.relevant_chunk_ids for case in answerable))
        for case in cases:
            if case.category == "answerable":
                self.assertTrue(
                    set(case.relevant_chunk_ids) <= chunks_by_id.keys()
                )
        cases_by_question = {case.question: case for case in answerable}
        self.assertIn(
            "Anyone!",
            chunks_by_id[
                cases_by_question[
                    "Who can join the Computer Club?"
                ].relevant_chunk_ids[0]
            ].text,
        )
        self.assertIn(
            "Faculty Advisor",
            chunks_by_id[
                cases_by_question[
                    "Who is the faculty advisor?"
                ].relevant_chunk_ids[0]
            ].text,
        )
        self.assertIn(
            "President",
            chunks_by_id[
                cases_by_question["Who is the president?"].relevant_chunk_ids[
                    0
                ]
            ].text,
        )

    def test_entity_queries_have_entity_bearing_gold_chunks(self) -> None:
        chunks_by_id = {
            chunk.id: chunk
            for chunk in load_knowledge_base(Path("knowledge_base"))
        }
        cases = load_eval_cases(Path("src/chitchat_nao/rag/eval_corpus.json"))

        for case in cases:
            if (
                case.category != "answerable"
                or "Computer Club" not in case.question
            ):
                continue
            with self.subTest(question=case.question):
                self.assertTrue(
                    all(
                        "Computer Club" in chunks_by_id[chunk_id].text
                        for chunk_id in case.relevant_chunk_ids
                    )
                )

    def test_evaluation_reports_each_case_and_recall_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "eval.json"
            corpus.write_text(
                json.dumps(
                    [
                        {
                            "question": "who",
                            "category": "answerable",
                            "relevant_chunk_ids": ["known"],
                            "expected_answer_contains": ["answer"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            chunks = [
                DocumentChunk(
                    "known", "answer", "source.md", "Section", "hash"
                )
            ]

            report = evaluate_corpus(
                corpus,
                lambda _: [
                    RetrievedContext(
                        chunks[0].id,
                        chunks[0].text,
                        chunks[0].source_path,
                        chunks[0].section,
                        1.0,
                        1,
                    )
                ],
            )

        self.assertEqual(report.cases[0].retrieved_ids, ["known"])
        self.assertEqual(report.cases[0].retrieved_sources, ["source.md"])
        self.assertEqual(report.recall_at_1, 1.0)
        self.assertEqual(report.recall_at_2, 1.0)
        self.assertEqual(report.mrr, 1.0)

    def test_evaluation_reports_scores_margin_and_gold_hit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "eval.json"
            corpus.write_text(
                json.dumps(
                    [
                        {
                            "question": "who",
                            "category": "answerable",
                            "relevant_chunk_ids": ["known"],
                            "expected_answer_contains": ["answer"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            report = evaluate_corpus(
                corpus,
                lambda _: [
                    RetrievedContext(
                        "other", "other", "source.md", "Section", 0.83, 1
                    ),
                    RetrievedContext(
                        "known", "answer", "source.md", "Section", 0.61, 2
                    ),
                ],
            )

        result = report.cases[0]
        self.assertEqual(result.top1_score, 0.83)
        self.assertEqual(result.top2_score, 0.61)
        margin = result.top2_margin
        self.assertIsNotNone(margin)
        assert margin is not None
        self.assertAlmostEqual(margin, 0.22)
        self.assertTrue(result.has_gold_hit)
        self.assertEqual(report.recall_at_1, 0.0)
        self.assertEqual(report.recall_at_2, 1.0)
        self.assertEqual(report.mrr, 0.5)
        rendered = format_report(report)
        self.assertIn("top1_score=0.83", rendered)
        self.assertIn("top2_score=0.61", rendered)
        self.assertIn("top2_margin=", rendered)
        self.assertIn("gold_hit=True", rendered)

    def test_starter_answerable_cases_cover_each_meaningful_chunk(
        self,
    ) -> None:
        chunks = load_knowledge_base(Path("knowledge_base"))
        cases = load_eval_cases(Path("src/chitchat_nao/rag/eval_corpus.json"))
        gold_ids = {
            chunk_id
            for case in cases
            if case.category == "answerable"
            for chunk_id in case.relevant_chunk_ids
        }
        chunks_by_marker = {
            marker: next(chunk for chunk in chunks if marker in chunk.text)
            for marker in (
                "The Computer Club at Quincy College",
                "Anyone!",
                "Faculty Advisor",
                "President",
                "Vice President",
                "Treasurer",
                "Secretary",
            )
        }

        for marker, chunk in chunks_by_marker.items():
            with self.subTest(marker=marker):
                self.assertIn(chunk.id, gold_ids)

    def test_starter_evaluation_exits_at_top_two(self) -> None:
        knowledge_base = load_knowledge_base(Path("knowledge_base"))
        cases = load_eval_cases(Path("src/chitchat_nao/rag/eval_corpus.json"))
        provider = WordOverlapProvider(
            [chunk.text for chunk in knowledge_base]
            + [case.question for case in cases]
        )
        retriever = Retriever(knowledge_base, provider)
        report = evaluate_corpus(
            Path("src/chitchat_nao/rag/eval_corpus.json"),
            lambda question: retriever.search(question, top_k=2),
        )

        answerable = [
            case for case in report.cases if case.category == "answerable"
        ]
        self.assertTrue(answerable)
        self.assertTrue(all(case.has_gold_hit for case in answerable))
        self.assertEqual(report.recall_at_2, 1.0)

    def test_recall_counts_relevant_ids_and_mrr_uses_first_relevant_rank(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "eval.json"
            corpus.write_text(
                json.dumps(
                    [
                        {
                            "question": "partial",
                            "category": "answerable",
                            "relevant_chunk_ids": ["a", "b"],
                            "expected_answer_contains": ["partial"],
                        },
                        {
                            "question": "rank two",
                            "category": "answerable",
                            "relevant_chunk_ids": ["c"],
                            "expected_answer_contains": ["rank two"],
                        },
                        {
                            "question": "none",
                            "category": "answerable",
                            "relevant_chunk_ids": ["d"],
                            "expected_answer_contains": ["none"],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            retrieved = {
                "partial": ["a", "x"],
                "rank two": ["x", "c"],
                "none": ["x", "y"],
            }

            def search(question: str) -> list[RetrievedContext]:
                return [
                    RetrievedContext(
                        identifier,
                        identifier,
                        "source.md",
                        "Section",
                        1.0,
                        rank,
                    )
                    for rank, identifier in enumerate(
                        retrieved[question], start=1
                    )
                ]

            report = evaluate_corpus(corpus, search)

        self.assertAlmostEqual(report.recall_at_1, 1 / 6)
        self.assertEqual(report.recall_at_2, 0.5)
        self.assertEqual(report.mrr, 0.5)
        self.assertEqual(report.cases[1].recall_at_1, 0.0)
        self.assertEqual(report.cases[1].recall_at_2, 1.0)

    def test_empty_relevance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "eval.json"
            corpus.write_text(
                '[{"question": "unanswerable", '
                '"category": "answerable", '
                '"relevant_chunk_ids": []}]',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "relevant_chunk_ids"):
                load_eval_cases(corpus)

    def test_non_answerable_relevance_error_identifies_question(self) -> None:
        for category in ("unanswered", "ambiguous"):
            question = f"{category} question"
            with (
                self.subTest(category=category),
                tempfile.TemporaryDirectory() as directory,
            ):
                corpus = Path(directory) / "eval.json"
                corpus.write_text(
                    json.dumps(
                        [
                            {
                                "question": question,
                                "category": category,
                                "relevant_chunk_ids": ["id"],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, re.escape(question)):
                    load_eval_cases(corpus)

    def test_answerable_missing_phrases_error_identifies_question(
        self,
    ) -> None:
        question = "answerable without expected phrases"
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "eval.json"
            corpus.write_text(
                json.dumps(
                    [
                        {
                            "question": question,
                            "category": "answerable",
                            "relevant_chunk_ids": ["id"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, re.escape(question)):
                load_eval_cases(corpus)

    def test_non_answerable_phrases_error_identifies_question(self) -> None:
        question = "unanswered with expected phrases"
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "eval.json"
            corpus.write_text(
                json.dumps(
                    [
                        {
                            "question": question,
                            "category": "unanswered",
                            "expected_answer_contains": ["answer"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, re.escape(question)):
                load_eval_cases(corpus)

    def test_report_labels_per_case_and_aggregate_recall(self) -> None:
        report = format_report(
            EvaluationReport(
                [
                    EvalCaseResult(
                        "q", ["chunk-a"], ["source.md"], 0.5, 0.5, 1.0
                    )
                ],
                0.5,
                0.5,
                1.0,
            )
        )

        self.assertIn("Recall@1=0.500", report)
        self.assertIn("retrieved_ids=['chunk-a']", report)

    def test_eval_cases_are_loaded_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "eval.json"
            corpus.write_text(
                '[{"question": "q", '
                '"category": "answerable", '
                '"relevant_chunk_ids": ["id"], '
                '"expected_answer_contains": ["answer"]}]',
                encoding="utf-8",
            )

            cases = load_eval_cases(corpus)

        self.assertEqual(cases[0].question, "q")
        self.assertEqual(cases[0].relevant_chunk_ids, ["id"])

    def test_mixed_categories_keep_metrics_answerable_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "eval.json"
            corpus.write_text(
                json.dumps(
                    [
                        {
                            "question": "answered",
                            "category": "answerable",
                            "relevant_chunk_ids": ["a", "b"],
                            "expected_answer_contains": ["answered"],
                        },
                        {"question": "unknown", "category": "unanswered"},
                        {"question": "ambiguous", "category": "ambiguous"},
                    ]
                ),
                encoding="utf-8",
            )
            search_results = {
                "answered": ["a", "x"],
                "unknown": ["x"],
                "ambiguous": ["x"],
            }

            def search(question: str) -> list[RetrievedContext]:
                return [
                    RetrievedContext(
                        identifier,
                        identifier,
                        "source.md",
                        "Section",
                        1.0,
                        rank,
                    )
                    for rank, identifier in enumerate(
                        search_results[question], start=1
                    )
                ]

            report = evaluate_corpus(corpus, search)

        self.assertEqual(report.recall_at_1, 0.5)
        self.assertEqual(report.recall_at_2, 0.5)
        self.assertEqual(report.mrr, 1.0)
        self.assertEqual(
            [case.category for case in report.cases],
            ["answerable", "unanswered", "ambiguous"],
        )
        self.assertIsNone(report.cases[1].recall_at_1)
        self.assertIn("category=unanswered", format_report(report))
        self.assertIn("category=ambiguous", format_report(report))

    def test_malformed_category_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "eval.json"
            corpus.write_text(
                '[{"question": "q", "category": "unknown"}]',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "category"):
                load_eval_cases(corpus)

    def test_non_string_categories_are_rejected_clearly(self) -> None:
        for category in ([], {}, 1):
            with self.subTest(category=category):
                with tempfile.TemporaryDirectory() as directory:
                    corpus = Path(directory) / "eval.json"
                    corpus.write_text(
                        json.dumps([{"question": "q", "category": category}]),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, "category"):
                        load_eval_cases(corpus)

    def test_main_inspection_uses_injected_retrieval_and_skips_evaluation(
        self,
    ) -> None:
        class FakeRetriever:
            def __init__(
                self, chunks: list[DocumentChunk], provider: object
            ) -> None:
                self.chunks = chunks
                self.provider = provider

            def search(
                self, question: str, top_k: int
            ) -> list[RetrievedContext]:
                return [
                    RetrievedContext(
                        "fake-id", "fake text", "fake.md", "Fake", 0.9, 1
                    )
                ]

        class FakeProvider:
            pass

        from chitchat_nao.rag import evaluator

        output = StringIO()
        original_argv = sys.argv
        try:
            sys.argv = ["evaluator", "--question", "arbitrary question"]
            with redirect_stdout(output):
                evaluator.main(
                    provider_factory=FakeProvider,
                    retriever_factory=FakeRetriever,
                )
        finally:
            sys.argv = original_argv

        rendered = output.getvalue()
        self.assertIn("rank=1", rendered)
        self.assertIn("source=fake.md", rendered)
        self.assertIn("section=Fake", rendered)
        self.assertIn("text=fake text", rendered)

    def test_inspection_format_includes_ranked_context_fields(self) -> None:
        report = format_inspection(
            "club question",
            [RetrievedContext("id", "answer text", "faq.md", "FAQ", 0.75, 1)],
        )

        self.assertIn("rank=1", report)
        self.assertIn("score=0.750", report)
        self.assertIn("source=faq.md", report)
        self.assertIn("section=FAQ", report)
        self.assertIn("text=answer text", report)
