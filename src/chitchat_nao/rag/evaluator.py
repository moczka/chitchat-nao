import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .embedding import EmbeddingProvider
from .embedding import SentenceTransformerProvider
from .ingest import ingest_markdown
from .models import DocumentChunk, RetrievedContext
from .retrieval import Retriever


@dataclass(frozen=True)
class EvalCase:
    question: str
    relevant_chunk_ids: list[str]
    category: str = "answerable"
    expected_answer_contains: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalCaseResult:
    question: str
    retrieved_ids: list[str]
    retrieved_sources: list[str]
    recall_at_1: float | None
    recall_at_2: float | None
    reciprocal_rank: float | None
    category: str = "answerable"
    top1_score: float | None = None
    top2_score: float | None = None
    top2_margin: float | None = None
    has_gold_hit: bool | None = None


@dataclass(frozen=True)
class EvaluationReport:
    cases: list[EvalCaseResult]
    recall_at_1: float
    recall_at_2: float
    mrr: float


def load_eval_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("evaluation corpus must be a JSON list")
    cases: list[EvalCase] = []
    valid_categories = {"answerable", "unanswered", "ambiguous"}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each evaluation case must be an object")
        question = item.get("question")
        category = item.get("category")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("each evaluation case needs a non-empty question")
        if not isinstance(category, str) or category not in valid_categories:
            raise ValueError("each evaluation case needs a valid category")
        relevant_ids = item.get("relevant_chunk_ids", [])
        if not isinstance(relevant_ids, list) or not all(
            isinstance(identifier, str) and identifier
            for identifier in relevant_ids
        ):
            raise ValueError(
                "relevant_chunk_ids must be a list of non-empty strings"
            )
        if category == "answerable" and not relevant_ids:
            raise ValueError("answerable cases need relevant_chunk_ids")
        if category != "answerable" and relevant_ids:
            raise ValueError(
                f"{category} case for question {question!r} cannot include "
                "relevant_chunk_ids"
            )
        expected_phrases = item.get("expected_answer_contains", [])
        if not isinstance(expected_phrases, list) or not all(
            isinstance(phrase, str) and phrase.strip()
            for phrase in expected_phrases
        ):
            raise ValueError(
                "expected_answer_contains must be a list of non-empty strings"
            )
        if category == "answerable" and not expected_phrases:
            raise ValueError(
                f"answerable case for question {question!r} needs "
                "expected_answer_contains"
            )
        if category != "answerable" and expected_phrases:
            raise ValueError(
                f"{category} case for question {question!r} cannot include "
                "expected_answer_contains"
            )
        cases.append(
            EvalCase(question, relevant_ids, category, expected_phrases)
        )
    return cases


def load_knowledge_base(knowledge_base: Path) -> list[DocumentChunk]:
    root = knowledge_base.resolve()
    chunks: list[DocumentChunk] = []
    for path in sorted(root.glob("**/*.md")):
        chunks.extend(ingest_markdown(path, path.relative_to(root).as_posix()))
    return chunks


def evaluate_corpus(
    path: Path,
    search: Callable[[str], list[RetrievedContext]],
) -> EvaluationReport:
    cases = load_eval_cases(path)
    results: list[EvalCaseResult] = []
    recalls_1: list[float] = []
    recalls_2: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        retrieved = search(case.question)
        ids = [item.id for item in retrieved]
        sources = [item.source_path for item in retrieved]
        if case.category != "answerable":
            results.append(
                EvalCaseResult(
                    case.question,
                    ids,
                    sources,
                    None,
                    None,
                    None,
                    case.category,
                )
            )
            continue
        relevant = set(case.relevant_chunk_ids)
        position = next(
            (
                index + 1
                for index, item_id in enumerate(ids)
                if item_id in relevant
            ),
            None,
        )
        reciprocal = 1.0 / position if position is not None else 0.0
        recall_at_1 = len(set(ids[:1]) & relevant) / len(relevant)
        recall_at_2 = len(set(ids[:2]) & relevant) / len(relevant)
        top1_score = retrieved[0].score if retrieved else None
        top2_score = retrieved[1].score if len(retrieved) > 1 else None
        # Keep the diagnostic margin as the raw top1 - top2 score difference.
        top2_margin = (
            top1_score - top2_score
            if top1_score is not None and top2_score is not None
            else None
        )
        results.append(
            EvalCaseResult(
                case.question,
                ids,
                sources,
                recall_at_1,
                recall_at_2,
                reciprocal,
                case.category,
                top1_score,
                top2_score,
                top2_margin,
                bool(set(ids) & relevant),
            )
        )
        recalls_1.append(recall_at_1)
        recalls_2.append(recall_at_2)
        reciprocal_ranks.append(reciprocal)
    count = len(recalls_1) or 1
    return EvaluationReport(
        results,
        sum(recalls_1) / count,
        sum(recalls_2) / count,
        sum(reciprocal_ranks) / count,
    )


def format_report(report: EvaluationReport) -> str:
    lines: list[str] = []
    for case in report.cases:
        lines.append(f"QUESTION {case.question}")
        if case.category != "answerable":
            lines.append(
                f"  category={case.category} "
                f"retrieved_ids={case.retrieved_ids} "
                f"sources={case.retrieved_sources} inspection_only=true"
            )
            continue
        lines.append(
            f"  retrieved_ids={case.retrieved_ids} "
            f"sources={case.retrieved_sources} "
            f"top1_score={case.top1_score} "
            f"top2_score={case.top2_score} "
            f"top2_margin={case.top2_margin} "
            f"gold_hit={case.has_gold_hit} "
            f"Recall@1={case.recall_at_1:.3f} "
            f"Recall@2={case.recall_at_2:.3f} "
            f"MRR={case.reciprocal_rank:.3f}"
        )
    lines.append(
        f"Recall@1={report.recall_at_1:.3f} "
        f"Recall@2={report.recall_at_2:.3f} MRR={report.mrr:.3f}"
    )
    return "\n".join(lines)


def format_inspection(question: str, results: list[RetrievedContext]) -> str:
    lines = [f"QUESTION {question}"]
    for result in results:
        lines.append(
            f"rank={result.rank} score={result.score:.3f} "
            f"source={result.source_path} section={result.section} "
            f"text={result.text}"
        )
    return "\n".join(lines)


def main(
    provider_factory: Callable[[], EmbeddingProvider] | None = None,
    retriever_factory: Callable[..., Retriever] = Retriever,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval against the preliminary Markdown "
            "knowledge base."
        )
    )
    parser.add_argument(
        "--knowledge-base", type=Path, default=Path("knowledge_base")
    )
    parser.add_argument(
        "--eval-corpus",
        type=Path,
        default=Path(__file__).with_name("eval_corpus.json"),
    )
    parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--question")
    args = parser.parse_args()
    chunks = load_knowledge_base(args.knowledge_base)
    provider = (
        provider_factory()
        if provider_factory is not None
        else SentenceTransformerProvider(args.model)
    )
    retriever = retriever_factory(chunks, provider)
    if args.question is not None:
        print(
            format_inspection(
                args.question, retriever.search(args.question, args.top_k)
            )
        )
        return
    report = evaluate_corpus(
        args.eval_corpus,
        lambda question: retriever.search(question, args.top_k),
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
