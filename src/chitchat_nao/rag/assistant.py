"""Command-line assistant combining the existing retrieval components."""

import argparse
import re
from pathlib import Path
from typing import Callable

from .embedding import EmbeddingProvider, SentenceTransformerProvider
from .evaluator import load_knowledge_base
from .generation import (
    DEFAULT_MODEL_PATH,
    GenerationRequest,
    LocalLlamaCppGenerator,
)
from .models import RetrievedContext
from .retrieval import Retriever

_CITATION_TOKEN = re.compile(r"\[([^\]\r\n]*)\]")
_SOURCE_LABEL = re.compile(r"S[1-9][0-9]*")
STRUCTURAL_VALIDATION_FALLBACK = (
    "[answer withheld: citation structural validation failed]"
)


def validate_generated_answer(
    answer: str, results: list[RetrievedContext]
) -> bool:
    """Allow only source labels from this request's retrieval results."""
    if not results:
        return False
    citations = _CITATION_TOKEN.findall(answer)
    if not citations:
        return False
    unmatched_brackets = _CITATION_TOKEN.sub("", answer)
    if "[" in unmatched_brackets or "]" in unmatched_brackets:
        return False
    allowed_labels = {
        f"S{position}" for position in range(1, len(results) + 1)
    }
    return all(
        _SOURCE_LABEL.fullmatch(citation) is not None
        and citation in allowed_labels
        for citation in citations
    )


def main(
    *,
    embedding_provider_factory: Callable[[], EmbeddingProvider] | None = None,
    generator_factory: Callable[[Path], LocalLlamaCppGenerator] | None = None,
    retriever_factory: Callable[..., Retriever] = Retriever,
) -> None:
    parser = argparse.ArgumentParser(
        description="Ask the local RAG assistant."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    chunks = load_knowledge_base(Path("knowledge_base"))
    provider = (
        embedding_provider_factory()
        if embedding_provider_factory is not None
        else SentenceTransformerProvider()
    )
    retriever = retriever_factory(chunks, provider)
    results = retriever.search(args.question, args.top_k)
    if not results:
        print(STRUCTURAL_VALIDATION_FALLBACK)
        return

    generator = (
        generator_factory(args.model)
        if generator_factory is not None
        else LocalLlamaCppGenerator(args.model)
    )
    answer = generator.generate(GenerationRequest(args.question, results))

    print(
        answer
        if validate_generated_answer(answer, results)
        else STRUCTURAL_VALIDATION_FALLBACK
    )
    for result in results:
        print(
            f"rank={result.rank} score={result.score:.3f} "
            f"source={result.source_path} id={result.id} "
            f"section={result.section}"
        )


if __name__ == "__main__":
    main()
