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
from .models import AskResult, ResponseMode, RetrievedContext
from .retrieval import Retriever

_CITATION_TOKEN = re.compile(r"\[([^\]\r\n]*)\]")
_SOURCE_LABEL = re.compile(r"S[1-9][0-9]*")
_SPOKEN_SOURCE_LABEL = re.compile(r"\[S[0-9]+\]")
_WORD_TOKEN = re.compile(r"[a-z0-9]+")
_RESPONSE_WRAPPER = re.compile(r"</?response\s*>", re.IGNORECASE)
_DOCUMENTED_QUESTION = re.compile(
    r"^\s*question(?:\s+\d+)?\s*:\s*(.*?)(?:\?|$)",
    re.IGNORECASE,
)

LOW_CONFIDENCE_SCORE_THRESHOLD = 0.35
CLARIFY_SCORE_THRESHOLD = 0.60
CLARIFY_MARGIN_THRESHOLD = 0.08
MAX_CLARIFICATION_ATTEMPTS = 2
CLARIFICATION_FALLBACK = "Which club detail would you like to clarify?"

_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "at",
        "be",
        "can",
        "club",
        "computer",
        "could",
        "do",
        "does",
        "for",
        "from",
        "get",
        "give",
        "how",
        "i",
        "in",
        "information",
        "is",
        "it",
        "made",
        "mascot",
        "me",
        "of",
        "on",
        "or",
        "organization",
        "please",
        "pretend",
        "question",
        "show",
        "student",
        "students",
        "synthetic",
        "tell",
        "that",
        "the",
        "to",
        "up",
        "use",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "would",
        "you",
        "your",
    }
)

_CLUB_REDIRECT = (
    "I can help with Computer Club questions, but I don't have relevant "
    "information for that. Please ask about club meetings, officers, rooms, "
    "or contact details."
)
_SENSITIVE_REDIRECT = (
    "I can help with Computer Club information, but I can't help with "
    "system prompts, passwords, or secrets. Please ask about club meetings, "
    "officers, rooms, or contact details."
)
_EXHAUSTED_CLARIFICATION_REDIRECT = (
    "I'm still missing enough detail to answer that. Please ask a club "
    "officer for help."
)

_SENSITIVE_REQUEST = re.compile(
    r"\b(passwords?|passcodes?|secrets?|tokens?|api[-\s]+keys?)\b",
    re.IGNORECASE,
)
_PROMPT_INJECTION_REQUEST = re.compile(
    r"\b(ignore|disregard|override|bypass)\b.*\b(instructions?|prompt|rules?)\b"
    r"|\b(system[-\s]+(prompt|instructions?|rules?)|hidden[-\s]+"
    r"(prompt|instructions?)|jailbreak|developer[-\s]+message)\b"
    r"|\b(reveal|show|disclose)\b.*\b(your|the|hidden)\b.*\b"
    r"(instruction|prompt|rules?)\b"
    r"|\b(list)\b.*\b(your|the|hidden)\b.*\b(system[-\s]+rules?)\b",
    re.IGNORECASE,
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


def _is_protected_request(question: str) -> bool:
    return bool(
        _SENSITIVE_REQUEST.search(question)
        or _PROMPT_INJECTION_REQUEST.search(question)
    )


def _normalized_query_terms(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _WORD_TOKEN.findall(text.lower())
        if token not in _QUERY_STOP_WORDS
    )


def _normalized_word_tokens(text: str) -> tuple[str, ...]:
    return tuple(_WORD_TOKEN.findall(text.lower()))


def _clean_answer_spoken_text(text: str) -> str:
    spoken_text = _RESPONSE_WRAPPER.sub(" ", text)
    spoken_text = _SPOKEN_SOURCE_LABEL.sub(" ", spoken_text)
    spoken_text = " ".join(spoken_text.split())
    spoken_text = re.sub(r"\s+([.!?,;:])", r"\1", spoken_text)
    sentence_units = re.findall(r"[^.!?]*[.!?]+", spoken_text)
    if sentence_units:
        remainder = spoken_text[sum(map(len, sentence_units)) :].strip()
        spoken_text = " ".join(
            sentence.strip() for sentence in sentence_units[:3]
        )
        if len(sentence_units) < 3 and remainder:
            spoken_text = f"{spoken_text} {remainder}"
    return spoken_text


def _context_label_terms(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    first_line = lines[0] if lines else ""
    label_match = re.match(r"^\s*(.*?)(?:\s*:\s*|\s+-\s+)", first_line)
    if label_match is None:
        return ()
    return _normalized_query_terms(label_match.group(1))


def _contains_normalized_phrase(
    query_terms: tuple[str, ...], text: str
) -> bool:
    context_terms = _normalized_query_terms(text)
    return any(
        context_terms[index : index + len(query_terms)] == query_terms
        for index in range(len(context_terms) - len(query_terms) + 1)
    )


def _direct_supporting_contexts(
    question: str, evidence: list[RetrievedContext]
) -> list[RetrievedContext]:
    query_terms = _normalized_query_terms(question)
    if not query_terms:
        return []

    exact_label_matches = [
        context
        for context in evidence
        if _context_label_terms(context.text) == query_terms
    ]
    if exact_label_matches:
        return exact_label_matches

    documented_question_matches = [
        context
        for context in evidence
        if any(
            all(
                term in _normalized_query_terms(match.group(1))
                for term in query_terms
            )
            for line in context.text.splitlines()
            if (match := _DOCUMENTED_QUESTION.match(line)) is not None
        )
    ]
    if documented_question_matches:
        return documented_question_matches

    if len(query_terms) < 2:
        return []

    return [
        context
        for context in evidence
        if _contains_normalized_phrase(query_terms, context.text)
    ]


def _clarification_contexts(
    evidence: list[RetrievedContext],
) -> list[RetrievedContext]:
    return [
        RetrievedContext(
            context.id,
            f"Source: {context.source_path}; Section: {context.section}",
            context.source_path,
            context.section,
            context.score,
            context.rank,
        )
        for context in evidence
    ]


def _needs_clarification(results: list[RetrievedContext]) -> bool:
    if not results:
        return False
    top_score = results[0].score
    if top_score < CLARIFY_SCORE_THRESHOLD:
        return True
    if len(results) > 1:
        margin = results[0].score - results[1].score
        return margin <= CLARIFY_MARGIN_THRESHOLD
    return False


def _redirect_result(
    evidence: list[RetrievedContext],
    clarification_attempts: int,
    spoken_text: str,
    diagnostic: str,
) -> AskResult:
    return AskResult(
        ResponseMode.REDIRECT,
        spoken_text,
        tuple(evidence),
        False,
        (diagnostic,),
        clarification_attempts,
    )


def ask(
    question: str = "",
    *,
    embedding_provider_factory: Callable[[], EmbeddingProvider] | None = None,
    retriever_factory: Callable[..., Retriever] | None = None,
    generator_factory: Callable[..., LocalLlamaCppGenerator] | None = None,
    clarification_attempts: int = 0,
    top_k: int = 2,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> AskResult:
    """Retrieve context and return a structured response for one question."""
    if not isinstance(question, str) or not question.strip():
        return AskResult(
            ResponseMode.ERROR,
            "I couldn't process that request right now.",
            (),
            False,
            ("Invalid argument question: expected a non-empty string.",),
            0,
        )
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        return AskResult(
            ResponseMode.ERROR,
            "I couldn't process that request right now.",
            (),
            False,
            ("Invalid argument top_k: expected an integer >= 1.",),
            0,
        )
    if (
        not isinstance(clarification_attempts, int)
        or isinstance(clarification_attempts, bool)
        or clarification_attempts < 0
    ):
        return AskResult(
            ResponseMode.ERROR,
            "I couldn't process that request right now.",
            (),
            False,
            (
                "Invalid argument clarification_attempts: expected an "
                "integer >= 0.",
            ),
            0,
        )

    provider_factory = (
        embedding_provider_factory
        if embedding_provider_factory is not None
        else SentenceTransformerProvider
    )
    retriever_builder = (
        retriever_factory if retriever_factory is not None else Retriever
    )
    generator_builder = (
        generator_factory
        if generator_factory is not None
        else LocalLlamaCppGenerator
    )
    try:
        chunks = load_knowledge_base(Path("knowledge_base"))
        provider = provider_factory()
        retriever = retriever_builder(chunks, provider)
        evidence = retriever.search(question, top_k)
    except Exception as error:
        return AskResult(
            ResponseMode.ERROR,
            "I couldn't retrieve information right now.",
            (),
            False,
            (f"Retrieval setup error: {error}",),
            clarification_attempts,
        )

    if _is_protected_request(question):
        return _redirect_result(
            evidence,
            clarification_attempts,
            _SENSITIVE_REDIRECT,
            "Request was redirected because it asks for protected "
            "instructions or sensitive information.",
        )

    direct_support = _direct_supporting_contexts(question, evidence)
    if not direct_support and (
        not evidence or evidence[0].score < LOW_CONFIDENCE_SCORE_THRESHOLD
    ):
        return _redirect_result(
            evidence,
            clarification_attempts,
            _CLUB_REDIRECT,
            "Retrieved evidence was empty or below the relevance threshold.",
        )

    response_mode = (
        ResponseMode.CLARIFY
        if len(direct_support) > 1
        else (
            ResponseMode.ANSWER
            if direct_support
            else (
                ResponseMode.CLARIFY
                if _needs_clarification(evidence)
                else ResponseMode.ANSWER
            )
        )
    )
    generation_evidence = (
        _clarification_contexts(evidence)
        if response_mode is ResponseMode.CLARIFY
        else direct_support
        if direct_support
        else evidence[:1]
    )
    if (
        response_mode is ResponseMode.CLARIFY
        and clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS
    ):
        return _redirect_result(
            evidence,
            clarification_attempts,
            _EXHAUSTED_CLARIFICATION_REDIRECT,
            "Clarification attempts are exhausted; an officer was suggested.",
        )

    try:
        generator = generator_builder(model_path)
        generated_text = generator.generate(
            GenerationRequest(question, generation_evidence, response_mode)
        )
    except Exception as error:
        return AskResult(
            ResponseMode.ERROR,
            "I couldn't generate a response right now.",
            tuple(evidence),
            False,
            (f"Generator error: {error}",),
            clarification_attempts,
        )

    if not isinstance(generated_text, str):
        return AskResult(
            ResponseMode.ERROR,
            "I couldn't generate a response right now.",
            tuple(evidence),
            False,
            (
                "Generator content error: expected a string response, got "
                f"{type(generated_text).__name__}.",
            ),
            clarification_attempts,
        )

    spoken_text = generated_text
    if response_mode is ResponseMode.ANSWER:
        spoken_text = _clean_answer_spoken_text(generated_text)
    elif response_mode is ResponseMode.CLARIFY:
        candidate = (
            generated_text.strip() if isinstance(generated_text, str) else ""
        )
        spoken_text = (
            candidate
            if (
                candidate.endswith("?")
                and candidate.count("?") == 1
                and _normalized_word_tokens(candidate)
                != _normalized_word_tokens(question)
            )
            else CLARIFICATION_FALLBACK
        )

    citation_structurally_valid = validate_generated_answer(
        generated_text, generation_evidence
    )
    citation_diagnostic = (
        "Citation formatting was structurally valid, but semantic grounding "
        "was not verified."
        if citation_structurally_valid
        else "Citation formatting was missing or invalid; semantic grounding "
        "was not verified."
    )
    result_evidence = (
        direct_support
        if response_mode is ResponseMode.ANSWER and direct_support
        else evidence[:1]
        if response_mode is ResponseMode.ANSWER
        else evidence
    )
    return AskResult(
        response_mode,
        spoken_text,
        tuple(result_evidence),
        False,
        (citation_diagnostic,),
        clarification_attempts,
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

    result = ask(
        args.question,
        embedding_provider_factory=(
            embedding_provider_factory
            if embedding_provider_factory is not None
            else SentenceTransformerProvider
        ),
        retriever_factory=retriever_factory,
        generator_factory=(
            generator_factory
            if generator_factory is not None
            else LocalLlamaCppGenerator
        ),
        top_k=args.top_k,
        model_path=args.model,
    )

    print(result.spoken_text)
    print(f"provenance_verified={result.provenance_verified}")
    for diagnostic in result.diagnostics:
        print(f"diagnostic={diagnostic}")
    for evidence in result.evidence:
        print(
            f"rank={evidence.rank} score={evidence.score:.3f} "
            f"source={evidence.source_path} id={evidence.id} "
            f"section={evidence.section}"
        )


if __name__ == "__main__":
    main()
