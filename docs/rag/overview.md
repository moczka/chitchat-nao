# RAG System - Overview

This guide explains the current Retrieval-Augmented Generation (RAG) system in our project. You don't need any prior RAG/ML experience!

It describes the implementation that currently exists in this repository. It does not describe planned behavior/improvements.

For maintenance details, evaluation data, and test coverage, see [`evaluation-and-maintenance.md`](evaluation-and-maintenance.md).

## What the system does

The current system is a local, text-based Computer Club assistant:

```text
Type a question
      |
      v
Read the local Markdown knowledge base
      |
      v
Find the most relevant document chunks (top-k)
      |
      v
Route the question: answer / clarify / redirect / error
      |
      v
Give only the selected evidence to a local language model
      |
      v
Return a typed AskResult
      |
      v
Print the speech text, then provenance + diagnostics + ranked sources
```

This system does not need a physical NAO robot, a network connection to NAO, or a web server. The laptop-side RAG code is designed to remain independent of the robot layer.

## RAG in plain language

RAG stands for **Retrieval-Augmented Generation**.

A language model can produce fluent answers, but it does not automatically know the current facts in this project's Computer Club documents. RAG adds a retrieval step before generation:

1. Store the club information as small text chunks.
2. Convert the chunks into numeric representations called embeddings.
3. Convert the user's question into an embedding too.
4. Find the chunks whose embeddings are most similar to the question.
5. Put only those retrieved chunks into the language-model prompt.
6. Generate an answer from that supplied context.

This is different from putting every document into one giant prompt. Retrieval selects a small set of evidence for each question.

### A few important terms

**Chunk** - A small piece of a document, usually a paragraph associated with a Markdown heading. The current system searches chunks rather than whole files.

**Embedding** - A list of numbers representing text in a way that is useful for similarity comparison. Texts with related meanings may have vectors that point in similar directions.

**Similarity score** - A number used to rank chunks for a question. The current implementation uses cosine similarity through a dot product of normalized vectors. A higher score means the embedding model considered the texts more related; it is *not* a calibrated probability of "correctness".

**Top-k** - The number of highest-ranked chunks returned. With the default `top-k=2`, the retriever returns at most the two best chunks. The routing logic then decides which (if any) of those chunks may support an answer.

**Response mode** - The typed decision the system returns for a question: `ANSWER`, `CLARIFY`, `REDIRECT`, or `ERROR`. This is the integration contract a future robot adapter should switch on.

**Grounding** - The idea that an answer's claims are supported by the supplied source context. The current system does **not** verify grounding: `provenance_verified` is always `False`, and no diagnostic should be read as proof that every sentence of the answer is supported by the cited text.

**Hallucination** - An answer that sounds plausible but is unsupported or false. The routing policy reduces (but does not eliminate) the risk of unsupported answers by redirecting low-confidence queries and clarifying ambiguous ones before generation.

## Current architecture

```text
knowledge_base/*.md
        |
        v
evaluator.load_knowledge_base()
        |
        v
ingest_markdown() -> DocumentChunk objects
        |
        v
SentenceTransformerProvider
        |
        v
Retriever -> top-k RetrievedContext objects
        |
        v
assistant.ask(question, ...) -> route the evidence
        |
        v
GenerationRequest(question, selected contexts, mode)
        |
        v
LocalLlamaCppGenerator -> local GGUF model
        |
        v
AskResult (mode, spoken_text, evidence, provenance_verified, diagnostics)
        |
        v
CLI prints speech first, then provenance/diagnostics/ranked sources
```

`ask()` is the typed, stateless integration seam: a caller passes a question and gets back an `AskResult`. The generator receives the question and the **selected** retrieved contexts only. It does **not** read the knowledge-base directory itself. This boundary is important: for `ANSWER`, the result's `evidence` describes exactly the context supplied to generation; for `CLARIFY`, `AskResult.evidence` deliberately retains the original retrieved contexts for diagnostics, while the generator receives sanitized copies containing only `Source:`/`Section:` text, with no chunk contents.

The RAG code does not import NAOqi. A future robot adapter can consume `AskResult.spoken_text` and the mode field, but retrieval, routing, and generation can be developed and tested without the robot.

## The active knowledge base

The current knowledge base contains three Markdown files:

| File | Current information |
|---|---|
| `knowledge_base/club_overview.md` | The club is at Quincy College and lists Dr. Robert Pitts as faculty advisor. |
| `knowledge_base/faq.md` | Any Quincy College student and alumni may join, regardless of major or year. |
| `knowledge_base/officers.md` | President, vice president, treasurer, and secretary responsibilities. |

The current files produce seven meaningful chunks:

- one overview/advisor chunk;
- one FAQ/joining chunk;
- one officer-board introduction chunk;
- one president chunk;
- one vice-president chunk;
- one treasurer chunk;
- one secretary chunk.

This is a very small starter corpus and is to be expanded upon.

## Quick start

### Prerequisites

- Python 3.13;
- `uv`;
- the project checkout;
- access to download the SentenceTransformer model on first use;
- a local SmolLM2 GGUF file for generation.

The current generation default is:

```text
~/.local/share/chitchat-nao/models/SmolLM2-1.7B-Instruct-Q6_K.gguf
```

The GGUF is a local model artifact and is intentionally not committed to Git. If the file is absent, retrieval-only commands and the offline tests can still run, but the generation path cannot produce an answer.

The project has unrelated audio dependencies, including PyAudio. On systems without PortAudio headers, the `.venv` may be missing `sentence-transformers` because the audio stack failed to resolve. Use the isolated `--no-project` commands below for all RAG checks; do not try to rebuild the project environment just for RAG.

### Inspect retrieval without generation

This command answers the question: "Which chunks would the assistant retrieve?" It does not load the language model.

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  python -m chitchat_nao.rag.evaluator \
  --question "Who is the faculty advisor?" \
  --top-k 10
```

The output includes each result's rank, similarity score, source path, section, and text.

### Run the retrieval evaluation

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  python -m chitchat_nao.rag.evaluator \
  --knowledge-base knowledge_base --top-k 2
```

This runs the cases in `src/chitchat_nao/rag/eval_corpus.json` and prints per-case diagnostics plus Recall@1, Recall@2, and MRR.

### Run the offline tests

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  --with 'llama-cpp-python>=0.3.28' \
  python -m unittest discover -s tests -p 'test_*.py'
```

The tests use deterministic fakes and an isolated synthetic Markdown corpus. They do not need the real embedding model, the GGUF file, network access, or a physical robot. The suite currently contains **77 tests**.

### Run the local generation CLI

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  --with 'llama-cpp-python>=0.3.28' \
  python -m chitchat_nao.rag.assistant \
  --question "Who is the faculty advisor?" \
  --top-k 2 \
  --model "$HOME/.local/share/chitchat-nao/models/SmolLM2-1.7B-Instruct-Q6_K.gguf"
```

The assistant currently expects to be run from the repository root because it looks for `Path("knowledge_base")` relative to the current working directory. Only `--question`, `--model`, and `--top-k` exist; there is no `--knowledge-base` option for the assistant yet (the evaluator has one).

The first embedding run may download `sentence-transformers/all-MiniLM-L6-v2` and may show an unauthenticated Hugging Face warning. That warning is about model access, not about retrieval correctness, and it is completely non-blocking.

## How to read the output

The assistant prints the speech-ready text first, then provenance state, diagnostics, and ranked source evidence:

```text
Dr. Robert Pitts is the faculty advisor.
provenance_verified=False
diagnostic=Citation formatting was structurally valid, but semantic grounding was not verified.
rank=2 score=0.457 source=club_overview.md id=chunk-... section=Club Overview
```

Read these fields as follows:

- the first line is the **speech text**. `[S<n>]` source labels and outer `<response>` tags are stripped, and the text is capped at three sentences;
- `provenance_verified=False` is **always** printed. It never means an answer is verified; it means semantic grounding was not checked;
- `diagnostic=` explains the citation state and reminds the reader that grounding is unverified;
- `rank=` is the retrieval position (here rank 2 was the selected support);
- `score` is a raw cosine-similarity score, not a confidence percentage;
- `source` identifies the relative Markdown file;
- `id` is the deterministic canonical chunk ID;
- `section` is the heading context recorded during ingestion.

The diagnostics and ranked sources prove which chunks were selected: for `ANSWER`, the contexts actually supplied to generation; for `CLARIFY`, the original retrieved contexts are retained for diagnostics, while generation receives only `Source:`/`Section:` text. They do not prove that the model used them correctly or that every sentence in the answer is supported.

> **Note on `provenance_verified`.** The field is always `False` in the current implementation: no semantic verification is performed, and no output line should be read as proof that the answer is supported by the cited text. Its current purpose is explicit transparency — the reader can see that verification did not run — and a stable API seam for future work. A human review that happens *after* the CLI output is a separate review record; it cannot retroactively mark the already-returned `AskResult` as verified, because that object was constructed with `provenance_verified=False`. Only a future verifier that runs *before* the result is built could set the field to `True`.

## The `ask()` contract

`ask(question, ...)` in `assistant.py` is the typed, stateless integration seam:

```python
from chitchat_nao.rag import ask
result = ask("Who is the faculty advisor?")
```

The returned `AskResult` is a frozen dataclass with:

| Field | Meaning |
|---|---|
| `mode` | `ResponseMode`: `ANSWER`, `CLARIFY`, `REDIRECT`, or `ERROR` |
| `spoken_text` | The speech-ready string a robot/CLI should speak |
| `evidence` | Tuple of `RetrievedContext`. For `ANSWER`, the contexts actually supplied to generation; for `CLARIFY`, the original retrieved contexts retained for diagnostics (the generator receives sanitized `Source:`/`Section:`-only copies) |
| `provenance_verified` | Always `False` in the current implementation |
| `diagnostics` | Tuple of human-readable diagnostic strings |
| `clarification_attempts` | The caller-supplied count, echoed back unchanged |

`ask()` is stateless: nothing persists between calls, and the caller owns the clarification budget. If the result mode is `CLARIFY` and the caller has already used two attempts (`clarification_attempts >= 2`), the next call returns `REDIRECT` with a message suggesting the user ask a club officer.

Validation errors (missing/non-string/empty/whitespace question, invalid `top_k`, invalid `clarification_attempts`) return a structured `ERROR` without touching retrieval or generation. Setup, retrieval, and generator failures also return `ERROR` with a diagnostic and with whatever evidence was available.

## Deterministic routing rules

`ask()` applies a fixed, deterministic decision sequence:

| Situation | Route |
|---|---|
| Invalid input (bad question, `top_k`, or attempts) | `ERROR` |
| Request asks for passwords, secrets, or hidden system prompts | `REDIRECT` |
| Exactly one retrieved chunk gives unique direct textual support (label, documented question, or phrase match - even at rank 2) | `ANSWER` using only that chunk |
| More than one chunk gives competing direct support | `CLARIFY` |
| No direct support and the top score is below 0.35 | `REDIRECT` |
| No direct support, top score >= 0.60, and (single result or margin (top1 - top2) > 0.08) | `ANSWER` using rank-1 evidence only |
| No direct support, otherwise (weak or competing semantics) | `CLARIFY` |
| `CLARIFY` would be chosen but attempts are exhausted (>= 2) | `REDIRECT` (suggest an officer) |
| Setup, retrieval, or generator failure; non-string generator output | `ERROR` |

The thresholds live as constants in `assistant.py`:

```text
LOW_CONFIDENCE_SCORE_THRESHOLD = 0.35   # below this, redirect
CLARIFY_SCORE_THRESHOLD        = 0.60   # at/above this with clear margin, answer
CLARIFY_MARGIN_THRESHOLD       = 0.08   # margin must exceed this to answer
MAX_CLARIFICATION_ATTEMPTS     = 2      # caller-owned budget
```

When `CLARIFY` is chosen, the generator receives only `Source:`/`Section:` text for each retrieved chunk (no chunk contents), so it cannot leak or repeat facts. If the model echoes the question or produces anything other than a single clarifying question, the output falls back to:

```text
Which club detail would you like to clarify?
```

The direct-textual-support matcher is deliberately small and generic: normalized token/phrase matching and `Question N:` label matching against the retrieved text. It is not an NLP classifier and not a re-ranker.

## What is implemented in code

The main implementation files are:

- [`models.py`](../../src/chitchat_nao/rag/models.py) — shared `DocumentChunk`, `RetrievedContext`, `AskResult`, and `ResponseMode` types;
- [`ingest.py`](../../src/chitchat_nao/rag/ingest.py) — heading-aware, paragraph-aware Markdown ingestion;
- [`embedding.py`](../../src/chitchat_nao/rag/embedding.py) — embedding interface and SentenceTransformer provider;
- [`retrieval.py`](../../src/chitchat_nao/rag/retrieval.py) — normalized in-memory cosine search;
- [`generation.py`](../../src/chitchat_nao/rag/generation.py) — local GGUF model boundary, mode-aware prompt construction;
- [`assistant.py`](../../src/chitchat_nao/rag/assistant.py) — `ask()` routing policy, speech cleanup, citation diagnostics, CLI;
- [`evaluator.py`](../../src/chitchat_nao/rag/evaluator.py) — KB loading, evaluation, and retrieval inspection.

The maintenance guide explains how these pieces interact in more detail.

## Current measured retrieval state

The latest documented evaluation used the real SentenceTransformer provider against the three-file starter corpus:

```text
Recall@1 = 0.600
Recall@2 = 1.000
MRR      = 0.800
```

All 10 answerable starter questions retrieved a gold chunk by rank two. (Two president-related gold IDs in `eval_corpus.json` were corrected to the current canonical chunk ID; the numbers above reflect the corrected labels.) These numbers are evidence about retrieval, not about answer quality, refusal behavior, or semantic grounding.

> Here, a "gold" chunk means a chunk that the evaluation data identifies as the expected relevant source for a question.

## Current limitations (07/31/2026)

This section is intentionally prominent because a working demo can otherwise look more complete than it is.

### Citation state is diagnostic-only, not semantic proof

`provenance_verified` is always `False`. The diagnostic only reports whether the model's citation *formatting* was structurally valid. It never proves that all claims in the answer are supported by the cited source. Semantic grounding evaluation remains future work.

### The small model may paraphrase awkwardly

Even when generation receives a selected direct-support chunk, the small SmolLM2 model can paraphrase names or facts awkwardly (for example, rendering an officer's name imperfectly). The routing policy cannot detect this; diagnostics and sources do not verify the model's statements. Check spoken output manually when it matters.

### Unsupported and ambiguous questions rely on heuristics

The routing policy is deterministic but heuristic. It uses score thresholds and simple textual matching, not calibrated confidence or a learned re-ranker. It will not catch every unsupported question or every ambiguity, and the thresholds were not tuned on a large corpus.

### The system is small and non-persistent

The retriever embeds all chunks at startup and compares each question with every chunk in memory. There is no vector database, persistence, reranking, filtering, or indexing.

### The CLI is starter-corpus oriented

The assistant uses the repository-root `knowledge_base/` path directly. A configurable assistant KB path is a possible future improvement.

### No robot or speech integration is included

There is no NAO I/O, ASR, robot speech call, FastAPI service, conversation memory, or end-to-end spoken interaction in this current iteration. The `AskResult` contract (especially `mode` and `spoken_text`) is the intended seam for that future work.

## Related documentation

- [`evaluation-and-maintenance.md`](evaluation-and-maintenance.md) — implementation details, evaluation data, tests, and safe update workflow

This guide is intended to remain a stable explanation of the current implementation and should be updated when anything pertaining to the RAG system changes.
