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
Find the most relevant document chunks
      |
      v
Give those chunks to a local language model
      |
      v
Print the answer and ranked source diagnostics
```

This system does not need a physical NAO robot, a network connection to NAO, or a web server. The laptop-side RAG code is designed to remain independent of the robot layer.

## RAG in plain language

RAG stands for **Retrieval-Augmented Generation**.

A language model can produce fluent answers, but it does not automatically know the current facts in this project’s Computer Club documents. RAG adds a retrieval step before generation:

1. Store the club information as small text chunks.
2. Convert the chunks into numeric representations called embeddings.
3. Convert the user’s question into an embedding too.
4. Find the chunks whose embeddings are most similar to the question.
5. Put only those retrieved chunks into the language-model prompt.
6. Generate an answer from that supplied context.

This is different from putting every document into one giant prompt. Retrieval selects a small set of evidence for each question.

### A few important terms

**Chunk** - A small piece of a document, usually a paragraph associated with a Markdown heading. The current system searches chunks rather than whole files.

**Embedding** - A list of numbers representing text in a way that is useful for similarity comparison. Texts with related meanings may have vectors that point in similar directions.

**Similarity score** - A number used to rank chunks for a question. The current implementation uses cosine similarity through a dot product of normalized vectors. A higher score means the embedding model considered the texts more related; it is *not* a calibrated probability of "correctness".

**Top-k** - The number of highest-ranked chunks returned. With the default `top-k=2`, the generator receives at most the two best retrieved chunks.

**Grounding** - The idea that an answer’s claims are supported by the supplied source context. For example, an answer is "grounded" when its claims are supported by the retrieved text. The current system exposes source evidence, but it does not yet prove semantic grounding for every generated claim.

**Hallucination** - An answer that sounds plausible but is unsupported or false. A central future goal is to make unsupported questions produce a cautious refusal rather than an invented answer.

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
GenerationRequest(question, contexts)
        |
        v
LocalLlamaCppGenerator -> local GGUF model
        |
        v
assistant.py -> answer + source diagnostics
```

The generator receives the question and retrieved contexts. It does **not** read the knowledge-base directory itself. This boundary is important: the displayed retrieval results describe the evidence that was actually supplied to generation.

The RAG code does not import NAOqi. A future robot adapter can consume a final text answer, but retrieval and generation can be developed and tested without the robot.

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

The GGUF is a local model artifact and is intentionally not committed to Git. If the file is absent, retrieval-only commands and the offline tests can still run, but the generation CLI cannot produce an answer.

The project has unrelated audio dependencies, including PyAudio. On systems without PortAudio headers, use the isolated `--no-project` commands below so RAG verification does not attempt to build the audio stack.

### Inspect retrieval without generation

This command answers the question: “Which chunks would the assistant retrieve?” It does not load the language model.

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  python -m chitchat_nao.rag.evaluator \
  --question "Who is the faculty advisor?" \
  --top-k 10
```

The output includes each result’s rank, similarity score, source path, section, and text.

### Run the retrieval evaluation

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  python -m chitchat_nao.rag.evaluator
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

The tests use deterministic fakes. They do not need the real embedding model, the GGUF file, network access, or a physical robot.

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

The assistant currently expects to be run from the repository root because it looks for `Path("knowledge_base")` relative to the current working directory.

The first embedding run may download `sentence-transformers/all-MiniLM-L6-v2` and may show an unauthenticated Hugging Face warning. That warning is about model access, not about retrieval correctness.

## How to read the output

The assistant prints ranked source diagnostics similar to:

```text
rank=1 score=0.717 source=faq.md id=chunk-... section=Frequently Asked Questions
rank=2 score=0.572 source=officers.md id=chunk-... section=Officers
```

Read these fields as follows:

- `rank=1` is the highest-ranked retrieved chunk;
- `score` is a raw cosine-similarity score, not a confidence percentage;
- `source` identifies the relative Markdown file;
- `id` is the deterministic canonical chunk ID;
- `section` is the heading context recorded during ingestion.

The source diagnostics prove which chunks retrieval selected. They do not prove that the model used them correctly or that every sentence in its answer is supported.

## What is implemented in code

The main implementation files are:

- [`models.py`](../../src/chitchat_nao/rag/models.py) — shared `DocumentChunk` and `RetrievedContext` data types;
- [`ingest.py`](../../src/chitchat_nao/rag/ingest.py) — heading-aware, paragraph-aware Markdown ingestion;
- [`embedding.py`](../../src/chitchat_nao/rag/embedding.py) — embedding interface and SentenceTransformer provider;
- [`retrieval.py`](../../src/chitchat_nao/rag/retrieval.py) — normalized in-memory cosine search;
- [`generation.py`](../../src/chitchat_nao/rag/generation.py) — local GGUF model boundary and prompt construction;
- [`assistant.py`](../../src/chitchat_nao/rag/assistant.py) — CLI orchestration and current citation validation;
- [`evaluator.py`](../../src/chitchat_nao/rag/evaluator.py) — KB loading, evaluation, and retrieval inspection.

The maintenance guide explains how these pieces interact in more detail.

## Current measured retrieval state

The latest documented evaluation used the real SentenceTransformer provider against the three-file starter corpus:

```text
Recall@1 = 0.700
Recall@2 = 1.000
MRR      = 0.850
```

All 10 answerable starter questions retrieved a gold chunk by rank two. This is evidence about retrieval, not about answer quality, refusal behavior, or semantic grounding.

> Here, a "gold" chunk means a chunk that the evaluation data identifies as the expected relevant source for a question.

## Current limitations (07/30/2026)

This section is intentionally prominent because a working demo can otherwise look more complete than it is.

### Generation output can currently be withheld

The prompt asks the model to cite retrieved snippets with labels such as `[S1]` (where `S` stands for "source"). The CLI currently accepts an answer only when every bracketed citation is a valid label for the current retrieval results. If the model omits or misformats a citation, the answer is replaced by:

```text
[answer withheld: citation structural validation failed]
```

The small local model has already been observed to fail this structural requirement even after successful retrieval. Citation enforcement is still to be implemented because of this issue. Relaxing or bypassing this gate is a possibility.

### Citation labels are not semantic proof

Even a valid `[S1]` label only shows that the model emitted an allowed source label. It does not prove that all claims in the answer are supported by that source. Semantic grounding evaluation remains future work.

### Unsupported and ambiguous questions have no answer policy yet

The evaluation corpus contains unanswered and ambiguous examples, but the assistant does not yet:

- refuse before generation when evidence is insufficient;
- ask for clarification when a question is ambiguous;
- apply a calibrated score threshold;
- distinguish retrieval failure from generation failure through a structured result.

These cases are currently inspection-only in the evaluator.

### The system is small and non-persistent

The retriever embeds all chunks at startup and compares each question with every chunk in memory. There is no vector database, persistence, reranking, filtering, or indexing.

### The CLI is starter-corpus oriented

The evaluator supports `--knowledge-base`, but the assistant currently uses the repository-root `knowledge_base/` path directly. A configurable assistant KB path is a possible future improvement.

### No robot or speech integration is included

There is no NAO I/O, ASR, robot speech call, FastAPI service, conversation memory, or end-to-end spoken interaction in this current iteration, although the absence of the robot doesn't block current retrieval work.

## Related documentation

- [`evaluation-and-maintenance.md`](evaluation-and-maintenance.md) — implementation details, evaluation data, tests, and safe update workflow

This guide is intended to remain a stable explanation of the current implementation and should be updated when anything pertaining to the RAG system changes.
