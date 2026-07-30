# RAG System - Evaluation and Maintenance Guide

This guide is for contributors who need to understand, test, or safely modify the current RAG implementation. It assumes basic Python but little prior ML/RAG experience.

Start with [`overview.md`](overview.md) if you want the shorter conceptual explanation and quick-start commands first.

## Source-of-truth boundaries

The current RAG system has four related kinds of files:

1. **Knowledge sources** under `knowledge_base/` — the club facts that can be retrieved.
2. **Pipeline code** under `src/chitchat_nao/rag/` — ingestion, embeddings, retrieval, generation, and CLI/evaluation orchestration.
3. **Evaluation labels** in `src/chitchat_nao/rag/eval_corpus.json` — questions and gold chunk IDs used to measure retrieval.
4. **Tests** under `tests/` — deterministic checks of the pipeline mechanics.

The knowledge sources and evaluation labels are tightly coupled through deterministic chunk IDs. Treat changes to them as one review unit.

## Module map

| File | Responsibility | Input | Output |
|---|---|---|---|
| `models.py` | Shared immutable data contracts | Field values | `DocumentChunk`, `RetrievedContext` |
| `ingest.py` | Parse Markdown into chunks | One Markdown path | `list[DocumentChunk]` |
| `embedding.py` | Convert text to normalized vectors | Text list | NumPy array |
| `retrieval.py` | Rank chunks by cosine similarity | Chunks and question | `list[RetrievedContext]` |
| `generation.py` | Ask local GGUF model for text | Question and contexts | Answer string |
| `assistant.py` | User-facing pipeline CLI | CLI arguments | Answer/fallback and diagnostics |
| `evaluator.py` | Load KB and score retrieval | KB, eval corpus, questions | Reports or inspection output |
| `eval_corpus.json` | Retrieval labels | Questions and gold IDs | Input to evaluator |

The main data flow is:

```text
Markdown files
  -> DocumentChunk
  -> normalized chunk embeddings
  -> Retriever.search(question)
  -> RetrievedContext
  -> GenerationRequest
  -> local model answer
```

## How ingestion works

`ingest_markdown()` in `ingest.py` reads one UTF-8 Markdown file.

### Heading and paragraph rules

- H1 through H6 headings are recognized.
- Headings update the current section label.
- Nested headings are joined with ` > `, for example `Club > Officers`.
- Headings themselves are metadata, not chunk text.
- Non-empty lines are collected as content.
- A blank line ends the current content block and creates a chunk.
- Consecutive blank lines do not create empty chunks.
- A document with no non-empty content raises `ValueError`.

For example:

```markdown
## Officers

President - Jane Doe
- Directs the club

Treasurer - Alex Smith
- Manages finances
```

creates two text chunks under the `Officers` section.

### Chunk identity

The chunk’s `content_hash` is the SHA-256 hash of the chunk text. The canonical chunk ID is another SHA-256 hash over:

```text
relative source path
section
chunk index
chunk text
```

The resulting ID is prefixed with `chunk-`.

The index is part of the identity so two identical paragraphs in the same document still receive different deterministic IDs. This also means that a formatting-only change can have evaluation consequences:

- inserting a blank line can split one chunk into two;
- removing a blank line can merge chunks;
- changing a heading changes the section value;
- renaming or moving a file changes the source path;
- changing wording changes the text hash and canonical ID.

If any of those happen, existing gold IDs in `eval_corpus.json` may no longer exist.

## How knowledge-base loading keeps IDs reproducible

`load_knowledge_base()` in `evaluator.py`:

1. resolves the KB root;
2. finds all `**/*.md` files;
3. sorts paths deterministically;
4. passes each file’s POSIX path relative to the KB root into `ingest_markdown()`.

For the active root, `officers.md` is used as the source identity rather than an absolute machine-specific path. This is why the same KB can produce the same IDs from different current working directories.

## How embeddings and retrieval work

### Embedding provider seam

`EmbeddingProvider` is a small protocol with one method:

```python
embed(texts: list[str]) -> numpy.ndarray
```

The production implementation, `SentenceTransformerProvider`, constructs `sentence-transformers/all-MiniLM-L6-v2` and normalizes the returned vectors. The import is lazy so importing the package does not immediately load the external model.

Tests provide deterministic fake providers instead. This is important: retrieval mechanics can be tested without pretending that a fake vector is a production-quality semantic model.

### Normalization and cosine similarity

`normalize_vectors()` divides each vector by its L2 norm. Once both chunk vectors and the question vector are normalized, their dot product is cosine similarity:

```text
score = normalized_chunk_vector · normalized_question_vector
```

Zero vectors raise `ValueError` rather than producing invalid numbers. The production provider and `Retriever` normalize more than once; that is harmless but redundant and is not currently a reason to redesign the system.

### Retriever lifecycle

When `Retriever` is constructed, it embeds every chunk once. On each search it embeds the question, calculates scores against all stored chunk vectors, and sorts by:

1. descending score;
2. ascending canonical chunk ID as a deterministic tie-breaker.

`top_k` must be positive. Search is brute-force over the complete in-memory chunk list. There is no persistence, vector database, reranker, score threshold, metadata filter, or approximate index.

## How generation works

`generation.py` defines:

- `GenerationRequest(question, contexts)`;
- `LocalLlamaCppGenerator`;
- the default local GGUF path.

The generator validates that the GGUF file exists, but does not construct the model until the first call to `generate()`. It then caches the loaded model. Tests inject a fake factory to observe loading and generation parameters without loading a real model.

The prompt includes context labels in retrieval order:

```text
<context>
[S1] first retrieved chunk
[S2] second retrieved chunk
</context>
Answer the question using only the supplied context.
...
Question: ...
```

The current request-local labels are separate from canonical chunk IDs. Generation uses temperature `0.0`, seed `42`, and a default maximum of `256` tokens.

The prompt asks for citations, but a prompt instruction is not a guarantee that a small model will follow it. See the assistant section for the current validation behavior.

## How the assistant CLI works

`assistant.py` is the composition root for the current user-facing path:

1. parse `--question`, `--model`, and `--top-k`;
2. load `Path("knowledge_base")`;
3. construct the embedding provider;
4. construct the retriever and search;
5. short-circuit if no results exist;
6. construct the generator;
7. create a `GenerationRequest` and generate;
8. validate structural citations;
9. print the answer or fallback, followed by ranked source diagnostics.

The assistant currently has three injectable factories for tests: embedding provider, generator, and retriever. It does not yet return a structured `AskResult`; its public behavior is just printed text.

### Current citation behavior

`validate_generated_answer()` accepts only exact labels corresponding to current retrieval positions: `[S1]`, `[S2]`, and so on. It rejects:

- no bracketed citation;
- lowercase labels such as `[s1]`;
- leading-zero labels such as `[S01]`;
- labels outside the result range;
- canonical chunk hashes or other arbitrary bracketed content;
- unmatched brackets.

If validation fails, the answer is replaced by:

```text
[answer withheld: citation structural validation failed]
```

Citation enforcement, however, is deferred because the local SmolLM2 model often fails to emit accepted labels even after successful retrieval. The next task is to relax or bypass this gate while keeping source diagnostics and failure visibility. We shouldn't expand the citation parser before that decision.

## Evaluation corpus

The evaluation data lives in `src/chitchat_nao/rag/eval_corpus.json`.

Each case has:

```json
{
  "question": "Who is the faculty advisor?",
  "category": "answerable",
  "relevant_chunk_ids": ["chunk-..."],
  "expected_answer_contains": ["Faculty Advisor"]
}
```

### Categories

`answerable` cases require non-empty `relevant_chunk_ids` and non-empty `expected_answer_contains`.

`unanswered` cases represent plausible questions whose answers are not in the current KB. They must omit both fields.

`ambiguous` cases represent questions that need clarification or have more than one reasonable interpretation. They must also omit both fields.

The loader validates the shape and category rules. The retrieval evaluator uses the relevant chunk IDs for metrics. `expected_answer_contains` records answer-oriented expectations and is validated as data, but the current evaluator does not run generated answers or calculate answer phrase accuracy. Grounded-generation evaluation is future work.

The current corpus contains 10 answerable cases, two unanswered cases, and one ambiguous case. The answerable questions cover the meaningful starter chunks so that no major current fact is completely absent from retrieval evaluation.

## Retrieval metrics

Metrics are calculated for answerable cases only. Unanswered and ambiguous cases appear in reports with `inspection_only=true` and do not affect aggregate retrieval scores.

### Recall@1

For each answerable case, compare the gold `relevant_chunk_ids` with the first retrieved ID. If multiple gold IDs exist, the metric is the number of gold IDs found in the first result divided by the number of gold IDs.

The aggregate Recall@1 is the mean across answerable cases.

### Recall@2

The same calculation using the first two retrieved IDs. This is currently the most important starter-corpus exit check because the assistant normally supplies two chunks to generation.

### MRR

Mean Reciprocal Rank uses the rank of the first retrieved gold chunk:

```text
first relevant at rank 1 -> 1.0
first relevant at rank 2 -> 0.5
no relevant chunk        -> 0.0
```

### Per-case diagnostics

The report also includes:

- `top1_score` and `top2_score`;
- `top2_margin = top1_score - top2_score` when two results exist;
- `gold_hit`, indicating whether a returned result matched a gold ID;
- per-case Recall@1, Recall@2, and reciprocal rank.

The margin is a raw score difference. It is not a calibrated confidence value and is not a gold-aware margin.

### Current baseline

The latest documented real SentenceTransformer run against the three-file starter KB measured:

```text
Recall@1 = 0.700
Recall@2 = 1.000
MRR      = 0.850
```

All 10 answerable cases hit a gold chunk by rank two. These numbers are retrieval results only. They do not show that the model generated a correct answer, refused an unsupported question, clarified ambiguity, or used the evidence semantically.

## Test suite guide

Run the suite with:

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  --with 'llama-cpp-python>=0.3.28' \
  python -m unittest discover -s tests -p 'test_*.py'
```

The tests are intentionally independent of real model inference.

### `test_rag_ingestion.py`

Checks heading/paragraph chunking, stable IDs, duplicate paragraph handling, and rejection of empty documents.

### `test_rag_retrieval.py`

Uses fixed vectors to check ordering, score and metadata propagation, and deterministic chunk-ID tie-breaking.

### `test_rag_generation.py`

Uses a fake Llama object to check prompt delimiters and labels, lazy model loading, deterministic generation arguments, and missing-model validation.

### `test_rag_assistant.py`

Uses fake providers, retrievers, and generators to check factory wiring, output diagnostics, empty-result short-circuiting, and the current citation-validation contract. These tests must change when the citation-gate behavior changes.

### `test_rag_evaluation.py`

Checks CWD-independent IDs, gold-ID validity, evaluation schema rules, metrics, margins, report formatting, category behavior, and starter-corpus coverage.

## Safe knowledge-base update workflow

Use this sequence when adding or editing club information.

### 1. Edit only verified facts

Add or update Markdown under `knowledge_base/`. Keep headings meaningful and use blank lines to separate answer-sized chunks. Do not add placeholder facts to the active corpus.

### 2. Inspect retrieval before changing evaluation labels

Use a focused question:

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  python -m chitchat_nao.rag.evaluator \
  --question "YOUR QUESTION" --top-k 10
```

Check whether the intended chunk is present and whether its text is complete enough to support the question.

### 3. Reconcile chunk IDs

Run the offline tests. If a source edit changed a gold chunk ID, update the matching `relevant_chunk_ids` in `eval_corpus.json` using the newly produced canonical ID. Keep the KB edit and evaluation update together.

Do not manually guess a hash. Use the ingestion/evaluation output or a small Python inspection script so the ID is generated by the same code path.

### 4. Check evaluation semantics

For every new answerable question:

- identify at least one relevant chunk ID;
- provide answer-focused expected phrases;
- make sure the gold chunk actually contains the expected facts;
- ensure the question is represented by a realistic phrasing.

For unanswered or ambiguous questions, omit gold IDs and expected phrases. Remember that those categories are currently inspection-only and do not make the assistant refuse or clarify automatically.

### 5. Run the full retrieval evaluation

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  python -m chitchat_nao.rag.evaluator
```

Compare the result with the current baseline, and investigate any meaningful regression rather than hiding it with a threshold or top-k change.

### 6. Test the real answer path when relevant

If the change affects generation or prompt context, run the assistant command from [`overview.md`](overview.md) with known, unsupported, and ambiguous questions. Retrieval passing is not enough evidence for a correct answer path.

### 7. Review the complete diff

Before committing a KB/evaluation change:

```bash
git status --short
git diff --stat
git diff --check
```

Inspect source, KB, evaluation, and tests together. Never stage local GGUF files, virtual environments, caches, or unrelated edits.

## What is intentionally not here yet

The current implementation does not include:

- a structured `AskResult` contract;
- a tested refusal policy for unsupported questions;
- ambiguity clarification;
- semantic answer-grounding evaluation;
- causal nonce-corpus tests;
- a score threshold or calibrated confidence;
- a vector database or persistent index;
- reranking;
- FastAPI or another backend API;
- ASR, NAO speech, or robot integration;
- conversation memory or production observability.


## Maintenance checklist

For a normal code or KB change:

- [ ] Read the relevant source and tests before editing.
- [ ] Keep the generator dependent on retrieved contexts, not direct KB access.
- [ ] Keep KB and `eval_corpus.json` changes synchronized when IDs change.
- [ ] Run focused tests, then the full offline suite when pipeline behavior changes.
- [ ] Run retrieval evaluation after corpus or retrieval changes.
- [ ] Manually inspect real CLI output after generation changes.
- [ ] Do not claim semantic grounding from structural citations alone.
- [ ] Keep unanswered and ambiguous behavior described as incomplete until policy tests exist.
- [ ] Inspect Git status and the full diff before staging.
