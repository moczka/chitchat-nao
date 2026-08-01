# RAG System - Evaluation and Maintenance Guide

This guide is for contributors who need to understand, test, or safely modify the current RAG implementation. It assumes basic Python but little prior ML/RAG experience.

Start with [`overview.md`](overview.md) if you want the shorter conceptual explanation and quick-start commands first.

## Source-of-truth boundaries

The current RAG system has five related kinds of files:

1. **Knowledge sources** under `knowledge_base/` — the club facts that can be retrieved.
2. **Pipeline code** under `src/chitchat_nao/rag/` — ingestion, embeddings, retrieval, routing, generation, and CLI/evaluation orchestration.
3. **Evaluation labels** in `src/chitchat_nao/rag/eval_corpus.json` — questions and gold chunk IDs used to measure retrieval.
4. **Tests** under `tests/` — deterministic checks of the pipeline mechanics.
5. **Synthetic fixtures** under `tests/fixtures/` — isolated fake Markdown corpora and fake scenarios used only by tests.

The knowledge sources and evaluation labels are tightly coupled through deterministic chunk IDs. Treat changes to them as one review unit.

The synthetic fixtures must stay separate from the production KB and from the real evaluation metrics. They use clearly fake content (`synthetic://` source paths, `synthetic_*.md` filenames) so that a mistake can never be mistaken for a real club fact or inflate the real corpus evaluation.

## Module map

| File | Responsibility | Input | Output |
|---|---|---|---|
| `models.py` | Shared immutable data contracts | Field values | `DocumentChunk`, `RetrievedContext`, `AskResult`, `ResponseMode` |
| `ingest.py` | Parse Markdown into chunks | One Markdown path | `list[DocumentChunk]` |
| `embedding.py` | Convert text to normalized vectors | Text list | NumPy array |
| `retrieval.py` | Rank chunks by cosine similarity | Chunks and question | `list[RetrievedContext]` |
| `generation.py` | Ask local GGUF model for text | Question, selected contexts, response mode | Answer string |
| `assistant.py` | `ask()` routing policy, speech cleanup, diagnostics, CLI | Question and factories | `AskResult` |
| `evaluator.py` | Load KB and score retrieval | KB, eval corpus, questions | Reports or inspection output |
| `eval_corpus.json` | Retrieval labels | Questions and gold IDs | Input to evaluator |

The main data flow is:

```text
Markdown files
  -> DocumentChunk
  -> normalized chunk embeddings
  -> Retriever.search(question)
  -> RetrievedContext
  -> ask() routing decision (ANSWER / CLARIFY / REDIRECT / ERROR)
  -> GenerationRequest (only for ANSWER and CLARIFY)
  -> local model output -> cleaned speech text
  -> AskResult
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

The chunk's `content_hash` is the SHA-256 hash of the chunk text. The canonical chunk ID is another SHA-256 hash over:

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
4. passes each file's POSIX path relative to the KB root into `ingest_markdown()`.

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

- `GenerationRequest(question, contexts, response_mode)`;
- `LocalLlamaCppGenerator`;
- the default local GGUF path.

The generator validates that the GGUF file exists, but does not construct the model until the first call to `generate()`. It then caches the loaded model. Tests inject a fake Llama factory to observe loading and generation parameters without loading a real model.

The prompt includes context labels in retrieval order:

```text
<context>
[S1] first retrieved chunk
[S2] second retrieved chunk
</context>
...
Question: ...
```

The instruction depends on the response mode:

- `ANSWER` asks for one concise, speech-ready answer using only the supplied context;
- `CLARIFY` asks for one concise clarifying question using only the supplied context, and explicitly tells the model not to answer or add facts.

Generation uses temperature `0.0`, seed `42`, and a default maximum of `256` tokens. The current request-local labels are separate from canonical chunk IDs.

For `CLARIFY`, `assistant.py` deliberately sends only `Source:`/`Section:` text per chunk (no chunk contents), so a clarifying question cannot leak or repeat underlying facts.

## How the `ask()` pipeline works

`ask()` in `assistant.py` is the typed, stateless integration seam. A caller passes a question and optional factories, and receives an `AskResult`. The sequence is:

1. **Validate input.** A missing, non-string, empty, or whitespace-only question returns `ERROR` before any work. `top_k` must be an `int >= 1` and `clarification_attempts` an `int >= 0`; anything else also returns `ERROR`. No retrieval or generation is attempted.
2. **Load the KB and retrieve.** `load_knowledge_base(Path("knowledge_base"))`, construct the embedding provider and retriever, search with `top_k`. Any exception returns `ERROR` with a `Retrieval setup error:` diagnostic.
3. **Check protected requests.** Questions asking for passwords, secrets, API keys, or hidden system prompts (a small regex check) return `REDIRECT` without generation.
4. **Find direct textual support.** A deliberately small matcher checks whether any retrieved chunk *directly* contains the question's facts: an exact normalized label match (for example `President - ...` for "Who is the president?"), a documented `Question N:` line containing all query terms, or a contiguous normalized phrase match. This is generic token/phrase logic, not an NLP classifier or re-ranker.
5. **Route.** See the routing rules below.
6. **Generate (only for `ANSWER` and `CLARIFY`).** The generator receives only the selected support contexts, or only source/section metadata for clarification. Generator exceptions return `ERROR` with a `Generator error:` diagnostic plus the evidence that was available; a non-string return value is also `ERROR`.
7. **Clean the speech text.** For `ANSWER`, strip `[S<n>]` labels and outer `<response>` tags, normalize whitespace/punctuation, and keep at most three sentences. For `CLARIFY`, keep the model output only if it is a single question that is not an echo of the user's question; otherwise use the neutral fallback `Which club detail would you like to clarify?`.
8. **Attach citation diagnostics.** Citation formatting is validated structurally only (see below).
9. **Return `AskResult`.**

### Routing rules

| Situation | Route |
|---|---|
| Invalid input (bad question, `top_k`, or attempts) | `ERROR` |
| Request asks for passwords, secrets, or hidden system prompts | `REDIRECT` |
| Exactly one chunk gives unique direct textual support (even at rank 2) | `ANSWER` using only that chunk |
| More than one chunk gives competing direct support | `CLARIFY` |
| No direct support and top score < 0.35 | `REDIRECT` |
| No direct support, top score >= 0.60, and (single result or top1 - top2 margin > 0.08) | `ANSWER` using rank-1 evidence only |
| No direct support, otherwise (weak or competing semantics) | `CLARIFY` |
| `CLARIFY` would be chosen but attempts >= 2 | `REDIRECT` (suggest an officer) |
| Setup, retrieval, or generator failure; non-string generator output | `ERROR` |

Thresholds are named constants in `assistant.py`:

```text
LOW_CONFIDENCE_SCORE_THRESHOLD = 0.35
CLARIFY_SCORE_THRESHOLD        = 0.60
CLARIFY_MARGIN_THRESHOLD       = 0.08
MAX_CLARIFICATION_ATTEMPTS     = 2
```

The clarification budget is **caller-owned**: `ask()` never changes the attempt count. If a caller sees `CLARIFY`, it should re-ask with `clarification_attempts + 1`, and stop after two attempts because the third call returns `REDIRECT`.

### Current citation behavior

`validate_generated_answer()` checks only structure: are all bracketed citations exact current-result labels such as `[S1]`/`[S2]`? It rejects no citation, lowercase labels, leading-zero labels, out-of-range labels, canonical chunk hashes, and unmatched brackets. This result is **diagnostic-only**:

- the answer is never withheld for citation-formatting problems;
- `AskResult.provenance_verified` is always `False`;
- the diagnostic reads either `Citation formatting was structurally valid, but semantic grounding was not verified.` or `Citation formatting was missing or invalid; semantic grounding was not verified.`

A structurally valid label proves only that the model emitted an allowed label. It does not prove semantic support.

## The `AskResult` contract

`AskResult` is a frozen dataclass:

| Field | Meaning |
|---|---|
| `mode` | `ResponseMode` (`ANSWER`, `CLARIFY`, `REDIRECT`, `ERROR`) |
| `spoken_text` | Speech-ready string |
| `evidence` | Tuple of `RetrievedContext` actually selected/supplied |
| `provenance_verified` | Always `False` in the current implementation |
| `diagnostics` | Tuple of human-readable strings |
| `clarification_attempts` | Echoed back from the caller |

For `ANSWER`, `evidence` contains only the chunks that supported the answer (direct support, or rank-1 when answering from the semantic-score rule). For `CLARIFY` and `REDIRECT`, `evidence` contains the retrieved context. For input-validation `ERROR`, `evidence` is empty; for setup failures it is empty; for generation failures it retains whatever was retrieved.

The convenience export `chitchat_nao.rag.ask()` (in `__init__.py`) lazily dispatches to `assistant.ask()`, so robot-side integration can import one function.

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

**Correction note:** the two president-related entries (`Who is the president?` and `Who leads the Computer Club as president?`) had their gold IDs corrected to the current canonical president chunk ID:

```text
chunk-472fc9f75cc6e4b5e4d44263759f8d18e9b3b1602184f63287fc2d22b02eeecc
```

Always regenerate IDs from the actual KB output instead of guessing hashes (see the update workflow).

## Retrieval metrics

Metrics are calculated for answerable cases only. Unanswered and ambiguous cases appear in reports with `inspection_only=true` and do not affect aggregate retrieval scores.

### Recall@1

For each answerable case, compare the gold `relevant_chunk_ids` with the first retrieved ID. If multiple gold IDs exist, the metric is the number of gold IDs found in the first result divided by the number of gold IDs.

The aggregate Recall@1 is the mean across answerable cases.

### Recall@2

The same calculation using the first two retrieved IDs. This is the most important starter-corpus exit check because the retriever normally returns two chunks, one of which must support the answer.

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
Recall@1 = 0.600
Recall@2 = 1.000
MRR      = 0.800
```

All 10 answerable cases hit a gold chunk by rank two. These numbers are retrieval results only. They do not show that the model generated a correct answer, refused an unsupported question, clarified ambiguity, or used the evidence semantically.

Note that the routing policy does **not** use these gold labels: `ask()` decides from scores and direct textual matching alone. Retrieval metrics and routing behavior are evaluated separately.

## Test suite guide

Run the suite with:

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  --with 'llama-cpp-python>=0.3.28' \
  python -m unittest discover -s tests -p 'test_*.py'
```

The tests are intentionally independent of real model inference. The suite currently contains **77 tests**, and `ruff check src tests` plus `git diff --check` pass cleanly.

### `test_rag_ingestion.py`

Checks heading/paragraph chunking, stable IDs, duplicate paragraph handling, and rejection of empty documents.

### `test_rag_retrieval.py`

Uses fixed vectors to check ordering, score and metadata propagation, and deterministic chunk-ID tie-breaking.

### `test_rag_generation.py`

Uses a fake Llama object to check prompt delimiters and labels, the mode-dependent instructions (`ANSWER` vs `CLARIFY`), lazy model loading, deterministic generation arguments, and missing-model validation.

### `test_rag_assistant.py`

Uses fake providers, retrievers, and generators to check CLI wiring, output ordering (speech first, then provenance/diagnostics/ranked sources), empty-result redirect behavior, the diagnostic-only citation behavior, and generator-error reporting.

### `test_rag_core_contract.py`

The contract test for `ask()`. Uses an isolated JSON scenario set (`tests/fixtures/synthetic_rag_scenarios.json`, 27 scenarios across categories such as direct facts, paraphrases, competing/ambiguous evidence, weak relevance, unrelated, adversarial-unanswered, and nonce-fact contrast). It checks:

- `AskResult` is frozen and typed, and `ResponseMode` covers exactly the four modes;
- the routing rules above, including rank-2 direct support, exact-label preference, competing-direct-support clarification, the >= 0.60 + margin > 0.08 semantic rule, and the < 0.35 redirect;
- invalid input (question, `top_k`, `clarification_attempts`) returns `ERROR` without touching retrieval/generation;
- protected requests redirect without generation;
- clarification sends source/section-only contexts and falls back to `Which club detail would you like to clarify?` for non-questions or echoed questions;
- the caller-managed two-attempt budget (attempts 0 and 1 clarify, attempts >= 2 redirect);
- speech cleanup (tags/citations stripped, three-sentence cap);
- generator exceptions and non-string generator output return `ERROR` with visible diagnostics;
- an end-to-end policy test over the isolated Markdown corpus in `tests/fixtures/synthetic_retrieval_corpus/` using a deterministic word-overlap embedding provider.

### `test_rag_evaluation.py`

Checks CWD-independent IDs, gold-ID validity, evaluation schema rules, metrics, margins, report formatting, category behavior, and starter-corpus coverage. It also verifies that the isolated synthetic retrieval corpus (`tests/fixtures/synthetic_retrieval_corpus/`) is separate from the production `knowledge_base/` and that its answerable cases all hit gold by rank two.

## Synthetic test data rules

- Keep all fake data under `tests/fixtures/`. The two current fixtures are `synthetic_rag_scenarios.json` (scenario contracts for `ask()`) and `synthetic_retrieval_corpus/` (Markdown + eval labels for end-to-end policy tests).
- Use obviously fake content (`synthetic://` source paths, `synthetic_*.md` names, invented names such as Mira Vale). Never put real club facts in fixtures, and never point fixtures at `knowledge_base/`.
- Synthetic evaluation runs in tests must never be reported as production retrieval metrics, and real KB edits must never be validated only against synthetic data.

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

For unanswered or ambiguous questions, omit gold IDs and expected phrases. Remember that those categories remain inspection-only for metrics, even though the `ask()` policy now has separate routing behavior (redirect/clarify) that is tested through the synthetic fixtures.

### 5. Run the full retrieval evaluation

```bash
PYTHONPATH=src uv run --no-project --isolated \
  --with 'numpy>=2' \
  --with 'sentence-transformers>=5.1.2' \
  python -m chitchat_nao.rag.evaluator \
  --knowledge-base knowledge_base --top-k 2
```

Compare the result with the current baseline, and investigate any meaningful regression rather than hiding it with a threshold or top-k change.

### 6. Test the real answer path when relevant

If the change affects generation, routing, or prompt context, run the assistant command from [`overview.md`](overview.md) with known, unsupported, and ambiguous questions. Retrieval passing is not enough evidence for a correct answer path. Remember that `.venv` may lack `sentence-transformers`; use the isolated commands.

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

- semantic answer-grounding evaluation (citations are diagnostic-only; `provenance_verified` is always `False`);
- a calibrated confidence model (thresholds 0.35/0.60/0.08 are heuristics);
- a vector database or persistent index;
- reranking;
- FastAPI or another backend API;
- ASR, NAO speech, or robot integration (the `AskResult` contract is the intended seam);
- conversation memory or production observability.

Routing, validation, and the structured `AskResult` contract are implemented and tested; do not re-add earlier "not yet" wording for those.

## Maintenance checklist

For a normal code or KB change:

- [ ] Read the relevant source and tests before editing.
- [ ] Keep the generator dependent on retrieved/selected contexts, not direct KB access.
- [ ] Keep KB and `eval_corpus.json` changes synchronized when IDs change.
- [ ] Keep synthetic fixtures under `tests/fixtures/`, clearly fake, and separate from production KB/metrics.
- [ ] Run focused tests, then the full offline suite (currently 77 tests) when pipeline behavior changes.
- [ ] Run retrieval evaluation after corpus or retrieval changes.
- [ ] Manually inspect real CLI output after generation or routing changes.
- [ ] Do not claim semantic grounding from structural citations alone; `provenance_verified` is always `False`.
- [ ] If routing thresholds change, update `test_rag_core_contract.py` and re-run the synthetic scenarios.
- [ ] Inspect Git status and the full diff before staging.
