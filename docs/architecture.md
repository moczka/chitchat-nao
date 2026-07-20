# Chitchat NAO Architecture

## Purpose

Chitchat NAO is a modular AI assistant for a NAO6 humanoid robot. The robot acts as the embodied input/output device, while a laptop-hosted backend handles speech processing, retrieval-augmented question answering, model inference, identity/session state, and evaluation.

The first usable mode is CLI-first simulation:

```text
Human: When does Computer Club meet?
Robot says: "Computer Club meets every Friday..."
```

This keeps development possible without constant access to the physical robot.

## Core Goals

- Let users ask natural-language questions about Computer Club.
- Answer from a source-grounded RAG knowledge base.
- Support both CLI simulation and physical NAO6 output.
- Keep AI models swappable between local and hosted providers.
- Support future identity-aware interaction through face, voice, or manual session state.
- Avoid accidentally processing unrelated background conversations.

## High-Level System

```text
                    ┌────────────────────────────┐
                    │        CLI / Demo UI        │
                    │  "Robot says: ..." mode     │
                    └──────────────┬─────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                         AI Backend Service                          │
│                                                                     │
│  Conversation orchestration                                          │
│  Speech input pipeline                                               │
│  Active-speaker / interaction gating                                 │
│  Identity/session state                                              │
│  RAG retrieval                                                       │
│  Swappable answer generation providers                               │
│  Logging, metrics, and evaluations                                   │
└───────────────┬───────────────────────────────┬────────────────────┘
                │                               │
                │                               │
┌───────────────▼──────────────┐   ┌────────────▼─────────────┐
│      RAG Knowledge Base      │   │       Robot Adapter       │
│  docs, chunks, embeddings,   │   │  CLI adapter first        │
│  metadata, eval questions    │   │  NAOqi adapter later      │
└──────────────────────────────┘   └────────────┬─────────────┘
                                                │
                                  ┌─────────────▼─────────────┐
                                  │        Physical NAO6       │
                                  │ speech, camera, gesture    │
                                  └───────────────────────────┘
```

## Architectural Principles

1. **Robot-agnostic backend**
   The backend should not depend directly on NAOqi (e.g. doesn't require the robot to be physically present). It should speak to an abstracted robot interface. Every core feature should be testable without the physical robot.

2. **Laptop as the brain, NAO as the body**
   The laptop runs ASR, RAG, generation, logging, and evaluation. NAO provides embodied I/O.

3. **Stable contracts over model-specific code**
   RAG, model generation, ASR, VAD, and robot output should communicate through normalized request/response objects.

4. **Measured behavior**
   The project should track retrieval accuracy, hallucination/refusal behavior, ASR latency, wrong-speaker captures, and end-to-end response time.

## Main Components

### 1. Robot Adapter

The robot adapter exposes a common interface for CLI, mock, and physical NAO modes.

```python
class RobotIO:
    def say(self, text: str) -> None: ...
    def listen(self) -> object | None: ...
    def get_camera_frame(self) -> object | None: ...
    def set_eye_color(self, color: str) -> None: ...
    def gesture(self, name: str) -> None: ...
```

Implementations:

```text
CliRobotIO      # prints "Robot says: ..."
MockRobotIO     # deterministic tests
NaoqiRobotIO    # physical NAO bridge
```

The NAOqi adapter should remain thin. It should handle robot-specific calls such as text-to-speech, camera access, LEDs, gestures, and sound-localization signals. It should not own RAG, ASR, or answer generation.

### 2. AI Backend

The backend coordinates the full interaction loop:

```text
input event
  -> interaction gate
  -> speech processing or CLI text
  -> RAG retrieval
  -> answer generation
  -> response policy
  -> robot output
  -> logs / metrics
```

The orchestrator is the only component that should understand the full pipeline.

### 3. Model Providers

Answer generation must be provider-agnostic.

```python
class GeneratorProvider:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
```

Potential implementations:

```text
LocalLlamaCppGenerator
HostedApiGenerator
MockGenerator
RuleBasedGenerator
```

The RAG system should retrieve context before the generator is called. Model providers should receive a normalized prompt/context bundle rather than performing retrieval themselves.

### 4. RAG Knowledge Base

The RAG system should handle:

```text
source ingestion
  -> document normalization
  -> metadata preservation
  -> chunking
  -> embeddings
  -> retrieval
  -> optional reranking
  -> context packing
  -> answer grounding
```

Expected sources include club schedules, officer lists, event docs, onboarding material, project descriptions, FAQs, and meeting notes.

Answers should be grounded in retrieved sources. When evidence is weak or absent, the system should refuse or ask for clarification rather than inventing facts.

### 5. Speech Input Pipeline

The speech system should not immediately transcribe all room audio. It should gate audio before ASR.

Recommended pipeline:

```text
audio input
  -> interaction gate
  -> VAD
  -> optional direction / face / session checks
  -> ASR
  -> transcript object
```

Early modes:

```text
CLI text input
Push-to-talk laptop microphone
Wake phrase + short listening window
NAO-assisted directional listening
```

Crowd handling should be treated as active-speaker gating, not full speaker separation. The system should prefer asking the user to repeat over answering the wrong person.

### 6. Identity and Session State

Identity should be progressive and probabilistic.

Recommended order:

```text
manual CLI identity
short-lived active-speaker session
NAO face recognition or external face embeddings
optional speaker verification later
face + voice fusion only as stretch
```

The system should ideally support unknown users and low-confidence identity states.

## Core Data Contracts

### `UserUtterance`

```python
@dataclass
class UserUtterance:
    text: str
    input_mode: Literal["cli", "laptop_mic", "nao_mic"]
    asr_confidence: float | None
    speaker_session_id: str | None
    source_direction: float | None
    source_confidence: float | None
    overlap_detected: bool
```

### `RetrievedContext`

```python
@dataclass
class RetrievedContext:
    id: str
    text: str
    source_title: str
    source_path: str
    section: str | None
    score: float
    metadata: dict
```

### `GenerationRequest`

```python
@dataclass
class GenerationRequest:
    user_query: str
    system_instructions: str
    retrieved_contexts: list[RetrievedContext]
    conversation_history: list[object]
    user_identity: object | None
    generation_config: dict
```

### `GenerationResult`

```python
@dataclass
class GenerationResult:
    text: str
    model_name: str
    provider: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    refusal: bool
    cited_context_ids: list[str]
    raw_metadata: dict
```

### `RobotResponse`

```python
@dataclass
class RobotResponse:
    speech_text: str
    display_text: str
    gesture: str | None
    eye_color: str | None
```

## Key Risks

- NAOqi Python compatibility requires isolating the physical robot bridge from the modern Python backend.
- Always-on listening is likely to capture unrelated room speech unless intentionally gated.
- Face and voice recognition should not be treated as reliable identity without checking confidence.
- Small local models may need extra work to make "smart".

## Evaluation Targets

The project should eventually report:

```text
RAG retrieval recall
answer correctness
hallucination / unsupported claim rate
refusal accuracy on unanswerable questions
ASR latency
wrong-speaker capture rate
Voice activity detection (VAD) false activation rate
physical demo success rate
```
