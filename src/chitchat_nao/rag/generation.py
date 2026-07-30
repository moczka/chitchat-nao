"""Local, context-bounded answer generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import RetrievedContext

DEFAULT_MODEL_PATH = (
    Path.home()
    / ".local/share/chitchat-nao/models/SmolLM2-1.7B-Instruct-Q6_K.gguf"
)


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    contexts: list[RetrievedContext]


class LocalLlamaCppGenerator:
    """Generate an answer using a local llama.cpp model."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        *,
        llama_factory: Callable[[], object] | None = None,
        max_tokens: int = 256,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self._llama_factory = llama_factory
        self._max_tokens = max_tokens
        self._model: object | None = None

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        if self._llama_factory is not None:
            self._model = self._llama_factory()
        else:
            from llama_cpp import Llama

            self._model = Llama(model_path=str(self.model_path), verbose=False)
        return self._model

    @staticmethod
    def _prompt(request: GenerationRequest) -> str:
        context_lines = "\n".join(
            f"[S{position}] {context.text}"
            for position, context in enumerate(request.contexts, start=1)
        )
        return (
            "<context>\n"
            f"{context_lines}\n"
            "</context>\n"
            "Answer the question using only the supplied context.\n"
            "Include at least one citation in your answer. "
            "Use one or more exact source labels in the form [S<n>], "
            "such as [S1]. Only cite labels present in the supplied context. "
            "Do not use canonical chunk IDs or any other citation format.\n"
            f"Question: {request.question}"
        )

    def generate(self, request: GenerationRequest) -> str:
        model = self._load_model()
        response = model.create_chat_completion(  # type: ignore[attr-defined]
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions from retrieved context.",
                },
                {"role": "user", "content": self._prompt(request)},
            ],
            temperature=0.0,
            seed=42,
            max_tokens=self._max_tokens,
        )
        return response["choices"][0]["message"]["content"]  # type: ignore[index]
