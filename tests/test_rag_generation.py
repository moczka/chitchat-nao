import tempfile
import unittest
from pathlib import Path

from chitchat_nao.rag.generation import (
    DEFAULT_MODEL_PATH,
    GenerationRequest,
    LocalLlamaCppGenerator,
)
from chitchat_nao.rag.models import RetrievedContext


class GenerationTests(unittest.TestCase):
    def test_generator_sends_only_supplied_context_and_returns_text(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        class FakeLlama:
            def create_chat_completion(
                self, **kwargs: object
            ) -> dict[str, object]:
                captured.update(kwargs)
                return {
                    "choices": [{"message": {"content": " grounded answer "}}]
                }

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            first_chunk_id = "chunk-" + "a" * 64
            second_chunk_id = "chunk-" + "b" * 64
            generator = LocalLlamaCppGenerator(
                model_path, llama_factory=FakeLlama
            )
            answer = generator.generate(
                GenerationRequest(
                    "Who can join?",
                    [
                        RetrievedContext(
                            first_chunk_id,
                            "Anyone may join.",
                            "faq.md",
                            "FAQ",
                            0.9,
                            1,
                        ),
                        RetrievedContext(
                            second_chunk_id,
                            "Guests may attend events.",
                            "events.md",
                            "Events",
                            0.8,
                            2,
                        ),
                    ],
                )
            )

        self.assertEqual(answer, " grounded answer ")
        messages = captured["messages"]
        prompt = str(messages)
        self.assertIn("[S1] Anyone may join.", prompt)
        self.assertIn("[S2] Guests may attend events.", prompt)
        self.assertLess(prompt.index("[S1]"), prompt.index("[S2]"))
        self.assertNotIn(first_chunk_id, prompt)
        self.assertNotIn(second_chunk_id, prompt)
        self.assertIn("Anyone may join.", prompt)
        self.assertIn("at least one citation", prompt)
        self.assertIn("[S<n>]", prompt)
        self.assertIn(
            "Only cite labels present in the supplied context", prompt
        )
        self.assertNotIn("knowledge_base/", prompt)

    def test_model_is_loaded_once_and_only_on_generation(self) -> None:
        factory_calls = 0

        class FakeLlama:
            def create_chat_completion(
                self, **kwargs: object
            ) -> dict[str, object]:
                return {"choices": [{"message": {"content": "answer"}}]}

        def llama_factory() -> FakeLlama:
            nonlocal factory_calls
            factory_calls += 1
            return FakeLlama()

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            generator = LocalLlamaCppGenerator(
                model_path, llama_factory=llama_factory
            )
            self.assertEqual(factory_calls, 0)
            generator.generate(GenerationRequest("question", []))
            generator.generate(GenerationRequest("question", []))

        self.assertEqual(factory_calls, 1)

    def test_generation_call_is_deterministic_and_bounded(self) -> None:
        captured: dict[str, object] = {}

        class FakeLlama:
            def create_chat_completion(
                self, **kwargs: object
            ) -> dict[str, object]:
                captured.update(kwargs)
                return {"choices": [{"message": {"content": "answer"}}]}

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            generator = LocalLlamaCppGenerator(
                model_path, llama_factory=FakeLlama, max_tokens=37
            )
            generator.generate(GenerationRequest("question", []))

        self.assertEqual(captured["temperature"], 0.0)
        self.assertEqual(captured["seed"], 42)
        self.assertEqual(captured["max_tokens"], 37)
        messages = captured["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("<context>", messages[1]["content"])
        self.assertIn("</context>", messages[1]["content"])

    def test_default_model_path_and_missing_model_validation(self) -> None:
        self.assertEqual(
            DEFAULT_MODEL_PATH,
            Path.home()
            / (
                ".local/share/chitchat-nao/models/"
                "SmolLM2-1.7B-Instruct-Q6_K.gguf"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.gguf"
            with self.assertRaisesRegex(FileNotFoundError, "GGUF model"):
                LocalLlamaCppGenerator(missing, llama_factory=object)
