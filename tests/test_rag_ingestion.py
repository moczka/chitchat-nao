import tempfile
import unittest
from pathlib import Path

from chitchat_nao.rag.ingest import ingest_markdown


class MarkdownIngestionTests(unittest.TestCase):
    def test_heading_aware_chunks_preserve_source_and_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "club.md"
            path.write_text(
                "# Club\n\nWelcome to the club.\n\n"
                "## Officers\n\nThe board leads.\n",
                encoding="utf-8",
            )

            chunks = ingest_markdown(path)

        self.assertEqual(
            [chunk.section for chunk in chunks], ["Club", "Club > Officers"]
        )
        self.assertEqual(
            [chunk.source_path for chunk in chunks], ["club.md", "club.md"]
        )
        self.assertEqual(chunks[0].text, "Welcome to the club.")

    def test_chunk_ids_are_deterministic_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.md"
            second = Path(directory) / "first.md"
            first.write_text("# Heading\n\nStable text.\n", encoding="utf-8")
            first_chunks = ingest_markdown(first)
            second_chunks = ingest_markdown(second)

        self.assertEqual(first_chunks[0].id, second_chunks[0].id)
        self.assertTrue(first_chunks[0].id.startswith("chunk-"))
        self.assertEqual(len(first_chunks[0].content_hash), 64)

    def test_empty_document_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.md"
            path.write_text("# Heading\n\n  \n", encoding="utf-8")

            with self.assertRaises(ValueError):
                ingest_markdown(path)

    def test_blank_paragraphs_make_separate_chunks_within_a_section(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faq.md"
            path.write_text(
                "# FAQ\n\nFirst answer.\n\nSecond answer.\n", encoding="utf-8"
            )

            chunks = ingest_markdown(path)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["First answer.", "Second answer."],
        )
        self.assertEqual([chunk.section for chunk in chunks], ["FAQ", "FAQ"])

    def test_duplicate_paragraphs_have_distinct_reproducible_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faq.md"
            path.write_text(
                "# FAQ\n\nRepeated answer.\n\nRepeated answer.\n",
                encoding="utf-8",
            )

            first = ingest_markdown(path)
            second = ingest_markdown(path)

        self.assertEqual(
            [chunk.text for chunk in first], ["Repeated answer."] * 2
        )
        self.assertEqual(
            [chunk.id for chunk in first], [chunk.id for chunk in second]
        )
        self.assertEqual(len({chunk.id for chunk in first}), 2)
        self.assertEqual(first[0].content_hash, first[1].content_hash)
