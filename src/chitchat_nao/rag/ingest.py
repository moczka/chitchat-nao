import hashlib
import re
from pathlib import Path

from .models import DocumentChunk

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def ingest_markdown(
    path: Path, source_path: str | None = None
) -> list[DocumentChunk]:
    """Read Markdown into deterministic, non-empty section chunks."""
    raw = path.read_text(encoding="utf-8")
    display_path = source_path or path.name
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    current_lines: list[str] = []
    current_section = "Document"

    def finish() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_section, text))
        current_lines.clear()

    for line in raw.splitlines():
        match = _HEADING.match(line)
        if match:
            finish()
            level, title = len(match.group(1)), match.group(2).strip()
            heading_stack[:] = [
                (existing_level, name)
                for existing_level, name in heading_stack
                if existing_level < level
            ]
            heading_stack.append((level, title))
            current_section = " > ".join(name for _, name in heading_stack)
        elif line.strip():
            current_lines.append(line)
        elif current_lines:
            finish()
    finish()

    if not sections:
        raise ValueError(f"Markdown document has no content: {path}")

    chunks: list[DocumentChunk] = []
    for index, (section, text) in enumerate(sections):
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identity = f"{display_path}\n{section}\n{index}\n{text}".encode(
            "utf-8"
        )
        chunk_id = f"chunk-{hashlib.sha256(identity).hexdigest()}"
        chunks.append(
            DocumentChunk(chunk_id, text, display_path, section, content_hash)
        )
    return chunks
